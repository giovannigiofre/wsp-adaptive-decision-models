# Copied from the LLaDA repository (github.com/ML-GSAI/LLaDA), file `generate.py`,
# commit 9182493720ed723ef8031210d85959364e51cbe0.
# Original authors: the LLaDA team (Nie et al., "Large Language Diffusion Models", 2025).
# License: the LLaDA README states "LLaDA-8B-Base and LLaDA-8B-Instruct are under MIT LICENSE";
# the repository has no separate LICENSE file at that commit. Original copyright belongs to
# the LLaDA authors; reproduced here for research use with attribution.
#
# Changes in this repo (marked "# [phase1]"):
#   - kept `add_gumbel_noise`, `get_num_transfer_tokens` and the block/step loop of `generate`
#     unchanged in logic;
#   - dropped the iLLaDA end-of-think boost and `var_generate` (not used here);
#   - added scoring strategies `random`, `confidence` (= upstream `low_confidence`), `margin`,
#     `entropy` via `strategy_scores`;
#   - added instrumentation: per answer position, the step at which it was committed and the
#     max-softmax probability ("commit confidence") of the committed token at that step;
#     optional per-step snapshots of the answer tokens (for GIFs);
#   - returns a dict instead of only the token tensor (`generate_instrumented`).
import torch
import numpy as np
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    # [phase1] MPS has no float64; do the noise in float32 on MPS (temperature>0 is not used here).
    dt = torch.float32 if logits.device.type == 'mps' else torch.float64
    logits = logits.to(dt)
    noise = torch.rand_like(logits, dtype=dt)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


# [phase1] strategy names; `low_confidence` is accepted as an alias of `confidence`.
STRATEGIES = ('random', 'confidence', 'margin', 'entropy')


def strategy_scores(logits, x0, strategy, generator=None):
    """[phase1] Score per position; higher = commit earlier.

    logits: (b, l, V) float; x0: (b, l) argmax/sampled tokens.
    Returns (scores, conf) where conf is the softmax probability of x0 (always computed,
    used for instrumentation) and scores is the strategy's ranking score.
    """
    p = F.softmax(logits.float(), dim=-1)
    conf = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    if strategy in ('confidence', 'low_confidence'):
        scores = conf
    elif strategy == 'random':
        scores = torch.rand(x0.shape, device=x0.device, generator=generator)
    elif strategy == 'margin':
        top2 = torch.topk(p, k=2, dim=-1).values
        scores = top2[..., 0] - top2[..., 1]
    elif strategy == 'entropy':
        logp = torch.log(p.clamp_min(1e-30))
        scores = (p * logp).sum(-1)  # = -entropy; lowest entropy committed first
    else:
        raise NotImplementedError(strategy)
    return scores, conf


def select_transfer(scores, mask_index, k):
    """[phase1] Boolean mask of the top-k scoring masked positions per row (k: (b,) ints)."""
    s = torch.where(mask_index, scores, torch.full_like(scores, -np.inf))
    transfer = torch.zeros_like(mask_index)
    for j in range(s.shape[0]):
        kj = int(k[j])
        if kj > 0:
            _, idx = torch.topk(s[j], k=kj)
            transfer[j, idx] = True
    return transfer


@torch.no_grad()
def generate_instrumented(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128,
                          temperature=0., cfg_scale=0., remasking='confidence', mask_id=126336,
                          record_frames=False, seed=None):
    '''
    Args (as upstream `generate`):
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: [phase1] one of STRATEGIES (or 'low_confidence').
        mask_id: The toke id of [MASK] is 126336.
    [phase1] Returns dict with
        x: (b, L+gen_length) tokens,
        commit_step: (b, gen_length) global step index (0-based) at which each answer position was committed,
        commit_conf: (b, gen_length) softmax prob of the committed token at commit time,
        frames: list of (gen_length,) answer-token arrays after each step (if record_frames),
        nfe: number of model forward passes.
    '''
    gen = None
    if seed is not None and remasking == 'random':
        gen = torch.Generator(device=model.device if model.device.type != 'mps' else 'cpu').manual_seed(seed)

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    P = prompt.shape[1]
    commit_step = torch.full((prompt.shape[0], gen_length), -1, dtype=torch.long)  # [phase1]
    commit_conf = torch.full((prompt.shape[0], gen_length), float('nan'))          # [phase1]
    frames = []                                                                     # [phase1]
    nfe = 0
    global_step = 0

    for num_block in range(num_blocks):
        block_mask_index = (x[:, P + num_block * block_length: P + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                attention_mask_ = None
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits
            nfe += 1

            # [phase1] only the answer region matters for selection; slice to save memory.
            logits = logits[:, P:]
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, gen_length

            scores, conf = strategy_scores(logits, x0, remasking, generator=gen)  # [phase1]
            if gen is not None and scores.device != x0.device:
                scores = scores.to(x0.device)
            scores = scores.float()
            scores[:, (num_block + 1) * block_length:] = -np.inf

            ans_mask = mask_index[:, P:]
            x0 = torch.where(ans_mask, x0, x[:, P:])
            transfer_index = select_transfer(scores, ans_mask, num_transfer_tokens[:, i])
            x[:, P:][transfer_index] = x0[transfer_index]

            # [phase1] instrumentation
            t_cpu = transfer_index.cpu()
            commit_step[t_cpu] = global_step
            commit_conf[t_cpu] = conf.float().cpu()[t_cpu]
            if record_frames:
                frames.append(x[0, P:].cpu().numpy().copy())
            global_step += 1

    return {'x': x, 'commit_step': commit_step, 'commit_conf': commit_conf, 'frames': frames, 'nfe': nfe}


@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    """Upstream-compatible signature: returns only the token tensor."""
    return generate_instrumented(model, prompt, attention_mask, steps, gen_length, block_length,
                                 temperature, cfg_scale, remasking, mask_id)['x']
