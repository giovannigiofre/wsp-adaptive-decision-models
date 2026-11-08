"""Heatmaps of commit step / commit confidence and GIFs of the decoding process (step 1.1)."""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MASK_CHAR = "░"  # ░


def commit_heatmap(commit_steps: dict, commit_confs: dict, title: str, path, block_length: int = 32):
    """commit_steps/confs: {strategy_name: 1D array over answer positions}. One row per strategy."""
    names = list(commit_steps)
    L = len(next(iter(commit_steps.values())))
    fig, axes = plt.subplots(2, 1, figsize=(min(14, 2 + L / 12), 1.2 + 0.55 * len(names) * 2), constrained_layout=True)
    for ax, data, cmap, lab in [(axes[0], commit_steps, "viridis", "commit step"),
                                (axes[1], commit_confs, "magma", "commit confidence (max prob)")]:
        M = np.stack([np.asarray(data[n], dtype=float) for n in names])
        im = ax.imshow(M, aspect="auto", cmap=cmap, interpolation="nearest",
                       vmin=0 if "conf" in lab else None, vmax=1 if "conf" in lab else None)
        ax.set_yticks(range(len(names)), names)
        for b in range(block_length, L, block_length):
            ax.axvline(b - 0.5, color="white", lw=0.8, ls="--")
        ax.set_xlabel("answer position (token index)")
        fig.colorbar(im, ax=ax, label=lab, fraction=0.03)
    axes[0].set_title(title, fontsize=10)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _font(size=14):
    for p in ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf",
              "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Courier New.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def decode_frames(tokenizer, frames, mask_id: int) -> list[str]:
    """Convert per-step answer-token arrays into display strings, masks shown as ░."""
    out = []
    for fr in frames:
        pieces = []
        for t in fr:
            if t == mask_id:
                pieces.append(MASK_CHAR)
            else:
                s = tokenizer.decode([int(t)], skip_special_tokens=False)
                if s in ("<|eot_id|>", "<|endoftext|>"):
                    s = "∅"  # ∅ for end-of-text padding
                pieces.append(s)
        out.append("".join(pieces))
    return out


def text_gif(texts: list[str], path, header: str = "", width_chars: int = 80, max_frames: int = 64,
             duration_ms: int = 120):
    """Render frames of text to a small GIF (subsample frames to at most max_frames)."""
    if len(texts) > max_frames:
        idx = np.unique(np.linspace(0, len(texts) - 1, max_frames).round().astype(int))
        texts = [texts[i] for i in idx]
    font = _font(13)
    wrapped = [textwrap.wrap(t.replace("\n", " ↵ "), width_chars) or [""] for t in texts]
    n_lines = max(len(w) for w in wrapped) + 2
    W, lh = 8 * width_chars + 20, 17
    H = n_lines * lh + 16
    imgs = []
    for k, lines in enumerate(wrapped):
        im = Image.new("P", (W, H), 0)
        im.putpalette([255, 255, 255, 20, 20, 20, 120, 120, 120] + [0] * (256 * 3 - 9))
        d = ImageDraw.Draw(im)
        d.text((10, 6), f"{header}  step {k + 1}/{len(wrapped)}", fill=2, font=font)
        for j, ln in enumerate(lines):
            d.text((10, 6 + (j + 1.5) * lh), ln, fill=1, font=font)
        imgs.append(im)
    imgs[0].save(path, save_all=True, append_images=imgs[1:] + [imgs[-1]] * 10, duration=duration_ms,
                 loop=0, optimize=True)
