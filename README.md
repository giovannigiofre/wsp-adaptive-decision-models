# WSP — Adaptive Decision Models

Research on adaptive execution for repeated decision calls in agentic systems. The objective is **cost per correct decision**, not speed alone. The full parent model remains the fallback.

## Research question
Can functional discovery and specialist execution paths, selected adaptively using request complexity, uncertainty, and resource state, reduce decision cost without unacceptable quality loss?

Routing and sparse activation are established directions; WSP does not claim novelty for either alone. Any contribution must be demonstrated experimentally.

## Initial candidate
- Model: [shenjunhao/OpenJev-4B](https://huggingface.co/shenjunhao/OpenJev-4B)
- Code: [OpenJev](https://github.com/ejhshen/OpenJev)
- Dataset: [OpenJevData-140k](https://huggingface.co/datasets/shenjunhao/OpenJevData-140k)

Verify architecture, tokenizer, prompt / output schema, license, immutable revisions, code commit, and checkpoint hashes before implementation. The associated dataset is not an independent test set until provenance and contamination checks establish that it is.

## Status
Bootstrap stage. No experimental results or performance claims yet.

## Structure
- `docs/`: white paper, roadmap, literature.
- `configs/`: versioned configurations.
- `src/`: implementation.
- `experiments/`: reproducible experiment protocols.
- `benchmarks/`: independent workloads and metrics.
- `scripts/`: environment utilities.
- `results/`: compact result summaries.
- `checkpoints/`: manifests only; no weights.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/inspect_environment.py
```

Initial hardware target: Apple Silicon, 24 GB unified memory. Evaluate MPS and CPU first; cloud CUDA is optional. Record hardware, runtime, precision, seeds, revisions, quality, latency, throughput, memory, cost, fallback rate, and failure analysis.

Original code and documentation are MIT-licensed. Third-party models, datasets, and dependencies retain their own terms.
