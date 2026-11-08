# WSP — Adaptive Decision Models
## Research white paper · v0.1 · 6 November 2026

### Abstract
Agentic systems repeatedly call models to choose tools, rank actions, score alternatives, or select workflow branches. A uniform full-model policy can waste resources on routine decisions while difficult cases still require the full model. WSP investigates whether useful functional components and specialist execution paths can be discovered within a decision model and selected adaptively. The objective is cost per correct decision, with the parent model retained as fallback.

### 1. Research question
Can functional discovery plus adaptive selection of specialist execution paths reduce inference and orchestration cost on repeated agentic decision workloads while preserving decision quality?

### 2. Hypotheses
- H1: requests differ measurably in difficulty, uncertainty, and computational requirements.
- H2: some specialist paths handle identifiable request families at lower cost with acceptable quality.
- H3: an adaptive controller outperforms fixed routing.
- H4: uncertainty gates and full-model fallback recover quality on difficult or unfamiliar inputs.
- H5: state-aware control helps only when actions affect future state such as model loading, cache residency, batching, queues, or memory pressure.

### 3. Scope
In scope: decision-oriented inference, frozen-parent-model analysis, evidence-backed specialist paths, adaptive selection, safe fallback, and repeated-workload evaluation. Initial non-goals: training a foundation model, arbitrary pruning without evidence, and deployment of unsupervised high-impact decisions.

### 4. Initial candidate
`shenjunhao/OpenJev-4B`, associated code OpenJev and dataset OpenJevData-140k. Verify architecture, tokenizer, prompt template, input/output schema, license, upstream code commit, checkpoint revision, runtime support, and file hashes. Treat the associated dataset as development material only until training overlap and provenance are understood.

### 5. Proposed architecture
1. **Request normalizer:** validate and canonicalize task type, context, options, constraints, and output schema.
2. **Functional discovery:** inspect modules, profile activations, analyze representations, and run controlled ablations / interventions.
3. **Execution-path library:** version candidate paths and record supported tasks, model revision, quality, cost, hardware compatibility, and failure modes.
4. **Adaptive controller:** select a path using request features, uncertainty, outcomes, and resource state.
5. **Verification and fallback:** validate schema and confidence; escalate uncertain or out-of-distribution requests to the full model.
6. **Telemetry:** log correctness, latency, memory, queueing, cache effects, and fallback outcomes.

### 6. Learning formulation
Begin with a contextual bandit if each request can be evaluated independently. Use task type, input length, candidate count, uncertainty, and runtime context as features; execution paths are actions. Consider sequential RL (e.g. PPO) only if actions change future system state and delayed reward can be defined. Do not use RL merely because the system is adaptive.

### 7. Experimental plan
- **A — Baseline:** full model; measure quality, calibration, latency distributions, throughput, peak memory, and cost; separate cold and warm runs.
- **B — Discovery:** inspect module graph; capture activations; test candidate components with controlled ablations and interventions.
- **C — Specialists:** implement only evidence-backed paths; compare on identical examples.
- **D — Selection:** compare rule-based routing, confidence escalation, and contextual bandit; tune on validation only.
- **E — Agentic workloads:** repeated traces with realistic mixtures, cold/warm cache, batching, concurrency, and queueing.

### 8. Metrics
Primary: **cost per correct decision** = total inference/orchestration cost divided by number of correct decisions, with accounting boundaries declared. Also report task quality, quality at fixed budget, cost at fixed quality, p50/p95 latency, throughput, peak memory, fallback rate, invalid-output rate, calibration, selective risk, and distribution-shift robustness. Include confidence intervals and failed requests.

### 9. Baselines and ablations
Compare full model, fixed specialist, rule-based complexity routing, confidence-only escalation, adaptive selection without resource features, and adaptive selection with resource features. Ablate discovery, path count, controller context, calibration, fallback, and stateful scheduling. Include simpler precision and batching baselines where relevant.

### 10. Risks
Architecture mismatch; dataset leakage; hidden routing / verification overhead; quality collapse on rare cases; hardware-specific conclusions; unnecessary controller complexity; novelty overclaim. Mitigate through verification, independent splits, end-to-end accounting, fallback, replication, simple baselines, and a systematic literature matrix.

### 11. Compute
Initial target: Apple Silicon laptop, 24 GB unified memory; test MPS and CPU. Do not assume external SSD streaming makes a larger model practical—measure I/O, memory, and latency. Optional cloud CUDA budget: USD 100, with a predefined experimental question and shutdown limit.

### 12. Success criterion
A reproducible improvement in cost per correct decision while meeting a predefined quality floor on independent held-out workloads, with uncertainty estimates, ablations, and failure analysis. A negative result is still informative.

### 13. Immediate next steps
Pin and verify model/code/tokenizer; reproduce full-model baseline; build independent labelled workload; establish cost/quality logging; only then begin functional discovery.
