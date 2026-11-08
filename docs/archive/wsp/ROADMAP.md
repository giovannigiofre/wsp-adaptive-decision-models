# WSP roadmap

## Phase 0 — Reproducibility and verification
- [ ] Verify architecture, tokenizer, prompt/output schema, license, upstream code.
- [ ] Pin immutable model, code, and dataset revisions; record hashes.
- [ ] Verify Apple Silicon runtime support and document environment.
- [ ] Define independent test data and leakage controls.
**Exit:** reproducible model load and documented inference contract.

## Phase 1 — Full-model baseline
- [ ] Implement canonical request/response schema.
- [ ] Measure quality, calibration, latency, throughput, memory, and cost.
- [ ] Separate cold start and warm execution.
**Exit:** repeatable baseline and failure taxonomy.

## Phase 2 — Functional discovery
- [ ] Inspect actual module graph and operations.
- [ ] Capture activations on development data.
- [ ] Test candidate components with controlled ablations/interventions.
**Exit:** reproducible candidate components or documented negative result.

## Phase 3 — Specialist paths
- [ ] Implement a minimal evidence-backed set.
- [ ] Version paths and document operating regions.
- [ ] Measure quality and resource use against parent model.
**Exit:** validated specialist path or documented failure.

## Phase 4 — Adaptive selection
- [ ] Establish rule-based routing baseline.
- [ ] Add calibrated uncertainty escalation and full-model fallback.
- [ ] Evaluate contextual bandit if independent-request assumptions hold.
**Exit:** fair comparison with fixed and full-model policies.

## Phase 5 — Resource-aware scheduling
- [ ] Measure loading, cache, batching, concurrency, queueing, memory pressure.
- [ ] Add resource features only when useful.
- [ ] Consider sequential RL only with material state transitions.
**Exit:** demonstrated stateful-control benefit or decision to remain stateless.

## Phase 6 — Agentic benchmark
- [ ] Build repeated multi-step decision traces.
- [ ] Test task mixtures and distribution shifts.
- [ ] Report cost per correct decision, quality constraints, ablations, and failures.
**Exit:** reproducible report with explicit limitations.
