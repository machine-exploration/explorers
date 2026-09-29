# Roadmap

The roadmap covers the whole org, so it lives in one place: [machine-exploration/public/ROADMAP.md](https://github.com/machine-exploration/public/blob/main/ROADMAP.md).

The steps that happen in this repository:

- **Step 0 — One library:** this repository becomes the workspace for `explorers.core`, `explorers.learning` and `explorers.populations`.
- **T1, T2, T3 — Training observability (first product line):** read LoRA checkpoints of hosted RL runs, `watch` a run, and the proof that internals warn before evals.
- **P0 — Do small models hack and coordinate?** The first scenario on models from 0.6B to 4B.
- **P1 — Detectors at a matched false-positive rate:** [#1](https://github.com/machine-exploration/explorers/pull/1).
- **P2 — Read activations for episodes.**
- **P3 — v0.1: monitors at inference,** with one assumption-audit table per monitored feature.
- **J1, J2 — The join:** monitors under training, and transfer across models.
