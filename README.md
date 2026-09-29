# explorers

> **White-box experimentation at scale**: observe, measure and intervene on the internal computation of models, across training and deployment.

`explorers` is the open library of [Machine Exploration](https://github.com/machine-exploration/public), which is building the infrastructure for a science of intelligence. The vision is in the [org README](https://github.com/machine-exploration/public#readme), and the plan in the [roadmap](https://github.com/machine-exploration/public/blob/main/ROADMAP.md).

**Status: pre-alpha.** Nothing is released yet. The API will change.

---

## Layout

`explorers` is the open-source interface of Machine Exploration: read, write and trace the streams of computation inside a model, and describe experiments as studies that run the same way on any backend.

| Part | What it does | Where it is today |
|---|---|---|
| `explorers.core` | Examples, model states, observables, the engine (one forward pass per state), the content-addressed store, analyses | [mechanics](https://github.com/machine-exploration/mechanics) `packages/core` |
| Streams and studies | `read` / `write` / `trace` on named streams; `Study` over models × checkpoints × examples | planned (roadmap E1, E2) |
| Backends | Hugging Face / PyTorch first; then NNsight or TransformerLens | planned (E3) |
| Methods | probes, lenses (the Jacobian lens exists), patching, attribution, sparse autoencoders | partly in [mechanics](https://github.com/machine-exploration/mechanics) `packages/learning` |
| `explorers.populations` | Scenarios, the multi-agent runtime, episodes, labels, detectors | **this repository**, as the prototype package `explorers` |

The next step ([roadmap step 0](https://github.com/machine-exploration/public/blob/main/ROADMAP.md#step-0--one-library--next)) moves all library code into this repository. Until then, do not install this repository next to `mechanics`: its `explorers/__init__.py` hides `explorers.core`.

## What is here today

The agent-side prototype (`explorers.populations` after step 0):

- `explorers.scenario`: scenario format v0 with fail-closed validation.
- `explorers.episode`: the immutable episode model and a JSONL store.
- `explorers.methods`: the `Method` / `Annotation` interface, with `hack_label` and `propagation`.
- `explorers.runtime.verifiers`: compiling a scenario for, and converting traces from, the [verifiers](https://github.com/PrimeIntellect-ai/verifiers) runtime (through the pinned fork [machine-exploration/verifiers](https://github.com/machine-exploration/verifiers)).
- `explorers.exploration` and `explorers.metrics`: the study unit, its report, and Wilson rates.

The first scenario is the Hugging Face incident in miniature: agents on impossible tasks, a shared message board, a sealed sandbox. Because the tasks are impossible, any passing solution is a hack, and a hack that spreads through the board is coordination. The first question: **does anything inside the models signal the coordination before it shows in behaviour, better than text monitors at the same false-positive rate?**

## Development

```bash
uv sync --extra dev
uv run pytest            # GPU tests are skipped by default (marker: gpu)
```

## Rules for scenarios

- **Contained:** scenarios that push agents to hack run with no network, no shared cache and no path between episodes.
- **Not published yet:** such scenarios and their traces are published only after the publication rules are settled.
- **Audit isolation:** oversight monitors run outside the trainer's process and write to an append-only store that the reward code cannot read.

## Contributing

Contributors and coding agents: read [AGENTS.md](AGENTS.md) first. Design notes are in [docs/](docs/).
