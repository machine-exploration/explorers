# explorers

> **Measure what happens inside models over time**: across training (learning mechanics) and across turns in agent populations (oversight).

`explorers` is the open library of [Machine Exploration](https://github.com/machine-exploration/public). The vision is in the [org README](https://github.com/machine-exploration/public#readme), and the plan in the [roadmap](https://github.com/machine-exploration/public/blob/main/ROADMAP.md).

**Status: pre-alpha.** Nothing is released yet. The API will change.

---

## Layout

The library has one core and two programs on top of it:

| Package | Clock | What it does | Where it is today |
|---|---|---|---|
| `explorers.core` | both | Examples, states, observables, the engine, the content-addressed store, analyses | [mechanics](https://github.com/machine-exploration/mechanics) `packages/core` |
| `explorers.learning` | training time | Pythia checkpoints, toy tasks, probes, the quanta study | [mechanics](https://github.com/machine-exploration/mechanics) `packages/learning` |
| `explorers.populations` | interaction time | Scenarios, the multi-agent runtime, episodes, labels, detectors | **this repository**, as the prototype package `explorers` |

The next step ([roadmap step 0](https://github.com/machine-exploration/public/blob/main/ROADMAP.md#step-0--one-library--next)) moves all three into this repository as one workspace. Until then, do not install this repository next to `mechanics`: its `explorers/__init__.py` hides `explorers.core`.

## What is here today

The populations prototype:

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
