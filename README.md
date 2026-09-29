# explorers

> **White-box experimentation at scale**: observe, measure and intervene on the internal computation of models, across training and deployment.

`explorers` is the open library of [Machine Exploration](https://github.com/machine-exploration/public), which is building the infrastructure for a science of intelligence. The vision is in the [org README](https://github.com/machine-exploration/public#readme), and the plan in the [roadmap](https://github.com/machine-exploration/public/blob/main/ROADMAP.md).

**Status: pre-alpha.** Nothing is released yet. The API will change.

---

## Layout

`explorers` is the open-source interface of Machine Exploration: read, write and trace the streams of computation inside a model, and describe experiments as studies that run the same way on any backend. One repository, one `uv` workspace, three packages that share the `explorers` namespace:

| Package | Import | What it holds |
|---|---|---|
| `packages/core` | `explorers.core` | Examples identified by content, model states along training, observables that declare what they read, the engine (one forward pass per state; gradient reads for the Jacobian lens), the content-addressed store, analyses |
| `packages/learning` | `explorers.learning` | Pythia checkpoints, toy tasks with known answers, probes, the probe sweep |
| `packages/populations` | `explorers.populations` | The scenario format, the multi-agent runtime (through the pinned [verifiers](https://github.com/machine-exploration/verifiers) fork), episodes, labels, detectors |

Planned (roadmap [E1–E3](https://github.com/machine-exploration/public/blob/main/ROADMAP.md#phase-1--the-primitive)): named streams with `read` / `write` / `trace`, `Study` over models × checkpoints × examples, and backends beyond Hugging Face / PyTorch. The Runtime that executes studies at scale will be open source too.

Design notes: [docs/design.md](docs/design.md) (core primitives), [docs/jlens.md](docs/jlens.md) (the Jacobian lens), [docs/pythia.md](docs/pythia.md) (checkpoints and hidden states), [docs/desk-spikes-2026-09-27.md](docs/desk-spikes-2026-09-27.md) (reading activations next to a trainer).

## Example: when are quanta learned?

```python
from explorers.core import analysis, observe
from explorers.core.engine import over
from explorers.learning import toy

task = toy.MultitaskLookup(n_tasks=16, n_symbols=16, alpha=1.3)   # tasks with Zipf frequencies
run = toy.train(task, steps=1500, every=50)                       # a few minutes on a CPU
ds = over(run, [observe.example_loss, observe.stable_rank, observe.update_norm], task.examples())
on = analysis.onsets(ds.example_loss)            # when, and how suddenly, each example is learned
analysis.spearman(ds.task_frequency, on.onset)   # frequent tasks are learned first: negative
```

Research that uses the library (studies, datasets, papers) lives in [mechanics](https://github.com/machine-exploration/mechanics).

## Development

```bash
uv sync
uv run pytest            # tiny models on a CPU, no downloads; GPU tests are skipped (marker: gpu)
```

## Rules for scenarios

- **Contained:** scenarios that push agents to hack run with no network, no shared cache and no path between episodes.
- **Not published yet:** such scenarios and their traces are published only after the publication rules are settled.
- **Audit isolation:** oversight monitors run outside the trainer's process and write to an append-only store that the reward code cannot read.

## Contributing

Contributors and coding agents: read [AGENTS.md](AGENTS.md) first. Design notes are in [docs/](docs/).
