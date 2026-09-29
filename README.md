# explorers

> **White-box experimentation at scale**: observe, measure and intervene on the internal computation of models, across training and deployment.

`explorers` is the open library of [Machine Exploration](https://github.com/machine-exploration/public), which is building the infrastructure for a science of intelligence. The vision is in the [org README](https://github.com/machine-exploration/public#readme), and the plan in the [roadmap](https://github.com/machine-exploration/public/blob/main/ROADMAP.md).

**Status: pre-alpha.** Nothing is released yet. The API will change.

---

## The interface

```python
import explorers as ex

model = ex.open("EleutherAI/pythia-70m", revision="step143000")

with model.trace(tokens) as run:                      # declared, then run once on exit
    resid = run.stream("residual")
    x = resid.read(layer=3, position=-1)
    resid.write(layer=3, position=-1, fn=ex.steering.add(direction, 4.0))
x.value, run.logits
```

Streams are named the same way on every architecture: `residual[L]` (entering block L; `residual[n]` is after the last block, before the final norm), `attn_out[L]`, `mlp_out[L]`. Known layouts: GPT-NeoX (Pythia), Llama (also Qwen, Mistral, OLMo), GPT-2.

A **study** is the unit of work: reads, writes, measurements and patching over models (for example the checkpoints of a run) × examples.

```python
study = ex.Study(models=checkpoints, examples=corrupt)
study.read("residual", layers="*", position=-1)
study.patch(source=clean, stream="residual", metric=ex.patching.logit_diff(correct, wrong))
ds = study.compute()                                   # an xarray Dataset with a `model` dimension
```

Design and the checks behind it: [docs/interface.md](docs/interface.md). Other notes: [docs/design.md](docs/design.md) (`explorers.core`: observables over training runs), [docs/jlens.md](docs/jlens.md) (the Jacobian lens), [docs/pythia.md](docs/pythia.md), [docs/desk-spikes-2026-09-27.md](docs/desk-spikes-2026-09-27.md).

## Layout

One package, `explorers`:

| Module | What it holds |
|---|---|
| `explorers` (`model`, `trace`, `study`) | `open`, `Model`, `Trace`, streams, `Study` |
| `explorers.methods` | probes, steering, patching metrics, sparse autoencoders |
| `explorers.core` | examples identified by content, model states along training, observables, the engine (one forward pass per state; gradient reads for the Jacobian lens), the content-addressed store, analyses |
| `explorers.learning` | Pythia checkpoints, toy tasks with known answers, the probe sweep |
| `explorers.populations` | the scenario format, the multi-agent runtime (through the pinned [verifiers](https://github.com/machine-exploration/verifiers) fork), episodes, labels, detectors |

Planned: a second backend (roadmap E3), then the planner and the Runtime that executes studies at scale (S1, S2, R1). The Runtime will be open source too. Research that uses the library lives in [mechanics](https://github.com/machine-exploration/mechanics).

## Development

```bash
uv sync                  # the library with torch, transformers and the agent-side extras
uv run pytest            # tiny models on a CPU, no downloads; GPU tests are skipped (marker: gpu)
```

Extras for users: `explorers[torch]` (models and traces), `explorers[populations]` (scenarios), `explorers[verifiers]` (to play scenarios).

## Rules for scenarios

- **Contained:** scenarios that push agents to hack run with no network, no shared cache and no path between episodes.
- **Not published yet:** such scenarios and their traces are published only after the publication rules are settled.
- **Audit isolation:** oversight monitors run outside the trainer's process and write to an append-only store that the reward code cannot read.

## Contributing

Contributors and coding agents: read [AGENTS.md](AGENTS.md) first. Design notes are in [docs/](docs/).
