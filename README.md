# explorers

> **The open lab for the science of deep learning:** train models, read and change their internals, and run experiments at any scale without thinking about infrastructure. Today it reads and changes models across every checkpoint of a run; training in the same loop is next.

`explorers` is the open-source lab of [Machine Exploration](https://github.com/machine-exploration/public), which is building a science of deep learning: the laws by which training creates the computation inside a model, joining learning mechanics (the dynamics of training) with mechanistic interpretability (what training produces). The experiments live in [mechanics](https://github.com/machine-exploration/mechanics); they climb a ladder from small models pretrained on designed data to Pythia, fine-tuning and reinforcement learning. The vision is in the [org README](https://github.com/machine-exploration/public#readme), the plan in the [roadmap](https://github.com/machine-exploration/public/blob/main/ROADMAP.md).

**Status: pre-alpha.** Nothing is released yet. The API will change.

---

## Today: read models and the checkpoints of a run

```python
import explorers as ex

model = ex.open("EleutherAI/pythia-70m", revision="step143000")

with model.trace(tokens) as run:                      # declared, then run once on exit
    resid = run.stream("residual")
    x = resid.read(layer=3, position=-1)
    resid.write(layer=3, position=-1, fn=ex.ops.Add(direction, 4.0))
x.value, run.logits
```

A **experiment** is the unit of work: reads, writes, measures and patching over models (for example the checkpoints of a run) × examples, one model at a time, with results cached by content.

```python
experiment = ex.Experiment(ex.checkpoints("EleutherAI/pythia-70m", steps=[0, 1000, 143000]), examples)
experiment.read("residual", layers="*", position=-1, reduce=ex.ops.Norm())
experiment.measure(ex.measures.loss, ex.measures.jlens_error(layers=range(1, 6)))
experiment.patch(source=clean, metric=ex.measures.logit_diff(correct, wrong))
ds = experiment.compute(store="runs/store")                  # an xarray Dataset indexed by step
```

Six concepts: **Model** (named streams: `residual`, `attn_out`, `mlp_out`, the same on GPT-NeoX, Llama, GPT-2 and Qwen 3.5+), **Stream**, **Trace** (one forward pass), **Op** (interventions and reductions as data), **Measure** (named, versioned functions of what a model computed), **Experiment**. The whole design fits on one page: [docs/interface.md](docs/interface.md). Other notes: [docs/monitors.md](docs/monitors.md) (concept, logit-lens and probe monitors, detection at a matched false-positive rate), [docs/episodes.md](docs/episodes.md) (replaying agent rollouts exactly), [docs/reward-hacking-probes.md](docs/reward-hacking-probes.md) (the reward-hacking protocol), [docs/prime.md](docs/prime.md) (reading prime-rl runs), [docs/jlens.md](docs/jlens.md) (the Jacobian lens), [docs/pythia.md](docs/pythia.md) (checkpoints), [docs/desk-spikes-2026-09-27.md](docs/desk-spikes-2026-09-27.md) (reading activations next to a trainer).

Supporting modules: `data` (examples identified by content), `state` (training runs), `store`, `analysis` (onsets, rank correlation, AUROC), `toy` (tasks with known answers), `methods` (probes, sparse autoencoders). The agent side, `explorers.populations` (scenarios, a multi-agent runtime through the pinned [verifiers](https://github.com/machine-exploration/verifiers) fork), is frozen behind the `populations` extra.

It works with any training stack through thin adapters. Today it reads Hugging Face checkpoints and, as the first integration, [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) runs: `ex.archive_adapters` and `ex.adapters` read a run's LoRA adapters as checkpoints ([docs/prime.md](docs/prime.md)), and `ex.episodes.replay` turns the rollouts of an eval or an RL run into examples, token for token ([docs/episodes.md](docs/episodes.md)). Monitors are measures: `concept_monitor` (a Jacobian-lens direction from words, no labels), its logit-lens baseline, and `probe_monitor` (difference of means), scored with `analysis.detection_at_fpr` and `analysis.auroc` ([docs/monitors.md](docs/monitors.md)). Planned: the training API below, with Modal as its first backend. Research that uses the library lives in [mechanics](https://github.com/machine-exploration/mechanics).

## Next: train, read and intervene in one loop

*A design, not yet built; the section above is what exists today.* Explorers becomes a Tinker-like API for the science of deep learning: train a model, read it and change it in the same loop. During training the activations are computed anyway, so reading them costs almost nothing.

```python
m = ex.model(ex.configs.tiny(layers=4, d=256), seed=0, init_scale=0.5)
for step, batch in env.batches():                                  # data order recorded exactly
    out = m.forward_backward(batch, reads={"residual[*]": ex.project(V)}, do=None)
    m.optim_step()

@ex.sweep(size=[1, 2, 4], seed=range(5), init_scale=[0.1, 1.0])
def onset_law(check, size, seed, init_scale):
    m = ex.model(ex.configs.tiny(scale=size), seed=seed, init_scale=init_scale)
    return ex.train(m, planted_tasks(sliver=check), steps=100 if check else 20_000, every=100,
                    measure=[per_task_accuracy, readability])

onset_law.check()                     # one grid point, on the CPU, on a sliver of the data
R = onset_law.run(on=ex.Modal())      # 30 runs on GPU workers; stored runs are never recomputed
```

- **Primitives:** `model`, `forward_backward(reads=, do=)`, `optim_step`, `sample`, `stream` (one model's activations fed to another training loop, nothing stored), `save`/`load`, and `@sweep` over a grid of sizes, seeds and checkpoints.
- **Environments** in Prime Intellect's format (data in a recorded order plus a rubric), read through an adapter so the core does not import verifiers.
- **Sweeps and backends:** `@ex.sweep` is built: a grid of runs keyed by content, checked on the CPU, sent to a backend. Locally only the check runs; full runs go to GPU workers, Modal first. Small models ship their whole loop to the backend; large ones take one call per primitive on a resident model.
- **Fast iteration:** activations cached once when an experiment reuses them, sweeps in one line, warm GPUs while you iterate, only changed grid points recomputed, results streamed back while runs go, a smoke mode before every full run, resume for long jobs.
- **Rules that carry over:** the same experiment gives the same result on every backend within a stated tolerance; every result reproduces from config, seed, data order and code version; cost comes back with every run.

The first experiment it serves is [`glp-activation`](https://github.com/machine-exploration/mechanics/tree/main/experiments/glp-activation) in mechanics, a generative model of activations fitted across training; the plan is O1 in the [roadmap](https://github.com/machine-exploration/public/blob/main/ROADMAP.md).

## Development

```bash
uv sync                  # the library with torch, transformers and the agent-side extras
uv run pytest            # tiny models on a CPU, no downloads; GPU tests are skipped (marker: gpu)
```

Extras for users: `explorers[torch]` (models and traces), `explorers[populations]` (scenarios), `explorers[verifiers]` (to play scenarios).

## Rules for scenarios

- **Contained:** scenarios that push agents to hack run with no network, no shared cache and no path between episodes.
- **Not published yet:** such scenarios and their traces are published only after the publication rules are settled. One exception: [`impossible_code`](https://github.com/machine-exploration/verifiers/tree/main/environments/impossible_code), a port of ImpossibleBench, whose tasks are already public.
- **Audit isolation:** oversight monitors run outside the trainer's process and write to an append-only store that the reward code cannot read.

## Contributing

Contributors and coding agents: read [AGENTS.md](AGENTS.md) first. Design notes are in [docs/](docs/).
