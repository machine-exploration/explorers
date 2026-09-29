# explorers

A learning mechanics library: measure what models learn, when, and why, across training.

**Status: pre-alpha.** The API will change.

Part of [Machine Exploration](https://github.com/machine-exploration/public): see the [vision](https://github.com/machine-exploration/public#readme) and the [roadmap](https://github.com/machine-exploration/public/blob/main/ROADMAP.md).

**Mechanics** is Machine Exploration's research program: how training creates representations, algorithms and circuits. Today this repository also holds the library code (`explorers-core`, `explorers-learning`). In roadmap step 0 that code moves to [machine-exploration/explorers](https://github.com/machine-exploration/explorers), and this repository keeps the research: experiments, datasets and papers, built on `explorers`.

## Idea

Training runs are trajectories. `explorers` treats a run as a sequence of model states (published
checkpoints, or snapshots of a live run), measures observables at each state on a fixed set of
examples, and analyses the resulting curves: when does something change, how suddenly, and what
predicts it.

Five primitives do the work:

- **`Examples`**: token sequences with per-example metadata. Ids are content hashes, so results
  about the same example line up across runs and machines.
- **`State` / `Step` / `Trajectory`**: a model at one point of training (loaded lazily), a
  transition with its gradient, and an ordered run with coordinates such as size or seed.
- **`Observable`**: a pure, versioned function of a state and the examples that declares what it
  reads (`weights`, `token_loss`, `hidden:<layer>`, `step`) and the dimensions it returns.
- **`over` / `across`**: measure observables along a run or several runs. Each state is loaded once
  and gets at most one forward pass, however many observables ask for it.
- **`Store`**: results keyed by their content, so re-runs skip finished work and result folders merge.

Results are `xarray` datasets with named dimensions (`run`, `step`, `example`, `position`, `param`).
See [docs/design.md](docs/design.md).

## Example: when are quanta learned?

```python
from explorers.core import analysis, observe
from explorers.core.engine import over
from explorers.learning import toy

task = toy.MultitaskLookup(n_tasks=16, n_symbols=16, alpha=1.3)   # tasks with Zipf frequencies
run = toy.train(task, steps=1500, every=50)                       # a few minutes on a CPU
ds = over(run, [observe.example_loss, observe.stable_rank, observe.update_norm], task.examples())

on = analysis.onsets(ds.example_loss)            # when and how suddenly each example is learned
analysis.spearman(ds.task_frequency, on.onset)   # frequent tasks are learned first: negative
```

The same analysis on Pythia checkpoints is in [examples/quanta_pythia.py](examples/quanta_pythia.py).

## Writing an observable

```python
from explorers.core import observable
import numpy as np

@observable(reads=["hidden:3"], dims=("example",))
def layer3_norm(ctx):
    return np.linalg.norm(ctx.hidden[3], axis=-1).mean(axis=1)
```

## Development

```bash
uv sync
uv run pytest
```

The tests train tiny models on a CPU and never download anything.
