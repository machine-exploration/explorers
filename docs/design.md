# Design: primitives for learning mechanics

`explorers` measures how models change during training and finds the regularities in those
changes. This note records the primitives, why they were chosen, and how the first study (quanta)
exercises them. Written 2026-09-29.

## The primitives

| Primitive | What it is | Why it scales |
|---|---|---|
| `Examples` (`data.py`) | fixed-length token sequences, a loss mask, and per-example metadata columns. Each example's id is a hash of its tokens. | Results are indexed by content, so the same example lines up across runs, machines and people. Questions about data (frequency, source, difficulty) are metadata joins, not new code. |
| `State`, `Step`, `Trajectory` (`state.py`) | a model at one point of training (lazy loader + content key); one transition with its gradient; an ordered run with coordinates (size, lr, seed). | Checkpoint suites give states; live and toy runs also give steps. Dynamics observables read steps; everything else reads states. |
| `Observable` (`observe.py`) | a pure, versioned function of a state (or step) and the examples, declaring what it reads and the dims it returns. | New science is a new small function, not a change to the core. Declared reads let the engine share work. |
| `over` / `across` (`engine.py`) | measure observables along a trajectory, or along several runs. | One model load and at most one batched forward pass per state, whatever the number of observables. |
| `Store` (`store.py`) | results keyed by (state content, observable name+version+params, examples fingerprint). | Re-runs skip finished work; result folders from different machines merge by copying. |
| analyses (`analysis.py`) | functions of labelled arrays only: `onsets`, `spearman`, `cluster_curves`. | They never touch a model, so they work the same on toy runs, Pythia or live training. |

Results are `xarray` objects with named dimensions: `step`, `run`, `example`, `position`,
`param`… and the examples' ids and metadata as coordinates on `example`.

### Reads understood by the engine

- `weights`: every named parameter tensor, as NumPy.
- `token_loss`: `(example, position)` next-token loss. Position 0 and masked positions are NaN.
- `hidden:<L>`: `(example, position, d_model)` residual stream after block `L` (0 = embeddings).
- `step`: weights before and after the step, and the gradient at the state before it.

## First study: quanta

The quantization model of neural scaling (Michaud et al., 2023) says skills are learned as
discrete quanta, in order of how often they are used. `explorers` tests it across training time.

1. **Toy, known quanta** (`toy.MultitaskLookup`, `examples/quanta_toy.py`, `tests/test_quanta_toy.py`).
   Tasks are lookup tables drawn with Zipf frequencies. Measured on 16 tasks x 16 symbols,
   1500 steps, a state every 50 steps: frequent tasks drop first (rank correlation between task
   frequency and onset step -0.74), and drops are sudden (median sharpness 0.74).
2. **Pythia** (`examples/quanta_pythia.py`). Every (window, position) of a text sample is a sample;
   its next-token loss is measured at each checkpoint; `onsets` gives when and how suddenly it
   drops; `cluster_curves` groups samples that drop together into candidate quanta; target-token
   frequency in the sample is the first metadata column to correlate with onset.
   Not yet run: Hugging Face is not reachable from the development container.

## Known limits of this prototype

- `token_loss` and `hidden:<L>` keep full arrays in memory; long windows or many layers on big
  samples will need streaming reducers.
- Step observables load both states' weights; for live runs of large models the engine should read
  them from the training process instead of snapshots.
- `sweep.py` and the probe pipeline predate these primitives and should become observables.
- The store has no garbage collection and no index; it is a folder of `.npy` + `.json` pairs.

## References read for this design

Cloned outside the repository and read, not imported (see `AGENTS.md`):

- `timaeus-research/devinterp` at `fbbf4c54e1f6ee46acb149f004df261fb05055c6`: one function per
  estimator (`llc(model, dataset, observables=...)`), results as xarray with named dims, a zarr
  cache keyed by config. Works on one model at a time; no trajectory object.
- `CalculatedContent/WeightWatcher` at `a5940f79e9d5bbc37c82360c6f2f8a3aefe6543a`: data-free
  per-layer spectral metrics (`alpha`, `stable_rank`, `log_spectral_norm`) of one snapshot.
- `microsoft/mup` at `19814971934ef91dd546f88e913fc963e096d11c`: a parametrisation plus coordinate
  checks across widths and steps; the experiment is a grid of runs.
