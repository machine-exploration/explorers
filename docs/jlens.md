# The Jacobian lens (J-lens)

Durable note on the J-lens reads and observables in `explorers.core`. Written 2026-09-29.

## What it measures

The J-lens reads out what an internal activation is disposed to make the model say
([Anthropic, 2026](https://transformer-circuits.pub/2026/workspace/)):

```
lens_L(h) = unembed(J_L @ h),   J_L = E[ d final / d h_L ]
```

- `h_L` is the residual stream after block `L` (`hidden:<L>`, 0 = embeddings).
- `final` is the residual after the last block, **before** the final norm. In Hugging Face
  models, `hidden_states[-1]` is after the final norm, so the engine captures `final` with a
  forward pre-hook on the final norm (`FINAL_NORMS` in `engine.py`).
- `unembed` is the model's own final norm followed by its output embedding.

## Estimator

For each output dimension `k`, a one-hot cotangent at `k` is set at every lens position of every
example at once, and one backward pass gives row `k` of the Jacobian at every source position.
By causality, the row at source position `p` is `sum_{t >= p} d final[t, k] / d h_L[p]`. Rows
are averaged over source positions and examples. Cost: `d_model` backward passes per batch.

- **Lens positions:** from `skip_first` (default 16: early positions behave as attention sinks)
  to the one before last (the last has no next token).
- **Fit and evaluation rows:** if the examples carry a `split` column, `J` is fitted on
  `split == "fit"` and the lens is evaluated on the other rows; otherwise both use every row.
- **Frozen models:** the graph is rooted at the embedding output, so it works when parameters do
  not require gradients.

## Reads and observables

| Name | Kind | What it gives |
|---|---|---|
| `final` | read | `(example, position, d_model)` before the final norm |
| `unembed` | read | `ctx.unembed_topk(h, k)`: top-k token ids of the model's own decoding |
| `jacobian:<L>:<skip>` | read | `J_L`, `(d_model, d_model)` |
| `observe.jacobian(L, skip_first)` | observable | the fitted `J_L`, stored like any result |
| `observe.jlens_error(layers, skip_first)` | observable | per layer: how often the lens top-1 differs from the model's own top-1 at the same position |
| `observe.logit_lens_error(layers, skip_first)` | observable | the same without transport (`J` = identity): the baseline |

Both error curves fall as a layer's content becomes what the model says, so they work directly
with `analysis.onsets` across checkpoints.

## Validation

- `packages/learning/tests/test_jlens.py`: the estimator equals a brute-force Jacobian computed
  one scalar at a time; fitting uses only the fit rows; frozen parameters give the same `J`;
  results are cached in the store; `final` is before the norm.
- Against the reference implementation, `anthropics/jacobian-lens` at
  `581d398613e5602a5af361e1c34d3a92ea82ba8e` (cloned outside the repository, read and run, not
  imported): on a random 3-layer GPT-NeoX, `J` agrees within 1.2e-7 at every source layer, and
  the top-5 readouts on held-out prompts are identical at 54/54 (layer, position) pairs.
- Note: that reference's built-in Pythia layout names the output layer `embed_out`; current
  `transformers` names it `lm_head`, so the check passed an explicit layout.

## Example: Q1 on Pythia (not yet run: needs Hugging Face access)

```python
from explorers.core import analysis, observe
from explorers.core.data import Examples
from explorers.core.engine import over
from explorers.learning import from_checkpoints, pythia

run = from_checkpoints(pythia("70m", n=24), device="cuda")
examples = Examples.from_texts(texts, tokenizer, seq_len=128, max_examples=200)
examples = examples.with_meta(split=np.where(np.arange(len(examples)) < 100, "fit", "eval"))
layers = range(1, 6)
ds = over(run, [observe.jlens_error(layers), observe.logit_lens_error(layers)], examples,
          store="runs/store", device="cuda")
on = analysis.onsets(ds.jlens_error)      # when each layer's readout becomes what the model says
```
