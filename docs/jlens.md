# The Jacobian lens (J-lens)

Durable note on the Jacobian lens in `explorers` (reads served by `execute.py`, measures in
`measures.py`). Written 2026-09-29; names updated after the lean pass.

## What it measures

The J-lens reads out what an internal activation is disposed to make the model say
([Anthropic, 2026](https://transformer-circuits.pub/2026/workspace/)):

```
lens_L(h) = unembed(J_L @ h),   J_L = E[ d final / d h_L ]
```

- `h_L` is the residual stream entering block `L` (`residual:<L>`, 0 = embeddings).
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
| `jacobian:<L>:<skip>[:<target>]` | read | `J_L`, `(d_model, d_model)`; target `final` (default, `residual[n]`) or `penultimate` (`residual[n-1]`, the paper's default) |
| `unembed` | read | also `ctx.unembed_apply(h, fn)` (a function of the logits, on the device) and `ctx.unembed_matrix` (vocab, d): `W_U diag(gain)`, with LayerNorm's centering folded in |
| `measures.jacobian(L, skip_first)` | measure | the fitted `J_L`, stored like any result |
| `measures.jlens_error(layers, skip_first)` | measure | per layer: how often the lens top-1 differs from the model's own top-1 at the same position |
| `measures.logit_lens_error(layers, skip_first)` | measure | the same without transport (`J` = identity): the baseline |
| `measures.jlens_dimension(layers, skip_first, target, share=0.9)` | measure | per layer: fraction of dimensions holding 90 % of the variance of the J-lens vectors `W J_L` |
| `measures.jlens_cka(layers, skip_first, target)` | measure | (layer, layer2): linear CKA between the J-lens vector sets of two layers |
| `measures.lens_kurtosis(layers, skip_first, target, lens)` | measure | per layer: median excess kurtosis of the readout logits; `lens="logit"` for the baseline |
| `measures.lens_persistence(layers, skip_first, target, lens, offsets)` | measure | (layer, offset): log of how much more often the top-1 readout repeats D positions later in the same text than in the next text |

Both error curves fall as a layer's content becomes what the model says: they track the late
"motor" regime. The four workspace signatures (Gurnee et al. 2026, section 4.1) track the
workspace itself: dimension, kurtosis and persistence rise at its onset, and CKA shows the
early / workspace / motor blocks. Dimension and CKA need only `W` and `J`; kurtosis and
persistence read the evaluation rows. For `analysis.onsets` (which finds drops), pass the negated
curve of a rising signature.

Checks (`tests/test_jlens.py`), exact by construction: the Jacobian to the penultimate target
matches the brute-force estimator, and at `L = n-1` it is the identity; `unembed_matrix` reproduces
the logits through the norm; `dimension_fraction`, `linear_cka` (1 under orthogonal transforms and
rescaling) and `repeat_rate` (0 for position-locked tokens) on constructed inputs; with `J` =
identity every J-lens signature equals its logit-lens version, and kurtosis equals a direct
computation.

## Validation

- `tests/test_jlens.py`: the estimator equals a brute-force Jacobian computed
  one scalar at a time; fitting uses only the fit rows; frozen parameters give the same `J`;
  results are cached in the store; `final` is before the norm.
- Against the reference implementation, `anthropics/jacobian-lens` at
  `581d398613e5602a5af361e1c34d3a92ea82ba8e` (cloned outside the repository, read and run, not
  imported): on a random 3-layer GPT-NeoX, `J` agrees within 1.2e-7 at every source layer, and
  the top-5 readouts on held-out prompts are identical at 54/54 (layer, position) pairs.
- Note: that reference's built-in Pythia layout names the output layer `embed_out`; current
  `transformers` names it `lm_head`, so the check passed an explicit layout.

## What the paper says about the method

From Gurnee et al., "Verbalizable Representations Form a Global Workspace in Language Models"
(Transformer Circuits, July 2026), read 2026-09-29:

- **Target layer.** Their default Jacobian is taken at the **penultimate** layer's residual, not
  the final one; including the last block adds noisy readouts (§A.7). Ours uses `final`
  (pre-norm, after the last block). Aggregation: mean over positions, then over prompts; median and
  frozen-QK variants change little. Frozen QK can increase causal effects.
- **Data.** 1,000 sequences of 128 tokens by default; the J-lens beats the logit and tuned lens
  with as few as 10 prompts (Fig. 59).
- **Next-token agreement is not the target.** The J-lens is the *worst* of the three lenses at
  predicting the model's next token through most layers, by design (Fig. 55). So `jlens_error`
  (disagreement with the model's top-1) measures the late "motor" regime, where every lens
  converges on the output. It does not measure the workspace.
- **Workspace signatures** (§4.1, Fig. 28), all computable from a fitted `J`:
  excess kurtosis of readouts; top-1 autocorrelation across positions against a shuffled null;
  effective dimensionality of `W_U J_L` (collapsed before the workspace, fans out at its onset);
  CKA between layers of the J-lens vectors (block structure: early, workspace, motor);
  occupancy by sparse non-negative decomposition (about 25 active vectors).
- **Recovery of known intermediates** (pass@k on multi-hop, arithmetic, poetry, typo prompts,
  §A.6) is their main quality check against the logit and tuned lens.
- **Training dynamics are open** (§9.1): the space is present in base models before post-training
  (§6), but "we do not know how much earlier in pretraining it emerges, whether it appears
  gradually or abruptly, or how it scales with model size".

## Example: Q1 on Pythia (see mechanics/experiments/q1_verbalizable_space)

```python
import numpy as np
import explorers as ex

name = "EleutherAI/pythia-70m"
examples = ex.Examples.from_texts(texts, tokenizer, seq_len=128, max_examples=200)
examples = examples.with_meta(split=np.where(np.arange(len(examples)) < 100, "fit", "eval"))
layers = range(1, 6)
ds = (ex.Study(ex.checkpoints(name, steps=ex.pick(ex.pythia_steps(), 24), device="cuda"), examples, dim_batch=8)
      .measure(ex.measures.jlens_error(layers), ex.measures.logit_lens_error(layers))
      .compute(store="runs/store"))
on = ex.analysis.onsets(ds.jlens_error)   # when each layer's readout becomes what the model says
```
