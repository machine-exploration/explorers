# The interface: streams, traces and studies

Design note for roadmap steps E1 (streams) and E2 (studies). Written 2026-09-29.

## The model

```
Model  →  Trace (one execution)  →  Stream  →  read / write at (layer, position)
Study  =  models × examples × {reads, writes, measures, patches}  →  compute()
```

- **`ex.open(repo_or_module, revision=...)`** returns a `Model`: the network, its tokenizer, and a
  *layout* that says where its blocks, sublayers, final norm and embedding are. Known layouts:
  GPT-NeoX (Pythia), Llama (also Qwen, Mistral, OLMo) and GPT-2. A model's `key` is `name@revision`
  for published checkpoints, or a hash of its parameters.
- **Streams**, named the same way on every architecture:

  | Stream | Layers | Meaning |
  |---|---|---|
  | `residual[L]` | 0 … n | the residual entering block L; `residual[n]` is after the last block, **before** the final norm |
  | `attn_out[L]` | 0 … n−1 | the output of the attention sublayer of block L |
  | `mlp_out[L]` | 0 … n−1 | the output of the MLP sublayer of block L |

  `residual[L]` equals Hugging Face's `hidden_states[L]` for L < n. For L = n, Hugging Face returns
  the value after the final norm; we keep the value before it, because that is what the Jacobian
  lens and the unembedding read (`model.unembed` applies the norm and the output embedding).

## Declare, then execute

Inside `with model.trace(tokens) as run:` nothing runs. Reads and writes are recorded; on exit the
model runs **once**, with forward hooks at the declared sites. This is the property the Runtime needs:
a whole experiment is known before it runs, so it can be planned.

- **Order at one site:** writes apply in the order declared; reads see the value after all writes
  (what the rest of the model sees).
- **Writes:** `fn=` (tensor → tensor, e.g. `ex.steering.add(direction, scale)`) or `value=` (a
  tensor broadcast to the selected slice: patching).
- **Gradients:** `trace(..., grad=True)` keeps the graph, rooted at the embedding output so frozen
  models work. Compute a metric from `run.logits`, call `.backward()`, read `value.grad`.
- **Hooks:** residual sites are forward pre-hooks on the block (or the final norm for L = n); sublayer
  sites are forward hooks on the attention or MLP module (tuple outputs: the first element).

## Studies

A `Study` declares reads, writes, measures (`fn(logits, ids) -> (batch,)`) and patches over one or
more models and a set of examples, and `compute()` returns an `xarray.Dataset` with a `model`
dimension. Models can be `Model`s, zero-argument loaders (one model in memory at a time), or an
`explorers.core.Trajectory` (the coordinate is then the training step).

`patch(source, stream, layers, positions, metric, method)` reports the normalized effect
`(m_patched − m_corrupt) / (m_clean − m_corrupt)` for every site and example:

- `method="exact"`: one run per site, each batched over examples. The clean activations are computed
  once per batch and reused for every site.
- `method="attribution"`: one forward and one backward pass for all sites:
  `(a_clean − a_corrupt) · ∂metric/∂a` at the corrupt run.

Execution is direct today. Planning it (sharing computation up to each patched site, batching
counterfactual branches, caching) is roadmap S2; the results must not change when it does.

## The five canonical examples, and why the checks are exact

`tests/test_interface.py`, on tiny random GPT-NeoX, Llama and GPT-2 models built locally:

| Example | Check | Why it must hold for any weights |
|---|---|---|
| Linear probe | accuracy 1 on a planted label (is the first token in a set?) read at `residual[0]`, position 0 | that read is the token embedding; 16 random embeddings in 32 dimensions are linearly independent, so every labelling is separable |
| Steering | adding a large multiple of the unembedding direction of token t (times the norm weight) to `residual[n]` raises the logit of t | the normalized residual aligns with that direction |
| Activation patching | effect 1 at `residual[0]`, position 0 and at `residual[n]`, last position; effect 0 at every other position of those two layers | clean and corrupt differ only at position 0; `residual[n]` at other positions does not reach the last logit |
| Attribution patching | 0 where clean and corrupt activations are equal, and where there is no path to the metric; equal to exact patching (1) at `residual[n]`, last position, when the final norm is removed | the approximation is exact where the metric is linear in the site |
| Sparse autoencoder read | features equal `relu((x − b_dec) W_enc + b_enc)` on read activations; an identity SAE round-trips | definition |

Reads also match the model: `residual[L]` equals `hidden_states[L]`, the final norm of `residual[n]`
equals the last hidden state, and `unembed(residual[n])` equals the logits, on all three layouts.
A write changes exactly its target, leaves earlier layers and earlier positions unchanged (causality),
and a zero write leaves the logits unchanged.

## Not yet

- Padding and attention masks: traces take token ids of equal length.
- Backends other than Hugging Face / PyTorch (E3), and a result store for studies.
- `explorers.core.over` (observables over trajectories) and `Study` are two execution paths; `Study`
  should absorb `over` once it caches results by content.
- `explorers.learning.sweep` and `activations` predate both and should become studies.
