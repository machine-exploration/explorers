# The design of `explorers`, on one page

Written 2026-09-29, after the lean pass. The thesis: a neural network is a learned computation; its
weights are the program, and its internal streams carry the running state. The rule: few concepts,
each essential; one way to do each thing; data structures first.

## Six concepts

| Concept | Module | What it is |
|---|---|---|
| **Model** | `model.py` | A network with named streams. `ex.open(repo, revision)`, `ex.checkpoints(repo, steps)` (lazy handles whose key is known before download), or `ex.adapters(base, path)` (a prime-rl run's LoRA adapters, merged at load; `docs/prime.md`). |
| **Stream** | `model.py`, `trace.py` | `residual[L]` (entering block L; `residual[n]` after the last block, before the final norm), `attn_out[L]`, `mlp_out[L]`. Same names on GPT-NeoX, Llama (Qwen, Mistral, OLMo) and GPT-2. |
| **Trace** | `trace.py` | One forward pass. Reads and writes are declared inside `with model.trace(tokens)` and run on exit. |
| **Op** | `ops.py` | What a write does to a stream (`Add`, `Set`, `Scale`, `Ablate`, `ProjectOut`) or what a read keeps (`Project`, `Norm`). Data: JSON round trip. |
| **Measure** | `measures.py` | A named, versioned function of what a model computed. It declares its reads (`token_loss`, `logits:p`, `residual:L`, `weights`, `unembed`, `jacobian:L:skip[:target]`, `step`, …). |
| **Study** | `study.py` | The unit of work: `read`, `write`, `measure`, `patch` over models × examples. `compute(store=...)` runs it. |

Everything else supports these: `execute.py` (serves the reads of measures), `data.py` (examples
identified by content), `state.py` (training runs as states and steps), `store.py` (results by
content key), `analysis.py` (onsets, rank correlation, AUROC; arrays only, never models), `toy.py`
(tasks with known answers), `methods/` (probes, sparse autoencoders).

## One execution path

A study executes one model at a time:

1. **Reads** (`study.read`): one traced pass per batch; each read keeps only its selection, or a
   reduction computed on the device.
2. **Measures** (`study.measure`): `execute.serve` runs one traced pass per batch with the study's
   writes applied, and fills a `Context` with every declared read; each measure is a pure function of
   it. Jacobians (for the lens) add one backward sweep on the unmodified model.
3. **Patches** (`study.patch`): exact (one pass per site, batched over examples, clean activations
   computed once per batch) or attribution (one forward and one backward pass for all sites).
4. **Step measures** (`update_norm`, `grad_norm`) run on the training steps of a `Trajectory`.

Declare-then-execute is what scale needs: a whole study is known before it runs, so it can be
sharded and planned. Results must not change when it is.

## Traces

- At one site, writes apply in the order declared; reads see the value after all writes.
- `write(fn=op)` or `write(value=tensor)` (stored as `ops.Set`: patching).
- A read copies only what it selects, or keeps only `reduce=` computed on the device. One position
  of one layer costs batch × d floats.
- `trace(grad=True)` keeps the graph, rooted at the embedding output (frozen models work); compute a
  metric from `run.logits`, call `.backward()`, read `value.grad`.
- Hooks: residual sites are forward pre-hooks on the block (or the final norm for L = n); sublayer
  sites are forward hooks on the attention or MLP module.

## Studies as data

- `spec()` is the study as JSON (`explorers.study/v0`): model keys, the examples' fingerprint and
  shape, reads, writes (ops), measures (name, version, parameters), patches. It refuses Python
  callables in writes and reductions.
- `key()` hashes the spec. Equal studies get equal keys on any machine; any change to a parameter,
  the weights or the examples changes it.
- `compute(store=folder)` stores each output of each model under
  `sha256(study key | model key | output)`. A model whose outputs are all stored is not loaded, so a
  study over many checkpoints resumes where it stopped. Change a measure's code, bump its `version`.
- Results are an `xarray.Dataset`. The first dimension is `step` when every model is a checkpoint
  with a step (a `Trajectory`, or `ex.checkpoints`), otherwise `model`. Example metadata columns
  become coordinates on `example`.

## Canonical and derived data

- **Canonical** (stored or referenced, never recomputed): the weights (a checkpoint, by its model key
  `name@revision`), the examples (by fingerprint), and the study (by its spec and key).
- **Derived:** streams are a function of weights and examples, so they are recomputed, not stored;
  a read keeps only its selection or reduction. Results are a function of all three, so they are
  cached by content key and can always be rebuilt.
- Weights are not a seventh concept: a Model holds them and a Measure reads them (`weights`, and
  `step` for before, after and the gradient). Today they carry PyTorch parameter names. Planned, when
  the first weight-space measure needs it: uniform names across layouts, as for streams
  (`attn.W_Q[L]`, `attn.W_K[L]`, `attn.W_V[L]`, `attn.W_O[L]`, `mlp.W_in[L]`, `mlp.W_out[L]`, `embed`,
  `unembed`), GPT-NeoX first, with the same exact checks (a named weight equals the parameter it maps).

## Checks, exact by construction

`tests/test_interface.py` and `tests/test_ops.py`, on tiny random GPT-NeoX, Llama and GPT-2 models
built locally, hold for any weights:

| Example | Check |
|---|---|
| Reads | `residual[L]` equals Hugging Face's `hidden_states[L]`; the final norm of `residual[n]` equals the last hidden state; `unembed(residual[n])` equals the logits |
| Writes | a write changes exactly its target; earlier layers and positions are unchanged; a zero write changes nothing; writes apply to measures |
| Linear probe | accuracy 1 on a planted label read at `residual[0]` (token embeddings are linearly independent) |
| Steering | adding a large multiple of a token's unembedding direction to `residual[n]` raises that token's logit |
| Activation patching | effect 1 at the only position where clean and corrupt differ and at the last position of `residual[n]`; 0 where the site cannot reach the metric |
| Attribution patching | 0 where activations are equal or there is no path; equal to exact patching where the metric is linear in the site |
| Sparse autoencoder | features equal the definition; an identity SAE round-trips |

The Jacobian lens has its own checks (docs/jlens.md). The toy quanta result is a test too: frequent
tasks are learned first.

## Not yet

- Padding and attention masks: traces take token ids of equal length.
- Episode replay (eval and RL episodes as examples), the concept lens, a second backend, a planner.
- The agent side (`populations`) is frozen behind its extra; it predates this design and keeps its own
  `Method` interface until the agent questions return.
