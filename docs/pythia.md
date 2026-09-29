# Pythia checkpoints and hidden states

Durable notes on the backend `explorers` reads from. Written 2026-09-28; the tiny-model tests in
`tests/test_hf.py` pin the behaviour described under "Reading hidden states".

## Checkpoints

- Repos: `EleutherAI/pythia-<size>` and `EleutherAI/pythia-<size>-deduped` (trained on the
  deduplicated Pile). Sizes: 14m, 31m, 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b.
- Each checkpoint is a git branch of the repo, named `step<N>`. Pass it as `revision=` to
  `from_pretrained`. `main` is the final checkpoint, `step143000`.
- 154 checkpoints: `step0` (initialisation), `step1`, `step2`, `step4` … `step512` (powers of two),
  then `step1000` to `step143000` every 1000 steps. `pythia_steps()` returns exactly this list.
- All sizes saw the same data in the same order, so checkpoint N of two sizes are directly comparable.
- `pick(steps, n)` spreads `n` checkpoints evenly in log-step, keeping step 0 and the last one.
  Most change happens early, so linear spacing would waste checkpoints on the flat tail.

## Reading hidden states

- `model(**enc, output_hidden_states=True).hidden_states` is a tuple of `num_hidden_layers + 1`
  tensors of shape `(batch, seq, d_model)`. Index 0 is the embedding output; index L is the
  residual stream after block L. `explorers` uses the same numbering for `layers`.
- Pythia's tokenizer has no pad token. `HFSource` sets `pad_token = eos_token` and pads on the
  right. With a causal model, right padding never changes the states of real tokens, so the
  `last` position is `attention_mask.sum(1) - 1` and `mean` averages over the mask.
- Activations are returned as float32 NumPy arrays whatever the model dtype.

## Memory on a 12 GB GPU (rough, weights only)

| Model | fp16 weights |
|---|---|
| pythia-410m | ~0.8 GB |
| pythia-1.4b | ~2.8 GB |
| pythia-2.8b | ~5.6 GB |
| pythia-6.9b | ~14 GB: does not fit in fp16 |

Short probing texts add little on top. Each checkpoint is a separate download of the full weights,
so the Hugging Face cache grows by one model size per checkpoint read.
