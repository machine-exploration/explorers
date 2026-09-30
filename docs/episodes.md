# Episodes: replaying rollouts as examples

Durable note on `explorers.episodes`, the adapter that turns verifiers episode records into
`Examples`. Read at `machine-exploration/verifiers` commit
`e66ec52966327e79011f4c3bb34a0f9817ce32b8` (our pinned fork; `verifiers/v1/episode.py`,
`trace.py`, `graph.py`) and prime-rl `d5f29c0` (`monitors/file/traces/chunks.py`). Written
2026-09-30. Neither package is imported: the adapter reads plain JSON.

## The record

prime-rl's file monitor writes every eval and RL episode, once, to
`<run>/monitors/file/traces/stream/` as chunked JSONL: `NNNNN.jsonl` while live, sealed into
`NNNNN.jsonl.zst` (seekable zstd) when full. A line is `Episode.to_record()`:

- the episode: `id`, `env.id`, `task`, `group`, `run`, `traces`;
- a trace: `id`, `rewards` (name → `{score, weight}`), `metrics`, `nodes`, `calls`, `stop_condition`;
- a node (`MessageNode`): `parent` (index of its predecessor, None for a root), `message`,
  `sampled`, `token_ids` (the tokens it adds to the sequence: leading template scaffold plus its
  own tokens), `mask` (per token: True if the model sampled it), `logprobs` (one per sampled
  token, rounded to 4 decimals in the record), `advantages`.

A root-to-leaf path is a branch (compaction and subagents fork the graph). Concatenating its nodes'
`token_ids` reproduces the exact `prompt + completion` tokens the model saw.

**Token ids exist only when the model ran behind a server that reports tokens** (prime-rl's
inference server, the renderer client). An eval that relays to an OpenAI-style API records text
only; such a branch would need re-rendering through the chat template, which is not exact, so
`replay` refuses it.

## The adapter

```python
import explorers as ex

records = ex.episodes.read_stream("outputs/my-run/monitors/file/traces/stream")
examples, logprobs = ex.episodes.replay(records, pad_id=tokenizer.pad_token_id,
                                        label=lambda episode, trace: trace["rewards"]["pass"]["score"] > 0)
ds = ex.Study(model, examples).measure(ex.measures.concept_monitor(40, word_ids),
                                       ex.measures.probe_monitor(40)).compute()
```

- One example per branch, right-padded with `pad_id` (a causal model's earlier positions never see
  the padding); `loss_mask` = the sampled tokens, so monitors score the model's own turns.
- Metadata: `episode`, `trace`, `branch`, `env`, `reward` (score × weight, summed), and `label`
  from the given function.
- `logprobs`: the recorded sampling log-probabilities at sampled positions, NaN elsewhere.
- Sealed chunks need `zstandard`.

Check (`tests/test_episodes.py`): tokens sampled from a model and recorded as verifiers records them
replay to the same log-probabilities (`-token_loss` at the sampled positions). On a real run the
same check compares with the recorded values: within 1e-4 (the record's rounding) plus the
numerical difference between the inference server and the replaying model (bf16), and at the
sampling temperature (the recorded values are at temperature 1 only if the run sampled at 1).

## Recording token ids with vLLM, and the check on a real model

Done 2026-09-30 on `Qwen/Qwen3.8-27B` (bf16, one RTX PRO 6000 96 GB), vLLM 0.30.0, verifiers fork
`18f2608b`, explorers `ef27b0d`.

- **Server:** `vllm serve <model> --enable-scale-out` mounts `/inference/v1/generate` (token ids in
  and out); without the flag the route answers 404. Qwen 3.5+ are hybrid (linear-attention state per
  sequence): `--max-num-seqs` must fit the state blocks (256 on 96 GB at 32k context; the default
  1024 refuses to start). An image without `nvcc` cannot JIT FlashInfer's sampler:
  `VLLM_USE_FLASHINFER_SAMPLER=0`.
- **Client:** `vf-eval` with `[client] type = "train"`, `base_url = "http://localhost:8000/v1"`, a
  non-Prime `api_key_var`, and `[client.renderer] name = "qwen3.8"` (the renderer map has the
  model, with tool calls). The default agent runtime is Prime sandboxes (`type = "prime"`, needs a
  Prime login); `type = "subprocess"` runs tools on the host, unisolated.
- **Record:** `vf-eval` writes `traces.jsonl` (the full episode, full-precision logprobs); its nodes
  have the fields `replay` reads (`parent` is omitted on roots, where it is None).
- **Check:** 3 `impossible_code` rollouts (2 hacks), 2,762 sampled tokens, replayed with
  `ex.open(hf, device="cuda")` and `token_loss`: |replayed − recorded| logprob median 0.0003, mean
  0.013, p99 0.12, max 0.32. Aligned tokens (a shift would give large gaps everywhere); the rest is
  the bf16 difference between vLLM's kernels and transformers' through 64 layers. Load 34 s, replay
  43 s. Install `flash-linear-attention` for the linear-attention layers (the reference fallback is
  slow).
- **Timeouts lose the episode:** an episode that exceeds `--env.timeout.episode` is recorded with
  `traces: []` and a `TimeoutError`: nothing to replay or label. Bound rollouts by turns and tokens
  instead. And size concurrency to the KV cache: 32 concurrent rollouts of ~12k tokens on a
  366k-token cache kept it ≥95% full, 28 requests waiting, ~400 tok/s shared (2026-09-30 run).
