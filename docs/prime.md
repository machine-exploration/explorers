# Prime Intellect as the runtime

Durable note on how a `prime-rl` run lays out what `explorers` reads. Read at
`PrimeIntellect-ai/prime-rl` commit `d5f29c072a6731a17897f5aed661e0b0e6053d8c` (cloned outside the
repository, 2026-09-30, not imported).

## The split

| Layer | Who | Role |
|---|---|---|
| Runtime | Prime Intellect: pods, `prime-rl` (trainer, orchestrator), `verifiers` (environments, rubrics), vLLM | Trains, serves and scores the run; writes weights and rollouts to disk |
| White-box layer | `explorers` | Opens those weights and measures what is inside, on the same pods |
| Research | `mechanics` | Studies built from both |

`verifiers` talks to the model through an OpenAI-style API: it sees text, never weights or
activations. White-box access happens where the weights are: the trainer, or what it writes.

## What a run writes (`<run_dir>` = the run's output directory)

| Path | What | Lifetime |
|---|---|---|
| `checkpoints/step_N/trainer/` | FSDP-sharded model, optimizer, scheduler (torch DCP), for resuming | Kept per `[ckpt]` config |
| `checkpoints/step_N/orchestrator/` | Orchestrator progress, per-environment data state | Same |
| `broadcasts/step_N/` | The policy handed to inference at step N | **Only `N` and `N-1` are kept** (`WeightSender._clean`) |
| `monitors/file/traces/stream/` | Every episode (rollout) as a verifiers record, chunked JSONL, with an index beside it | Whole run |

- **Full fine-tune:** export a DCP checkpoint to Hugging Face safetensors with
  `tools/convert_dcp_to_bf16.py <run>/checkpoints/step_N` (writes `<ckpt>/weights`). It refuses LoRA
  checkpoints.
- **LoRA:** the NCCL transport is not supported, so broadcasts go through the filesystem, and
  `broadcasts/step_N/` holds the raw adapter: `adapter_model.safetensors` and `adapter_config.json`
  (`r`, `lora_alpha`, `target_modules`, `base_model_name_or_path`). A broadcast is complete when
  its `.finished` marker exists.
- **Adapter tensors:** keys are `<module>.lora_A.weight` (rank × in) and `<module>.lora_B.weight`
  (out × rank), with `<module>` the Hugging Face module path (e.g.
  `model.layers.0.self_attn.q_proj`). The layer computes `base(x) + (alpha / r) · B A x`
  (`set_multilora_scaling`), so merging is `W ← W + (alpha / r) · B A`. MoE expert adapters use a
  per-expert format and are not merged by `explorers` yet.

## How explorers reads a run

Broadcasts are transient, so the adapters must be copied out while the run trains:

```python
import explorers as ex

ex.archive_adapters("outputs/my-run", "runs/my-run/adapters")   # idempotent; copies finished broadcasts
refs = ex.adapters("Qwen/Qwen3-8B", "runs/my-run/adapters", device="cuda", dtype="bfloat16")
ds = ex.Study(refs, examples).measure(...).compute(store="runs/store")
```

- `archive_adapters` copies each `broadcasts/step_N/` with a `.finished` marker that is not yet in
  the destination. Run it in a loop beside the trainer (a broadcast lives about two steps).
- `adapters` returns one `ModelRef` per `step_N`: base model + adapter, merged into the weights at
  load (`W + (alpha / r) B A`, computed in float32). The study coordinate is the step; the key is
  the base's key plus a hash of the adapter file, so results cache by content.
- Merging is checked against running the adapter unmerged (`tests/test_prime.py`).

## Not yet

- The rollouts under `monitors/file/traces/` as `Examples` (the prompts and completions the run
  actually trained on).
- A measure hook inside the trainer (run a study on the live weights every N steps, no copy).
- MoE expert adapters; `modules_to_save`.
