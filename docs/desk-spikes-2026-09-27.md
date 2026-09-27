# Desk spikes, 2026-09-27

Moved out of the README on 2026-09-27. This is desk research from before the multi-agent pivot. It evaluated vllm-lens, prime-rl and NDIF; it did not choose any of them. What still holds: the sidecar design (score finished rollouts in a separate process) works whatever reads the activations.

Code read: vllm-lens @7d252f7, prime-rl @f87fd0e, vLLM v0.29.0, nnsight @260c555. Not run: no GPU in that session.

**B — vllm-lens inside prime-rl: three conflicts**
1. *Worker extension.* prime-rl sets `worker_extension_cls` for weight sync, and vllm-lens only injects its own when none is set, so hooks never install. Fix: a combined class `(NCCLWeightUpdateWorker, HiddenStatesExtension)`. Method names don't collide.
2. *Model runner.* vllm-lens needs the V1 runner. The prime-rl `rl` entrypoint switches to V2 whenever train sampling truncates (top_p/top_k), for sampling masks. vLLM 0.29 still has every V1 internal vllm-lens reads, but vllm-lens is locked and tested at vLLM **0.19**, while prime-rl pins **0.29**.
3. *Eager mode.* vllm-lens forces `enforce_eager`, which removes CUDA graphs from the rollout path. The published ~20% overhead was measured on OPT-30B with 1 layer and short Alpaca completions, so it doesn't transfer to small models or long agentic rollouts.
- Weight sync is in place and layer by layer, so hooks on layer modules should survive updates (static reasoning, not run).

**The way around it: a sidecar scorer.** Leave the rollout path untouched (`VLLM_LENS_DISABLE=1` on prime-rl's server). vllm-lens ships that kill switch for exactly this case. Send completed rollouts (token ids) to a separate scorer: vllm-lens, HF or NDIF, loaded with the same checkpoint. It does one prefill pass (`max_tokens=1`) with a persistent hook that projects onto the probes and returns scalars only. Under teacher forcing, a causal model's activations equal the ones it had while generating (up to kernel numerics).
- **Pros:** no conflicts with prime-rl; CUDA graphs kept; audit isolation comes free (separate process); one code path for prime-rl, Tinker checkpoints and NDIF (the "dataset sweep" pattern); with prefix caching, scoring each finished turn is near-real-time.
- **Cons:** a second copy of the weights; scores lag the policy by one checkpoint; no warning *within* a turn.

**A — 4070 budget** (0.9 util minus ~1 GB overhead; KV = 2·layers·kv_heads·head_dim·2 B)

| Model | Weights | 12 GB: KV room | 8 GB: KV room | Residual / token / layer |
|---|---|---|---|---|
| Qwen3-0.6B | 1.2 GB | ~75k tok | ~44k tok | 2 KB |
| Qwen3-1.7B | 3.4 GB | ~56k tok | ~24k tok | 4 KB |
| Qwen3-4B | 8.0 GB | ~12k tok | doesn't fit (bf16) | 5 KB |

Use plain Qwen3, not the hybrid Qwen3.5, for the spikes (fewer variables). Needs Linux or WSL2 and Python 3.12. prime-rl RL defaults to **2 GPUs** (1 trainer + 1 inference), so full RL on a single 4070 is marginal (Qwen3-0.6B, LoRA, split GPU). RL is out of Phase 1; when it comes back (Phase 3), rent 2 GPUs or use Tinker, then score on the 4070.

**C — NDIF.** Free pilot tier (key at login.ndif.us). Shared queue: HOT/WARM models, and a WARM model deploys on first request. No quota is documented. Tools run client-side, so each agent turn is one remote request that waits in the queue, which makes live multi-turn SWE rollouts impractical. Plan: generate rollouts elsewhere, then score them on NDIF with the dataset-sweep pattern (reduce server-side, download scores only). Still unknown without an account: queue latency, maximum sequence length per request, eligibility outside the US.
