# Episode format `explorers.episode/v0`

One JSON object per line in `episodes.jsonl`. An episode is immutable once written;
a trailing truncated line is treated as a crash mid-write and skipped with a warning,
a malformed line anywhere else fails the read.

```
Episode
├─ id                    string
├─ format                "explorers.episode/v0"
├─ scenario, scenario_version
├─ model                 checkpoint reference served by the runtime
├─ seed
├─ status                "ok" | "infra_error" | "timeout"
├─ rounds
├─ board                 BoardMessage[] — the shared channel, verbatim
├─ rollouts              Rollout[] — one per agent, sorted by agent_id
└─ source                {"backend", "commit", "run_dir"} — provenance for replay
```

```
Rollout
├─ agent_id
├─ status                "ok" | "stalled" | "timeout" | "error"
├─ token_ids             int[] — the rollout's full token stream (final branch)
├─ tokens_recorded       bool — false when the backend returned no ids
├─ spans                 Span[] — the agent's turns, in order
└─ metrics               instrumented results (tests_pass, tests_modified)
```

```
Span
├─ id, kind ("turn" | "board_post"), agent_id, round
├─ text
├─ source_agent          set for kind "board_post" — who wrote it
└─ token_range           [start, end) into the rollout's token_ids, or null
```

## Span ↔ token alignment

`token_range` is what makes an episode white-box-capable: it lets a method point a
probe at exactly the tokens of "the round where agent 3 picked up the cheat" instead
of scoring the whole rollout. A method that needs activations re-derives them from
`token_ids` plus `model` (same checkpoint, teacher-forced prefill — see
`docs/desk-spikes-2026-09-27.md`, the sidecar design); the range says which positions
to read.

The verifiers backend emits a range for a round when, and only when, the trace grew
append-only during that round: the range is the number of tokens on the final branch
before and after `interaction.turn(...)` (`Branch.token_ids` concatenates each node's
`token_ids` along the root-to-leaf path). A renderer that rewrites history mid-rollout
(stripped reasoning, re-rendered context) makes the count shrink or stall, and the
round is recorded with `token_range: null` rather than a wrong range.

Range covers everything the backend appended that round — the fed-in user prompt and
the sampled output together. Methods that need sampled-only positions should intersect
with the prompt boundary themselves; we do not store a per-token mask in v0.

## Annotations (sibling format, same store discipline)

Methods do not write into episodes. A method's `run(episode)` returns `Annotation`
records — `{episode_id, agent_id?, span_ids, type, name, value, method, method_version}`
— and anything that wants them recomputed caches by `(episode id, method name,
method version)`. `type` "score" with a float value is what a detector emits for every
scored rollout; "label" is categorical (ground truth or discrete findings).

## Versioning

Additive fields stay inside `v0` — old stores still parse (`token_range` was added
this way). A breaking change bumps the format string and gets a log entry naming the
old and new version and the reason.
