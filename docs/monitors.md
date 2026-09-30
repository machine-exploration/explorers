# Monitors

Durable note on the monitors in `explorers` (`measures.py`, `analysis.py`). Written 2026-09-30.

A monitor turns a model's activations on an example (a text, a replayed episode) into one score:
it projects residual:L on a direction at every scored position, then reduces the positions. Two
directions are compared on the same examples, at the same cost per token:

| Monitor | Direction at layer L | Labels | Fit cost |
|---|---|---|---|
| `concept_monitor(L, token_ids)` | v_t = W[t] @ J_L for each token t; the position's score is the max over t | None | One backward pass per token, over unlabelled examples |
| `probe_monitor(L, label="label")` | Difference of means: mean over positive fit examples of their mean scored activation, minus the same for negatives | A 0/1 column `label` | Forward passes only |

`W` is `unembed_matrix` (the linear part of the final norm and unembedding), so h . v_t is the
linear part of token t's J-lens logit at that position: the concept lens (Gurnee et al. 2026) read
as a probe, without labels. The final norm's per-position scale is left out; it is a positive factor
at each position, so it changes scores across positions and examples but not the sign.

- **Concept rows** come from the read `concept:<L>:<skip>:<target>:<ids>`: the same backward sweep
  as the full Jacobian (`execute.jacobians`), with the cotangent set to W[t] instead of a one-hot,
  so it costs k passes instead of d. Fitted on `split == "fit"` rows.
- **Scored positions**: the examples' loss mask when there is one (the model's own turns in a
  replayed episode), else from `skip_first` to the one before last.
- **Reduction** over positions: `max` (default), `mean` or `last`.
- **Detection**: `analysis.detection_at_fpr(y, scores, fpr)` is the fraction of positives flagged at
  the threshold where at most `fpr` of negatives are flagged (strictly above the threshold, so ties
  never exceed the budget). Compare monitors at the same `fpr`; `analysis.auroc` gives the ranking
  quality over all thresholds.

Checks (`tests/test_monitors.py`), exact by construction: the concept rows equal W[t] @ J_L from
the full Jacobian; `concept_monitor` equals the direct projection; `probe_monitor` equals a direct
difference of means and ignores the labels of evaluation rows; `reduce_positions` and
`detection_at_fpr` on constructed inputs, ties included.

Compared against: docs/reward-hacking-probes.md (the Goodfire protocol). `concept_monitor(..., lens="logit")` is the same monitor with W[t] alone (nothing fitted), the baseline.

Not yet: episode replay (episodes from `monitors/file/traces` of an eval or RL run as examples with
their loss mask, docs/prime.md), the exact readout with the norm scale, aggregation over several
layers, and running on 70B+ models (the Prime trainer backend).
