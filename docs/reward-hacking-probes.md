# Reward-hacking probes: the Goodfire protocol, and our comparison

Durable note on Bergen, Bhalla, Lee et al., "Monitoring and Discovering Reward Hacking with
Internal Representations during LLM Evaluations" (Goodfire, arXiv:2609.19101v1, 16 Sep 2026),
read 2026-09-30 from the HTML the founder provided. It is the baseline our monitors are compared to
(docs/monitors.md).

## Their protocol

| Part | What they do |
|---|---|
| Models | Kimi K3, GLM 5.2, Qwen 3.8 Max (up to about 2.8T parameters); DeepSeek-V4-Pro in the readout table; Qwen 3.8 27B for steering |
| Environments | SWE-bench Verified, DeepSWE, ImpossibleBench (LiveCodeBench split, tools harness); transfer to τ-bench and ShoppingBench |
| Ground truth | An LLM judge (GPT-5.6 Sol, high effort) with a rubric tailored per benchmark, labelling every step (reasoning, messages, tool calls) with a supporting quote; run 3 times, only unanimous labels kept (96–99% of passages) |
| Baseline monitor | The same model with a generic rubric, one pass, a flag per passage |
| Probe | Difference of means at one layer: mean residual over token positions of hacking spans minus the same over matched honest spans; a token-level score h . w, flagged above a threshold |
| Probe data | Off-policy synthetic minimal pairs: 5,285 templated 9-turn SWE-agent conversations, five hack families (test-suite, validator, score-file tampering, information acquisition, reporting), 54 conditions; activations harvested at chosen locations (end of a hacking tool call, end of reading a prohibited file, around patches, claims to the user, the closing sentence), several poolings |
| Selection | Many probes (condition × location × layer); per model, the two with the highest AUROC on held-out rollouts (250 SWE-bench, 280 DeepSWE); results reported on the rest |
| Comparison | Passage level; the probe threshold set to the monitor's false-positive rate, true-positive rates compared; ROC curves; cascade (probe, then monitor) and probe OR monitor |

Headline numbers: at the monitor's false-positive rate, Kimi K3 probes catch 3.0 and 33.1 points
more hacks on DeepSWE and ImpossibleBench and 6.6 fewer on SWE-bench; GLM 5.2 7.9 fewer on
DeepSWE. Probe OR monitor raises Kimi K3's DeepSWE true-positive rate from 41.1% to 80.2% for 1
point more false positives. Kimi's SWE-bench-selected probe reaches AUROC 0.93 and 0.94 on DeepSWE
and ImpossibleBench. Probe hits on reasoning predict later hacks after resampling; about half of the
probe's "false positives" are shortcut deliberations the judge missed.

What matters for us: **their probe directions decode, under the logit lens, as "cheating",
"cheat", "hack", "hardcoded", "dishonest", "illicit", "sneak"** (their Table 1, layers 44–83),
although those words appear nowhere in the probe data. The paper names no code or data release and
does not mention Prime Intellect.

## Our comparison

The concept monitor builds its direction from those words directly, with no data: v_t = W[t] @ J_L
(the J-lens) or W[t] (the logit lens). The question: does a direction from words alone detect
hacking as well as a difference of means built from 5,285 designed conversations?

1. **Zero-cost check:** cosine similarity, at the same layer, between a difference-of-means
   direction and the concept directions for their Table 1 words.
2. **Detection:** on rollouts with labels, all three monitors (`probe_monitor`,
   `concept_monitor`, `concept_monitor(lens="logit")`) scored with `analysis.auroc` and
   `analysis.detection_at_fpr` at the same false-positive rates.
3. **Labels cheaply:** ImpossibleBench's tasks cannot be solved honestly, so a passing rollout is a
   hack by construction; rollout-level labels need no judge. Passage-level labels need a judge.
4. **Model ladder:** a model that fits one GPU first (Qwen 3.8 27B is the one they steer), their
   scale only with the Prime trainer backend.

Differences from their protocol to keep in mind: our probe baseline fits on on-policy labelled
rollouts (split), theirs on off-policy synthetic pairs; we score rollouts before passages.
