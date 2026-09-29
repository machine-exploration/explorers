# Roadmap

This file lists the stages of the work, in order. Each stage has a question, a deliverable, and a condition that says when it is done. There are no dates.

## How the two horizons connect

The link from oversight to a science of deep learning goes through mechanistic interpretability. A white-box monitor is an interpretability claim put to work: "this feature, read at this site, signals this behaviour." Learning mechanics can support that claim in two ways ([Simon et al., §3.1](https://arxiv.org/abs/2604.21691)).

- **It can test the assumptions a monitor rests on.** Every monitor assumes some of the core assumptions of interpretability: linear representability, locality, sparsity, compositionality. On a fixed model we can test each assumption empirically (stage 2). A theory of learning would say in which regimes each assumption holds, and so predict when a monitor will fail on a new model. That theory is still open (open direction 4), so today this link gives audits, not predictions.
- **It can explain how the monitored feature forms and changes during training.** Does the probe still work on the next checkpoint? When does the feature form? Does training against the monitor hide it? These are questions about how representations change under gradient updates, which is the subject of learning mechanics. This link is testable now (stage 3).

So the bridge is **the assumptions behind monitors, and monitors under training**. Stages 0 to 2 are oversight and interpretability, with assumption audits. The dynamics enter at stage 3.

### The role of learning mechanics

Learning mechanics gives predictions about how representations change during training. We use those predictions as hypotheses about monitored features, and we test them. We do not build the theory ourselves at first. The numbers refer to the [open directions](https://learningmechanics.pub/openquestions/).

| Prediction from learning mechanics | Direction | Hypothesis about a monitor |
|---|---|---|
| Learning proceeds as a sequence of units | 11 | **Formation.** The hack feature becomes decodable at a discrete step, before the hack rate rises. |
| Weight updates are approximately low-rank | 12 | **Drift.** A probe loses accuracy when a large low-rank update hits the layer it reads. |
| Optimisation is biased toward simple solutions | 3 | **Obfuscation.** Under pressure against a probe, the model moves the feature along the cheapest direction. This predicts which monitors survive. |
| Models trained differently learn similar representations | 10 | **Transfer.** A probe trained on one model works on another after a representation alignment. |

### Learning mechanics ⇄ mechanistic interpretability, applied to monitors

**1. The assumptions behind a monitor.** Each core assumption of interpretability is a place where a monitor can fail. We audit each one for each monitored feature, with a `Method` that returns annotations like any other.

| Assumption | How a monitor depends on it | How the monitor fails if it is false | Audit | What learning mechanics would add |
|---|---|---|---|---|
| **Linear representability:** a feature is a direction in activation space | A linear probe reads one direction | The feature is nonlinear or spread over many directions; the probe misses it | Gap between a linear and a nonlinear probe; steering along the probe direction changes the behaviour | The regimes (data statistics, training) where features are linear |
| **Locality:** a feature lives at a few sites | The probe reads one layer | The feature is spread over layers, or moves to another layer during training | Probe sweep over layers; activation patching per site; the best site at each checkpoint | When and why a computation localises |
| **Sparsity:** a feature is active on few inputs | Alert thresholds and dictionary (SAE) features assume rare activation | A dense feature gives a high false-positive rate; feature absorption gives misses | Activation frequency of the monitored feature on benign episodes | When training gives sparse codes (superposition) |
| **Compositionality:** complex features are built from simpler ones | A coordination monitor can combine simpler monitors (hack intent, reads the board, copies a message) | Coordination is a new feature that the parts do not predict | Compare a direct coordination probe with a combination of component probes | How compositions form in training (in-context learning, induction heads) |

**2. How mechanisms form.** Interpretability describes what a model computes. Learning mechanics can explain how and why it came to compute it. "Nothing in biology makes sense except in the light of evolution" ([Simon et al.](https://arxiv.org/abs/2604.21691), after Saphra): a monitor may make sense only in the light of the training that built its feature. The audits above run at each checkpoint in stage 3. One hypothesis links the two parts: **obfuscation is a change in which assumption holds.** For example, a feature that is linear and local early in training becomes distributed under pressure against a linear probe.

**3. The reverse direction: phenomena for theory.** Interpretability gives learning mechanics concrete targets: induction heads, Fourier features, feature geometry. Our scenarios give new ones, with labels for free and controlled data structure: hack features, coordination features, and how they form under RL. Each phenomenon goes into a public catalogue with the run that reproduces it, its measurements, and the open question it poses to theory. Phenomena from hacking scenarios follow the publication rules (README, Principles). Phenomena from model organisms can be published earlier.

## Stages

### Stage 0 — Do small models hack and coordinate?

- **Question:** Do models that fit one GPU (0.6B to 4B) hack and coordinate in the first scenario?
- **Deliverable:** Hack rate, propagation rate and board activity for each model, with Wilson intervals, on 200 episodes per model. Labels checked by hand on a sample of 50.
- **Done when:** One model shows a hack rate high enough to train a probe on. If none does, move to larger open models on rented GPUs, or start with a single-agent impossible-task setting.
- **Exists:** Scenario format v0, runtime through verifiers, `hack_label`, `propagation` (phrase-echo heuristic v0), `Exploration.report`.

### Stage 1 — Read activations

- **Question:** Can we read the activations of every recorded rollout?
- **Deliverable:** A sidecar scorer. It loads the same checkpoint, runs one prefill pass over the recorded token ids, and returns activations at named sites. It runs in its own process (audit isolation). Design in [docs/desk-spikes-2026-09-27.md](docs/desk-spikes-2026-09-27.md).
- **Done when:** The prefill reproduces the log-probs of the sampled tokens within numerical tolerance.

### Stage 2 — v0.1: monitors at inference

- **Question:** Does anything inside the models signal hacking and coordination before it shows in behaviour, and better than text monitors?
- **Deliverable:** On the same episodes: an LLM judge, a chain-of-thought monitor, a linear probe, a nonlinear probe, and a steering test along the probe direction (open direction 4). A judge replaces the phrase-echo heuristic for coordination. Assumption audits (linearity, locality, sparsity, compositionality) for each monitored feature.
- **Done when:** Results at 1 % and 5 % false-positive rate, with intervals, with train and test split by task, and one audit table per monitored feature. Published whether positive or negative.

### Stage 3 — Monitors under training (the bridge)

- **Question:** How do monitored features form, drift and hide during training?
- **Deliverable:** RL on hackable tasks with a dense checkpoint series. At each checkpoint: probe accuracy, hack rate on a held-out set, rank of the weight updates in the layers the probe reads, loss and reward steps, and the four assumption audits. A second run trains against the probe (obfuscation). Test whether obfuscation shows as a change in which assumption holds.
- **Baseline:** Re-evaluating the probe on labelled held-out data at each checkpoint. A dynamics signal has value only if it tells us something this baseline does not, or tells it earlier or without labels.
- **Done when:** Each of formation, drift and obfuscation has a result against the baseline.
- **Needs:** Two GPUs for RL (or a hosted trainer). A format change: `Annotation` gains a site and a training step (versioned).

### Stage 4 — Transfer

- **Question:** Does a probe trained on one model work on another (other seed, other size)?
- **Deliverable:** Detection quality of a transferred probe, with and without representation alignment (open direction 10).

### Stage 5 — The wider science

- **Question:** Do the results of stage 3 hold beyond hacking?
- **Deliverable:** The same measurements on model organisms where the mechanism is known (grokking of modular addition as a tool check), then on open models with public training checkpoints (Pythia, OLMo). A public catalogue of phenomena for theorists: each entry with its run, its measurements and its open question.
- **Starts only if:** Stage 3 shows that dynamics measures predict something the baseline does not.

### Stage 6 — More scenarios, shared hub

- A second scenario family (virtual marketplaces) and larger open-weight models.
- A shared hub for scenarios, runs, methods and results.

## Known risks

- Small models may not hack or coordinate. Stage 0 tests this first.
- The coordination label is a heuristic until the judge replaces it in stage 2.
- There is no activation path yet. Stage 1 builds it.
- RL does not fit well on one consumer GPU. Stage 3 needs rented compute.
- Dense checkpoints are large. Store only the layers that are read, or low-rank deltas, and keep full checkpoints at log-spaced steps.
