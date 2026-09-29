# Roadmap

This file lists the stages of the work, in order. Each stage has a question, a deliverable, and a condition that says when it is done. There are no dates.

## How the two horizons connect

The link from oversight to a science of deep learning is narrow. We state it precisely.

- **On a fixed model, at inference time,** you trust a monitor through evaluation (detection at a fixed false-positive rate, on held-out tasks) and through interpretability (steering and ablation along the probe direction). Learning mechanics adds nothing there.
- **When the model changes under training,** new questions appear. Does the probe still work on the next checkpoint? When does the monitored feature form? Does training against the monitor hide the feature? These are questions about how representations change under gradient updates. That is the subject of learning mechanics.

So the bridge is **monitors under training**. Stages 0 to 2 are oversight and interpretability. The science enters at stage 3.

### The role of learning mechanics

Learning mechanics gives predictions about how representations change during training. We use those predictions as hypotheses about monitored features, and we test them. We do not build the theory ourselves at first. The numbers refer to the [open directions](https://learningmechanics.pub/openquestions/).

| Prediction from learning mechanics | Direction | Hypothesis about a monitor |
|---|---|---|
| Learning proceeds as a sequence of units | 11 | **Formation.** The hack feature becomes decodable at a discrete step, before the hack rate rises. |
| Weight updates are approximately low-rank | 12 | **Drift.** A probe loses accuracy when a large low-rank update hits the layer it reads. |
| Optimisation is biased toward simple solutions | 3 | **Obfuscation.** Under pressure against a probe, the model moves the feature along the cheapest direction. This predicts which monitors survive. |
| Models trained differently learn similar representations | 10 | **Transfer.** A probe trained on one model works on another after a representation alignment. |

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
- **Deliverable:** On the same episodes: an LLM judge, a chain-of-thought monitor, a linear probe, a nonlinear probe, and a steering test along the probe direction (open direction 4). A judge replaces the phrase-echo heuristic for coordination.
- **Done when:** Results at 1 % and 5 % false-positive rate, with intervals, with train and test split by task. Published whether positive or negative.

### Stage 3 — Monitors under training (the bridge)

- **Question:** How do monitored features form, drift and hide during training?
- **Deliverable:** RL on hackable tasks with a dense checkpoint series. At each checkpoint: probe accuracy, hack rate on a held-out set, rank of the weight updates in the layers the probe reads, loss and reward steps. A second run trains against the probe (obfuscation).
- **Baseline:** Re-evaluating the probe on labelled held-out data at each checkpoint. A dynamics signal has value only if it tells us something this baseline does not, or tells it earlier or without labels.
- **Done when:** Each of formation, drift and obfuscation has a result against the baseline.
- **Needs:** Two GPUs for RL (or a hosted trainer). A format change: `Annotation` gains a site and a training step (versioned).

### Stage 4 — Transfer

- **Question:** Does a probe trained on one model work on another (other seed, other size)?
- **Deliverable:** Detection quality of a transferred probe, with and without representation alignment (open direction 10).

### Stage 5 — The wider science

- **Question:** Do the results of stage 3 hold beyond hacking?
- **Deliverable:** The same measurements on model organisms where the mechanism is known (grokking of modular addition as a tool check), then on open models with public training checkpoints (Pythia, OLMo).
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
