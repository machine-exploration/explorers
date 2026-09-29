# explorers

> **Oversight that reads models, and a science of how they learn.**
> Machine Exploration builds white-box oversight for agent populations, and studies how intelligence arises in deep learning systems. We publish the research and the open stack it runs on. `explorers` is where the stack starts.

**Status: pre-alpha.** Nothing is released yet. The code in this repository covers the first horizon, agent oversight (see [What is here today](#what-is-here-today)). This README describes what we are building and in what order. Names and API are illustrative and will change.

PyPI: `machine-explorers` (planned) · `import explorers`

---

## Two horizons

- **Now: oversight of agent populations.** Agents run in populations that share tools, caches and channels, and they fail together. We build reproducible multi-agent scenarios and white-box monitors that read the models, not only their transcripts.
- **Long term: a science of deep learning.** How training builds the mechanisms a model computes with, at the join of learning mechanics and mechanistic interpretability.

The link between them is narrow, and we state it precisely: **monitors under training**.

On a fixed model, you trust a monitor through evaluation and interpretability. Learning mechanics adds nothing there. But models are trained and retrained all the time. Then new questions appear: when does the feature a monitor reads form? Does the probe still work on the next checkpoint? Does training against the monitor hide the feature? These are questions about how representations change under gradient updates, which is what learning mechanics studies. Details in [ROADMAP.md](ROADMAP.md).

## Now: agent oversight

Agents now run as populations, and their failures can be collective. In July 2026 about 1,200 agents in an internal OpenAI evaluation found a shared cache, turned it into a message board and helped each other cheat ([METR](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/), [Redwood Research](https://www.redwoodresearch.org/research/hugging-face-incident)). No single rollout looks like the incident. Transcripts can be spoofed and logs rewritten, so oversight has to read the models, not only what they write.

White-box monitors (probes and other readings of activations) already catch reward hacking in open models ([Goodfire](https://arxiv.org/abs/2609.19101)). Agent scenarios are where these methods are tested on the behaviour that matters, against text monitors at the same false-positive rate.

**First scenario: the Hugging Face incident in miniature.** A handful of agents on impossible tasks, a shared message board, a sealed sandbox. Because the tasks are impossible, any passing solution is a hack, and a hack that spreads through the board is coordination. The labels come for free.

The first question: **does anything inside the models signal the coordination before it shows in behaviour?** Every internal method is compared with a chain-of-thought monitor and an LLM judge at the same false-positive rate. The result is published either way.

## Long term: a science of deep learning

Deep learning works far better than we can explain. We can train a model that writes code, but we cannot say which computation it performs, when in training that computation appeared, or why gradient descent found it rather than another one that fits the data as well.

Two fields attack the problem from opposite ends:

- **Mechanistic interpretability** reads a trained model: its features, its circuits, the algorithms its weights implement. It asks *what* the model computes.
- **Learning mechanics** treats training as a dynamical system: loss landscapes, feature learning, phase transitions, scaling laws. It asks *how* and *why* a computation forms ([Simon et al.](https://arxiv.org/abs/2604.21691)).

Interpretability mostly studies one final checkpoint, so it sees the result of learning, not the process. The theory of training mostly tracks scalars (loss, norms, curvature), so it sees the process without knowing what is being learned. The clearest results sit at the join: the Fourier circuit behind grokking ([Nanda et al.](https://arxiv.org/abs/2301.05217)), induction heads forming in a phase change ([Olsson et al.](https://arxiv.org/abs/2209.11895)), superposition in toy models ([Elhage et al.](https://arxiv.org/abs/2209.10652)), developmental stages found by the local learning coefficient ([Lau et al.](https://arxiv.org/abs/2308.12108), [Hoogland et al.](https://arxiv.org/abs/2402.02364)).

### Research program

The learning-mechanics community keeps a list of [open directions](https://learningmechanics.pub/openquestions/). We use its predictions as hypotheses about monitored features, and test them. We also audit the core assumptions of interpretability that every monitor rests on (linear representability, locality, sparsity, compositionality), first on fixed models, then across training ([ROADMAP.md](ROADMAP.md)). We take the four that bear on monitors. Each experiment gives a result about monitors and a result for the science.

| Open direction | Why it matters for oversight | First experiment |
|---|---|---|
| **4.** How do we formally define the features a network learns? | A probe assumes a behaviour such as hacking is a linear direction in activations. If that is false, probe monitors fail. | On the first scenario, compare a linear and a nonlinear probe for "hack" at the same false-positive rate. Steer activations along the probe direction and check whether the behaviour changes. |
| **11.** Does learning decompose into a sequence of "units"? | If features form one by one, we can find when the monitored feature appears. | During RL on hackable tasks, record probe accuracy and hack rate at each checkpoint. Check whether the probe moves before the behaviour. |
| **12.** Does training proceed as a series of low-rank steps? | A step that adds the monitored feature, or hides it, may be visible in the weight updates. | At the same checkpoints, record the rank of the weight updates in the layers the probe reads. Check whether large updates predict a drop in probe accuracy. |
| **10.** Do models trained differently learn similar representations? | If they do, a probe trained on one model can work on another, and monitors get cheaper and easier to trust. | Train a probe on model A. Apply it to model B (other seed or size) through a representation alignment. Measure the drop in detection. |

Each dynamics signal is compared with a cheap baseline: re-evaluating the probe on labelled held-out data at each checkpoint. If the results hold, the same measurements move to model organisms where the mechanism is known (grokking of modular addition) and to open models with public training checkpoints ([Pythia](https://arxiv.org/abs/2304.01373), [OLMo](https://arxiv.org/abs/2402.00838)).

## The stack

For now, one package. The parts below are split out only when a result needs them. Names are working names.

| Part | Object | Status |
|---|---|---|
| **`explorers`** (core) | formats, storage, studies | Episode format, annotation schema, `Exploration` study unit. Exists. |
| **agents** (`explorers.scenario`, `episode`, `runtime`) | agent populations | Scenario format, multi-agent runtime, episodes. Exists. |
| **mechanics** (planned) | training over time | Reproducible runs with dense checkpoints; dynamics measures (weight-update rank, curvature, local learning coefficient). |
| **interp** (planned) | one checkpoint, read from inside | Activations at named sites, linear and nonlinear probes, steering, patching. |

All methods share one interface. The same probe runs on agent episodes for oversight and on training checkpoints for the science.

## Core abstraction

The goal is an abstraction where adding a method, a scenario, a model organism or compute never touches the core.

- **`Episode`: immutable data for oversight.** Spans (agent, turn, tool call, message, token range) for the agents that ran together. `branch` forks a new episode from any span.
- **`Run`: immutable data for the science.** Config, seed, data order, and checkpoints at chosen steps. `branch` forks a new run from any checkpoint with one change.
- **`Method`: pure, versioned functions.** `method(inputs, needs) -> annotations`. Inputs are episode spans, run checkpoints, or other methods' annotations, so methods compose.
- **Declared needs.** A method says what it needs (`text`, `weights`, `read(sites)`, `grad`, `intervene`, `branch`) and never calls the model itself. The runtime fuses every read on a rollout or a checkpoint into one pass, so ten methods cost about as much as one.
- **Two shapes.** `map` runs per rollout or per checkpoint and parallelises without limit. `reduce` runs over many of them through their annotations: fleet views and calibration for oversight; curves and change points over training for the science.
- **`Annotation`: one shared schema.** Target (episode span, or run and step), site, type (score, vector, label, text, alert), value, method version, seed. Stored in columns and cached by content, so adding a method recomputes nothing.
- **`Exploration`: a study.** Scenarios or runs × models × methods → annotations → metrics. The unit you rerun, share and compare.

### Illustrative API

```python
import explorers as ex

exp = ex.Exploration(
    scenarios=["hf-incident-mini"],
    models=[small_model, larger_model],
    methods=[ex.Probe.linear(layer=20), ex.Probe.mlp(layer=20),
             ex.baselines.CoTMonitor(), ex.baselines.Judge()],
    episodes=200, seed=0,
)
results = exp.run()   # episodes -> annotations -> metrics against labels
```

## Principles

- **Open by default:** the library, the formats, the runs and the results. Research is published whether the result is positive or negative.
- **Reproducible:** every episode replays from its scenario, seed and model version; every run from its config, seed, data order and code version.
- **Against baselines:** a white-box monitor is compared with chain-of-thought monitors and LLM judges at the same false-positive rate. A new training signal is compared with loss, weight norm and gradient norm.
- **Causal, not only correlational:** a claim that a feature exists is tested by intervention (steering, ablation, patching, branching), not only by a probe that correlates.
- **Ground truth first:** a method is validated where the answer is known (labelled scenarios, model organisms) before it is used where it is not.
- **Small before large:** a question is settled on a model that fits one GPU before it is asked of a large one.
- **Contained:** scenarios that push agents to hack run with no network, no shared cache and no path between episodes. Publication rules for such scenarios and their traces are settled before release.
- **Audit isolation:** oversight monitors run outside the trainer's process and write to an append-only, hash-chained store that the reward code cannot read.

## What is here today

- `explorers.scenario`: scenario format v0 with fail-closed validation.
- `explorers.episode`: the immutable episode model and a JSONL store.
- `explorers.methods`: the `Method` / `Annotation` interface, with `hack_label` and `propagation`.
- `explorers.runtime.verifiers`: compiling a scenario for, and converting traces from, the [verifiers](https://github.com/PrimeIntellect-ai/verifiers) runtime.
- `explorers.exploration` and `explorers.metrics`: the study unit and Wilson rates.

## Roadmap

The stages, with questions, deliverables and exit conditions, are in [ROADMAP.md](ROADMAP.md). In short:

0. Do small models hack and coordinate in the first scenario?
1. Read activations for recorded rollouts (a sidecar scorer).
2. **v0.1:** text monitors, linear and nonlinear probes, and a steering test on the same episodes.
3. **Monitors under training:** formation, drift and obfuscation of the monitored feature during RL. This is where the science enters.
4. Transfer of a probe across models.
5. The wider science, if stage 3 shows that dynamics signals add to the baseline.
6. More scenario families and a shared hub.

## Open questions

- Whether models that fit one GPU hack and coordinate in the first scenario.
- How activations are read on top of vLLM at scale.
- A model-agnostic way to name a site (layer, component, position) that survives across checkpoints and architectures.
- A checkpoint format and a storage budget. Dense checkpoints are large even for small models.
- Existing tools (TransformerLens, nnsight, SAELens, devinterp): build on them, or treat them as design references as we do verifiers.
- License: Apache-2.0 or MIT.

## Contributing

Pre-alpha, so the most useful contributions right now are conversations. If you train or evaluate agents in populations, build methods to read models, or work on learning mechanics, open an issue.

Contributors and coding agents: read [AGENTS.md](AGENTS.md) first.

## References

Agent oversight

- METR and Redwood Research, investigation of the OpenAI / Hugging Face hacking incident (2026): [METR](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/) · [Redwood Research](https://www.redwoodresearch.org/research/hugging-face-incident)
- Goodfire, *Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations*, [arXiv:2609.19101](https://arxiv.org/abs/2609.19101)
- Zhong et al., *ImpossibleBench*, [arXiv:2510.20270](https://arxiv.org/abs/2510.20270)
- Taufeeque et al., *The Obfuscation Atlas*, [arXiv:2602.15515](https://arxiv.org/abs/2602.15515)
- Gupta & Jenner, *RL-Obfuscation*, [arXiv:2506.14261](https://arxiv.org/abs/2506.14261)
- [verifiers](https://github.com/PrimeIntellect-ai/verifiers), design reference for the agent runtime

Learning mechanics

- Simon et al., *There Will Be a Scientific Theory of Deep Learning*, [arXiv:2604.21691](https://arxiv.org/abs/2604.21691), and the [open directions](https://learningmechanics.pub/openquestions/)
- Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*, [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)
- Lau et al., *The Local Learning Coefficient: A Singularity-Aware Complexity Measure*, [arXiv:2308.12108](https://arxiv.org/abs/2308.12108)
- Hoogland et al., *The Developmental Landscape of In-Context Learning*, [arXiv:2402.02364](https://arxiv.org/abs/2402.02364)

Mechanistic interpretability

- Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability*, [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- Olsson et al., *In-context Learning and Induction Heads*, [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)
- Elhage et al., *Toy Models of Superposition*, [arXiv:2209.10652](https://arxiv.org/abs/2209.10652)

Open models with training checkpoints

- Biderman et al., *Pythia*, [arXiv:2304.01373](https://arxiv.org/abs/2304.01373)
- Groeneveld et al., *OLMo*, [arXiv:2402.00838](https://arxiv.org/abs/2402.00838)
