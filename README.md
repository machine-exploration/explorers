# explorers

> **Toward a science of deep learning.**
> Machine Exploration studies how intelligence arises in deep learning systems: how training builds the mechanisms a model computes with, and how to read those mechanisms once they exist. We start where reading models matters today: overseeing agent populations with white-box methods. We publish the research and the open stack it runs on. `explorers` is where the stack starts.

**Status: pre-alpha.** Nothing is released yet. The mission changed on 2026-09-29; the code in this repository so far covers the first horizon, agent oversight (see [What is here today](#what-is-here-today)). This README describes what we are building and in what order. Names and API are illustrative and will change.

PyPI: `machine-explorers` (planned) · `import explorers`

---

## Two horizons

- **Now: oversight of agent populations.** Agents run in populations that share tools, caches and channels, and they fail together. We build reproducible multi-agent scenarios and white-box monitors that read the models, not only their transcripts. This is where a result is useful soonest, and the code here today does it.
- **Long term: a science of deep learning.** How training builds the mechanisms a model computes with, studied at the join of learning mechanics and mechanistic interpretability.

The two feed each other. Oversight gives the science its most pressing question: when can a monitor that reads activations be trusted? The science answers it: when the feature the monitor reads forms in training, whether it causes the behaviour, whether it survives further training. Both run on one stack, with one `Method` interface and one annotation schema.

## Now: agent oversight

Agents now run as populations that share tools, caches and channels, and their failures can be collective: in July 2026 about 1,200 agents in an internal OpenAI evaluation found a shared cache, turned it into a message board and helped each other cheat ([METR](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/), [Redwood Research](https://www.redwoodresearch.org/research/hugging-face-incident)). Transcripts can be spoofed and logs rewritten, so oversight has to read the models, not only what they write.

White-box monitors (probes and other readings of activations) already catch reward hacking in open models ([Goodfire](https://arxiv.org/abs/2609.19101)). The science decides whether to trust them: when the feature a probe reads forms in training, whether it is causal, whether it survives further training or is obfuscated under pressure. Agent scenarios are where the stack's methods are tested on the behaviour that matters, against text monitors at the same false-positive rate.

**First scenario: the Hugging Face incident in miniature.** A handful of agents on impossible tasks, a shared message board, a sealed sandbox. Because the tasks are impossible, any passing solution is a hack, and a hack that spreads through the board is coordination. The labels come for free. The first question: **does anything inside the models signal the coordination before it shows in behaviour?** Every internal method is compared with a chain-of-thought monitor and an LLM judge at the same false-positive rate, and the result is published either way.

## Long term: a science of deep learning

Deep learning works far better than we can explain. We can train a model that writes code, but we cannot say which computation it performs, when in training that computation appeared, or why gradient descent found it rather than another one that fits the data as well. Engineering has outrun science.

Two fields attack the problem from opposite ends:

- **Mechanistic interpretability** reads a trained model: its features, its circuits, the algorithms its weights implement. It asks *what* the model computes.
- **Learning mechanics** treats training as a dynamical system: loss landscapes, feature learning, phase transitions, scaling laws. It asks *how* and *why* a computation forms.

Each lacks what the other has. Interpretability mostly studies one final checkpoint, so it sees the result of learning, not the process. The theory of training mostly tracks scalars (loss, norms, curvature), so it sees the process without knowing what is being learned.

The bridges that exist are few, small, and among the clearest results in the field:

- Grokking of modular addition is explained by a Fourier circuit that forms well before test loss drops ([Nanda et al.](https://arxiv.org/abs/2301.05217)).
- Induction heads form in a phase change that is visible as a bump in the loss curve ([Olsson et al.](https://arxiv.org/abs/2209.11895)).
- Toy models show when and how features are stored in superposition ([Elhage et al.](https://arxiv.org/abs/2209.10652)).
- The local learning coefficient from singular learning theory finds developmental stages in training that line up with what the model learns ([Lau et al.](https://arxiv.org/abs/2308.12108), [Hoogland et al.](https://arxiv.org/abs/2402.02364)).

> Our bet: the science of deep learning is made at the join. Measure mechanisms across training time, and explain their formation with dynamics.

### Research program

The questions, in the order we take them:

1. **When do mechanisms form?** Track features and circuits across dense checkpoints. Is formation sudden or gradual, and does anything in the weights or activations signal it before the loss does?
2. **Why this mechanism?** What in the data, architecture, optimizer and scale selects one algorithm among the many that fit the training set?
3. **What is universal?** Which mechanisms recur across seeds, sizes and architectures, and which are accidents of one run?
4. **Does it hold at scale?** Test what small models show on open model suites that publish their training checkpoints ([Pythia](https://arxiv.org/abs/2304.01373), [OLMo](https://arxiv.org/abs/2402.00838)).
5. **From description to prediction.** Predict what a run will learn, and when, before training it.

The method: start with **model organisms**, small tasks where the mechanism is known or can be found (modular arithmetic, sparse parity, in-context learning, toy superposition). Validate every measurement there against ground truth. Then carry it to open models where the answer is not known.

## The stack

Package names are working names.

| Part | Object | What it does |
|---|---|---|
| **`explorers`** (core) | formats, storage, studies | The episode and run formats, the annotation schema, the `Exploration` study unit, content-addressed caching. |
| **`agents`** (today `explorers.scenario`, `episode`, `runtime`) | agent populations | The scenario format, the multi-agent runtime and its episodes. Oversight methods run on these episodes. |
| **`mechanics`** | training over time | Reproducible, instrumented training runs with dense (log-spaced) checkpoints. Measurements of dynamics: loss decompositions, weight and gradient statistics, curvature (Hessian spectrum), local learning coefficient, effective rank. |
| **`interp`** (working name) | one checkpoint, read from inside | Reading activations at named sites, linear probes, dictionaries (sparse autoencoders), attribution, activation patching and ablation. |

All methods share one interface. A probe from `interp` runs on agent episodes for oversight and on training checkpoints for the science; a dynamics measure and a mechanism measure run on the same checkpoints and land in the same table, indexed by training step.

## Core abstraction

The goal is an abstraction where adding a method, a model organism, or compute never touches the core. It carries over from the agent-oversight code: immutable records, pure versioned methods, one annotation schema.

- **`Episode`: immutable data for oversight.** Spans (agent, turn, tool call, message, token range) for the agents that ran together. `branch` forks a new episode from any span.
- **`Run`: immutable data for the science.** Config, seed, data order, and checkpoints at chosen steps, with optimizer state. Nothing edits a run; `branch` forks a new one from any checkpoint with one change (data, learning rate, seed). That is the intervention experiment for dynamics.
- **`Method`: pure, versioned functions.** `method(inputs, needs) -> annotations`. Inputs are episode spans, run checkpoints, or other methods' annotations, so methods compose.
- **Declared needs.** A method says what it needs (`weights`, `read(sites)` for activations, `grad`, `hvp` for curvature, `intervene`, `branch`) and never calls the model itself. The runtime fuses every read on a rollout or a checkpoint into one pass, so ten methods cost about as much as one.
- **Two shapes.** `map` runs per rollout or per checkpoint, independently, and parallelises without limit. `reduce` runs over many of them through their annotations: fleet views and calibration for oversight; curves, change-point detection and universality across seeds for the science.
- **`Annotation`: one shared schema.** Target (episode span, or run and step), site, type (score, vector, label, text), value, method version, seed. Stored in columns and cached by content, so adding a method recomputes nothing.
- **`Exploration`: a study.** Scenarios or runs × models × methods → annotations → metrics against labels, or results over training time. The unit you rerun, share and compare.

### Illustrative API

```python
import explorers as ex
from explorers import mechanics, interp

run = mechanics.train(
    "modular-addition", model="transformer-1l", seed=0,
    checkpoints=mechanics.logspace(1, 50_000, n=200),
)

exp = ex.Exploration(
    runs=[run],
    methods=[
        mechanics.LLC(),
        mechanics.HessianTop(k=5),
        interp.Probe.linear(site="resid.1"),
        interp.FourierCircuit(),
    ],
)
results = exp.run()   # checkpoints -> annotations -> curves over training time
```

## First science study: grokking on one timeline

Reproduce grokking of modular addition end to end, with dense checkpoints, and put every measurement on the same time axis: train and test loss, weight norm, curvature, local learning coefficient, a linear probe, and the Fourier circuit's own progress measures.

The circuit is known, so this is ground truth for the stack. The first question: **which measurements move first, and does any of them signal the transition before test loss drops?** The result is published either way. Next: induction heads in small attention-only transformers, then the Pythia and OLMo checkpoints.

## Principles

- **Open by default:** the library, the formats, the runs and the results. Research is published whether the result is positive or negative.
- **Reproducible:** every result replays from its run config, seed, data order and code version.
- **Ground truth first:** a method is validated on a model organism where the mechanism is known before it is used where it is not.
- **Causal, not only correlational:** a claim that a mechanism exists is tested by intervention (ablation, patching, branching the run), not only by a probe that correlates.
- **Against baselines:** a new signal is compared with the simple ones (loss, weight norm, gradient norm) at the same budget. A white-box monitor is compared with chain-of-thought monitors and LLM judges at the same false-positive rate.
- **Small before large:** a question is settled on a model that fits one GPU before it is asked of a large one.
- **Contained:** scenarios that push agents to hack run with no network, no shared cache and no path between episodes. Publication rules for such scenarios and their traces are settled before release.
- **Audit isolation:** oversight monitors run outside the trainer's process and write to an append-only, hash-chained store that the reward code cannot read.

## What is here today

The current code is the first horizon, agent oversight: reproducible multi-agent scenarios where agents can hack and coordinate, with labels that come for free.

- `explorers.scenario`: scenario format v0 with fail-closed validation.
- `explorers.episode`: the immutable episode model and a JSONL store.
- `explorers.methods`: the `Method` / `Annotation` interface, with `hack_label` and `propagation`.
- `explorers.runtime.verifiers`: compiling a scenario for, and converting traces from, the [verifiers](https://github.com/PrimeIntellect-ai/verifiers) runtime.
- `explorers.exploration` and `explorers.metrics`: the study unit and Wilson rates.

The code's core ideas (immutable records, pure versioned methods, one annotation schema) are the ones the new stack generalises from episodes to training runs.

## Roadmap

1. **The first scenario**, hand-rolled, to measure whether agents hack and coordinate.
2. **v0.1:** the scenario format, the runtime, the first scenario, and two methods of different kinds (a judge on text and a probe on activations) running on the same episodes through one interface.
3. **Grokking on one timeline:** the first science study, on the same `Method` interface extended to training checkpoints.
4. **Where monitors come from:** follow the features that oversight probes read back through training, on model organisms first, then open model suites with public checkpoints (Pythia, OLMo).
5. **A second scenario family** (virtual marketplaces) and larger open-weight models.
6. **A shared hub** for scenarios, runs, methods and results.

## Open questions

- Names, and the package split: one package with subpackages, or separate packages.
- A checkpoint format and a storage budget. Dense checkpoints are large even for small models.
- Existing tools (TransformerLens, nnsight, SAELens, devinterp): build on them, or treat them as design references as we do verifiers.
- A model-agnostic way to name a site (layer, component, position) that survives across checkpoints and architectures.
- License: Apache-2.0 or MIT.

## Contributing

Pre-alpha, so the most useful contributions right now are conversations. If you train or evaluate agents in populations, build methods to read models, or study training dynamics and the theory of deep learning, and want a shared, reproducible substrate for it, open an issue.

Contributors and coding agents: read [AGENTS.md](AGENTS.md) first.

## References

Learning mechanics

- Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*, [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)
- Barak et al., *Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit*, [arXiv:2207.08799](https://arxiv.org/abs/2207.08799)
- Cohen et al., *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*, [arXiv:2103.00065](https://arxiv.org/abs/2103.00065)
- Yang & Hu, *Feature Learning in Infinite-Width Neural Networks*, [arXiv:2011.14522](https://arxiv.org/abs/2011.14522)
- Kaplan et al., *Scaling Laws for Neural Language Models*, [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Lau et al., *The Local Learning Coefficient: A Singularity-Aware Complexity Measure*, [arXiv:2308.12108](https://arxiv.org/abs/2308.12108)
- Hoogland et al., *The Developmental Landscape of In-Context Learning*, [arXiv:2402.02364](https://arxiv.org/abs/2402.02364)

Mechanistic interpretability

- Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability*, [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- Olsson et al., *In-context Learning and Induction Heads*, [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)
- Elhage et al., *Toy Models of Superposition*, [arXiv:2209.10652](https://arxiv.org/abs/2209.10652)
- Bricken et al., *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*, [transformer-circuits.pub](https://transformer-circuits.pub/2023/monosemantic-features)

Open models with training checkpoints

- Biderman et al., *Pythia*, [arXiv:2304.01373](https://arxiv.org/abs/2304.01373)
- Groeneveld et al., *OLMo*, [arXiv:2402.00838](https://arxiv.org/abs/2402.00838)

Agent oversight

- METR and Redwood Research, investigation of the OpenAI / Hugging Face hacking incident (2026): [METR](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/) · [Redwood Research](https://www.redwoodresearch.org/research/hugging-face-incident)
- Goodfire, *Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations*, [arXiv:2609.19101](https://arxiv.org/abs/2609.19101)
- Zhong et al., *ImpossibleBench*, [arXiv:2510.20270](https://arxiv.org/abs/2510.20270)
- Taufeeque et al., *The Obfuscation Atlas*, [arXiv:2602.15515](https://arxiv.org/abs/2602.15515)
- Gupta & Jenner, *RL-Obfuscation*, [arXiv:2506.14261](https://arxiv.org/abs/2506.14261)
- [verifiers](https://github.com/PrimeIntellect-ai/verifiers), design reference for the agent runtime
