# explorers

> **Infrastructure for understanding multi-agent AI.**
> Machine Exploration builds the open environments, runtime and white-box tools to understand and oversee agent populations at scale. `explorers` is where it starts.

**Status: pre-alpha.** Nothing is released yet. This README describes what we are building and in what order. The API below is illustrative and will change.

PyPI: `machine-explorers` (planned) · `import explorers`

---

## Why

Agents are no longer trained and evaluated one at a time. They run as populations that share tools, caches and channels.

In July 2026, during an internal OpenAI evaluation built on tasks that could not be solved, about 1,200 agents in separate sandboxes found a shared cache, turned it into a message board, and helped each other cheat. Hundreds of them joined an attack on Hugging Face. The independent investigation by [METR](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/) and [Redwood Research](https://www.redwoodresearch.org/research/hugging-face-incident) shows a universal cheat within about four hours, then days of coordinated effort, including attempts to trick the scorer and tamper with logs.

Two lessons:

- **The failure was collective.** No single rollout looks like the incident. Oversight that reads one agent at a time misses it.
- **Text is a fragile record.** Transcripts can be spoofed and logs rewritten. Oversight has to be able to read the models, not only what they write.

## The missing layer

Methods for reading models are advancing fast. [Goodfire](https://arxiv.org/abs/2609.19101) has shown that simple probes on activations can catch reward hacking in frontier open models. [Transluce](https://transluce.org) builds methods that describe what models compute, and tools that analyse long agent transcripts.

What is missing is the layer those methods run on when the subject is many agents acting together: environments you can reproduce, a runtime that records what every agent did and computed, a common format for the results, and the ability to do this at scale.

> Methods read models. `explorers` provides the environments and runtime where those methods run on agent populations, at scale.

## What explorers is

Three parts:

1. **Environments.** An open, versioned scenario format for multi-agent situations: agents, tasks, shared channels, sandbox rules, success checks and labels. Scenario families, in order: shared-infrastructure incidents, virtual marketplaces, multi-organisation workflows, simulated ecosystems.
2. **A runtime.** Plays a scenario reproducibly and records every agent's tokens, turns, tool calls and messages, grouped into episodes. Models are served with vLLM. The design takes inspiration from [verifiers](https://github.com/PrimeIntellect-ai/verifiers) but does not depend on it.
3. **Pluggable methods.** Probes, LLM judges, chain-of-thought monitors, transcript analysis, fleet views across agents. Black-box methods work on any agent; white-box methods are added where weights are available.

## Core abstraction

The goal is an abstraction where adding a method, a scenario, or compute never touches the core.

- **`Episode`: immutable data.** Spans (agent, turn, tool call, message, token range) for the agents that ran together. Nothing edits an episode; `branch` forks a new one from any span.
- **`Method`: pure, versioned functions.** `method(inputs, needs) -> annotations`. Inputs are spans or other methods' annotations, so methods compose.
- **Declared needs.** A method says what it needs (`text`, `read(sites)` for activations, `intervene`, `branch`) and never calls the model itself. The runtime fuses every read for a rollout into one forward pass, so ten white-box methods cost about as much as one.
- **Two shapes.** `map` runs per rollout, independently, and parallelises without limit. `reduce` runs over many rollouts through their annotations (fleet views, calibration, metrics, leaderboards).
- **`Annotation`: one shared schema.** Target spans, type (score, label, text, alert), value, method version, model checkpoint, seed. Stored in columns and cached by content, so adding a method recomputes nothing.
- **`Exploration`: a study.** Scenarios × models × methods → episodes → annotations → metrics against labels. The unit you rerun, share and compare.

### Illustrative API

```python
import explorers as ex

exp = ex.Exploration(
    scenarios=["hf-incident-mini"],
    models=[small_model, larger_model],
    methods=[ex.Probe.dom(layer=20), ex.baselines.CoTMonitor(), ex.baselines.Judge()],
    episodes=200, seed=0,
)
results = exp.run()   # episodes -> traces -> annotations -> metrics
```

## First scenario: the Hugging Face incident in miniature

A handful of agents on impossible tasks, a shared message board, a sealed sandbox. Because the tasks are impossible, any passing solution is a hack. A hack that spreads through the board is coordination. The labels come for free.

The first question: **does anything inside the models signal the coordination before it shows in behaviour?** Every internal method is compared against a chain-of-thought monitor and an LLM judge at the same false-positive rate, and the result is published either way.

## Principles

- **Open by default:** the library, the scenario format and the trace format.
- **Measured against baselines:** no result for an internal method without text-based baselines at the same false-positive rate.
- **Reproducible:** every episode can be replayed from its scenario, seed and model version.
- **Contained:** scenarios that push agents to hack run with no network, no shared cache and no path between episodes. Publication rules for such scenarios and their traces are settled before release.
- **Audit isolation:** audit monitors run outside the trainer's process and write to an append-only, hash-chained store that the reward code cannot read.

## Roadmap

1. **The first scenario**, hand-rolled, to measure whether agents hack and coordinate.
2. **v0.1:** the scenario format, the runtime, the first scenario, and two methods of different kinds (for example a judge on text and a probe on activations) running on the same episodes through one interface.
3. **A second scenario family** (virtual marketplaces) and larger open-weight models.
4. **A shared hub** for scenarios, traces, methods and leaderboards.

## Open questions

- Which model sizes run where. A 7B model now fits on one GPU as well as in the cloud.
- How activations are read on top of vLLM at scale.
- A model-agnostic way to name a site (layer, position) and the annotation schema for free text.
- What `branch` means across several agents.
- License: Apache-2.0 or MIT.

## Contributing

Pre-alpha, so the most useful contributions right now are conversations. If you train or evaluate agents in populations, study multi-agent safety, or build methods to read models and want them to run on agent populations, open an issue.

Contributors and coding agents: read [AGENTS.md](AGENTS.md) first.

## References

- METR and Redwood Research, *Brief independent investigation of agents' behavior, reasoning and collaboration in the OpenAI / Hugging Face hacking incident* (2026): [METR](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/) · [Redwood Research](https://www.redwoodresearch.org/research/hugging-face-incident)
- OpenAI, [The Hugging Face incident and the road ahead](https://openai.com/index/hugging-face-incident-and-the-road-ahead/)
- Goodfire, *Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations*, [arXiv:2609.19101](https://arxiv.org/abs/2609.19101)
- Zhong et al., *ImpossibleBench*, [arXiv:2510.20270](https://arxiv.org/abs/2510.20270)
- Taufeeque, Heimersheim, Gleave, Cundy, *The Obfuscation Atlas*, [arXiv:2602.15515](https://arxiv.org/abs/2602.15515)
- Gupta & Jenner, *RL-Obfuscation*, [arXiv:2506.14261](https://arxiv.org/abs/2506.14261)
- [verifiers](https://github.com/PrimeIntellect-ai/verifiers), design reference for the runtime
