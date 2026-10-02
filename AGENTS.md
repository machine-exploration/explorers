# Agent working agreement

This repo is the `explorers` library, the open stack for Machine Exploration's mission: understand how intelligence arises in deep learning systems and build a science of deep learning, through learning mechanics (training dynamics) and mechanistic interpretability. Two horizons: now, white-box oversight of training and post-training runs of large models; long term, the science. The agent-side code (`populations`) is frozen until the agent questions return: keep its tests passing, do not extend it.

The org vision and roadmap live in [machine-exploration/public](https://github.com/machine-exploration/public) (README.md, ROADMAP.md). Update them there, not here. Thesis: a neural network is a learned computation: its weights are the program, and its internal streams carry the running state; Machine Exploration builds the lab to train, read, write and trace both. Works with any training stack: whatever trains, serves and scores a model stays where it is, and Explorers, the open-source white-box layer, reads what a run writes (checkpoints, adapters, rollouts) through a thin adapter per stack and measures it. The first integration is Prime Intellect (pods, prime-rl, verifiers, vLLM); keep the core stack-neutral, with stack-specific code only in its adapter; Mechanics is the science: learning mechanics joined with mechanistic interpretability, climbing a ladder from controlled pretraining to RL. Explorers is the lab it runs in: today a library that reads models and runs; by design (org README, "The lab"; not yet built) a Tinker-like API to train, read and intervene in one loop, with environments in Prime Intellect's format read through an adapter and Modal as the first backend. We build no production trainer and no inference engine. The org's repositories:

- `explorers` (this repo): the open-source library, one package `explorers` (`src/explorers`). Six concepts: Model, Stream, Trace, Op, Measure, Study; the design is one page, `docs/interface.md`, and must stay one page. One way to do each thing: `Study` is the only entry point, `execute.serve` the only place that runs a model for measures. Remove before adding. `import explorers` must not import torch. `explorers.populations` (the agent side) is frozen behind the `populations` extra.
- `mechanics`: the science (experiments, datasets, figures, papers); first experiment: `glp-activation` (a generative model of activations, fitted across training), then the onset law in controlled pretraining (q2). It depends on `explorers` and holds no library code.
- `public`: vision, roadmap, and later research notes and results.
- `verifiers`, `vllm`: pinned forks of upstream projects. No local changes, except our own environments in the verifiers fork, under `environments/` (first: `impossible_code`, the Monitor Arena's first track).

Library rules: every check of a method is exact by construction where possible (see `docs/interface.md`); every study must give the same result on every supported backend within a stated tolerance; speed claims are measured against existing tools (NNsight, TransformerLens) on the same study. TransformerLens and NNsight may be backends, behind an optional extra, imported only in their backend module.

The public README of this repo is [README.md](README.md). Run the tests with `uv run pytest` from the root. The detailed internal README (founder plan) is `private/README.md`, the first use case is `private/first-use-case.md`, and the chronology is `private/LOG.md`. All are local only, ignored by git, never published. Keep strategy and plans (runway, funding, competitors, decision points, timelines) there, not in tracked files. Ask the founder before making any new file public.

## Log every session

`private/LOG.md` is the chronology. Append to it; do not rewrite earlier entries.

Before changing anything, append a dated entry with:

- the intent
- what you are about to read or edit

After the change, append:

- what was done
- why
- which files moved
- the decision
- the next step

If a session learns how a runtime, a backend, a training setup, a measurement method, or a reference framework works, write that into `docs/` in the same session. The log is the chronology. The doc is the durable note.

## Git

Commit and push directly to `main`. Do not open feature branches or pull requests unless the founder asks for one.

## Backends and reference frameworks

[verifiers](https://github.com/PrimeIntellect-ai/verifiers) is the runtime backend for scenarios. It is an optional extra of `explorers` (`verifiers`), installed from the pinned fork `machine-exploration/verifiers`, and imported lazily and only in `explorers.populations.runtime.verifiers`. Do not import it anywhere else, and do not vendor its code.

prime-rl is the first training stack we read, not a dependency: `explorers` reads the files a run writes (`docs/prime.md`) and never imports it. Inspect, devinterp, WeightWatcher and mup are design references, not dependencies.

To study a reference, clone it outside this git tree (for example a sibling folder `../verifiers`). Record the commit you read in the log and in the design note, so the note can be checked against the code later. Do not vendor or copy its code into this repo; write our own.

## Formats

The scenario format and the trace format are the parts others will depend on. Version them. A breaking change needs a log entry that names the old and new version and the reason.

## Scenarios

Scenarios that make agents hack or coordinate run with no network, no shared package cache, and no path between episodes. Do not publish a scenario or its traces before the publication rules (`private/README.md` §10, README "Principles") are settled.

## Research results

A result is published with the run that reproduces it: config, seed, data order, code version. Validate a method on a model organism with a known mechanism before using it where the mechanism is unknown. Negative results are published too.
