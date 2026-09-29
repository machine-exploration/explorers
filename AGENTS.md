# Agent working agreement

This repo is the `explorers` library, the open stack for Machine Exploration's mission: understand how intelligence arises in deep learning systems and build a science of deep learning, through learning mechanics (training dynamics) and mechanistic interpretability. Two horizons: now, oversight of agent populations with white-box methods; long term, the science. Keep the oversight code working and moving.

The org vision and roadmap live in [machine-exploration/public](https://github.com/machine-exploration/public) (README.md, ROADMAP.md). Update them there, not here. The org's repositories:

- `explorers` (this repo): the populations prototype today (scenario format, multi-agent runtime, episodes, labels, detectors); the home of the unified workspace after roadmap step 0.
- `mechanics`: the shared core (`explorers-core`: examples, states, observables, engine, store) and `explorers-learning`. It moves here in step 0. Until then, this repo's `explorers/__init__.py` hides the `explorers.core` namespace, so do not install both in one environment.
- `public`: vision, roadmap, and later research notes and results.
- `verifiers`, `vllm`: pinned forks of upstream projects. No local changes.

The public README of this repo is [README.md](README.md). The detailed internal README (founder plan) is `private/README.md`, the first use case is `private/first-use-case.md`, and the chronology is `private/LOG.md`. All are local only, ignored by git, never published. Keep strategy and plans (runway, funding, competitors, decision points, timelines) there, not in tracked files. Ask the founder before making any new file public.

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

[verifiers](https://github.com/PrimeIntellect-ai/verifiers) is the runtime backend for scenarios. It is an optional extra (`verifiers`), installed from the pinned fork `machine-exploration/verifiers`, and imported lazily and only in `explorers.runtime.verifiers`. Do not import it anywhere else, and do not vendor its code.

Inspect, prime-rl, devinterp, WeightWatcher and mup are design references, not dependencies.

To study a reference, clone it outside this git tree (for example a sibling folder `../verifiers`). Record the commit you read in the log and in the design note, so the note can be checked against the code later. Do not vendor or copy its code into this repo; write our own.

## Formats

The scenario format and the trace format are the parts others will depend on. Version them. A breaking change needs a log entry that names the old and new version and the reason.

## Scenarios

Scenarios that make agents hack or coordinate run with no network, no shared package cache, and no path between episodes. Do not publish a scenario or its traces before the publication rules (`private/README.md` §10, README "Principles") are settled.

## Research results

A result is published with the run that reproduces it: config, seed, data order, code version. Validate a method on a model organism with a known mechanism before using it where the mechanism is unknown. Negative results are published too.
