# Agent working agreement

This repo is the `explorers` library, the open stack for Machine Exploration's mission: understand how intelligence arises in deep learning systems and build a science of deep learning, through learning mechanics (training dynamics) and mechanistic interpretability. Two horizons: now, oversight of agent populations with white-box methods (the existing code: scenario format, multi-agent runtime, monitors); long term, the science. Keep the oversight code working and moving; the science shares its `Method` / `Annotation` interface. The public README is [README.md](README.md). The detailed internal README (founder plan) is `private/README.md`, the first use case is `private/first-use-case.md`, and the chronology is `private/LOG.md`. All are local only, ignored by git, never published. Keep strategy and plans (runway, funding, competitors, decision points, timelines) there, not in tracked files. Ask the founder before making any new file public.

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

## Reference frameworks, not dependencies

[verifiers](https://github.com/PrimeIntellect-ai/verifiers) is a design reference for the runtime (environment, rubric, multi-turn rollout, token-id capture). Explorers does not import it and has no adapter for it. The same goes for Inspect and prime-rl.

To study a reference, clone it outside this git tree (for example a sibling folder `../verifiers`). Record the commit you read in the log and in the design note, so the note can be checked against the code later. Do not vendor or copy its code into this repo; write our own.

## Formats

The scenario format and the trace format are the parts others will depend on. Version them. A breaking change needs a log entry that names the old and new version and the reason.

## Scenarios

Scenarios that make agents hack or coordinate run with no network, no shared package cache, and no path between episodes. Do not publish a scenario or its traces before the publication rules (`private/README.md` §10, README "Principles") are settled.

## Research results

A result is published with the run that reproduces it: config, seed, data order, code version. Validate a method on a model organism with a known mechanism before using it where the mechanism is unknown. Negative results are published too.
