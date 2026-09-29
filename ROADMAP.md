# Roadmap

The roadmap covers the whole org, so it lives in one place: [machine-exploration/public/ROADMAP.md](https://github.com/machine-exploration/public/blob/main/ROADMAP.md).

The steps that happen in this repository:

- **Step 0 — One library:** all library code moves here (`core`, `learning`, `populations`); `mechanics` becomes the research program.
- **E1 — Streams:** `read`, `write`, `trace` on named streams, on Hugging Face / PyTorch.
- **E2 — The study:** reads, writes and measurements over models × checkpoints × examples; five canonical examples (probe, steering, activation patching, attribution patching, sparse autoencoder read).
- **E3 — A second backend,** with the same results within a stated tolerance.
- **S1, S2 — The scaling proof:** the patching benchmark against existing tools, and the planner.
- **P0, P1 — The agent side:** do small models exploit tasks; detectors at a matched false-positive rate ([#1](https://github.com/machine-exploration/explorers/pull/1)).
