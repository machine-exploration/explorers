"""Watch quanta form during Pythia's pretraining.

Every (window, position) of a text sample is one sample, as in the quantization model of neural
scaling. For each checkpoint we measure its next-token loss, then ask when each sample's loss
drops, how suddenly, and whether frequent targets are learned first.

    python examples/quanta_pythia.py texts.txt --size 70m --checkpoints 24 --device cuda

`texts.txt` holds one document per line (a few hundred Pile-like documents is enough).
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from explorers.core import analysis, observe
from explorers.core.data import Examples
from explorers.core.engine import over
from explorers.learning import from_checkpoints, pythia

p = argparse.ArgumentParser()
p.add_argument("texts", type=Path)
p.add_argument("--size", default="70m")
p.add_argument("--checkpoints", type=int, default=24)
p.add_argument("--seq-len", type=int, default=128)
p.add_argument("--max-examples", type=int, default=256)
p.add_argument("--device", default="cpu")
p.add_argument("--dtype", default="float32")
p.add_argument("--clusters", type=int, default=8)
args = p.parse_args()

from transformers import AutoTokenizer  # noqa: E402

checkpoints = pythia(args.size, n=args.checkpoints)
tok = AutoTokenizer.from_pretrained(checkpoints[-1].model, revision=checkpoints[-1].revision)
texts = [line for line in args.texts.read_text(encoding="utf-8").splitlines() if line.strip()]
examples = Examples.from_texts(texts, tok, seq_len=args.seq_len, name=args.texts.stem,
                               max_examples=args.max_examples)

# Data-centred metadata: how often each window's tokens occur in this sample.
counts = Counter(examples.tokens.ravel().tolist())
freq = np.vectorize(counts.__getitem__)(examples.tokens).astype(float)

trajectory = from_checkpoints(checkpoints, device=args.device, dtype=args.dtype, coords={"size": args.size})
ds = over(trajectory, [observe.token_loss, observe.loss, observe.stable_rank], examples,
          store="runs/store", device=args.device)
ds = ds.assign(target_frequency=(("example", "position"), freq))

samples = ds.token_loss.isel(position=slice(1, None)).stack(sample=("example", "position"))
on = analysis.onsets(samples, min_drop=0.5)
target_freq = ds.target_frequency.isel(position=slice(1, None)).stack(sample=("example", "position"))

learned = np.isfinite(on.onset.values)
print(f"{learned.sum()} of {learned.size} samples drop by at least 0.5 nats")
print(f"rank correlation, target frequency vs onset: {analysis.spearman(target_freq, on.onset):+.2f}")
print(f"median sharpness: {float(on.sharpness.median()):.2f}  "
      f"(share of samples with sharpness > 0.5: {float((on.sharpness > 0.5).mean()):.2f})")
clusters = analysis.cluster_curves(samples.where(np.isfinite(on.onset)), k=args.clusters)
for c in range(args.clusters):
    members = (clusters == c).values
    if members.any():
        print(f"cluster {c}: {members.sum():>6} samples, median onset step {np.nanmedian(on.onset.values[members]):>8.0f}")
