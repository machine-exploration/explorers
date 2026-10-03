"""Replay recorded impossible_code episodes and compare log-probabilities (docs/episodes.md).

The model's log-probability of each sampled token, replayed by explorers, against the one vLLM
recorded while sampling. Run with the vLLM server stopped:

    python examples/episodes_4070/replay_check.py <vf-eval output dir> [cpu|cuda]

On a 12 GB card use cpu: `serve` builds a full-vocabulary fp32 log-softmax per branch, which does
not fit next to the 4B weights.
"""

import glob
import json
import sys
import time

import numpy as np

import explorers as ex
from explorers.execute import serve

path = glob.glob(sys.argv[1] + "/**/traces.jsonl", recursive=True)[0]
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
records = [json.loads(line) for line in open(path) if line.strip()]

t0 = time.time()
model = ex.open("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", device=device, dtype="bfloat16")
t1 = time.time()
examples, recorded = ex.episodes.replay(records, pad_id=model.tokenizer.pad_token_id or 0)
ctx = serve(model, examples, {"token_loss"}, batch_size=1)
t2 = time.time()

at = examples.loss_mask & ~np.isnan(recorded)
d = np.abs(-ctx.token_loss[at] - recorded[at])
print(f"branches {examples.tokens.shape[0]}, max len {examples.tokens.shape[1]}, sampled tokens {at.sum()}")
print(f"|replayed - recorded| logprob: median {np.median(d):.4f}  mean {d.mean():.4f}  "
      f"p99 {np.quantile(d, .99):.3f}  max {d.max():.3f}")
print(f"load {t1 - t0:.0f}s  replay {t2 - t1:.0f}s")
