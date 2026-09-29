"""The quanta hypothesis on a toy where each quantum is known. Runs on a CPU in about two minutes.

    python examples/quanta_toy.py
"""

from explorers.core import analysis, observe
from explorers.learning import toy
from explorers.core.engine import over

task = toy.MultitaskLookup(n_tasks=16, n_symbols=16, alpha=1.3)
trajectory = toy.train(task, steps=1500, every=50)
ds = over(trajectory, [observe.example_loss, observe.stable_rank, observe.update_norm],
          task.examples(), store="runs/store")

on = analysis.onsets(ds.example_loss)
print("median onset step per task (task 0 is the most frequent):")
print(on.onset.groupby("task").median().round().values)
print(f"rank correlation, task frequency vs onset: {analysis.spearman(ds.task_frequency, on.onset):+.2f}")
print(f"median sharpness (1 = one sudden drop): {float(on.sharpness.median()):.2f}")
