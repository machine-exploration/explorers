"""End to end: the quanta hypothesis on a task where each quantum is known.

Tasks are drawn with Zipf frequencies; the hypothesis predicts frequent tasks are learned first.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import explorers as ex  # noqa: E402
from explorers import analysis, measures, toy  # noqa: E402


def test_frequent_quanta_are_learned_first():
    task = toy.MultitaskLookup(n_tasks=8, n_symbols=8, alpha=1.5, seed=0)
    traj = toy.train(task, steps=900, every=60, batch_size=64, seed=0)
    ds = ex.Study(traj, task.examples()).measure(measures.example_loss).compute()
    on = analysis.onsets(ds.example_loss)
    assert np.isfinite(on.onset).mean() > 0.8                          # most examples were learned
    rho = analysis.spearman(ds.task_frequency, on.onset)
    assert rho < -0.5                                                   # frequent -> earlier onset
    per_task = on.onset.groupby("task").median().values
    assert np.nanmedian(per_task[:2]) < np.nanmedian(per_task[-2:])
