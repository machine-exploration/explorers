import itertools

import numpy as np
import pytest

from explorers.analysis import auroc, cluster_bootstrap
from explorers.methods.probes import DiffMeans, Logistic
from explorers.model import pick, pythia_steps


def test_pythia_schedule():
    steps = pythia_steps()
    assert len(steps) == 154
    assert steps[:12] == [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
    assert steps[-1] == 143000


def test_pick_is_log_spread_and_keeps_ends():
    chosen = pick(pythia_steps(), 8)
    assert len(chosen) == 8 and chosen[0] == 0 and chosen[-1] == 143000
    assert sum(s <= 1000 for s in chosen) >= 3          # early training gets its share
    assert pick([0, 5, 9], 10) == [0, 5, 9]


def brute_auroc(y, s):
    pos = [v for v, t in zip(s, y) if t]
    neg = [v for v, t in zip(s, y) if not t]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


def test_auroc_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.random(60) < 0.4
        s = rng.integers(0, 5, 60).astype(float)            # many ties
        assert auroc(y, s) == pytest.approx(brute_auroc(y, s))
    assert auroc([True, True], [1.0, 2.0]) is None


def test_auroc_all_ties_is_fast_and_half():
    y = np.arange(200_000) % 10 == 0
    assert auroc(y, np.zeros(len(y))) == pytest.approx(0.5)


def test_cluster_bootstrap_brackets_point():
    rng = np.random.default_rng(0)
    y = rng.random(300) < 0.5
    s = y + rng.normal(0, 1, 300)
    groups = [str(i // 3) for i in range(300)]
    lo, hi = cluster_bootstrap(y, s, groups, resamples=200)
    assert lo <= auroc(y, s) <= hi


@pytest.mark.parametrize("probe", [DiffMeans(), Logistic(l2=1.0)])
def test_probes_find_a_planted_direction(probe):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 32))
    y = rng.random(400) < 0.5
    X[:, 3] += np.where(y, 1.5, -1.5)
    probe.fit(X[:300], y[:300])
    assert auroc(y[300:], probe.score(X[300:])) > 0.9
