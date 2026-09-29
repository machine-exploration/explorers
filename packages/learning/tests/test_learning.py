import itertools

import numpy as np
import pytest

from explorers.learning import (
    Checkpoint, DiffMeans, Logistic, Sweep, number_comparison, pick,
    pythia, pythia_steps, read_records, split_by_group, write_records,
)
from explorers.core.metrics import auroc, cluster_bootstrap
from explorers.learning.activations import Cache
from explorers.learning.datasets import from_jsonl


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


def test_pythia_checkpoints():
    cks = pythia("70m", n=4)
    assert cks[0] == Checkpoint("EleutherAI/pythia-70m", "step0", 0)
    assert cks[-1].revision == "step143000"
    with pytest.raises(ValueError):
        pythia("7b")


def test_number_comparison_groups_both_orders():
    ds = number_comparison(n=40, seed=1)
    assert len(ds.texts) == 40 and sum(ds.labels) == 20
    assert ds.fingerprint == number_comparison(n=40, seed=1).fingerprint
    train, test = split_by_group(ds.groups, 0.25, seed=0)
    assert not {ds.groups[i] for i in train} & {ds.groups[i] for i in test}
    assert sorted(train + test) == list(range(40))


def test_from_jsonl(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"text": "a", "label": 1, "group": "g"}\n\n{"text": "b", "label": 0}\n')
    ds = from_jsonl(p)
    assert ds.name == "d" and ds.labels == [True, False] and ds.groups == ["g", "2"]


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


class PlantedSource:
    """Fake activations: the label becomes more readable as training goes on and deeper in."""

    def __init__(self, dataset, d=16):
        self.y, self.d, self.calls = np.array(dataset.labels), d, 0

    def read(self, checkpoint, dataset, layers):
        self.calls += 1
        rng = np.random.default_rng(checkpoint.step)
        strength = np.log1p(checkpoint.step) / np.log1p(143000)
        out = {}
        for layer in layers:
            X = rng.normal(size=(len(self.y), self.d))
            X[:, 0] += np.where(self.y, 1, -1) * 2 * strength * (layer + 1) / len(layers)
            out[layer] = X
        return out


def test_sweep_records_and_roundtrip(tmp_path):
    ds = number_comparison(n=200)
    cks = pythia("70m", n=3)
    source = PlantedSource(ds)
    records = Sweep(cks, ds, layers=[0, 3], probes=[DiffMeans()], source=source, resamples=50).run()
    assert len(records) == 6 and source.calls == 3
    by = {(r.step, r.layer): r.auroc for r in records}
    assert by[(0, 3)] < 0.7 < by[(143000, 3)]               # readable only after training
    assert all(r.ci_low <= r.auroc <= r.ci_high for r in records)
    write_records(tmp_path / "r.jsonl", records)
    assert read_records(tmp_path / "r.jsonl") == records


def test_read_records_rejects_other_formats(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"format": "explorers.result/v9"}\n')
    with pytest.raises(ValueError, match="v9"):
        read_records(p)


def test_cache_roundtrip(tmp_path):
    cache, ds, ck = Cache(tmp_path), number_comparison(n=10), Checkpoint("m", "step1", 1)
    assert cache.get(ck, ds, 2, "last") is None
    cache.put(np.ones((10, 4), dtype=np.float32), ck, ds, 2, "last")
    assert cache.get(ck, ds, 2, "last").shape == (10, 4)
    assert cache.path(ck, ds, 2, "last") != cache.path(ck, ds, 2, "mean")
