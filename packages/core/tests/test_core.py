"""explorers.core on its own: no torch, no other explorers package."""

import numpy as np
import pytest
import xarray as xr

from explorers.core import Examples, State, Store, analysis, observable
from explorers.core.observe import Context
from explorers.core.store import result_key


def examples(n=6, seq=3):
    tokens = np.arange(n * seq).reshape(n, seq)
    return Examples(tokens=tokens, meta={"group": np.arange(n) % 2})


def test_example_ids_follow_content():
    ex = examples()
    assert len(set(ex.ids)) == len(ex)
    again = Examples(ex.tokens.copy(), None, dict(ex.meta), name="other name")
    assert list(again.ids) == list(ex.ids) and again.fingerprint == ex.fingerprint
    assert ex.with_meta(extra=np.zeros(len(ex))).fingerprint != ex.fingerprint
    with pytest.raises(ValueError):
        Examples(ex.tokens, meta={"bad": np.zeros(2)})
    with pytest.raises(ValueError):
        Examples(ex.tokens, loss_mask=np.ones((1, 1), dtype=bool))


def test_observable_checks_declared_dims():
    ctx = Context(state=None, examples=examples(), token_loss=np.ones((6, 3)))

    @observable(reads=["token_loss"], dims=("example",))
    def per_example(ctx):
        return ctx.token_loss.mean(1)

    @observable(reads=["token_loss"], dims=("example",))
    def wrong(ctx):
        return ctx.token_loss.mean()

    assert per_example(ctx).dims == ("example",) and per_example(ctx).name == "per_example"
    with pytest.raises(ValueError, match="declared"):
        wrong(ctx)


def test_store_roundtrip_and_keys(tmp_path):
    @observable(reads=["weights"], dims=("param",), version="0")
    def norms(ctx):
        return xr.DataArray([1.0, 2.0], dims=("param",), coords={"param": ["a", "b"]})

    ex, store = examples(), Store(tmp_path)
    key = result_key("params:abc", norms, ex.fingerprint)
    assert store.get(key) is None
    store.put(key, norms(Context(state=None, examples=ex)), {"note": "test"})
    back = store.get(key)
    assert list(back.param.values) == ["a", "b"] and back.values.tolist() == [1.0, 2.0]
    assert result_key("params:abd", norms, ex.fingerprint) != key          # other state
    assert result_key("params:abc", norms, ex.with_meta(x=np.zeros(6)).fingerprint) != key


def test_state_is_lazy():
    calls = []
    s = State(run="r", step=3, key="k", load=lambda: calls.append(1))
    assert calls == [] and s == State(run="r", step=3, key="k", load=lambda: None)


def test_onsets_sharp_and_gradual():
    steps = np.array([0, 10, 20, 30, 40, 50])
    sudden = [2, 2, 2, 0, 0, 0]
    gradual = [2, 1.6, 1.2, 0.8, 0.4, 0]
    flat = [1, 1, 1, 1, 1, 1]
    curves = xr.DataArray(np.array([sudden, gradual, flat]).T, dims=("step", "example"),
                          coords={"step": steps, "task": ("example", [0, 1, 2])})
    on = analysis.onsets(curves)
    assert on.sharpness.values[0] == pytest.approx(1.0)
    assert on.sharpness.values[1] == pytest.approx(0.2)
    assert 20 < on.onset.values[0] <= 30 and 20 <= on.onset.values[1] <= 30
    assert np.isnan(on.onset.values[2])
    assert list(on.task.values) == [0, 1, 2]


def test_spearman_and_clusters():
    assert analysis.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert analysis.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(analysis.spearman([1, 2, 3, 4], [5, 5, 5, 5]))
    steps = np.arange(8)
    early = [[1] + [0] * 7] * 5
    late = [[1] * 6 + [0] * 2] * 5
    curves = xr.DataArray(np.array(early + late, dtype=float).T, dims=("step", "example"), coords={"step": steps})
    assert list(analysis.cluster_curves(curves, k=2).values) == [0] * 5 + [1] * 5
