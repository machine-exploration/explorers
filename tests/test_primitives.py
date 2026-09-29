"""Core primitives driven by a real (tiny) model: engine, cache, runs. Needs torch and explorers-learning."""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("torch")
pytest.importorskip("transformers")

from explorers.core import analysis, observe  # noqa: E402
from explorers.learning import toy  # noqa: E402
from explorers.core.engine import across, over  # noqa: E402
from explorers.core.observe import observable  # noqa: E402
from explorers.core.state import State, Trajectory, snapshot  # noqa: E402
from explorers.core.store import Store  # noqa: E402

TASK = toy.MultitaskLookup(n_tasks=4, n_symbols=4, alpha=1.0)


@pytest.fixture(scope="module")
def short_run():
    return toy.train(TASK, steps=60, every=20, batch_size=32)


def counting(traj):
    """Same trajectory, but count how often each state is loaded."""
    calls = []

    def wrap(s):
        def load():
            calls.append(s.step)
            return s.load()
        return State(run=s.run, step=s.step, key=s.key, load=load)
    return Trajectory(traj.run, [wrap(s) for s in traj.states], traj.steps, traj.coords), calls



def test_snapshot_keys_follow_content():
    m = toy.tiny_gpt(TASK.vocab_size, seed=0)
    a, b = snapshot(m, "r", 0), snapshot(toy.tiny_gpt(TASK.vocab_size, seed=0), "r", 5)
    assert a.key == b.key                                    # same parameters, same key
    assert snapshot(toy.tiny_gpt(TASK.vocab_size, seed=1), "r", 0).key != a.key


def test_over_shapes_and_example_coords(short_run):
    ex = TASK.examples()
    ds = over(short_run, [observe.example_loss, observe.loss, observe.stable_rank, observe.update_norm], ex)
    assert list(ds.step.values) == [0, 20, 40, 60]
    assert ds.example_loss.dims == ("step", "example") and ds.example_loss.shape == (4, 16)
    assert list(ds.example_id.values) == list(ex.ids)
    assert "task_frequency" in ds.coords
    assert np.isnan(ds.update_norm.sel(step=0)).all()       # no step leads into the first state
    assert not np.isnan(ds.update_norm.sel(step=20)).any()
    assert float(ds.loss.sel(step=60)) < float(ds.loss.sel(step=0))


def test_one_load_per_state_and_cache_skips_models(short_run, tmp_path):
    traj, calls = counting(short_run)
    ex = TASK.examples()
    obs = [observe.example_loss, observe.loss, observe.stable_rank, observe.hidden_norm(1)]
    first = over(traj, obs, ex, store=tmp_path)
    assert calls == [0, 20, 40, 60]                          # four observables, one load each
    second = over(traj, obs, ex, store=Store(tmp_path))
    assert calls == [0, 20, 40, 60]                          # everything came from the store
    xr.testing.assert_allclose(first.example_loss, second.example_loss)
    assert list(second.param.values) == list(first.param.values)


def test_new_observable_version_is_recomputed(short_run, tmp_path):
    traj, calls = counting(short_run)
    ex = TASK.examples()
    over(traj, [observe.loss], ex, store=tmp_path)

    @observable(reads=["token_loss"], dims=(), version="1", name="loss")
    def loss_v1(ctx):
        return float(np.nanmean(ctx.token_loss))

    over(traj, [loss_v1], ex, store=tmp_path)
    assert len(calls) == 8



def test_across_stacks_runs_with_coords(short_run):
    other = toy.train(TASK, steps=60, every=20, batch_size=32, seed=1)
    ds = across([short_run, other], [observe.loss], TASK.examples())
    assert ds.loss.dims == ("run", "step") and ds.sizes["run"] == 2
    assert list(ds.seed.values) == [0, 1]
