"""Studies over a real (tiny) training run: shapes, one load per state, the cache, step measures."""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("torch")
pytest.importorskip("transformers")

import explorers as ex  # noqa: E402
from explorers import measures, toy  # noqa: E402
from explorers.measures import measure  # noqa: E402
from explorers.state import State, Trajectory, snapshot  # noqa: E402

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


def test_shapes_coords_and_step_measures(short_run):
    examples = TASK.examples()
    ds = (ex.Study(short_run, examples)
          .measure(measures.example_loss, measures.loss, measures.stable_rank, measures.update_norm).compute())
    assert list(ds.step.values) == [0, 20, 40, 60]
    assert ds.example_loss.dims == ("step", "example") and ds.example_loss.shape == (4, 16)
    assert list(ds.example_id.values) == list(examples.ids)
    assert "task_frequency" in ds.coords
    assert np.isnan(ds.update_norm.sel(step=0)).all()        # no step leads into the first state
    assert not np.isnan(ds.update_norm.sel(step=20)).any()
    assert float(ds.loss.sel(step=60)) < float(ds.loss.sel(step=0))


def test_one_load_per_state_and_cache_skips_models(short_run, tmp_path):
    traj, calls = counting(short_run)
    study = lambda: ex.Study(traj, TASK.examples()).measure(
        measures.example_loss, measures.loss, measures.stable_rank, measures.residual_norm(1))
    first = study().compute(store=tmp_path)
    assert calls == [0, 20, 40, 60]                          # four measures, one load each
    second = study().compute(store=tmp_path)
    assert calls == [0, 20, 40, 60]                          # everything came from the store
    xr.testing.assert_allclose(first.example_loss, second.example_loss)
    assert list(second.param.values) == list(first.param.values)


def test_new_measure_version_is_recomputed(short_run, tmp_path):
    traj, calls = counting(short_run)
    ex.Study(traj, TASK.examples()).measure(measures.loss).compute(store=tmp_path)

    @measure(reads=["token_loss"], dims=(), version="1", name="loss")
    def loss_v1(ctx):
        return float(np.nanmean(ctx.token_loss))

    ex.Study(traj, TASK.examples()).measure(loss_v1).compute(store=tmp_path)
    assert len(calls) == 8


def test_step_measures_need_a_trajectory():
    with pytest.raises(ValueError, match="Trajectory"):
        ex.Study([ex.open(toy.tiny_gpt(TASK.vocab_size))], TASK.examples()).measure(measures.update_norm)
