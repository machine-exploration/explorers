"""Sweeps: grids of runs keyed by content, checked on the CPU, sent to a backend."""

from concurrent.futures import Future

import pytest

import explorers as ex
from explorers.sweep import Backend


class InProcess(Backend):
    """A test backend: runs each submitted run in this process and counts them."""

    def __init__(self):
        self.calls = []

    def submit(self, fn, run):
        self.calls.append(run.kwargs)
        f = Future()
        f.set_result(fn(check=False, **run.kwargs))
        return f


def make(scale=1):
    @ex.sweep(a=[1, 2], seed=range(2))
    def toy(check, a, seed):
        n = 3 if check else 100
        return {"total": sum(range(n)) * a * scale + seed, "n": n}
    return toy


def test_grid_is_a_cartesian_product_in_a_fixed_order():
    runs = make().runs()
    assert [r.kwargs for r in runs] == [{"a": 1, "seed": 0}, {"a": 1, "seed": 1}, {"a": 2, "seed": 0},
                                        {"a": 2, "seed": 1}]
    assert len({r.key for r in runs}) == 4


def test_check_runs_one_point_on_a_sliver():
    assert make().check() == {"total": 3, "n": 3}


def test_run_sends_every_point_to_the_backend():
    backend = InProcess()
    results = make().run(on=backend)
    assert len(backend.calls) == 4
    assert results.select(a=2, seed=1).rows() == [{"a": 2, "seed": 1, "result": {"total": 9901, "n": 100}}]


def test_stored_runs_are_not_computed_again(tmp_path):
    first, second = InProcess(), InProcess()
    make().run(on=first, store=tmp_path)
    again = make().run(on=second, store=tmp_path)
    assert len(first.calls) == 4 and second.calls == []
    assert again.rows() == make().run(on=InProcess()).rows()


def test_keys_follow_the_parameters():
    runs = make().runs()
    assert runs[0].key != runs[1].key
    assert make().runs()[0].key == runs[0].key


def test_local_is_only_a_check():
    with pytest.raises(TypeError, match="only `check"):
        make().run(on="local")


def test_a_sweep_needs_a_check_argument_and_json_results():
    with pytest.raises(TypeError, match="check"):
        ex.sweep(a=[1])(lambda a: a)
    bad = ex.sweep(a=[1])(lambda check, a: object())
    with pytest.raises(TypeError, match="JSON"):
        bad.check()
