"""Sweeps: one function over a grid of parameters, checked on the CPU, run on GPU workers.

    @ex.sweep(depth=[3, 6, 12], lr=[1e-4, 3e-4], seed=range(2))
    def glp(check, depth, lr, seed):
        ...                                    # check=True: a sliver of the data, on the CPU
        return {"loss": loss, "probe": score}

    glp.check()                                # one grid point, locally, on the CPU
    results = glp.run(on=backend, store="runs/sweeps")

A sweep is a grid of runs; a run is one grid point. A run's key hashes the function's source, the
sweep's name and the point's parameters, so a run whose result is stored is never computed again,
and changing the code or a parameter gives a new key. Nothing heavy runs locally: `check` runs the
first grid point with `check=True`, in this process, and `run` sends every run to a backend (GPU
workers; the Modal backend is next). Run results are JSON values (numbers, strings, lists, dicts).
The key covers the function's own source, not what it calls: when a helper changes, bump a
parameter (for example `version=[2]`).
"""

import hashlib
import inspect
import itertools
import json
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Run:
    """One grid point of a sweep."""
    sweep: str
    code: str        # sha256 of the function's source
    params: tuple    # sorted (name, value) pairs

    @property
    def key(self) -> str:
        blob = json.dumps({"format": "explorers.run/v1", "sweep": self.sweep, "code": self.code,
                           "params": dict(self.params)}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    @property
    def kwargs(self) -> dict:
        return dict(self.params)


class Backend(ABC):
    """Where runs execute. `submit` returns a future of the run's JSON result."""

    @abstractmethod
    def submit(self, fn, run: Run) -> Future: ...


class Results:
    """Run results in grid order; `select(**params)` filters by parameter values."""

    def __init__(self, items: list[tuple[Run, object]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def select(self, **params) -> "Results":
        return Results([(r, v) for r, v in self.items if all(r.kwargs.get(k) == x for k, x in params.items())])

    def rows(self) -> list[dict]:
        """One dict per run: its parameters, plus its result under `result`."""
        return [{**r.kwargs, "result": v} for r, v in self.items]


class Sweep:
    def __init__(self, fn, grid: dict):
        if "check" not in inspect.signature(fn).parameters:
            raise TypeError(f"{fn.__name__} must take a `check` argument (True: a sliver of the data, on the CPU)")
        if "check" in grid:
            raise ValueError("`check` is set by the sweep, not by the grid")
        self.fn = fn
        self.name = fn.__qualname__
        self.grid = {k: _json_values(k, v) for k, v in grid.items()}
        try:
            source = inspect.getsource(fn)
        except OSError as e:
            raise ValueError(f"the source of {self.name} is needed to key its runs") from e
        self.code = hashlib.sha256(source.encode()).hexdigest()

    def runs(self) -> list[Run]:
        names = sorted(self.grid)
        return [Run(self.name, self.code, tuple(zip(names, values)))
                for values in itertools.product(*(self.grid[n] for n in names))]

    def check(self):
        """Run the first grid point in this process with `check=True` (CPU, a sliver of the data)."""
        return _jsonable(self.fn(check=True, **self.runs()[0].kwargs), self.name)

    def run(self, on: Backend, store=None, check: bool = True) -> Results:
        """Check locally first, then send every run whose result is not stored to the backend."""
        if not isinstance(on, Backend):
            raise TypeError("runs go to a Backend (GPU workers); local execution is only `check()`")
        if check:
            self.check()
        root = Path(store) if store is not None else None
        out, pending = {}, {}
        for r in self.runs():
            cached = _load(root, r) if root is not None else _MISSING
            if cached is not _MISSING:
                out[r] = cached
            else:
                pending[r] = on.submit(self.fn, r)
        for r, fut in pending.items():
            value = _jsonable(fut.result(), self.name)
            if root is not None:
                _save(root, r, value)
            out[r] = value
        return Results([(r, out[r]) for r in self.runs()])

    def __call__(self, **kwargs):
        return self.fn(**kwargs)


def sweep(**grid):
    """Decorator: `@sweep(a=[1, 2], seed=range(3))` turns a function into a Sweep over the grid."""
    return lambda fn: Sweep(fn, grid)


_MISSING = object()


def _json_values(name, values) -> list:
    values = list(values)
    if not values:
        raise ValueError(f"grid parameter {name!r} has no values")
    for v in values:
        json.dumps(v)  # raises TypeError for values that cannot be keyed
    return values


def _jsonable(value, name):
    try:
        return json.loads(json.dumps(value))
    except TypeError as e:
        raise TypeError(f"{name} must return a JSON value (numbers, strings, lists, dicts): {e}") from e


def _path(root: Path, r: Run) -> Path:
    return root / r.key[:2] / f"{r.key}.json"


def _load(root: Path, r: Run):
    p = _path(root, r)
    return json.loads(p.read_text(encoding="utf-8"))["result"] if p.exists() else _MISSING


def _save(root: Path, r: Run, value) -> None:
    p = _path(root, r)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"sweep": r.sweep, "code": r.code, "params": r.kwargs, "result": value}),
                   encoding="utf-8")
    tmp.replace(p)
