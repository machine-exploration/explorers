"""Observables: pure, versioned functions of a state (or a step) and the examples.

An observable declares what it reads. The engine reads each thing once per state and hands it
to every observable that asked for it, so ten observables cost about one forward pass.

Reads:
  "weights"      named parameter tensors
  "token_loss"   (example, position) next-token loss; position 0 is NaN, masked positions NaN
  "hidden:<L>"   (example, position, d_model) residual stream after block L (0 = embeddings)
  "step"         a Step: its states' weights and the gradient (step observables only)
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class Observable:
    name: str
    fn: Callable = field(compare=False, repr=False)
    reads: frozenset[str]
    dims: tuple[str, ...]
    version: str = "0"
    params: tuple = ()           # (key, value) pairs that change the result; part of the cache key

    @property
    def on_steps(self) -> bool:
        return "step" in self.reads

    def __call__(self, ctx) -> xr.DataArray:
        out = self.fn(ctx)
        if not isinstance(out, xr.DataArray):
            out = np.asarray(out, dtype=np.float64)
            if out.ndim != len(self.dims):
                raise ValueError(f"observable {self.name!r} returned {out.ndim} dims, declared {self.dims}")
            out = xr.DataArray(out, dims=self.dims)
        if tuple(out.dims) != self.dims:
            raise ValueError(f"observable {self.name!r} returned dims {out.dims}, declared {self.dims}")
        return out.rename(self.name)


def observable(reads, dims=(), version="0", name=None, **params):
    """Decorate `fn(ctx) -> array | DataArray` into an Observable."""
    def wrap(fn):
        return Observable(name=name or fn.__name__, fn=fn, reads=frozenset(reads), dims=tuple(dims),
                          version=version, params=tuple(sorted(params.items())))
    return wrap


@dataclass
class Context:
    """What one state (or step) offers to observables. Filled by the engine from the declared reads."""
    state: object
    examples: object
    weights: dict = field(default_factory=dict)          # name -> np.ndarray
    token_loss: np.ndarray | None = None                 # (example, position)
    hidden: dict = field(default_factory=dict)           # layer -> (example, position, d)
    step: object = None
    before: dict = field(default_factory=dict)           # weights before the step
    after: dict = field(default_factory=dict)            # weights after the step
    grad: dict = field(default_factory=dict)             # name -> np.ndarray


def _matrices(weights: dict) -> list[str]:
    return [k for k, w in weights.items() if w.ndim == 2]


# --- built-in observables -----------------------------------------------------------------

@observable(reads=["token_loss"], dims=("example", "position"))
def token_loss(ctx):
    return ctx.token_loss


@observable(reads=["token_loss"], dims=())
def loss(ctx):
    return float(np.nanmean(ctx.token_loss))


@observable(reads=["token_loss"], dims=("example",))
def example_loss(ctx):
    """Mean loss over each example's counted positions."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(ctx.token_loss, axis=1)


@observable(reads=["weights"], dims=("param",))
def weight_norm(ctx):
    names = _matrices(ctx.weights)
    return xr.DataArray([float(np.linalg.norm(ctx.weights[n])) for n in names], dims=("param",),
                        coords={"param": names})


@observable(reads=["weights"], dims=("param",))
def stable_rank(ctx):
    """||W||_F^2 / ||W||_2^2 for every weight matrix: how many directions carry the weight."""
    names = _matrices(ctx.weights)
    vals = []
    for n in names:
        s = np.linalg.svd(ctx.weights[n].astype(np.float64), compute_uv=False)
        vals.append(float((s ** 2).sum() / (s[0] ** 2)) if s[0] > 0 else 0.0)
    return xr.DataArray(vals, dims=("param",), coords={"param": names})


@observable(reads=["step"], dims=("param",))
def update_norm(ctx):
    names = _matrices(ctx.after)
    return xr.DataArray([float(np.linalg.norm(ctx.after[n] - ctx.before[n])) for n in names],
                        dims=("param",), coords={"param": names})


@observable(reads=["step"], dims=("param",))
def grad_norm(ctx):
    names = [n for n in _matrices(ctx.after) if n in ctx.grad]
    return xr.DataArray([float(np.linalg.norm(ctx.grad[n])) for n in names], dims=("param",),
                        coords={"param": names})


def hidden_norm(layer: int):
    """Mean residual-stream norm at `layer`, per example: a simple activation-scale read."""
    @observable(reads=[f"hidden:{layer}"], dims=("example",), name=f"hidden_norm_{layer}", layer=layer)
    def fn(ctx):
        return np.linalg.norm(ctx.hidden[layer], axis=-1).mean(axis=1)
    return fn
