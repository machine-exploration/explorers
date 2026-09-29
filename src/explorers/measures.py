"""Measures: pure, versioned functions of what a model computed on the examples.

A measure declares what it reads; a study serves every read once per model and hands it to every
measure that asked, so ten measures cost about one traced forward pass. Writes declared on the study
apply to that pass: a measure sees the model as the study modified it.

Reads:
  "weights"                named parameter arrays
  "token_loss"             (example, position) next-token loss; position 0 and masked positions NaN
  "logits:<p>"             (example, vocab) logits at position p (e.g. "logits:-1")
  "<stream>:<L>"           (example, position, d) a stream: residual, attn_out, mlp_out;
                           "residual:final" is residual[n_layers], before the final norm
  "unembed"                ctx.unembed_topk(h, k): the model's own final norm + unembedding, top-k ids
  "jacobian:<L>:<skip>"    (d, d) average Jacobian of residual:final with respect to residual:L
                           (docs/jlens.md); fitted on the unmodified model
  "step"                   a training step: weights before and after, and the gradient
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class Measure:
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
                raise ValueError(f"measure {self.name!r} returned {out.ndim} dims, declared {self.dims}")
            out = xr.DataArray(out, dims=self.dims)
        if tuple(out.dims) != self.dims:
            raise ValueError(f"measure {self.name!r} returned dims {out.dims}, declared {self.dims}")
        return out.rename(self.name)


def measure(reads, dims=(), version="0", name=None, **params):
    """Decorate `fn(ctx) -> array | DataArray` into a Measure."""
    def wrap(fn):
        return Measure(name=name or fn.__name__, fn=fn, reads=frozenset(reads), dims=tuple(dims),
                       version=version, params=tuple(sorted(params.items())))
    return wrap


@dataclass
class Context:
    """What one model (or one training step) offers to measures. Filled from the declared reads."""
    examples: object
    state: object = None
    n_layers: int | None = None
    weights: dict = field(default_factory=dict)          # name -> array
    token_loss: np.ndarray | None = None                 # (example, position)
    logits: dict = field(default_factory=dict)           # position -> (example, vocab)
    streams: dict = field(default_factory=dict)          # (stream, layer) -> (example, position, d)
    unembed_topk: Callable | None = None                 # (..., d) -> (..., k) token ids
    jacobian: dict = field(default_factory=dict)         # (layer, skip_first) -> (d, d)
    step: object = None
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    grad: dict = field(default_factory=dict)

    def stream(self, name: str, layer) -> np.ndarray:
        return self.streams[(name, self.n_layers if layer == "final" else layer)]

    @property
    def final(self) -> np.ndarray:
        return self.stream("residual", "final")


def _matrices(weights: dict) -> list[str]:
    return [k for k, w in weights.items() if w.ndim == 2]


# --- losses and weights --------------------------------------------------------------------------

@measure(reads=["token_loss"], dims=("example", "position"))
def token_loss(ctx):
    return ctx.token_loss


@measure(reads=["token_loss"], dims=())
def loss(ctx):
    return float(np.nanmean(ctx.token_loss))


@measure(reads=["token_loss"], dims=("example",))
def example_loss(ctx):
    """Mean loss over each example's counted positions."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(ctx.token_loss, axis=1)


@measure(reads=["weights"], dims=("param",))
def weight_norm(ctx):
    names = _matrices(ctx.weights)
    return xr.DataArray([float(np.linalg.norm(ctx.weights[n])) for n in names], dims=("param",),
                        coords={"param": names})


@measure(reads=["weights"], dims=("param",))
def stable_rank(ctx):
    """||W||_F^2 / ||W||_2^2 for every weight matrix: how many directions carry the weight."""
    names = _matrices(ctx.weights)
    vals = []
    for n in names:
        s = np.linalg.svd(ctx.weights[n].astype(np.float64), compute_uv=False)
        vals.append(float((s ** 2).sum() / (s[0] ** 2)) if s[0] > 0 else 0.0)
    return xr.DataArray(vals, dims=("param",), coords={"param": names})


@measure(reads=["step"], dims=("param",))
def update_norm(ctx):
    names = _matrices(ctx.after)
    return xr.DataArray([float(np.linalg.norm(ctx.after[n] - ctx.before[n])) for n in names],
                        dims=("param",), coords={"param": names})


@measure(reads=["step"], dims=("param",))
def grad_norm(ctx):
    names = [n for n in _matrices(ctx.after) if n in ctx.grad]
    return xr.DataArray([float(np.linalg.norm(ctx.grad[n])) for n in names], dims=("param",),
                        coords={"param": names})


def residual_norm(layer: int):
    """Mean residual-stream norm at `layer`, per example: a simple activation-scale read."""
    @measure(reads=[f"residual:{layer}"], dims=("example",), name=f"residual_norm_{layer}", layer=layer)
    def fn(ctx):
        return np.linalg.norm(ctx.stream("residual", layer), axis=-1).mean(axis=1)
    return fn


def logit_diff(correct, wrong, position: int = -1):
    """logit[correct] - logit[wrong] at `position`, per example. Token ids: one per example or one for
    all. Works on numpy and torch logits, so patching can differentiate through it."""
    correct_, wrong_ = np.asarray(correct), np.asarray(wrong)

    @measure(reads=[f"logits:{position}"], dims=("example",), name="logit_diff",
             correct=correct_.tolist(), wrong=wrong_.tolist(), position=position)
    def fn(ctx):
        last = ctx.logits[position]
        rows = np.arange(len(last))
        c = np.broadcast_to(correct_, rows.shape).astype(np.int64)          # copies: writable indices
        w = np.broadcast_to(wrong_, rows.shape).astype(np.int64)
        return last[rows, c] - last[rows, w]
    return fn


# --- the Jacobian lens (docs/jlens.md) ------------------------------------------------------------
#
# lens_L(h) = unembed(J_L @ h). Fitting and evaluation rows: if the examples carry a `split` column,
# J is fitted on the rows where split == "fit" and the lens is evaluated on the other rows. Without
# it, both use every row (in-sample). Positions: `skip_first` leading positions and the last one are
# excluded.

def split_rows(examples) -> tuple[np.ndarray, np.ndarray]:
    split = examples.meta.get("split")
    if split is None:
        every = np.ones(len(examples), dtype=bool)
        return every, every
    fit = np.asarray(split) == "fit"
    return fit, ~fit


def lens_positions(seq_len: int, skip_first: int) -> np.ndarray:
    """Source (and target) positions the lens uses: from `skip_first` to the one before last."""
    if skip_first < 0 or seq_len - 1 <= skip_first:
        raise ValueError(f"no lens positions for seq_len={seq_len}, skip_first={skip_first}")
    return np.arange(skip_first, seq_len - 1)


def jacobian(layer: int, skip_first: int = 16):
    """The fitted J_L itself, (out, in). Storing it makes the lens part of the result."""
    @measure(reads=[f"jacobian:{layer}:{skip_first}"], dims=("out", "in"),
             name=f"jacobian_{layer}", layer=layer, skip_first=skip_first)
    def fn(ctx):
        return ctx.jacobian[(layer, skip_first)]
    return fn


def _lens_error(ctx, layers, skip_first, transport: bool) -> xr.DataArray:
    _, rows = split_rows(ctx.examples)
    pos = lens_positions(ctx.final.shape[1], skip_first)
    target = ctx.unembed_topk(ctx.final[rows][:, pos], 1)[..., 0]
    errors = []
    for layer in layers:
        h = ctx.stream("residual", layer)[rows][:, pos]
        if transport:
            h = h @ ctx.jacobian[(layer, skip_first)].T
        errors.append(float(np.mean(ctx.unembed_topk(h, 1)[..., 0] != target)))
    return xr.DataArray(errors, dims=("layer",), coords={"layer": list(layers)})


def jlens_error(layers, skip_first: int = 16):
    """Per layer: how often the J-lens top-1 token differs from the model's own top-1 next-token
    prediction at the same position. Falls as the layer's content becomes what the model says."""
    layers = tuple(layers)
    reads = ["residual:final", "unembed"] + [f"residual:{l}" for l in layers] + [f"jacobian:{l}:{skip_first}" for l in layers]

    @measure(reads=reads, dims=("layer",), name="jlens_error", layers=layers, skip_first=skip_first)
    def fn(ctx):
        return _lens_error(ctx, layers, skip_first, transport=True)
    return fn


def logit_lens_error(layers, skip_first: int = 16):
    """The same measure without transport (J = identity): the logit lens, as a baseline."""
    layers = tuple(layers)
    reads = ["residual:final", "unembed"] + [f"residual:{l}" for l in layers]

    @measure(reads=reads, dims=("layer",), name="logit_lens_error", layers=layers, skip_first=skip_first)
    def fn(ctx):
        return _lens_error(ctx, layers, skip_first, transport=False)
    return fn
