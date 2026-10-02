"""Measures: pure, versioned functions of what a model computed on the examples.

A measure declares what it reads; an experiment serves every read once per model and hands it to every
measure that asked, so ten measures cost about one traced forward pass. Writes declared on the experiment
apply to that pass: a measure sees the model as the experiment modified it.

Reads:
  "weights"                named parameter arrays
  "token_loss"             (example, position) next-token loss; position 0 and masked positions NaN
  "logits:<p>"             (example, vocab) logits at position p (e.g. "logits:-1")
  "<stream>:<L>"           (example, position, d) a stream: residual, attn_out, mlp_out;
                           "residual:final" is residual[n_layers], before the final norm
  "unembed"                the model's own final norm + unembedding: ctx.unembed_topk(h, k) (top-k ids),
                           ctx.unembed_apply(h, fn) (fn of the logits, on the device),
                           ctx.unembed_matrix (vocab, d), its linear part
  "jacobian:<L>:<skip>[:<target>]"  (d, d) average Jacobian of the target residual (final, the
                           default, or penultimate) with respect to residual:L (docs/jlens.md);
                           fitted on the unmodified model
  "concept:<L>:<skip>:<target>:<ids>"  (k, d) rows W[ids] @ J_L for token ids "12,40,7": the
                           concept lens, one backward pass per token instead of d
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
    unembed_apply: Callable | None = None                # (h, fn) -> fn(logits of h)
    unembed_matrix: np.ndarray | None = None             # (vocab, d)
    jacobian: dict = field(default_factory=dict)         # (layer, skip_first, target) -> (d, d)
    cache: dict = field(default_factory=dict)            # values shared by measures of one model
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


def _jread(layer, skip_first, target):
    return f"jacobian:{layer}:{skip_first}:{target}"


def jacobian(layer: int, skip_first: int = 16, target: str = "final"):
    """The fitted J_L itself, (out, in). Storing it makes the lens part of the result."""
    @measure(reads=[_jread(layer, skip_first, target)], dims=("out", "in"),
             name=f"jacobian_{layer}", layer=layer, skip_first=skip_first, target=target)
    def fn(ctx):
        return ctx.jacobian[(layer, skip_first, target)]
    return fn


def _lens_residuals(ctx, layer, skip_first, target, lens):
    """Evaluation rows of residual:L at the lens positions, transported by J_L for the J-lens."""
    _, rows = split_rows(ctx.examples)
    h = ctx.stream("residual", layer)
    h = h[rows][:, lens_positions(h.shape[1], skip_first)]
    return h @ ctx.jacobian[(layer, skip_first, target)].T if lens == "jacobian" else h


def _lens_reads(layers, skip_first, target, lens, final=False):
    reads = ["unembed"] + [f"residual:{l}" for l in layers] + (["residual:final"] if final else [])
    return reads + ([_jread(l, skip_first, target) for l in layers] if lens == "jacobian" else [])


def _lens_error(ctx, layers, skip_first, target, lens) -> xr.DataArray:
    _, rows = split_rows(ctx.examples)
    pos = lens_positions(ctx.final.shape[1], skip_first)
    said = ctx.unembed_topk(ctx.final[rows][:, pos], 1)[..., 0]
    errors = [float(np.mean(ctx.unembed_topk(_lens_residuals(ctx, l, skip_first, target, lens), 1)[..., 0] != said))
              for l in layers]
    return xr.DataArray(errors, dims=("layer",), coords={"layer": list(layers)})


def jlens_error(layers, skip_first: int = 16, target: str = "final"):
    """Per layer: how often the J-lens top-1 token differs from the model's own top-1 next-token
    prediction at the same position. It falls as a layer's content becomes what the model is about
    to say: the late, "motor" regime. It is not a workspace measure (the J-lens is not built to
    predict the next token); see the workspace signatures below."""
    layers = tuple(layers)

    @measure(reads=_lens_reads(layers, skip_first, target, "jacobian", final=True), dims=("layer",),
             name="jlens_error", layers=layers, skip_first=skip_first, target=target)
    def fn(ctx):
        return _lens_error(ctx, layers, skip_first, target, "jacobian")
    return fn


def logit_lens_error(layers, skip_first: int = 16):
    """The same measure without transport (J = identity): the logit lens, as a baseline."""
    layers = tuple(layers)

    @measure(reads=_lens_reads(layers, skip_first, None, "logit", final=True), dims=("layer",),
             name="logit_lens_error", layers=layers, skip_first=skip_first)
    def fn(ctx):
        return _lens_error(ctx, layers, skip_first, None, "logit")
    return fn


# --- workspace signatures (Gurnee et al. 2026, section 4.1; docs/jlens.md) ------------------------
#
# The J-lens vectors at layer L are the rows of V_L = W J_L (vocab, d), W = ctx.unembed_matrix.
# Where the workspace forms, the paper sees: V_L fans out from a low-dimensional subspace
# (dimension), readouts become peaked on few tokens (kurtosis), the top readout persists across
# positions (autocorrelation), and layers group into early / workspace / motor blocks (CKA).
# Dimension and CKA need only W and J (no data beyond the fit); kurtosis and autocorrelation read
# the evaluation rows.

def _unembed_cov(ctx) -> np.ndarray:
    """(d, d) covariance of the rows of W over the vocabulary; V_L's is J_L^T C J_L."""
    if "unembed_cov" not in ctx.cache:
        w = ctx.unembed_matrix.astype(np.float64)
        w = w - w.mean(axis=0, keepdims=True)
        ctx.cache["unembed_cov"] = w.T @ w / len(w)
    return ctx.cache["unembed_cov"]


def dimension_fraction(cov: np.ndarray, share: float) -> float:
    """Fraction of dimensions whose principal components hold `share` of the variance."""
    ev = np.sort(np.clip(np.linalg.eigvalsh((cov + cov.T) / 2), 0, None))[::-1]
    if ev.sum() == 0:
        return 0.0
    k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), share - 1e-12)) + 1
    return min(k, len(ev)) / len(ev)


def linear_cka(j_a: np.ndarray, j_b: np.ndarray, cov: np.ndarray) -> float:
    """Linear CKA between the vector sets W J_a and W J_b (rows indexed by token), from the
    covariance of W: ||J_b^T C J_a||^2 / (||J_a^T C J_a|| ||J_b^T C J_b||). 1 when one set is an
    orthogonal transform or rescaling of the other."""
    ab = j_b.T @ cov @ j_a
    aa, bb = j_a.T @ cov @ j_a, j_b.T @ cov @ j_b
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.sum(ab ** 2) / den) if den > 0 else float("nan")


def repeat_rate(top1: np.ndarray, offsets) -> np.ndarray:
    """Per offset D: log of (how often top1[i, t] == top1[i, t + D]) over (how often top1[i, t] ==
    top1[i + 1, t + D], the next example: a null with the same positions but no shared context).
    Add-one smoothed; 0 means no more persistence than chance."""
    n, T = top1.shape
    out = []
    for off in offsets:
        if off >= T or n < 2:
            out.append(np.nan)
            continue
        a, same, other = top1[:, :-off], top1[:, off:], np.roll(top1, -1, axis=0)[:, off:]
        total = a.size + 2
        out.append(np.log((np.sum(a == same) + 1) / total) - np.log((np.sum(a == other) + 1) / total))
    return np.asarray(out)


def jlens_dimension(layers, skip_first: int = 16, target: str = "final", share: float = 0.9):
    """Per layer: the fraction of residual dimensions holding `share` of the variance of the J-lens
    vectors. Small before the workspace (the lens collapses to a subspace), rises at its onset."""
    layers = tuple(layers)

    @measure(reads=["unembed"] + [_jread(l, skip_first, target) for l in layers], dims=("layer",),
             name="jlens_dimension", layers=layers, skip_first=skip_first, target=target, share=share)
    def fn(ctx):
        cov = _unembed_cov(ctx)
        vals = []
        for l in layers:
            j = ctx.jacobian[(l, skip_first, target)].astype(np.float64)
            vals.append(dimension_fraction(j.T @ cov @ j, share))
        return xr.DataArray(vals, dims=("layer",), coords={"layer": list(layers)})
    return fn


def jlens_cka(layers, skip_first: int = 16, target: str = "final"):
    """(layer, layer2): linear CKA between the J-lens vector sets of two layers. Blocks along the
    diagonal are the early, workspace and motor regimes."""
    layers = tuple(layers)

    @measure(reads=["unembed"] + [_jread(l, skip_first, target) for l in layers], dims=("layer", "layer2"),
             name="jlens_cka", layers=layers, skip_first=skip_first, target=target)
    def fn(ctx):
        cov = _unembed_cov(ctx)
        js = [ctx.jacobian[(l, skip_first, target)].astype(np.float64) for l in layers]
        m = np.array([[linear_cka(a, b, cov) for b in js] for a in js])
        return xr.DataArray(m, dims=("layer", "layer2"), coords={"layer": list(layers), "layer2": list(layers)})
    return fn


def _excess_kurtosis(z):
    z = z - z.mean(dim=-1, keepdim=True)
    m2 = (z ** 2).mean(dim=-1)
    return (z ** 4).mean(dim=-1) / m2.clamp_min(1e-30) ** 2 - 3.0


def lens_kurtosis(layers, skip_first: int = 16, target: str = "final", lens: str = "jacobian"):
    """Per layer: median over evaluation positions of the excess kurtosis of the lens logits over
    the vocabulary. Near 0 when readouts look like noise; high when a few tokens stand out.
    `lens="logit"` gives the logit-lens baseline."""
    layers = tuple(layers)

    @measure(reads=_lens_reads(layers, skip_first, target, lens), dims=("layer",),
             name=f"{'jlens' if lens == 'jacobian' else 'logit_lens'}_kurtosis", layers=layers,
             skip_first=skip_first, target=target if lens == "jacobian" else None, lens=lens)
    def fn(ctx):
        vals = [float(np.median(ctx.unembed_apply(_lens_residuals(ctx, l, skip_first, target, lens), _excess_kurtosis)))
                for l in layers]
        return xr.DataArray(vals, dims=("layer",), coords={"layer": list(layers)})
    return fn


def lens_persistence(layers, skip_first: int = 16, target: str = "final", lens: str = "jacobian",
                     offsets=(1, 2, 4, 8)):
    """(layer, offset): how much more often the lens top-1 token repeats D positions later in the
    same text than across texts (`repeat_rate`). Near 0 for token-local content; high when a
    layer carries content that persists across the text."""
    layers, offsets = tuple(layers), tuple(offsets)

    @measure(reads=_lens_reads(layers, skip_first, target, lens), dims=("layer", "offset"),
             name=f"{'jlens' if lens == 'jacobian' else 'logit_lens'}_persistence", layers=layers,
             skip_first=skip_first, target=target if lens == "jacobian" else None, lens=lens, offsets=offsets)
    def fn(ctx):
        vals = [repeat_rate(ctx.unembed_topk(_lens_residuals(ctx, l, skip_first, target, lens), 1)[..., 0], offsets)
                for l in layers]
        return xr.DataArray(np.array(vals), dims=("layer", "offset"),
                            coords={"layer": list(layers), "offset": list(offsets)})
    return fn


# --- monitors: one direction per layer, scored per example (docs/monitors.md) ---------------------
#
# A monitor reads residual:L, projects every scored position on a direction, and reduces the
# positions of an example to one score. Scored positions: the loss mask when the examples carry one
# (the model's own turns in a replayed episode), else from `skip_first` to the one before last.
# The concept monitor's directions come from the J-lens without labels; the probe's from labels.

REDUCTIONS = ("max", "mean", "last")


def scored_positions(examples, skip_first: int) -> np.ndarray:
    """(example, position) bool: the positions a monitor scores."""
    n, seq = examples.tokens.shape
    if examples.loss_mask is not None:
        return np.asarray(examples.loss_mask, dtype=bool)
    mask = np.zeros((n, seq), dtype=bool)
    mask[:, lens_positions(seq, skip_first)] = True
    return mask


def reduce_positions(scores: np.ndarray, mask: np.ndarray, how: str) -> np.ndarray:
    """(example, position) scores to one per example over the masked positions; NaN if none."""
    if how not in REDUCTIONS:
        raise ValueError(f"unknown reduction {how!r}; one of {REDUCTIONS}")
    out = np.full(len(scores), np.nan)
    for i, (row, m) in enumerate(zip(scores, mask)):
        if m.any():
            out[i] = row[m].max() if how == "max" else row[m].mean() if how == "mean" else row[np.flatnonzero(m)[-1]]
    return out


def concept_monitor(layer: int, token_ids, skip_first: int = 16, target: str = "final", reduce: str = "max",
                    lens: str = "jacobian"):
    """Per example: the concept lens as a monitor, with no labels. Each token's score at a position
    is h . v_t, v_t = W[t] @ J_L (the linear part of its J-lens logit); the position's score is the
    max over the tokens, reduced over positions. `lens="logit"` uses v_t = W[t] (the logit lens:
    nothing fitted), the baseline."""
    ids = tuple(int(t) for t in token_ids)
    read = f"concept:{layer}:{skip_first}:{target}:{','.join(map(str, ids))}"
    reads = [f"residual:{layer}"] + ([read] if lens == "jacobian" else ["unembed"])
    name = f"{'concept' if lens == 'jacobian' else 'logit_concept'}_monitor_{layer}"

    @measure(reads=reads, dims=("example",), name=name, layer=layer, token_ids=ids, skip_first=skip_first,
             target=target if lens == "jacobian" else None, reduce=reduce, lens=lens)
    def fn(ctx):
        v = ctx.jacobian[(layer, skip_first, target, ids)] if lens == "jacobian" else ctx.unembed_matrix[list(ids)]
        per_token = (ctx.stream("residual", layer) @ v.T).max(axis=-1)
        return reduce_positions(per_token, scored_positions(ctx.examples, skip_first), reduce)
    return fn


def probe_monitor(layer: int, label: str = "label", skip_first: int = 16, reduce: str = "max"):
    """Per example: a difference-of-means probe as a monitor. The direction is the mean over
    positive fit examples of their mean scored activation, minus the same for negatives (fit rows:
    `split == "fit"`, else every row). Every example is then scored like the concept monitor."""
    from explorers.methods.probes import DiffMeans

    @measure(reads=[f"residual:{layer}"], dims=("example",), name=f"probe_monitor_{layer}",
             layer=layer, label=label, skip_first=skip_first, reduce=reduce)
    def fn(ctx):
        h = ctx.stream("residual", layer)
        mask = scored_positions(ctx.examples, skip_first)
        fit, _ = split_rows(ctx.examples)
        means = np.stack([row[m].mean(axis=0) for row, m in zip(h, mask)])
        probe = DiffMeans().fit(means[fit], np.asarray(ctx.examples.meta[label])[fit])
        return reduce_positions(h @ probe.direction, mask, reduce)
    return fn
