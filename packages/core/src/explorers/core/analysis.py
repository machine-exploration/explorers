"""Analyses of curves over training. They only see labelled arrays, never models."""

import numpy as np
import xarray as xr


def onsets(curves: xr.DataArray, frac: float = 0.5, min_drop: float = 0.1) -> xr.Dataset:
    """When does each curve (a loss going down over `step`) make its drop?

    For every sample along the other dimensions:
      drop       first value minus last value
      onset      first step where the curve has covered `frac` of its drop, interpolated in log-step
      sharpness  largest drop between two consecutive steps, divided by the total drop:
                 near 1 for a sudden ("monogenic") transition, small for a gradual one
    Samples that drop by less than `min_drop` get NaN onset and sharpness.
    """
    c = curves.transpose("step", ...)
    steps = np.asarray(c["step"].values, dtype=float)
    vals = np.asarray(c.values, dtype=float).reshape(len(steps), -1)
    first, last = vals[0], vals[-1]
    drop = first - last
    target = first - frac * drop
    logs = np.log1p(steps)
    onset = np.full(vals.shape[1], np.nan)
    sharp = np.full(vals.shape[1], np.nan)
    for j in np.where(drop >= min_drop)[0]:
        below = np.where(vals[:, j] <= target[j])[0]
        if len(below) == 0:
            continue
        k = below[0]
        if k == 0:
            onset[j] = steps[0]
        else:
            y0, y1 = vals[k - 1, j], vals[k, j]
            t = (y0 - target[j]) / (y0 - y1) if y0 != y1 else 1.0
            onset[j] = np.expm1(logs[k - 1] + t * (logs[k] - logs[k - 1]))
        sharp[j] = np.max(-np.diff(vals[:, j])) / drop[j]
    rest = [d for d in c.dims if d != "step"]
    shape = [c.sizes[d] for d in rest]
    coords = {name: coord for name, coord in c.coords.items() if "step" not in coord.dims}
    return xr.Dataset({"drop": (rest, drop.reshape(shape)), "onset": (rest, onset.reshape(shape)),
                       "sharpness": (rest, sharp.reshape(shape))}, coords=coords)


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Rank correlation, ignoring pairs where either value is NaN."""
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return float("nan")
    ra, rb = _ranks(a[ok]), _ranks(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")                     # a constant side has no ranking to correlate
    return float(np.corrcoef(ra, rb)[0, 1])


def cluster_curves(curves: xr.DataArray, k: int, seed: int = 0, iters: int = 100) -> xr.DataArray:
    """Group samples whose curves fall at the same time: candidate quanta.

    Each curve is rescaled to go from 1 (first step) to 0 (last step), so only its timing
    counts, then k-means groups them. Curves with no drop are labelled -1."""
    c = curves.transpose("step", ...)
    vals = np.asarray(c.values, dtype=float).reshape(c.sizes["step"], -1).T      # (sample, step)
    drop = vals[:, 0] - vals[:, -1]
    ok = (drop > 1e-9) & ~np.isnan(vals).any(axis=1)
    X = (vals[ok] - vals[ok, -1:]) / drop[ok, None]
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), size=min(k, len(X)), replace=False)]
    for _ in range(iters):
        assign = ((X[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        new = np.array([X[assign == i].mean(0) if (assign == i).any() else centers[i] for i in range(len(centers))])
        if np.allclose(new, centers):
            break
        centers = new
    # order clusters by when they fall: cluster 0 learns first
    fall = (centers > 0.5).sum(1)
    relabel = np.argsort(np.argsort(fall))
    labels = np.full(len(vals), -1)
    labels[ok] = relabel[assign]
    rest = [d for d in c.dims if d != "step"]
    coords = {name: coord for name, coord in c.coords.items() if "step" not in coord.dims}
    return xr.DataArray(labels.reshape([c.sizes[d] for d in rest]), dims=rest, coords=coords, name="cluster")
