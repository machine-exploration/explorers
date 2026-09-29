from collections.abc import Callable

import numpy as np


def auroc(y: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank-based AUROC with ties counted as half. O(n log n). None when a class is empty."""
    y = np.asarray(y, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores))
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1   # average rank of the tie block, 1-based
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cluster_bootstrap(y: np.ndarray, scores: np.ndarray, groups: list[str],
                      stat: Callable[[np.ndarray, np.ndarray], float | None] = auroc,
                      resamples: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    """95% percentile interval for `stat`, resampling whole groups (texts in a group correlate)."""
    y, scores = np.asarray(y), np.asarray(scores)
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    members = [np.array(ix) for ix in by_group.values()]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(resamples):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        s = stat(y[idx], scores[idx])
        if s is not None:
            stats.append(s)
    if not stats:
        return None
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)
