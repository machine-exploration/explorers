"""Labelled texts to probe for. A group ties texts that must land in the same split."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    name: str
    version: str
    texts: list[str]
    labels: list[bool]
    groups: list[str]

    def __post_init__(self):
        if not (len(self.texts) == len(self.labels) == len(self.groups)):
            raise ValueError("texts, labels and groups must have the same length")

    @property
    def fingerprint(self) -> str:
        """Content hash: two datasets with the same texts, labels and groups share it."""
        h = hashlib.sha256()
        for t, y, g in zip(self.texts, self.labels, self.groups):
            h.update(json.dumps([t, y, g]).encode())
        return h.hexdigest()[:16]


def number_comparison(n: int = 400, seed: int = 0) -> Dataset:
    """'<a> is greater than <b>.' labelled a > b. Both orders of a pair share a group, so a
    probe cannot pass the test split by remembering the pair."""
    rng = random.Random(seed)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < n // 2:
        a, b = rng.sample(range(1, 100), 2)
        pairs.add((min(a, b), max(a, b)))
    texts, labels, groups = [], [], []
    for lo, hi in sorted(pairs):
        for a, b in ((hi, lo), (lo, hi)):
            texts.append(f"{a} is greater than {b}.")
            labels.append(a > b)
            groups.append(f"{lo}-{hi}")
    return Dataset(name="number_comparison", version="0", texts=texts, labels=labels, groups=groups)


def from_jsonl(path: Path, name: str | None = None, version: str = "0") -> Dataset:
    """One `{"text", "label", "group"?}` object per line; a missing group is the line number."""
    texts, labels, groups = [], [], []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        texts.append(row["text"])
        labels.append(bool(row["label"]))
        groups.append(str(row.get("group", i)))
    return Dataset(name=name or Path(path).stem, version=version, texts=texts, labels=labels, groups=groups)


def split_by_group(groups: list[str], test_frac: float = 0.3, seed: int = 0) -> tuple[list[int], list[int]]:
    """Train and test indices with no group on both sides."""
    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    test_groups = set(unique[: max(1, round(len(unique) * test_frac))])
    train = [i for i, g in enumerate(groups) if g not in test_groups]
    test = [i for i, g in enumerate(groups) if g in test_groups]
    return train, test
