"""Checkpoints x layers x probes -> one record each. Records are the versioned output format."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from explorers.learning.activations import ActivationSource
from explorers.learning.checkpoints import Checkpoint
from explorers.learning.datasets import Dataset, split_by_group
from explorers.core.metrics import auroc, cluster_bootstrap
from explorers.methods.probes import Probe

FORMAT = "explorers.result/v0"


@dataclass(frozen=True)
class Record:
    model: str
    revision: str
    step: int
    layer: int
    dataset: str
    dataset_version: str
    dataset_fingerprint: str
    probe: str
    probe_version: str
    position: str
    n_train: int
    n_test: int
    auroc: float | None
    ci_low: float | None
    ci_high: float | None
    seed: int
    format: str = field(default=FORMAT)


@dataclass
class Sweep:
    checkpoints: list[Checkpoint]
    dataset: Dataset
    layers: list[int]
    probes: list[Probe]
    source: ActivationSource
    position: str = "last"
    test_frac: float = 0.3
    seed: int = 0
    resamples: int = 1000

    def run(self, on_record=None) -> list[Record]:
        train, test = split_by_group(self.dataset.groups, self.test_frac, self.seed)
        y = np.array(self.dataset.labels)
        test_groups = [self.dataset.groups[i] for i in test]
        records = []
        for ck in self.checkpoints:
            acts = self.source.read(ck, self.dataset, self.layers)
            for layer in self.layers:
                X = acts[layer]
                for probe in self.probes:
                    s = probe.fit(X[train], y[train]).score(X[test])
                    point = auroc(y[test], s)
                    ci = cluster_bootstrap(y[test], s, test_groups, resamples=self.resamples,
                                           seed=self.seed) if point is not None else None
                    r = Record(model=ck.model, revision=ck.revision, step=ck.step, layer=layer,
                               dataset=self.dataset.name, dataset_version=self.dataset.version,
                               dataset_fingerprint=self.dataset.fingerprint,
                               probe=probe.name, probe_version=probe.version, position=self.position,
                               n_train=len(train), n_test=len(test), auroc=point,
                               ci_low=ci[0] if ci else None, ci_high=ci[1] if ci else None, seed=self.seed)
                    records.append(r)
                    if on_record:
                        on_record(r)
        return records


def write_records(path: Path, records: list[Record]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


def read_records(path: Path) -> list[Record]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("format") != FORMAT:
            raise ValueError(f"unsupported record format {row.get('format')!r}; this version reads {FORMAT}")
        out.append(Record(**row))
    return out
