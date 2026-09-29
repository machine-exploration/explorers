"""explorers.learning: single models across training.

Checkpoint suites (Pythia) as trajectories, toy settings with known answers, labelled probing
datasets and probes. Built on explorers.core.
"""

from explorers.learning import toy
from explorers.learning.checkpoints import Checkpoint, from_checkpoints, pick, pythia, pythia_steps
from explorers.learning.datasets import Dataset, from_jsonl, number_comparison, split_by_group
from explorers.learning.probes import DiffMeans, Logistic, Probe
from explorers.learning.sweep import FORMAT, Record, Sweep, read_records, write_records

__all__ = [
    "FORMAT", "Checkpoint", "Dataset", "DiffMeans", "Logistic", "Probe", "Record", "Sweep",
    "from_checkpoints", "from_jsonl", "number_comparison", "pick", "pythia", "pythia_steps",
    "read_records", "split_by_group", "toy", "write_records",
]
