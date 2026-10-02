"""Interventions and reductions as data: what a trace does to a stream, and what a read keeps.

An experiment made only of these can be written to JSON, hashed, sent to another machine and planned.
Plain Python callables still work for local runs, but they make an experiment non-serializable.

Interventions (for `write`), tensor -> tensor of the same shape:
  Add(vector, scale)      h + scale * vector
  Set(value)              value, broadcast to the selection (patching)
  Scale(factor)           factor * h
  Ablate()                zeros
  ProjectOut(direction)   h minus its component along direction

Reductions (for `read(..., reduce=)`), applied on the device before anything is stored:
  Project(direction)      h · direction          (..., d) -> (...)
  Norm()                  ||h||                  (..., d) -> (...)

Measurements (losses, logit differences, lenses) are `explorers.measures`, not ops.
"""

import base64
import hashlib
import json
from dataclasses import dataclass, fields

import numpy as np

OPS: dict[str, type] = {}


def _register(cls):
    OPS[cls.__name__] = cls
    return cls


def _encode(x):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)) and all(isinstance(v, (int, float)) for v in x):
        return list(x)
    a = np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, dtype=np.float32)
    return {"array": base64.b64encode(np.ascontiguousarray(a).tobytes()).decode(), "shape": list(a.shape)}


def _decode(x):
    if isinstance(x, dict) and "array" in x:
        return np.frombuffer(base64.b64decode(x["array"]), dtype=np.float32).reshape(x["shape"]).copy()
    return x


def _tensor(x, like):
    import torch

    return torch.as_tensor(np.asarray(x) if not hasattr(x, "detach") else x, dtype=like.dtype, device=like.device)


class Op:
    """Base: a frozen dataclass that is callable and round-trips through JSON."""

    def to_dict(self) -> dict:
        return {"op": type(self).__name__, **{f.name: _encode(getattr(self, f.name)) for f in fields(self)}}

    @staticmethod
    def from_dict(d: dict) -> "Op":
        d = dict(d)
        cls = OPS[d.pop("op")]
        return cls(**{k: _decode(v) for k, v in d.items()})


@_register
@dataclass(frozen=True, eq=False)
class Add(Op):
    vector: object
    scale: float = 1.0

    def __call__(self, h):
        return h + self.scale * _tensor(self.vector, h)


@_register
@dataclass(frozen=True, eq=False)
class Set(Op):
    value: object

    def __call__(self, h):
        return _tensor(self.value, h).expand_as(h)


@_register
@dataclass(frozen=True, eq=False)
class Scale(Op):
    factor: float

    def __call__(self, h):
        return self.factor * h


@_register
@dataclass(frozen=True, eq=False)
class Ablate(Op):
    def __call__(self, h):
        return h * 0


@_register
@dataclass(frozen=True, eq=False)
class ProjectOut(Op):
    direction: object

    def __call__(self, h):
        u = _tensor(self.direction, h)
        u = u / u.norm()
        return h - (h @ u)[..., None] * u


@_register
@dataclass(frozen=True, eq=False)
class Project(Op):
    direction: object

    def __call__(self, h):
        return h @ _tensor(self.direction, h)


@_register
@dataclass(frozen=True, eq=False)
class Norm(Op):
    def __call__(self, h):
        return h.norm(dim=-1)


def is_data(x) -> bool:
    return x is None or isinstance(x, Op)


def spec_hash(spec: dict) -> str:
    """Content hash of a serialized experiment: equal specs, equal hashes, on any machine."""
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:32]
