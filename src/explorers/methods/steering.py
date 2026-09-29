"""Steering: add a direction to a stream. A write, nothing more."""

from explorers.ops import Add


def add(direction, scale: float = 1.0) -> Add:
    """An intervention for `write`: h -> h + scale * direction (broadcast over batch and positions).
    It is `explorers.ops` data, so a study that uses it stays serializable."""
    return Add(direction, scale)
