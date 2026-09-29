"""Steering: add a direction to a stream. A write, nothing more."""


def add(direction, scale: float = 1.0):
    """`fn` for `write`: h -> h + scale * direction (direction broadcast over batch and positions)."""
    def fn(h):
        import torch

        return h + scale * torch.as_tensor(direction, dtype=h.dtype, device=h.device)
    return fn
