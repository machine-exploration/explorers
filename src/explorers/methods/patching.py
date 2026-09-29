"""Metrics for patching studies: functions of the output logits, one value per example."""

from explorers.ops import LogitDiff


def logit_diff(correct, wrong, position: int = -1) -> LogitDiff:
    """logit[correct] - logit[wrong] at `position`, per example (`explorers.ops` data)."""
    return LogitDiff(correct, wrong, position)
