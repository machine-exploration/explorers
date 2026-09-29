"""Metrics for patching studies: functions of the output logits, one value per example."""


def logit_diff(correct, wrong, position: int = -1):
    """logit[correct] - logit[wrong] at `position`, per example. `correct` and `wrong` are token
    ids, one per example (or one for all)."""
    def metric(logits, ids):
        import torch

        last = logits[:, position]
        c = torch.as_tensor(correct, device=last.device).expand(len(last))
        w = torch.as_tensor(wrong, device=last.device).expand(len(last))
        rows = torch.arange(len(last), device=last.device)
        return last[rows, c] - last[rows, w]
    return metric
