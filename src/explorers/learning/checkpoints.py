"""Training checkpoints of fully open model suites. See docs/pythia.md for the naming."""

from dataclasses import dataclass

from explorers.core.state import State, Trajectory

PYTHIA_SIZES = ("14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b")


@dataclass(frozen=True)
class Checkpoint:
    model: str      # Hugging Face repo id
    revision: str   # branch holding this checkpoint
    step: int       # optimizer step


def pythia_steps() -> list[int]:
    """The 154 Pythia checkpoints: step 0, powers of two up to 512, then every 1000 up to 143000."""
    return [0] + [2 ** i for i in range(10)] + list(range(1000, 143001, 1000))


def pick(steps: list[int], n: int) -> list[int]:
    """`n` steps spread evenly in log-step, always keeping the first and the last.
    Early training changes fastest, so a log spread spends checkpoints where things move."""
    steps = sorted(set(steps))
    if n >= len(steps):
        return steps
    if n < 2:
        raise ValueError("pick at least 2 steps")
    chosen = {steps[0], steps[-1]}
    positive = [s for s in steps if s > 0]
    lo, hi = positive[0], positive[-1]
    k = n - len(chosen)
    for i in range(1, k + 1):
        target = lo * (hi / lo) ** (i / (k + 1))
        free = [s for s in steps if s not in chosen]
        chosen.add(min(free, key=lambda s: abs(s - target)))
    return sorted(chosen)


def pythia(size: str, n: int | None = None, deduped: bool = False) -> list[Checkpoint]:
    """Checkpoints of `EleutherAI/pythia-<size>[-deduped]`, all of them or `n` picked by `pick`."""
    if size not in PYTHIA_SIZES:
        raise ValueError(f"unknown Pythia size {size!r}; one of {', '.join(PYTHIA_SIZES)}")
    repo = f"EleutherAI/pythia-{size}" + ("-deduped" if deduped else "")
    steps = pythia_steps() if n is None else pick(pythia_steps(), n)
    return [Checkpoint(model=repo, revision=f"step{s}", step=s) for s in steps]


def from_checkpoints(checkpoints: list[Checkpoint], run: str | None = None, device: str = "cpu",
                     dtype: str = "float32", coords: dict | None = None) -> Trajectory:
    """A trajectory over published checkpoints, loaded from Hugging Face on demand."""
    def loader(ck: Checkpoint):
        def load():
            import torch
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(ck.model, revision=ck.revision,
                                                         torch_dtype=getattr(torch, dtype))
            return model.to(device).eval()
        return load

    run = run or checkpoints[0].model
    states = [State(run=run, step=ck.step, key=f"hf:{ck.model}@{ck.revision}", load=loader(ck))
              for ck in checkpoints]
    return Trajectory(run=run, states=states, coords=coords or {})
