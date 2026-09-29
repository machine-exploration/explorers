"""States and steps: the two objects of learning dynamics.

A `State` is the model at one point of training. It loads lazily, so a trajectory of 154
checkpoints costs nothing until something is measured. Its `key` names its content
(a checkpoint address, or a hash of the parameters), which is what results are cached under.

A `Step` is one transition between two states, with the gradient that produced it when the
run was observed live. Published checkpoint suites only give states; live and toy runs also
give steps.
"""

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class State:
    run: str
    step: int
    key: str
    load: Callable = field(compare=False, repr=False)   # () -> torch.nn.Module in eval mode


@dataclass(frozen=True)
class Step:
    run: str
    step: int                                          # the step of the state after the update
    before: State
    after: State
    grad: dict = field(compare=False, repr=False, default_factory=dict)   # param name -> tensor, at `before`


@dataclass
class Trajectory:
    run: str
    states: list[State]
    steps: list[Step] = field(default_factory=list)
    coords: dict = field(default_factory=dict)          # e.g. {"size": 70e6, "lr": 1e-3, "seed": 0}


def snapshot(model, run: str, step: int) -> State:
    """Freeze a model in memory as a State. The key hashes the parameters."""
    params = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    h = hashlib.sha256()
    for k in sorted(params):
        h.update(k.encode())
        h.update(params[k].numpy().tobytes())
    template = copy.deepcopy(model).cpu()

    def load():
        m = copy.deepcopy(template)
        m.load_state_dict(params)
        return m.eval()

    return State(run=run, step=step, key=f"params:{h.hexdigest()[:24]}", load=load)
