"""explorers.core: the primitives shared by every explorers package.

Examples (data indexed by content), State / Step / Trajectory (models over time), Observable
(pure functions that declare what they read), over / across (the engine), Store (content-addressed
results) and analyses of labelled curves. No dependency on any other explorers package.
"""

from explorers.core import analysis, observe
from explorers.core.data import Examples
from explorers.core.engine import across, over
from explorers.core.metrics import auroc, cluster_bootstrap
from explorers.core.observe import Context, Observable, observable
from explorers.core.state import State, Step, Trajectory, snapshot
from explorers.core.store import Store

__all__ = [
    "Context", "Examples", "Observable", "State", "Step", "Store", "Trajectory", "across", "analysis",
    "auroc", "cluster_bootstrap", "observable", "observe", "over", "snapshot",
]
