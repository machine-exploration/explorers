"""explorers: read, write and trace the internal computation of neural networks.

    import explorers as ex

    model = ex.open("EleutherAI/pythia-70m", revision="step143000")
    with model.trace(tokens) as run:
        x = run.stream("residual").read(layer=3, position=-1)

    experiment = ex.Experiment(ex.checkpoints("EleutherAI/pythia-70m", steps=[0, 1000, 143000]), examples)
    experiment.read("residual", layers="*", position=-1)
    experiment.measure(ex.measures.loss, ex.measures.jlens_error(layers=range(1, 6)))
    ds = experiment.compute(store="runs/store")

Concepts: a Model has named streams (residual, attn_out, mlp_out); a trace reads and writes them in
one forward pass; `ops` are interventions and reductions as data; `measures` are named, versioned
functions of what a model computed; an Experiment runs reads, writes, measures and patching over models ×
examples; a Client sends jobs to a Runtime (LocalRuntime on the CPU checks the contract; GPU
runtimes run the real jobs). Supporting modules: `data` (examples), `state` (training runs), `store`, `analysis`, `toy`
(tasks with known answers), `methods` (probes, sparse autoencoders). The agent side, `populations`,
needs the `populations` extra.
"""

from explorers import analysis, episodes, measures, ops
from explorers.data import Examples
from explorers.methods import sae
from explorers.model import (Model, ModelRef, adapters, archive_adapters, checkpoint, checkpoints, open, pick,
                             pythia_steps)
from explorers.experiment import Experiment
from explorers.runtime import Client, Job, LocalRuntime, Resources, Runtime
from explorers.trace import Trace, Value

__version__ = "0.3.0.dev0"
__all__ = ["Client", "Examples", "Experiment", "Job", "LocalRuntime", "Model", "ModelRef", "Resources", "Runtime",
           "Trace", "Value", "adapters", "analysis", "archive_adapters", "checkpoint", "checkpoints", "episodes",
           "measures", "open", "ops", "pick", "pythia_steps", "sae"]
