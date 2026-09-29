"""explorers: read, write and trace the internal computation of neural networks.

    import explorers as ex

    model = ex.open("EleutherAI/pythia-70m", revision="step1000")
    with model.trace(tokens) as run:
        x = run.stream("residual").read(layer=3, position=-1)

    study = ex.Study(models=[model], examples=tokens)
    study.read("residual", layers="*", position=-1)
    ds = study.compute()

Subpackages: `explorers.core` (examples, states along training, observables, the engine, the
content-addressed store, analyses), `explorers.learning` (checkpoint suites, toy tasks),
`explorers.methods` (probes, steering, patching, sparse autoencoders), `explorers.populations`
(scenarios and agent episodes; extra `populations`).
"""

from explorers import ops
from explorers.methods import patching, sae, steering
from explorers.model import Model, open
from explorers.study import Study
from explorers.trace import Trace, Value

__version__ = "0.2.0.dev0"
__all__ = ["Model", "Study", "Trace", "Value", "open", "ops", "patching", "sae", "steering"]
