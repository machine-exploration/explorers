"""Studies: the unit of work. Reads, writes, measurements and patching over models × examples.

    study = ex.Study(models=[m0, m1], examples=tokens)
    study.read("residual", layers=range(4), position=-1)
    study.write("residual", layer=2, position=-1, fn=ex.steering.add(direction, 4.0))
    study.measure("logit_diff", ex.patching.logit_diff(correct, wrong))
    study.patch(source=clean, stream="residual", layers="*", positions="*", metric=metric)
    ds = study.compute()

A study only declares; `compute` executes it. Models can be `Model`s, zero-argument callables that
return one (loaded one at a time and released), or a `explorers.core.Trajectory` of checkpoints.
Writes declared on the study apply to every run of the study, patched runs included.

Execution today is direct (one traced forward pass per batch, one per patched site). The plan is
the part the Runtime will optimise; the results must not change when it does.
"""

import numpy as np
import xarray as xr


def _as_tokens(examples):
    tokens = getattr(examples, "tokens", examples)
    ids = getattr(examples, "ids", None)
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError("examples must be (n_examples, seq_len) token ids, or explorers.core.Examples")
    return tokens, (np.asarray(ids) if ids is not None else np.arange(len(tokens)))


def _models(models):
    """[(coordinate, loader)] for any accepted form of `models`."""
    from explorers.model import Model

    states = getattr(models, "states", None)
    if states is not None:                                     # explorers.core.Trajectory
        return [(s.step, lambda s=s: Model(s.load(), name=s.run, revision=f"step{s.step}")) for s in states]
    if isinstance(models, Model) or callable(models) and not isinstance(models, (list, tuple)):
        models = [models]
    out = []
    for i, m in enumerate(models):
        if isinstance(m, Model):
            out.append((m.key, lambda m=m: m))
        else:
            out.append((i, m))
    return out


def _layers(layers, top: int):
    return list(range(top + 1)) if layers == "*" else list(layers)


class Study:
    def __init__(self, models, examples, batch_size: int = 16):
        self._models = _models(models)
        self.tokens, self.example_ids = _as_tokens(examples)
        self.batch_size = batch_size
        self.reads: list = []
        self.writes: list = []
        self.measures: list = []
        self.patches: list = []

    # --- declarations -------------------------------------------------------------------------

    def read(self, stream: str, layers, position=None, name: str | None = None) -> "Study":
        self.reads.append((name or stream, stream, layers, position))
        return self

    def write(self, stream: str, layer: int, position=None, fn=None, value=None) -> "Study":
        if (fn is None) == (value is None):
            raise ValueError("write needs exactly one of fn= or value=")
        self.writes.append((stream, layer, position, fn, value))
        return self

    def measure(self, name: str, fn) -> "Study":
        """`fn(logits, ids) -> (batch,)`: a scalar per example, from the output logits."""
        self.measures.append((name, fn))
        return self

    def patch(self, source, stream: str = "residual", layers="*", positions="*", metric=None,
              method: str = "exact", name: str = "patch") -> "Study":
        """Patch `stream` from the `source` examples (clean) into the study's examples (corrupt),
        one site (layer, position) at a time, and report the normalized effect on `metric`:
        (m_patched - m_corrupt) / (m_clean - m_corrupt), per example. 1 = the site restores the
        clean behaviour, 0 = no effect. `method="attribution"` approximates every site at once with
        one backward pass: (a_clean - a_corrupt) · d metric / d a, at the corrupt run."""
        if metric is None:
            raise ValueError("patch needs a metric(logits, ids) -> (batch,)")
        if method not in ("exact", "attribution"):
            raise ValueError("method is 'exact' or 'attribution'")
        src, _ = _as_tokens(source)
        if src.shape != self.tokens.shape:
            raise ValueError("source and target examples must have the same shape")
        self.patches.append((name, src, stream, layers, positions, metric, method))
        return self

    # --- execution ----------------------------------------------------------------------------

    def _trace(self, model, ids, reads=(), extra_writes=(), grad=False):
        run = model.trace(ids, grad=grad)
        for stream, layer, position, fn, value in list(self.writes) + list(extra_writes):
            run.stream(stream).write(layer, position, fn=fn, value=value)
        values = [run.stream(s).read(layer, position) for s, layer, position in reads]
        run.run()
        return run, values

    def _batches(self):
        for i in range(0, len(self.tokens), self.batch_size):
            yield slice(i, i + self.batch_size)

    def _compute_model(self, model) -> dict:
        import torch

        out: dict = {}
        top = {"residual": model.n_layers}
        read_specs = [(name, s, _layers(layers, top.get(s, model.n_layers - 1)), pos)
                      for name, s, layers, pos in self.reads]
        flat = [(s, layer, pos) for _, s, layers, pos in read_specs for layer in layers]
        if flat or self.measures:
            parts = {i: [] for i in range(len(flat))}
            meas = {name: [] for name, _ in self.measures}
            for b in self._batches():
                run, values = self._trace(model, self.tokens[b], flat)
                for i, v in enumerate(values):
                    parts[i].append(v.value.float().cpu().numpy())
                ids = run.ids
                for name, fn in self.measures:
                    meas[name].append(torch.as_tensor(fn(run.logits, ids)).float().cpu().numpy())
            k = 0
            for name, s, layers, pos in read_specs:
                arrs = [np.concatenate(parts[k + j]) for j in range(len(layers))]
                k += len(layers)
                stacked = np.stack(arrs, axis=1)                  # (example, layer, [position], d)
                dims = ("example", "layer", "position", "d") if stacked.ndim == 4 else ("example", "layer", "d")
                coords = {"layer": layers}
                if stacked.ndim == 4:
                    seq = self.tokens.shape[1]
                    coords["position"] = np.arange(seq)[pos if pos is not None else slice(None)]
                out[name] = xr.DataArray(stacked, dims=dims, coords=coords)
            for name, chunks in meas.items():
                out[name] = xr.DataArray(np.concatenate(chunks), dims=("example",))
        for name, src, stream, layers, positions, metric, method in self.patches:
            out[name] = self._patch(model, src, stream, layers, positions, metric, method)
        return out

    def _patch(self, model, src, stream, layers, positions, metric, method) -> xr.DataArray:
        import torch

        seq = self.tokens.shape[1]
        top = model.n_layers if stream == "residual" else model.n_layers - 1
        layers = _layers(layers, top)
        positions = list(range(seq)) if positions == "*" else list(positions)
        result = np.full((len(layers), len(positions), len(self.tokens)), np.nan)
        for b in self._batches():
            clean_run, clean_acts = self._trace(model, src[b], [(stream, layer, None) for layer in layers])
            m_clean = torch.as_tensor(metric(clean_run.logits, clean_run.ids)).float()
            if method == "exact":
                base, _ = self._trace(model, self.tokens[b])
                m_corrupt = torch.as_tensor(metric(base.logits, base.ids)).float()
                for li, layer in enumerate(layers):
                    for pi, pos in enumerate(positions):
                        patched, _ = self._trace(model, self.tokens[b], extra_writes=[
                            (stream, layer, pos, None, clean_acts[li].value[:, pos])])
                        m = torch.as_tensor(metric(patched.logits, patched.ids)).float()
                        result[li, pi, b] = ((m - m_corrupt) / (m_clean - m_corrupt)).cpu().numpy()
            else:
                run, acts = self._trace(model, self.tokens[b], [(stream, layer, None) for layer in layers], grad=True)
                m_corrupt = metric(run.logits, run.ids)
                m_corrupt.sum().backward()
                denom = (m_clean - m_corrupt.detach().float())
                for li in range(len(layers)):
                    diff = clean_acts[li].value.float() - acts[li].value.detach().float()
                    attr = (diff * acts[li].grad.float()).sum(-1)          # (batch, seq)
                    result[li, :, b] = (attr[:, positions] / denom[:, None]).T.cpu().numpy()
        return xr.DataArray(result, dims=("layer", "position", "example"),
                            coords={"layer": layers, "position": positions})

    def compute(self) -> xr.Dataset:
        per_model, coords = [], []
        for coord, load in self._models:
            model = load()
            per_model.append(xr.Dataset(self._compute_model(model)))
            coords.append(coord)
            del model
        ds = xr.concat(per_model, dim="model")
        ds = ds.assign_coords(model=coords)
        if "example" in ds.dims:
            ds = ds.assign_coords(example=self.example_ids)
        return ds
