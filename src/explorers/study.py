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

A study made only of `explorers.ops` data (interventions, reductions, metrics) serializes to JSON
with `spec()` and has a content `key()`: that is what can be stored, compared and sent to a remote
executor. Python callables still work locally, but `spec()` refuses them.

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
    """[(coordinate, content key or None, loader)] for any accepted form of `models`."""
    from explorers.model import Model

    states = getattr(models, "states", None)
    if states is not None:                                     # explorers.core.Trajectory
        return [(s.step, s.key, lambda s=s: Model(s.load(), name=s.run, revision=f"step{s.step}")) for s in states]
    if isinstance(models, Model) or callable(models) and not isinstance(models, (list, tuple)):
        models = [models]
    out = []
    for i, m in enumerate(models):
        if isinstance(m, Model):
            out.append((m.key, m.key, lambda m=m: m))
        else:
            out.append((i, None, m))
    return out


def _fingerprint(examples) -> str:
    fp = getattr(examples, "fingerprint", None)
    if fp is not None:
        return fp
    import hashlib

    return hashlib.sha256(np.asarray(examples, dtype=np.int64).tobytes()).hexdigest()[:16]


def _position(p):
    if p is None or isinstance(p, int) or p == "*":
        return p
    if isinstance(p, (list, tuple)):
        return [int(x) for x in p]
    raise ValueError(f"position {p!r} cannot be serialized: use an int, a list of ints or None")


def _layers(layers, top: int):
    return list(range(top + 1)) if layers == "*" else list(layers)


class Study:
    def __init__(self, models, examples, batch_size: int = 16):
        self._models = _models(models)
        self.tokens, self.example_ids = _as_tokens(examples)
        self._fingerprint = _fingerprint(examples)
        self.batch_size = batch_size
        self.reads: list = []
        self.writes: list = []
        self.measures: list = []
        self.patches: list = []

    # --- declarations -------------------------------------------------------------------------

    def read(self, stream: str, layers, position=None, name: str | None = None, reduce=None) -> "Study":
        """`reduce` (e.g. `ops.Project(direction)`) runs on the device; only its output is kept."""
        self.reads.append((name or stream, stream, layers, position, reduce))
        return self

    def write(self, stream: str, layer: int, position=None, fn=None, value=None) -> "Study":
        if (fn is None) == (value is None):
            raise ValueError("write needs exactly one of fn= or value=")
        from explorers.ops import Set

        self.writes.append((stream, layer, position, fn if fn is not None else Set(value)))
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
        self.patches.append((name, src, stream, layers, positions, metric, method, _fingerprint(source)))
        return self

    # --- serialization -----------------------------------------------------------------------

    def spec(self) -> dict:
        """The study as JSON-ready data (format `explorers.study/v0`). Raises if any part is a Python
        callable instead of `explorers.ops` data, or if a model has no content key."""
        from explorers.ops import is_data

        def data(x, what):
            if not is_data(x):
                raise ValueError(f"{what} is a Python callable; use explorers.ops to make the study serializable")
            return None if x is None else x.to_dict()

        keys = [key for _, key, _ in self._models]
        if any(k is None for k in keys):
            raise ValueError("a model is a loader without a content key; pass Model objects or a Trajectory")
        layers = lambda ls: ls if ls == "*" else [int(x) for x in ls]  # noqa: E731
        return {
            "format": "explorers.study/v0",
            "models": keys,
            "examples": {"fingerprint": self._fingerprint, "shape": list(self.tokens.shape)},
            "reads": [{"name": n, "stream": s, "layers": layers(ls), "position": _position(p),
                       "reduce": data(r, f"read {n!r} reduce")} for n, s, ls, p, r in self.reads],
            "writes": [{"stream": s, "layer": int(layer), "position": _position(p), "op": data(fn, f"write {i}")}
                       for i, (s, layer, p, fn) in enumerate(self.writes)],
            "measures": [{"name": n, "metric": data(fn, f"measure {n!r}")} for n, fn in self.measures],
            "patches": [{"name": n, "source": fp, "stream": s, "layers": layers(ls), "positions": _position(ps),
                         "metric": data(m, f"patch {n!r} metric"), "method": method}
                        for n, _src, s, ls, ps, m, method, fp in self.patches],
        }

    def key(self) -> str:
        """Content key of the study: equal studies get equal keys, on any machine."""
        from explorers.ops import spec_hash

        return spec_hash(self.spec())

    # --- execution ----------------------------------------------------------------------------

    def _trace(self, model, ids, reads=(), extra_writes=(), grad=False):
        run = model.trace(ids, grad=grad)
        for stream, layer, position, fn in list(self.writes) + list(extra_writes):
            run.stream(stream).write(layer, position, fn=fn)
        values = [run.stream(s).read(layer, position, reduce) for s, layer, position, reduce in reads]
        run.run()
        return run, values

    def _batches(self):
        for i in range(0, len(self.tokens), self.batch_size):
            yield slice(i, i + self.batch_size)

    def _compute_model(self, model) -> dict:
        import torch

        out: dict = {}
        top = {"residual": model.n_layers}
        read_specs = [(name, s, _layers(layers, top.get(s, model.n_layers - 1)), pos, red)
                      for name, s, layers, pos, red in self.reads]
        flat = [(s, layer, pos, red) for _, s, layers, pos, red in read_specs for layer in layers]
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
            for name, s, layers, pos, red in read_specs:
                arrs = [np.concatenate(parts[k + j]) for j in range(len(layers))]
                k += len(layers)
                stacked = np.stack(arrs, axis=1)                  # (example, layer, [position], [d])
                has_pos = not isinstance(pos, int)
                dims = ("example", "layer") + (("position",) if has_pos else ())
                dims += ("d",) if stacked.ndim == len(dims) + 1 else ()
                coords = {"layer": layers}
                if has_pos:
                    seq = self.tokens.shape[1]
                    coords["position"] = np.arange(seq)[pos if pos is not None else slice(None)]
                out[name] = xr.DataArray(stacked, dims=dims, coords=coords)
            for name, chunks in meas.items():
                out[name] = xr.DataArray(np.concatenate(chunks), dims=("example",))
        for name, src, stream, layers, positions, metric, method, _fp in self.patches:
            out[name] = self._patch(model, src, stream, layers, positions, metric, method)
        return out

    def _patch(self, model, src, stream, layers, positions, metric, method) -> xr.DataArray:
        import torch

        from explorers.ops import Set

        seq = self.tokens.shape[1]
        top = model.n_layers if stream == "residual" else model.n_layers - 1
        layers = _layers(layers, top)
        positions = list(range(seq)) if positions == "*" else list(positions)
        result = np.full((len(layers), len(positions), len(self.tokens)), np.nan)
        for b in self._batches():
            clean_run, clean_acts = self._trace(model, src[b], [(stream, layer, None, None) for layer in layers])
            m_clean = torch.as_tensor(metric(clean_run.logits, clean_run.ids)).float()
            if method == "exact":
                base, _ = self._trace(model, self.tokens[b])
                m_corrupt = torch.as_tensor(metric(base.logits, base.ids)).float()
                for li, layer in enumerate(layers):
                    for pi, pos in enumerate(positions):
                        patched, _ = self._trace(model, self.tokens[b], extra_writes=[
                            (stream, layer, pos, Set(clean_acts[li].value[:, pos]))])
                        m = torch.as_tensor(metric(patched.logits, patched.ids)).float()
                        result[li, pi, b] = ((m - m_corrupt) / (m_clean - m_corrupt)).cpu().numpy()
            else:
                run, acts = self._trace(model, self.tokens[b], [(stream, layer, None, None) for layer in layers], grad=True)
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
        for coord, _key, load in self._models:
            model = load()
            per_model.append(xr.Dataset(self._compute_model(model)))
            coords.append(coord)
            del model
        ds = xr.concat(per_model, dim="model")
        ds = ds.assign_coords(model=coords)
        if "example" in ds.dims:
            ds = ds.assign_coords(example=self.example_ids)
        return ds
