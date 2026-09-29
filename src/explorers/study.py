"""Studies: the unit of work. Reads, writes, measures and patching over models × examples.

    study = ex.Study(models=ex.checkpoints("EleutherAI/pythia-70m", steps=[0, 1000, 143000]), examples=ex_)
    study.read("residual", layers="*", position=-1, reduce=ex.ops.Norm())
    study.write("residual", layer=2, position=-1, fn=ex.ops.Add(direction, 4.0))
    study.measure(ex.measures.loss, ex.measures.logit_diff(correct, wrong))
    study.patch(source=clean, stream="residual", metric=ex.measures.logit_diff(correct, wrong))
    ds = study.compute(store="runs/store")

A study only declares; `compute` executes it, one model at a time.

- **Models:** `Model`s, lazy handles from `ex.checkpoints(...)` (loaded one at a time, released
  after), zero-argument callables returning a `Model`, or an `explorers.state.Trajectory` (whose
  training steps also serve step measures such as `update_norm`). When every model is a checkpoint
  with a step, the result's first dimension is `step`; otherwise it is `model`.
- **Examples:** `explorers.data.Examples`, or a (n, seq) token array. Their metadata columns become
  coordinates on the `example` dimension.
- **Writes** apply to every run of the study: reads, measures and patched runs (Jacobians are fitted
  on the unmodified model and refuse writes).
- **`read`** returns raw stream values (or a reduction computed on the device); **`measure`** runs
  `explorers.measures` on everything one traced forward pass serves.
- **`compute(store=folder)`** caches each output of each model under (study key, model key, output),
  so a long study resumes where it stopped. **`spec()`** is the study as JSON (`explorers.study/v0`)
  and **`key()`** its hash; interventions and reductions must then be `explorers.ops` data.

Execution is direct: one traced forward pass per batch, one per patched site. The Runtime will plan
it; the results must not change when it does.
"""

import json

import numpy as np
import xarray as xr


def _examples(examples):
    from explorers.data import Examples

    if isinstance(examples, Examples):
        return examples
    tokens = np.asarray(examples)
    if tokens.ndim != 2:
        raise ValueError("examples must be explorers.data.Examples or (n_examples, seq_len) token ids")
    return Examples(tokens=tokens.astype(np.int64))


def _models(models):
    """[(coordinate, content key or None, loader, owned, step or None)]. `owned` models are loaded by
    the study and released after use."""
    from explorers.model import Model, ModelRef

    states = getattr(models, "states", None)
    if states is not None:                                     # a Trajectory
        return [(s.step, s.key, lambda s=s: Model(s.load(), name=s.run, revision=f"step{s.step}"), True, s.step)
                for s in states]
    if isinstance(models, (Model, ModelRef)) or callable(models) and not isinstance(models, (list, tuple)):
        models = [models]
    out = []
    for i, m in enumerate(models):
        if isinstance(m, Model):
            out.append((m.key, m.key, lambda m=m: m, False, None))
        elif isinstance(m, ModelRef):
            out.append((m.step if m.step is not None else m.key, m.key, m.load, True, m.step))
        else:
            out.append((i, None, m, True, None))
    return out


def _position(p):
    if p is None or isinstance(p, int) or p == "*":
        return p
    if isinstance(p, (list, tuple)):
        return [int(x) for x in p]
    raise ValueError(f"position {p!r} cannot be serialized: use an int, a list of ints or None")


def _layers(layers, top: int):
    return list(range(top + 1)) if layers == "*" else list(layers)


def _describe(m) -> dict:
    return {"name": m.name, "version": m.version, "params": json.loads(json.dumps(m.params, default=str))}


def _free():
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class Study:
    def __init__(self, models, examples, batch_size: int = 16, dim_batch: int = 1):
        """`dim_batch`: copies of each batch stacked when fitting Jacobians; more is faster on a GPU
        and uses more memory. It does not change results."""
        self._models = _models(models)
        self._steps = list(getattr(models, "steps", []) or [])
        self.examples = _examples(examples)
        self.batch_size, self.dim_batch = batch_size, dim_batch
        self.reads: list = []
        self.writes: list = []
        self.measures: list = []
        self.patches: list = []

    @property
    def tokens(self) -> np.ndarray:
        return self.examples.tokens

    # --- declarations -------------------------------------------------------------------------

    def read(self, stream: str, layers, position=None, name: str | None = None, reduce=None) -> "Study":
        """Raw stream values. `reduce` (e.g. `ops.Project(direction)`) runs on the device; only its
        output is kept."""
        self.reads.append((name or stream, stream, layers, position, reduce))
        return self

    def write(self, stream: str, layer: int, position=None, fn=None, value=None) -> "Study":
        if (fn is None) == (value is None):
            raise ValueError("write needs exactly one of fn= or value=")
        from explorers.ops import Set

        self.writes.append((stream, layer, position, fn if fn is not None else Set(value)))
        return self

    def measure(self, *measures) -> "Study":
        """`explorers.measures` (losses, weight statistics, lenses, logit differences), all served by
        one traced forward pass per batch (and one backward sweep for Jacobians)."""
        for m in measures:
            if m.on_steps and not self._steps:
                raise ValueError(f"{m.name} measures training steps; pass a Trajectory with steps")
        self.measures.extend(measures)
        return self

    def patch(self, source, stream: str = "residual", layers="*", positions="*", metric=None,
              method: str = "exact", name: str = "patch") -> "Study":
        """Patch `stream` from the `source` examples (clean) into the study's examples (corrupt), one
        site (layer, position) at a time, and report the normalized effect on `metric` (a measure of
        the logits, one value per example, e.g. `measures.logit_diff`):
        (m_patched - m_corrupt) / (m_clean - m_corrupt). 1 = the site restores the clean behaviour,
        0 = no effect. `method="attribution"` approximates every site with one backward pass:
        (a_clean - a_corrupt) · d metric / d a, at the corrupt run."""
        if metric is None or not all(r.startswith("logits:") for r in metric.reads):
            raise ValueError("patch needs a metric that reads logits, e.g. measures.logit_diff(correct, wrong)")
        if method not in ("exact", "attribution"):
            raise ValueError("method is 'exact' or 'attribution'")
        src = _examples(source)
        if src.tokens.shape != self.tokens.shape:
            raise ValueError("source and target examples must have the same shape")
        self.patches.append((name, src, stream, layers, positions, metric, method))
        return self

    # --- serialization -----------------------------------------------------------------------

    def spec(self) -> dict:
        """The study as JSON-ready data (format `explorers.study/v0`). Raises if an intervention or a
        reduction is a Python callable instead of `explorers.ops` data, or if a model has no key."""
        from explorers.ops import is_data

        def data(x, what):
            if not is_data(x):
                raise ValueError(f"{what} is a Python callable; use explorers.ops to make the study serializable")
            return None if x is None else x.to_dict()

        keys = [key for _, key, *_ in self._models]
        if any(k is None for k in keys):
            raise ValueError("a model is a loader without a content key; pass Models, checkpoints or a Trajectory")
        layers = lambda ls: ls if ls == "*" else [int(x) for x in ls]  # noqa: E731
        return {
            "format": "explorers.study/v0",
            "models": keys,
            "steps": [f"{s.before.key}->{s.after.key}" for s in self._steps] if any(m.on_steps for m in self.measures) else [],
            "examples": {"fingerprint": self.examples.fingerprint, "shape": list(self.tokens.shape)},
            "reads": [{"name": n, "stream": s, "layers": layers(ls), "position": _position(p),
                       "reduce": data(r, f"read {n!r} reduce")} for n, s, ls, p, r in self.reads],
            "writes": [{"stream": s, "layer": int(layer), "position": _position(p), "op": data(fn, f"write {i}")}
                       for i, (s, layer, p, fn) in enumerate(self.writes)],
            "measures": [_describe(m) for m in self.measures],
            "patches": [{"name": n, "source": src.fingerprint, "stream": s, "layers": layers(ls),
                         "positions": _position(ps), "metric": _describe(m), "method": method}
                        for n, src, s, ls, ps, m, method in self.patches],
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

    def _reads(self, model) -> dict:
        out = {}
        specs = [(name, s, _layers(layers, model.n_layers if s == "residual" else model.n_layers - 1), pos, red)
                 for name, s, layers, pos, red in self.reads]
        flat = [(s, layer, pos, red) for _, s, layers, pos, red in specs for layer in layers]
        parts = {i: [] for i in range(len(flat))}
        for b in self._batches():
            _, values = self._trace(model, self.tokens[b], flat)
            for i, v in enumerate(values):
                parts[i].append(v.value.float().cpu().numpy())
        k = 0
        for name, s, layers, pos, red in specs:
            stacked = np.stack([np.concatenate(parts[k + j]) for j in range(len(layers))], axis=1)
            k += len(layers)
            has_pos = not isinstance(pos, int)
            dims = ("example", "layer") + (("position",) if has_pos else ())
            dims += ("d",) if stacked.ndim == len(dims) + 1 else ()
            coords = {"layer": layers}
            if has_pos:
                coords["position"] = np.arange(self.tokens.shape[1])[pos if pos is not None else slice(None)]
            out[name] = xr.DataArray(stacked, dims=dims, coords=coords)
        return out

    def _compute_model(self, model) -> dict:
        from explorers.execute import serve

        out = self._reads(model) if self.reads else {}
        state_measures = [m for m in self.measures if not m.on_steps]
        if state_measures:
            reads = set().union(*(m.reads for m in state_measures))
            ctx = serve(model, self.examples, reads, self.writes, self.batch_size, self.dim_batch)
            out.update({m.name: m(ctx) for m in state_measures})
        for name, src, stream, layers, positions, metric, method in self.patches:
            out[name] = self._patch(model, src.tokens, stream, layers, positions, metric, method)
        return {k: v.reset_coords(drop=True) for k, v in out.items()}

    def _metric(self, metric, logits):
        from explorers.measures import Context

        positions = [int(r.split(":")[1]) for r in metric.reads]
        return metric.fn(Context(examples=None, logits={p: logits[:, p] for p in positions}))

    def _patch(self, model, src, stream, layers, positions, metric, method) -> xr.DataArray:
        from explorers.ops import Set

        seq = self.tokens.shape[1]
        layers = _layers(layers, model.n_layers if stream == "residual" else model.n_layers - 1)
        positions = list(range(seq)) if positions == "*" else list(positions)
        result = np.full((len(layers), len(positions), len(self.tokens)), np.nan)
        for b in self._batches():
            clean_run, clean_acts = self._trace(model, src[b], [(stream, layer, None, None) for layer in layers])
            m_clean = self._metric(metric, clean_run.logits).float()
            if method == "exact":
                base, _ = self._trace(model, self.tokens[b])
                m_corrupt = self._metric(metric, base.logits).float()
                for li, layer in enumerate(layers):
                    for pi, pos in enumerate(positions):
                        patched, _ = self._trace(model, self.tokens[b], extra_writes=[
                            (stream, layer, pos, Set(clean_acts[li].value[:, pos]))])
                        m = self._metric(metric, patched.logits).float()
                        result[li, pi, b] = ((m - m_corrupt) / (m_clean - m_corrupt)).cpu().numpy()
            else:
                run, acts = self._trace(model, self.tokens[b], [(stream, layer, None, None) for layer in layers], grad=True)
                m_corrupt = self._metric(metric, run.logits)
                m_corrupt.sum().backward()
                denom = m_clean - m_corrupt.detach().float()
                for li in range(len(layers)):
                    diff = clean_acts[li].value.float() - acts[li].value.detach().float()
                    attr = (diff * acts[li].grad.float()).sum(-1)          # (batch, seq)
                    result[li, :, b] = (attr[:, positions] / denom[:, None]).T.cpu().numpy()
        return xr.DataArray(result, dims=("layer", "position", "example"),
                            coords={"layer": layers, "position": positions})

    def _names(self) -> list[str]:
        return ([n for n, *_ in self.reads] + [m.name for m in self.measures if not m.on_steps]
                + [p[0] for p in self.patches])

    def _step_outputs(self, store, study_key) -> dict:
        """{after-step: {name: DataArray}} for step measures of a Trajectory."""
        import hashlib

        from explorers.execute import serve_step

        step_measures = [m for m in self.measures if m.on_steps]
        out = {}
        for step in self._steps if step_measures else []:
            unit = f"step:{step.before.key}->{step.after.key}"
            keys = {m.name: hashlib.sha256(f"{study_key}|{unit}|{m.name}".encode()).hexdigest()[:32] for m in step_measures}
            cached = {n: store.get(k) for n, k in keys.items()} if store is not None else {}
            if cached and all(v is not None for v in cached.values()):
                out[step.step] = cached
                continue
            ctx = serve_step(step, self.examples)
            out[step.step] = {m.name: m(ctx).reset_coords(drop=True) for m in step_measures}
            if store is not None:
                for n, da in out[step.step].items():
                    store.put(keys[n], da, {"study": study_key, "step": unit, "output": n})
        return out

    def compute(self, store=None, verbose: bool = False) -> xr.Dataset:
        """Run the study. With `store` (a folder), each output of each model is cached under
        (study key, model key, output name); models whose outputs are all cached are not loaded."""
        import hashlib
        import time
        from pathlib import Path

        from explorers.store import Store

        study_key = None
        if store is not None:
            store = store if isinstance(store, Store) else Store(Path(store))
            study_key = self.key()                     # raises if the study is not serializable
        steps = self._step_outputs(store, study_key)
        missing = {n: xr.full_like(da, np.nan, dtype=float) for n, da in next(iter(steps.values())).items()} if steps else {}
        per_model, coords = [], []
        for i, (coord, key, load, owned, step) in enumerate(self._models):
            t0 = time.time()
            names = self._names()
            ckeys = {n: hashlib.sha256(f"{study_key}|{key}|{n}".encode()).hexdigest()[:32] for n in names}
            cached = {n: store.get(k) for n, k in ckeys.items()} if store is not None and names else {}
            if not names:
                outputs, how = {}, "nothing to compute"
            elif cached and all(v is not None for v in cached.values()):
                outputs, how = cached, "cached"
            else:
                model = load()
                outputs, how = self._compute_model(model), "computed"
                if store is not None:
                    for n, da in outputs.items():
                        store.put(ckeys[n], da, {"study": study_key, "model": key, "output": n})
                del model
                if owned:
                    _free()
            outputs = {**outputs, **steps.get(step, missing)}              # no step leads into the first state
            per_model.append(xr.Dataset(outputs))
            coords.append(coord)
            if verbose:
                print(f"[{i + 1}/{len(self._models)}] {key or coord}: {how} in {time.time() - t0:.1f}s", flush=True)
        dim = "step" if all(m[4] is not None for m in self._models) else "model"
        ds = xr.concat(per_model, dim=dim, join="outer", coords="minimal", compat="override")
        ds = ds.assign_coords({dim: coords})
        if "example" in ds.dims:
            ds = ds.assign_coords(example=np.arange(len(self.examples)), example_id=("example", self.examples.ids),
                                  **{k: ("example", np.asarray(v)) for k, v in self.examples.meta.items()})
        return ds
