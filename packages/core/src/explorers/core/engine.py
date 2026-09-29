"""The engine: measure observables over a trajectory (`over`) or a set of runs (`across`).

For each state it serves every observable from one load and at most one batched forward pass,
and skips anything already in the store. Results come back as one xarray Dataset with a `step`
dimension (and a `run` dimension from `across`), plus `example_id` and the examples' metadata
as coordinates on the `example` dimension.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from explorers.core.data import Examples
from explorers.core.observe import Context, Observable
from explorers.core.state import Step, Trajectory
from explorers.core.store import Store, result_key


def _forward(model, examples: Examples, layers: list[int], batch_size: int, device: str):
    """Next-token loss for every (example, position), and hidden states for `layers`."""
    import torch

    n, seq = examples.tokens.shape
    loss = np.full((n, seq), np.nan, dtype=np.float32)
    hidden = {layer: [] for layer in layers}
    model = model.to(device)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            ids = torch.as_tensor(examples.tokens[i : i + batch_size], device=device)
            out = model(input_ids=ids, output_hidden_states=bool(layers))
            logp = torch.log_softmax(out.logits[:, :-1].float(), dim=-1)
            nll = -logp.gather(-1, ids[:, 1:, None]).squeeze(-1)
            loss[i : i + len(ids), 1:] = nll.cpu().numpy()
            for layer in layers:
                hidden[layer].append(out.hidden_states[layer].float().cpu().numpy())
    if examples.loss_mask is not None:
        loss[~examples.loss_mask] = np.nan
    return loss, {layer: np.concatenate(chunks) for layer, chunks in hidden.items()}


def _weights(model) -> dict:
    return {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}


def _label_examples(da: xr.DataArray, examples: Examples) -> xr.DataArray:
    if "example" in da.dims:
        da = da.assign_coords(example_id=("example", examples.ids),
                              **{k: ("example", np.asarray(v)) for k, v in examples.meta.items()})
    return da


def over(trajectory: Trajectory, observables: list[Observable], examples: Examples,
         store: Store | Path | None = None, batch_size: int = 16, device: str = "cpu") -> xr.Dataset:
    if store is not None and not isinstance(store, Store):
        store = Store(store)
    state_obs = [o for o in observables if not o.on_steps]
    step_obs = [o for o in observables if o.on_steps]
    columns: dict[str, list[xr.DataArray]] = {o.name: [] for o in observables}
    steps_seen: dict[str, list[int]] = {o.name: [] for o in observables}

    def lookup(obs, unit_key):
        return store.get(result_key(unit_key, obs, examples.fingerprint)) if store else None

    def record(obs, unit_key, unit_step, da, compute, provenance):
        if da is None:
            da = compute()
            if store:
                store.put(result_key(unit_key, obs, examples.fingerprint), da, provenance)
        columns[obs.name].append(_label_examples(da, examples))
        steps_seen[obs.name].append(unit_step)

    for state in trajectory.states:
        cached = {o.name: lookup(o, state.key) for o in state_obs}
        todo = [o for o in state_obs if cached[o.name] is None]
        ctx = Context(state=state, examples=examples)
        if todo:
            reads = set().union(*(o.reads for o in todo))
            model = state.load()
            if "weights" in reads:
                ctx.weights = _weights(model)
            layers = sorted(int(r.split(":")[1]) for r in reads if r.startswith("hidden:"))
            if "token_loss" in reads or layers:
                ctx.token_loss, ctx.hidden = _forward(model, examples, layers, batch_size, device)
            del model
        for o in state_obs:
            prov = {"run": state.run, "step": state.step, "state": state.key, "observable": o.name,
                    "version": o.version, "examples": examples.fingerprint}
            record(o, state.key, state.step, cached[o.name], lambda o=o: o(ctx), prov)

    for step in trajectory.steps:
        ctx = Context(state=step.after, examples=examples, step=step)
        loaded = False
        for o in step_obs:
            def compute(o=o):
                nonlocal loaded
                if not loaded:
                    ctx.before = _weights(step.before.load())
                    ctx.after = _weights(step.after.load())
                    ctx.grad = {k: v.detach().float().cpu().numpy() for k, v in step.grad.items()}
                    loaded = True
                return o(ctx)
            prov = {"run": step.run, "step": step.step, "state": step.after.key, "observable": o.name,
                    "version": o.version, "examples": examples.fingerprint}
            unit = f"step:{step.before.key}->{step.after.key}"
            record(o, unit, step.step, lookup(o, unit), compute, prov)

    arrays = {}
    for name, parts in columns.items():
        if parts:
            arrays[name] = xr.concat(parts, dim=xr.DataArray(steps_seen[name], dims="step", name="step"),
                                     coords="minimal", compat="override")
    ds = xr.Dataset(arrays)
    ds.attrs.update({"run": trajectory.run, "examples": examples.name,
                     "examples_fingerprint": examples.fingerprint})
    return ds.assign_coords(run=trajectory.run, **{k: v for k, v in trajectory.coords.items()})


def across(trajectories: list[Trajectory], observables: list[Observable], examples: Examples,
           **kwargs) -> xr.Dataset:
    """`over` for each run, stacked along a `run` dimension with each run's coords."""
    parts = [over(t, observables, examples, **kwargs) for t in trajectories]
    coord_names = sorted({k for t in trajectories for k in t.coords})
    parts = [p.drop_vars([c for c in coord_names if c in p.coords]) for p in parts]
    ds = xr.concat(parts, dim="run", coords="minimal", compat="override", join="outer")
    ds = ds.assign_coords(run=[t.run for t in trajectories])
    for c in coord_names:
        ds = ds.assign_coords({c: ("run", [t.coords.get(c, np.nan) for t in trajectories])})
    return ds
