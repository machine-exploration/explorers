"""Serve the reads of measures from one model: one traced forward pass per batch, with the study's
writes applied, plus one backward sweep when a Jacobian is asked for.

This is the only place that runs a model for measures. `Study` calls `serve` once per model and
`serve_step` once per training step.
"""

import numpy as np
import xarray as xr

from explorers.measures import Context, lens_positions, split_rows

STREAMS = ("residual", "attn_out", "mlp_out")
TARGETS = ("final", "penultimate")      # what a Jacobian differentiates: residual[n] or residual[n-1]


def parse_reads(reads: set[str], n_layers: int):
    """(streams {(name, layer)}, logit positions {p}, jacobians {(layer, skip, target)})."""
    streams, positions, jac = set(), set(), set()
    for r in reads:
        head, _, rest = r.partition(":")
        if head in STREAMS:
            streams.add((head, n_layers if rest == "final" else int(rest)))
        elif head == "logits":
            positions.add(int(rest))
        elif head == "jacobian":
            layer, skip, *target = rest.split(":")
            target = target[0] if target else "final"
            if target not in TARGETS:
                raise ValueError(f"unknown Jacobian target {target!r}; one of {TARGETS}")
            jac.add((int(layer), int(skip), target))
        elif r not in ("weights", "token_loss", "unembed", "step"):
            raise ValueError(f"unknown read {r!r}")
    return streams, positions, jac


def weights(module) -> dict:
    return {k: v.detach().float().cpu().numpy() for k, v in module.state_dict().items()}


def serve(model, examples, reads: set[str], writes=(), batch_size: int = 16, dim_batch: int = 1,
          state=None) -> Context:
    """A Context with every declared read, from `model` (an `explorers.Model`) on `examples`."""
    import torch

    streams, positions, jac = parse_reads(reads, model.n_layers)
    ctx = Context(examples=examples, state=state, n_layers=model.n_layers)
    if "weights" in reads:
        ctx.weights = weights(model.module)
    if streams or positions or "token_loss" in reads:
        tokens = examples.tokens
        loss_chunks, logit_chunks = [], {p: [] for p in positions}
        stream_chunks = {s: [] for s in streams}
        for i in range(0, len(tokens), batch_size):
            run = model.trace(tokens[i:i + batch_size])
            for stream, layer, position, fn in writes:
                run.stream(stream).write(layer, position, fn=fn)
            values = {s: run.stream(s[0]).read(s[1]) for s in streams}
            run.run()
            for s, v in values.items():
                stream_chunks[s].append(v.value.float().cpu().numpy())
            for p in positions:
                logit_chunks[p].append(run.logits[:, p].float().cpu().numpy())
            if "token_loss" in reads:
                ids = run.ids
                logp = torch.log_softmax(run.logits[:, :-1].float(), dim=-1)
                nll = -logp.gather(-1, ids[:, 1:, None]).squeeze(-1)
                chunk = np.full(ids.shape, np.nan, dtype=np.float32)
                chunk[:, 1:] = nll.cpu().numpy()
                loss_chunks.append(chunk)
        ctx.streams = {s: np.concatenate(c) for s, c in stream_chunks.items()}
        ctx.logits = {p: np.concatenate(c) for p, c in logit_chunks.items()}
        if "token_loss" in reads:
            ctx.token_loss = np.concatenate(loss_chunks)
            if examples.loss_mask is not None:
                ctx.token_loss[~examples.loss_mask] = np.nan
    if jac:
        if writes:
            raise ValueError("Jacobians are fitted on the unmodified model; they cannot be combined with writes")
        ctx.jacobian = jacobians(model, examples, jac, batch_size, dim_batch)
    if "unembed" in reads:
        ctx.unembed_apply = unembed_apply(model)
        ctx.unembed_topk = lambda h, k: ctx.unembed_apply(h, lambda z: z.topk(k, dim=-1).indices)
        ctx.unembed_matrix = unembed_matrix(model)
    return ctx


def serve_step(step, examples) -> Context:
    """A Context for a training step: weights before and after, and the gradient."""
    ctx = Context(examples=examples, state=step.after, step=step)
    ctx.before, ctx.after = weights(step.before.load()), weights(step.after.load())
    ctx.grad = {k: v.detach().float().cpu().numpy() for k, v in step.grad.items()}
    return ctx


def label(da: xr.DataArray, examples) -> xr.DataArray:
    if "example" in da.dims:
        da = da.assign_coords(example_id=("example", examples.ids),
                              **{k: ("example", np.asarray(v)) for k, v in examples.meta.items()})
    return da


# --- the Jacobian lens -----------------------------------------------------------------------------

def jacobians(model, examples, requests: set[tuple[int, int, str]], batch_size: int, dim_batch: int = 1) -> dict:
    """J_L for each requested (layer, skip_first, target): the average Jacobian of the target
    residual (`final`: residual[n], before the final norm; `penultimate`: residual[n-1], entering the
    last block) with respect to residual:L.

    For each output dimension k, a one-hot cotangent at k is set at every lens position of every
    example at once, and one backward pass gives row k at every source position. By causality, the
    row at source position p is sum_{t >= p} d target[t, k] / d residual_L[p]. Rows are averaged over
    source positions and examples. Fitted on the rows where `split == "fit"`, if that column exists.
    `dim_batch` > 1 stacks that many copies of each batch, so one backward pass gives that many rows
    (same result, fewer passes, `dim_batch` times the activation memory). The graph is rooted at the
    embedding output, so frozen models work.
    """
    import torch

    n_layers = model.n_layers
    depth = {"final": n_layers, "penultimate": n_layers - 1}
    for layer, _, target in requests:
        if not 0 <= layer <= depth[target]:
            raise ValueError(f"no Jacobian of {target} (residual[{depth[target]}]) with respect to residual[{layer}]")
    fit, _ = split_rows(examples)
    tokens = examples.tokens[fit]
    n, seq = tokens.shape
    layers = sorted({layer for layer, _, _ in requests})
    d = model.d_model
    sums = {r: torch.zeros(d, d, dtype=torch.float64) for r in requests}
    count = {skip: 0 for _, skip, _ in requests}
    captured = {}
    hooks = [model.norm.register_forward_pre_hook(lambda _m, args: captured.__setitem__("h", args[0])),
             model.embed.register_forward_hook(lambda _m, _i, out: out.requires_grad_(True))]
    try:
        for i in range(0, n, batch_size):
            ids = torch.as_tensor(tokens[i:i + batch_size], device=model.device)
            b = len(ids)
            with torch.enable_grad():
                out = model.module(input_ids=ids.repeat(dim_batch, 1), output_hidden_states=True)
                outputs = {"final": captured["h"], "penultimate": out.hidden_states[n_layers - 1]}
                sources = [out.hidden_states[layer] for layer in layers]
                skips = sorted(count)
                chunks = [(target, skip, k0) for target in sorted({t for _, _, t in requests})
                          for skip in skips for k0 in range(0, d, dim_batch)
                          if any((l, skip, target) in sums for l in layers)]
                cotangent = torch.zeros_like(sources[0])
                for c_idx, (target, skip, k0) in enumerate(chunks):
                    pos = torch.as_tensor(lens_positions(seq, skip), device=model.device)
                    rows = min(dim_batch, d - k0)
                    cotangent.zero_()
                    for j in range(rows):                                # copy j carries output dim k0 + j
                        cotangent[j * b:(j + 1) * b, pos, k0 + j] = 1.0
                    wanted = [l for l in layers if (l, skip, target) in sums]
                    grads = torch.autograd.grad(outputs[target], [sources[layers.index(l)] for l in wanted],
                                                cotangent, retain_graph=c_idx < len(chunks) - 1)
                    for layer, g in zip(wanted, grads):
                        g = g[:, pos].double()
                        for j in range(rows):
                            sums[(layer, skip, target)][k0 + j] += g[j * b:(j + 1) * b].sum(dim=(0, 1)).cpu()
                for skip in skips:
                    count[skip] += b * len(lens_positions(seq, skip))
    finally:
        for h in hooks:
            h.remove()
    return {r: (sums[r] / count[r[1]]).float().numpy() for r in requests}


def unembed_apply(model, chunk: int = 2048):
    """`apply(h, fn)`: the model's own final norm + unembedding on residuals h (..., d), in chunks on
    the model's device; `fn` maps the logits (rows, vocab) to (rows, ...). Returns numpy."""
    import torch

    head = model.module.get_output_embeddings()

    def apply(h: np.ndarray, fn) -> np.ndarray:
        flat = np.asarray(h, dtype=np.float32).reshape(-1, h.shape[-1])
        parts = []
        with torch.no_grad():
            for i in range(0, len(flat), chunk):
                x = torch.as_tensor(flat[i:i + chunk], device=model.device, dtype=head.weight.dtype)
                parts.append(fn(head(model.norm(x)).float()).cpu().numpy())
        out = np.concatenate(parts) if parts else np.empty((0,))
        return out.reshape(*h.shape[:-1], *out.shape[1:])

    return apply


def unembed_matrix(model) -> np.ndarray:
    """(vocab, d): the linear part of final norm + unembedding, W_U diag(gain), with the norm's
    mean-centering folded in for LayerNorm. Row t is the direction that raises token t's logit."""
    import torch

    w = model.module.get_output_embeddings().weight.detach().float().cpu().numpy()
    gain = getattr(model.norm, "weight", None)
    if gain is not None:
        w = w * gain.detach().float().cpu().numpy()[None, :]
    if isinstance(model.norm, torch.nn.LayerNorm):
        w = w - w.mean(axis=1, keepdims=True)                   # W diag(g) (I - 11^T / d)
    return w
