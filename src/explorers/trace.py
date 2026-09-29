"""Traces: declare reads and writes on streams, then execute them in one forward pass.

    with model.trace(tokens) as run:
        resid = run.stream("residual")
        x = resid.read(layer=3, position=-1)                     # a Value, filled on exit
        resid.write(layer=3, position=-1, fn=lambda h: h + d)
    x.value, run.logits

Rules:
- Nothing runs inside the `with` block. On exit, the model runs once with every declared write
  applied and every declared read captured.
- At one site, writes apply in the order they were declared; reads see the value after all writes
  at that site (what the rest of the model sees).
- `position` is an int, a list of ints, a slice, or None (every position).
- `write` takes `fn` (tensor -> tensor of the same shape) or `value` (a tensor broadcastable to the
  selected slice, e.g. (batch, d) for one position: this is how patching is written).
- With `grad=True`, the graph is kept (rooted at the embedding output, so frozen models work):
  compute a metric from `run.logits`, call `.backward()`, then read `value.grad`.
"""

from dataclasses import dataclass, field


def _index(position):
    return slice(None) if position is None else position


@dataclass
class Value:
    """What a read returns. `.value` and `.grad` are available after the trace has run."""
    stream: str
    layer: int
    position: object
    _full: object = field(default=None, repr=False)
    _ran: bool = field(default=False, repr=False)

    @property
    def value(self):
        if not self._ran:
            raise RuntimeError("the trace has not run yet: read .value after the `with` block")
        v = self._full[:, _index(self.position)]
        return v if self._full.requires_grad else v.detach()

    @property
    def grad(self):
        if self._full is None or self._full.grad is None:
            raise RuntimeError("no gradient: trace with grad=True and call .backward() on a metric")
        return self._full.grad[:, _index(self.position)]


class Stream:
    def __init__(self, trace: "Trace", name: str):
        self.trace, self.name = trace, name

    def read(self, layer: int, position=None) -> Value:
        self.trace.model.site(self.name, layer)          # validate now, not at run time
        v = Value(self.name, layer, position)
        self.trace._reads.setdefault((self.name, layer), []).append(v)
        return v

    def write(self, layer: int, position=None, fn=None, value=None) -> None:
        if (fn is None) == (value is None):
            raise ValueError("write needs exactly one of fn= or value=")
        self.trace.model.site(self.name, layer)
        self.trace._writes.setdefault((self.name, layer), []).append((position, fn, value))


class Trace:
    def __init__(self, model, ids, grad: bool = False):
        self.model, self.ids, self.grad = model, ids, grad
        self._reads: dict = {}
        self._writes: dict = {}
        self.logits = None

    def stream(self, name: str) -> Stream:
        return Stream(self, name)

    def __enter__(self) -> "Trace":
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.run()
        return False

    def _apply(self, key, h):
        import torch

        writes = self._writes.get(key, [])
        if writes:
            h = h.clone()
            for position, fn, value in writes:
                idx = _index(position)
                new = fn(h[:, idx]) if fn is not None else torch.as_tensor(value, dtype=h.dtype, device=h.device)
                h[:, idx] = new
        if self.grad and key in self._reads and h.requires_grad:
            h.retain_grad()
        for v in self._reads.get(key, []):
            v._full = h if self.grad else h.detach()
        return h

    def run(self) -> "Trace":
        import torch

        model = self.model
        handles = []
        for key in set(self._reads) | set(self._writes):
            module, kind = model.site(*key)
            if kind == "pre":
                def pre(_m, args, kwargs, key=key):
                    if args:
                        return (self._apply(key, args[0]),) + tuple(args[1:]), kwargs
                    kwargs = dict(kwargs)
                    kwargs["hidden_states"] = self._apply(key, kwargs["hidden_states"])
                    return args, kwargs
                handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
            else:
                def post(_m, _args, out, key=key):
                    if isinstance(out, tuple):
                        return (self._apply(key, out[0]),) + tuple(out[1:])
                    return self._apply(key, out)
                handles.append(module.register_forward_hook(post))
        if self.grad:
            handles.append(model.embed.register_forward_hook(lambda _m, _i, out: out.requires_grad_(True)))
        try:
            with torch.set_grad_enabled(self.grad):
                out = model.module(input_ids=self.ids)
        finally:
            for h in handles:
                h.remove()
        self.logits = out.logits if self.grad else out.logits.detach()
        for values in self._reads.values():
            for v in values:
                v._ran = True
        return self
