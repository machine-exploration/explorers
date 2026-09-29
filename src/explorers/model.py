"""Models: a network and where its streams live.

`open` wraps a Hugging Face causal language model (or any compatible module) in a `Model`. A model
knows where its blocks, attention, MLP, final norm, embedding and unembedding are, so streams can
be named the same way whatever the architecture:

  residual[L]   the residual stream entering block L; L = n_layers is the residual after the last
                block, before the final norm (what the unembedding reads, through the norm)
  attn_out[L]   the output of the attention sublayer of block L
  mlp_out[L]    the output of the MLP sublayer of block L

torch and transformers are imported only when a model is opened.
"""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    blocks: str
    attn: str
    mlp: str
    norm: str
    embed: str


#: Known layouts, tried in order. Llama covers Qwen, Mistral, OLMo and most recent decoders.
LAYOUTS = {
    "gpt_neox": Layout("gpt_neox.layers", "attention", "mlp", "gpt_neox.final_layer_norm", "gpt_neox.embed_in"),
    "llama": Layout("model.layers", "self_attn", "mlp", "model.norm", "model.embed_tokens"),
    "gpt2": Layout("transformer.h", "attn", "mlp", "transformer.ln_f", "transformer.wte"),
}

STREAMS = ("residual", "attn_out", "mlp_out")


def _get(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def find_layout(module) -> Layout:
    for layout in LAYOUTS.values():
        if all(_get(module, p) is not None for p in (layout.blocks, layout.norm, layout.embed)):
            return layout
    raise ValueError(f"unknown architecture {type(module).__name__}; known layouts: {', '.join(LAYOUTS)}")


class Model:
    """A network with named streams. Create it with `open`."""

    def __init__(self, module, tokenizer=None, name: str | None = None, revision: str | None = None,
                 device: str = "cpu", layout: Layout | None = None):
        self.module = module.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.layout = layout or find_layout(module)
        self.blocks = list(_get(module, self.layout.blocks))
        self.norm = _get(module, self.layout.norm)
        self.embed = _get(module, self.layout.embed)
        self.name = name or type(module).__name__
        self.revision = revision
        self._key = None

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    @property
    def d_model(self) -> int:
        return self.embed.weight.shape[1]

    @property
    def key(self) -> str:
        """Content key: `name@revision` for published checkpoints, else a hash of the parameters."""
        if self._key is None:
            if self.revision is not None:
                self._key = f"{self.name}@{self.revision}"
            else:
                h = hashlib.sha256()
                for k, v in sorted(self.module.state_dict().items()):
                    h.update(k.encode())
                    h.update(v.detach().float().cpu().numpy().tobytes())
                self._key = f"params:{h.hexdigest()[:24]}"
        return self._key

    def site(self, stream: str, layer: int):
        """(module, kind) where `stream[layer]` is seen: 'pre' hooks see a block's or the final
        norm's input, 'out' hooks see a sublayer's output."""
        if stream not in STREAMS:
            raise ValueError(f"unknown stream {stream!r}; streams: {', '.join(STREAMS)}")
        top = self.n_layers if stream == "residual" else self.n_layers - 1
        if not 0 <= layer <= top:
            raise ValueError(f"{stream}[{layer}] out of range: layers 0..{top}")
        if stream == "residual":
            return (self.norm if layer == self.n_layers else self.blocks[layer]), "pre"
        part = self.layout.attn if stream == "attn_out" else self.layout.mlp
        return getattr(self.blocks[layer], part), "out"

    def tokens(self, inputs):
        """Token ids as a (batch, seq) LongTensor on the model's device. Accepts a string, a list of
        strings of equal token length, or an int array of shape (seq,) or (batch, seq)."""
        import numpy as np
        import torch

        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(inputs, (list, tuple)) and inputs and isinstance(inputs[0], str):
            if self.tokenizer is None:
                raise ValueError("this model has no tokenizer; pass token ids")
            rows = [self.tokenizer(t)["input_ids"] for t in inputs]
            if len({len(r) for r in rows}) != 1:
                raise ValueError("texts have different token lengths; pass token ids of equal length")
            inputs = rows
        ids = torch.as_tensor(np.asarray(inputs), dtype=torch.long)
        if ids.ndim == 1:
            ids = ids[None]
        return ids.to(self.device)

    def trace(self, inputs, grad: bool = False):
        """Declare reads and writes on this model's streams; they run in one forward pass when the
        `with` block closes. See `explorers.trace.Trace`."""
        from explorers.trace import Trace

        return Trace(self, self.tokens(inputs), grad=grad)

    def unembed(self, residual):
        """The model's own final norm and unembedding applied to a final residual."""
        return self.module.get_output_embeddings()(self.norm(residual))

    def __repr__(self) -> str:
        return f"Model({self.key}, layers={self.n_layers}, d_model={self.d_model})"


def open(name_or_module, revision: str | None = None, device: str = "cpu", dtype=None,
         tokenizer=True) -> Model:
    """Open a model: a Hugging Face repo id (with an optional revision, e.g. a Pythia `step1000`),
    or an already-built module."""
    if not isinstance(name_or_module, str):
        return Model(name_or_module, tokenizer=None if tokenizer is True else tokenizer, device=device)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    module = AutoModelForCausalLM.from_pretrained(name_or_module, revision=revision, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(name_or_module, revision=revision) if tokenizer is True else tokenizer
    return Model(module, tokenizer=tok, name=name_or_module, revision=revision, device=device)
