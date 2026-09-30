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

import builtins
import functools
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    blocks: str
    attn: str          # the attention sublayer; "a|b" when blocks differ (hybrid models): the first present
    mlp: str
    norm: str
    embed: str
    norm_offset: float = 0.0     # the final norm's gain is norm_offset + weight (1 for Qwen 3.5: x̂ (1 + w))
    model_type: str | None = None    # match only a model whose config has this model_type


_QWEN3_5 = dict(attn="self_attn|linear_attn", mlp="mlp", norm_offset=1.0)

#: Known layouts, tried in order. Llama covers Qwen up to 3, Mistral, OLMo and most recent decoders.
#: Qwen 3.5 and later (3.6, 3.8) are hybrid (linear and full attention blocks) and their norms scale
#: by 1 + weight; the multimodal classes hold the text model under `model.language_model`.
LAYOUTS = {
    "gpt_neox": Layout("gpt_neox.layers", "attention", "mlp", "gpt_neox.final_layer_norm", "gpt_neox.embed_in"),
    "qwen3_5": Layout("model.language_model.layers", norm="model.language_model.norm",
                      embed="model.language_model.embed_tokens", model_type="qwen3_5", **_QWEN3_5),
    "qwen3_5_text": Layout("model.layers", norm="model.norm", embed="model.embed_tokens",
                           model_type="qwen3_5_text", **_QWEN3_5),
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
    model_type = getattr(getattr(module, "config", None), "model_type", None)
    for layout in LAYOUTS.values():
        if layout.model_type is not None and layout.model_type != model_type:
            continue
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
        parts = (self.layout.attn if stream == "attn_out" else self.layout.mlp).split("|")
        block = self.blocks[layer]
        return next(getattr(block, p) for p in parts if hasattr(block, p)), "out"

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


def _dtype(dtype):
    if dtype is None or not isinstance(dtype, str):
        return dtype
    import torch

    return getattr(torch, dtype)


def open(name_or_module, revision: str | None = None, device: str = "cpu", dtype=None,
         tokenizer=True, adapter: str | None = None) -> Model:
    """Open a model: a Hugging Face repo id (with an optional revision, e.g. a Pythia `step1000`),
    or an already-built module. `dtype` is a torch dtype or its name ("float16", "bfloat16").
    `adapter` is a LoRA adapter directory, merged into the weights (docs/prime.md)."""
    dtype = _dtype(dtype)
    if not isinstance(name_or_module, str):
        module = name_or_module if dtype is None else name_or_module.to(dtype)
        return Model(module, tokenizer=None if tokenizer is True else tokenizer, device=device)
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = {"dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype": dtype}
    module = AutoModelForCausalLM.from_pretrained(name_or_module, revision=revision, **kw)
    if adapter is not None:
        merge_lora(module, adapter)
    tok = AutoTokenizer.from_pretrained(name_or_module, revision=revision) if tokenizer is True else tokenizer
    return Model(module, tokenizer=tok, name=name_or_module, revision=revision, device=device)


@dataclass(frozen=True)
class ModelRef:
    """A model that is not loaded yet: a repo id, a revision, and optionally a LoRA adapter. Its key
    is known without downloading the model, so studies over checkpoints can be hashed and cached
    before they run."""
    name: str
    revision: str | None = None
    step: int | None = None
    device: str = "cpu"
    dtype: object = None
    adapter: str | None = None

    @property
    def key(self) -> str:
        base = f"{self.name}@{self.revision}"
        return base if self.adapter is None else f"{base}+{_file_hash(Path(self.adapter) / ADAPTER_WEIGHTS)}"

    def load(self) -> Model:
        return open(self.name, revision=self.revision, device=self.device, dtype=self.dtype, adapter=self.adapter)


def checkpoint(name: str, revision: str | None = None, device: str = "cpu", dtype=None) -> ModelRef:
    return ModelRef(name, revision, device=device, dtype=dtype)


def checkpoints(name: str, steps, device: str = "cpu", dtype=None, revision_format: str = "step{}") -> list[ModelRef]:
    """Lazy handles for the checkpoints of a suite, e.g. Pythia:
    `ex.checkpoints("EleutherAI/pythia-70m", steps=[0, 1000, 143000])`. Revisions default to
    `step<N>`; the study coordinate is the step."""
    return [ModelRef(name, revision_format.format(s), step=int(s), device=device, dtype=dtype) for s in steps]


# --- LoRA adapters of a prime-rl run (docs/prime.md) -----------------------------------------

ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"


def _file_hash(path: Path) -> str:
    st = path.stat()
    return _hash_contents(str(path), st.st_mtime_ns, st.st_size)


@functools.lru_cache(maxsize=None)
def _hash_contents(path: str, mtime_ns: int, size: int) -> str:
    h = hashlib.sha256()
    with builtins.open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def merge_lora(module, adapter) -> None:
    """Merge a LoRA adapter directory into `module` in place: `W <- W + (alpha / r) B A` for every
    adapted linear layer, computed in float32. Every adapter tensor must land on a linear layer of
    `module`; anything else raises, so a partly applied adapter cannot pass silently."""
    import torch
    from safetensors.torch import load_file

    adapter = Path(adapter)
    config = json.loads((adapter / ADAPTER_CONFIG).read_text())
    if config.get("modules_to_save"):
        raise NotImplementedError(f"adapter trains whole modules too: {config['modules_to_save']}")
    scale = config["lora_alpha"] / config["r"]
    tensors = load_file(adapter / ADAPTER_WEIGHTS)
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, t in tensors.items():
        name, _, part = key.removeprefix("base_model.model.").rpartition(".lora_")
        if part not in ("A.weight", "B.weight"):
            raise ValueError(f"not a LoRA tensor: {key}")
        pairs.setdefault(name, {})[part[0]] = t
    with torch.no_grad():
        for name, ab in pairs.items():
            layer = module.get_submodule(name)
            if not isinstance(layer, torch.nn.Linear) or set(ab) != {"A", "B"}:
                raise NotImplementedError(f"cannot merge the adapter of {name} ({type(layer).__name__})")
            w = layer.weight
            delta = scale * (ab["B"].to(w.device, torch.float32) @ ab["A"].to(w.device, torch.float32))
            w.copy_((w.float() + delta).to(w.dtype))


def adapters(name: str, path, revision: str | None = None, device: str = "cpu", dtype=None) -> list[ModelRef]:
    """One lazy handle per `step_<N>/` adapter directory under `path` (an archive written by
    `archive_adapters`), on base model `name`. The study coordinate is the step."""
    steps = sorted((int(d.name.removeprefix("step_")), d) for d in Path(path).glob("step_*")
                   if (d / ADAPTER_WEIGHTS).exists())
    return [ModelRef(name, revision, step=n, device=device, dtype=dtype, adapter=str(d)) for n, d in steps]


def archive_adapters(run_dir, dest) -> list[int]:
    """Copy each finished LoRA broadcast of a prime-rl run (`<run_dir>/broadcasts/step_<N>/`, done
    when `.finished` exists) to `<dest>/step_<N>/`, unless already there. prime-rl keeps only the
    last two broadcasts, so run this beside the trainer. Returns the steps copied."""
    dest = Path(dest)
    copied = []
    for d in sorted(Path(run_dir, "broadcasts").glob("step_*")):
        if not (d / ".finished").exists() or not (d / ADAPTER_WEIGHTS).exists() or (dest / d.name).exists():
            continue
        tmp = dest / f".{d.name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for f in (ADAPTER_WEIGHTS, ADAPTER_CONFIG):
            shutil.copy2(d / f, tmp / f)
        tmp.rename(dest / d.name)
        copied.append(int(d.name.removeprefix("step_")))
    return copied


# --- checkpoint schedules (docs/pythia.md) ---------------------------------------------------

def pythia_steps() -> list[int]:
    """The 154 Pythia checkpoints: step 0, powers of two up to 512, then every 1000 up to 143000."""
    return [0] + [2 ** i for i in range(10)] + list(range(1000, 143001, 1000))


def pick(steps: list[int], n: int) -> list[int]:
    """`n` steps spread evenly in log-step, always keeping the first and the last.
    Early training changes fastest, so a log spread spends checkpoints where things move."""
    steps = sorted(set(steps))
    if n >= len(steps):
        return steps
    if n < 2:
        raise ValueError("pick at least 2 steps")
    chosen = {steps[0], steps[-1]}
    positive = [s for s in steps if s > 0]
    lo, hi = positive[0], positive[-1]
    k = n - len(chosen)
    for i in range(1, k + 1):
        target = lo * (hi / lo) ** (i / (k + 1))
        free = [s for s in steps if s not in chosen]
        chosen.add(min(free, key=lambda s: abs(s - target)))
    return sorted(chosen)
