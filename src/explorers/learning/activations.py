"""Reading hidden states. One forward pass per batch serves every requested layer."""

import hashlib
import json
from pathlib import Path
from typing import Callable, Literal, Protocol

import numpy as np

from explorers.learning.checkpoints import Checkpoint
from explorers.learning.datasets import Dataset

Position = Literal["last", "mean"]


class ActivationSource(Protocol):
    def read(self, checkpoint: Checkpoint, dataset: Dataset, layers: list[int]) -> dict[int, np.ndarray]:
        """{layer: (n_texts, d_model) float32}. Layer 0 is the embedding output, layer L the
        residual stream after block L."""
        ...


class Cache:
    """One .npy per (checkpoint, dataset, layer, position). Keys hash everything that changes the values."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, checkpoint: Checkpoint, dataset: Dataset, layer: int, position: str) -> Path:
        key = json.dumps([checkpoint.model, checkpoint.revision, dataset.fingerprint, layer, position])
        return self.root / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".npy")

    def get(self, *key) -> np.ndarray | None:
        p = self.path(*key)
        return np.load(p) if p.exists() else None

    def put(self, value: np.ndarray, *key) -> None:
        p = self.path(*key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, value)
        tmp.replace(p)


def _load_hf(checkpoint: Checkpoint, device: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(checkpoint.model, revision=checkpoint.revision)
    model = AutoModelForCausalLM.from_pretrained(checkpoint.model, revision=checkpoint.revision,
                                                 torch_dtype=getattr(torch, dtype))
    return model.to(device).eval(), tok


class HFSource:
    """Hugging Face transformers backend. `loader(checkpoint) -> (model, tokenizer)` is
    swappable, so tests run on a tiny local model and never download."""

    def __init__(self, device: str = "cpu", dtype: str = "float32", batch_size: int = 16,
                 position: Position = "last", cache: Cache | None = None,
                 loader: Callable | None = None):
        self.device, self.dtype, self.batch_size, self.position = device, dtype, batch_size, position
        self.cache = cache
        self.loader = loader or (lambda ck: _load_hf(ck, device, dtype))

    def read(self, checkpoint, dataset, layers):
        out: dict[int, np.ndarray] = {}
        if self.cache:
            for layer in layers:
                hit = self.cache.get(checkpoint, dataset, layer, self.position)
                if hit is not None:
                    out[layer] = hit
        missing = [layer for layer in layers if layer not in out]
        if missing:
            computed = self._compute(checkpoint, dataset.texts, missing)
            for layer, value in computed.items():
                if self.cache:
                    self.cache.put(value, checkpoint, dataset, layer, self.position)
                out[layer] = value
        return {layer: out[layer] for layer in layers}

    def _compute(self, checkpoint, texts, layers):
        import torch

        model, tok = self.loader(checkpoint)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"       # positions below assume right padding
        chunks: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = tok(texts[i : i + self.batch_size], return_tensors="pt", padding=True).to(self.device)
                hidden = model(**enc, output_hidden_states=True).hidden_states
                mask = enc["attention_mask"]
                for layer in layers:
                    h = hidden[layer].float()
                    if self.position == "last":
                        last = mask.sum(1) - 1
                        v = h[torch.arange(h.shape[0], device=h.device), last]
                    else:
                        m = mask.unsqueeze(-1).to(h.dtype)
                        v = (h * m).sum(1) / m.sum(1)
                    chunks[layer].append(v.cpu().numpy().astype(np.float32))
        return {layer: np.concatenate(chunks[layer]) for layer in layers}
