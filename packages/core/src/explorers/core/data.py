"""Examples: the data every measurement is indexed by.

An example is a fixed-length token sequence. Its id is a hash of its tokens, so the same example
has the same id in every run, on every machine, and results about it can be joined and merged.
Per-example metadata (a task, a frequency, a source) travels with the examples as columns.
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, eq=False)
class Examples:
    tokens: np.ndarray                                  # (n, seq_len) int64
    loss_mask: np.ndarray | None = None                 # (n, seq_len) bool: positions whose loss counts
    meta: dict[str, np.ndarray] = field(default_factory=dict)   # per-example columns, length n
    name: str = "examples"

    def __post_init__(self):
        if self.tokens.ndim != 2:
            raise ValueError("tokens must be (n_examples, seq_len)")
        if self.loss_mask is not None and self.loss_mask.shape != self.tokens.shape:
            raise ValueError("loss_mask must have the same shape as tokens")
        for key, col in self.meta.items():
            if len(col) != len(self.tokens):
                raise ValueError(f"meta column {key!r} has {len(col)} rows for {len(self.tokens)} examples")

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def ids(self) -> np.ndarray:
        """One content id per example: identical token rows get identical ids."""
        return np.array([hashlib.sha256(row.astype(np.int64).tobytes()).hexdigest()[:12] for row in self.tokens])

    @property
    def fingerprint(self) -> str:
        """Content hash of the whole set: tokens, mask and metadata."""
        h = hashlib.sha256(self.tokens.astype(np.int64).tobytes())
        if self.loss_mask is not None:
            h.update(self.loss_mask.astype(np.bool_).tobytes())
        for key in sorted(self.meta):
            h.update(key.encode())
            h.update(np.asarray(self.meta[key]).astype(str).astype(bytes).tobytes())
        return h.hexdigest()[:16]

    def with_meta(self, **columns) -> "Examples":
        return Examples(self.tokens, self.loss_mask, {**self.meta, **columns}, self.name)

    @classmethod
    def from_texts(cls, texts: list[str], tokenizer, seq_len: int, name: str = "texts",
                   max_examples: int | None = None) -> "Examples":
        """Tokenise, concatenate with EOS between documents, and cut into windows of `seq_len`.
        Every window position after the first counts towards the loss."""
        ids: list[int] = []
        eos = tokenizer.eos_token_id
        for t in texts:
            ids.extend(tokenizer(t)["input_ids"])
            if eos is not None:
                ids.append(eos)
        n = len(ids) // seq_len
        if max_examples is not None:
            n = min(n, max_examples)
        if n == 0:
            raise ValueError(f"not enough tokens for one window of {seq_len}")
        tokens = np.array(ids[: n * seq_len], dtype=np.int64).reshape(n, seq_len)
        return cls(tokens=tokens, name=name)
