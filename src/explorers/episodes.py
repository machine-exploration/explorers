"""Episodes: replay agent rollouts as examples (docs/episodes.md).

Adapter for verifiers episode records, the JSON that prime-rl's file monitor writes for every eval
and RL episode (`monitors/file/traces/stream/`). It reads plain dicts: neither verifiers nor
prime-rl is imported.

A record is one episode; it holds traces; a trace is a graph of message nodes. Each node stores the
tokens it adds to the sequence the model saw (`token_ids`), which of them the model sampled
(`mask`) and their sampling log-probabilities (`logprobs`, one per sampled token). A path from a
root node to a leaf is one branch: concatenating its nodes gives the exact `prompt + completion`
tokens. Each branch becomes one example; its loss mask is the sampled tokens, so monitors score the
model's own turns.
"""

import json
from pathlib import Path

import numpy as np

from explorers.data import Examples


def branches(trace: dict) -> list[list[int]]:
    """Root-to-leaf paths of a trace's message graph, as node indices in order."""
    nodes = trace.get("nodes", [])
    parent = [n.get("parent") for n in nodes]
    has_child = {p for p in parent if p is not None}
    paths = []
    for leaf in range(len(nodes)):
        if leaf in has_child:
            continue
        path, node = [], leaf
        while node is not None:
            path.append(node)
            node = parent[node]
        paths.append(path[::-1])
    return paths


def reward(trace: dict) -> float:
    """Weighted sum of the trace's rewards (score × weight); NaN when it has none."""
    values = [r["score"] * r.get("weight", 1.0) for r in trace.get("rewards", {}).values() if r is not None]
    return float(sum(values)) if values else float("nan")


def replay(records, pad_id: int = 0, label=None, max_len: int | None = None):
    """(examples, logprobs) from episode records: one example per branch of every trace.

    `examples.tokens` are the branch's tokens, right-padded with `pad_id` (a causal model's earlier
    positions do not see padding); `examples.loss_mask` marks the tokens the model sampled.
    Metadata columns: `episode`, `trace`, `branch`, `env`, `reward`, and `label` when
    `label(record, trace)` is given. `logprobs` is (example, position): the recorded sampling
    log-probability at sampled positions, NaN elsewhere. A branch with no token ids (an eval that
    relayed to an API returning text only) raises: it would need re-rendering, which is not exact.
    Branches longer than `max_len` raise too, rather than being cut silently.
    """
    rows, masks, logps, meta = [], [], [], {k: [] for k in ("episode", "trace", "branch", "env", "reward")}
    labels = []
    for record in records:
        for t_index, trace in enumerate(record.get("traces", [])):
            nodes = trace.get("nodes", [])
            for b_index, path in enumerate(branches(trace)):
                ids, mask, lp = [], [], []
                for i in path:
                    node = nodes[i]
                    tok, m = node.get("token_ids", []), node.get("mask") or [False] * len(node.get("token_ids", []))
                    if len(m) != len(tok):
                        raise ValueError(f"episode {record.get('id')}: node {i} has {len(tok)} tokens and {len(m)} mask entries")
                    node_lp = iter(node.get("logprobs", []))
                    ids += tok
                    mask += m
                    lp += [next(node_lp, np.nan) if s else np.nan for s in m]
                if not ids:
                    raise ValueError(f"episode {record.get('id')}, trace {t_index}: branch {b_index} has no token ids "
                                     "(a text-only eval relay); it cannot be replayed exactly")
                if max_len is not None and len(ids) > max_len:
                    raise ValueError(f"episode {record.get('id')}: branch of {len(ids)} tokens exceeds max_len={max_len}")
                rows.append(ids)
                masks.append(mask)
                logps.append(lp)
                meta["episode"].append(record.get("id", ""))
                meta["trace"].append(trace.get("id", str(t_index)))
                meta["branch"].append(b_index)
                meta["env"].append((record.get("env") or {}).get("id", ""))
                meta["reward"].append(reward(trace))
                if label is not None:
                    labels.append(label(record, trace))
    if not rows:
        raise ValueError("no branches with tokens in these records")
    seq = max(len(r) for r in rows)
    tokens = np.full((len(rows), seq), pad_id, dtype=np.int64)
    loss_mask = np.zeros((len(rows), seq), dtype=bool)
    logprobs = np.full((len(rows), seq), np.nan)
    for i, (r, m, lp) in enumerate(zip(rows, masks, logps)):
        tokens[i, :len(r)], loss_mask[i, :len(r)], logprobs[i, :len(r)] = r, m, lp
    columns = {k: np.asarray(v) for k, v in meta.items()}
    if label is not None:
        columns["label"] = np.asarray(labels)
    return Examples(tokens=tokens, loss_mask=loss_mask, meta=columns, name="episodes"), logprobs


def read_stream(path):
    """Episode records from a chunked JSONL stream (prime-rl's `monitors/file/traces/stream`): plain
    `NNNNN.jsonl` chunks, and sealed `NNNNN.jsonl.zst` ones when `zstandard` is installed."""
    directory = Path(path)
    chunks = {}
    for p in directory.glob("*.jsonl*"):
        number = int(p.name.split(".", 1)[0])
        if p.suffix == ".jsonl" or number not in chunks:
            chunks[number] = p
    for number in sorted(chunks):
        p = chunks[number]
        if p.suffix == ".zst":
            try:
                import zstandard
            except ImportError as e:
                raise ImportError(f"{p.name} is zstd-compressed: install zstandard, or decompress it first") from e
            with open(p, "rb") as f:
                text = zstandard.ZstdDecompressor().stream_reader(f).read().decode()
        else:
            text = p.read_text()
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)
