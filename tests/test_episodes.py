"""Episode replay (docs/episodes.md): verifiers records become examples whose loss mask is the
model's own tokens, and replaying them reproduces the recorded sampling log-probabilities."""

import json

import numpy as np
import pytest

from explorers import episodes

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import explorers as ex  # noqa: E402
from explorers.execute import serve  # noqa: E402


def node(tokens, sampled=0, parent=None, logprobs=()):
    """A message node: `tokens`, the last `sampled` of them sampled by the model."""
    mask = [False] * (len(tokens) - sampled) + [True] * sampled
    return {"parent": parent, "message": {"role": "assistant" if sampled else "user", "content": ""},
            "token_ids": list(tokens), "mask": mask, "logprobs": list(logprobs)}


def record(traces, rid="ep0", env="impossible"):
    return {"id": rid, "env": {"id": env}, "traces": traces}


def test_branches_follow_parents():
    trace = {"nodes": [node([1, 2]), node([3], 1, parent=0), node([4], 1, parent=1), node([5], 1, parent=1)]}
    assert episodes.branches(trace) == [[0, 1, 2], [0, 1, 3]]


def test_replay_builds_one_example_per_branch():
    trace = {"id": "t0", "rewards": {"pass": {"score": 1.0, "weight": 2.0}},
             "nodes": [node([1, 2, 3]), node([9, 4, 5], 2, parent=0, logprobs=[-0.5, -0.25]),
                       node([6], 1, parent=0, logprobs=[-1.0])]}
    examples, logprobs = episodes.replay([record([trace])], pad_id=0, label=lambda r, t: t["rewards"]["pass"]["score"] > 0)
    np.testing.assert_array_equal(examples.tokens, [[1, 2, 3, 9, 4, 5], [1, 2, 3, 6, 0, 0]])
    np.testing.assert_array_equal(examples.loss_mask, [[0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 0]])
    np.testing.assert_array_equal(logprobs[0, 4:], [-0.5, -0.25])
    assert np.isnan(logprobs[0, :4]).all() and logprobs[1, 3] == -1.0 and np.isnan(logprobs[1, 4:]).all()
    assert list(examples.meta["reward"]) == [2.0, 2.0] and list(examples.meta["label"]) == [True, True]
    assert list(examples.meta["branch"]) == [0, 1] and list(examples.meta["env"]) == ["impossible"] * 2


def test_replay_refuses_what_it_cannot_reproduce():
    text_only = {"nodes": [{"parent": None, "message": {"role": "user", "content": "hi"}}]}
    with pytest.raises(ValueError, match="no token ids"):
        episodes.replay([record([text_only])])
    long = {"nodes": [node(list(range(10)))]}
    with pytest.raises(ValueError, match="exceeds max_len"):
        episodes.replay([record([long])], max_len=8)


def test_replayed_log_probabilities_equal_the_recorded_ones():
    """Sample from a tiny model, record the episode as verifiers would, replay it: the model's
    log-probabilities at the sampled positions are the recorded ones."""
    torch.manual_seed(0)
    cfg = transformers.GPTNeoXConfig(vocab_size=17, hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
                                     intermediate_size=16, max_position_embeddings=32)
    model = ex.open(transformers.GPTNeoXForCausalLM(cfg).eval())
    prompt, scaffold, sampled, lps = [3, 7, 1, 12], [5], [], []
    ids = prompt + scaffold                     # the assistant node opens with scaffold it did not sample
    with torch.no_grad():
        for _ in range(5):
            logp = torch.log_softmax(model.module(input_ids=torch.tensor([ids])).logits[0, -1].float(), -1)
            tok = int(torch.multinomial(logp.exp(), 1))
            sampled.append(tok)
            lps.append(float(logp[tok]))
            ids.append(tok)
    trace = {"nodes": [node(prompt), node(scaffold + sampled, len(sampled), parent=0, logprobs=lps)]}
    examples, recorded = episodes.replay([record([trace])])
    ctx = serve(model, examples, {"token_loss"}, batch_size=1)
    at = examples.loss_mask
    np.testing.assert_allclose(-ctx.token_loss[at], recorded[at], rtol=1e-5, atol=1e-5)


def test_read_stream_reads_chunks_in_order(tmp_path):
    (tmp_path / "00001.jsonl").write_text(json.dumps({"id": "b"}) + "\n")
    (tmp_path / "00000.jsonl").write_text(json.dumps({"id": "a"}) + "\n\n" + json.dumps({"id": "a2"}) + "\n")
    assert [r["id"] for r in episodes.read_stream(tmp_path)] == ["a", "a2", "b"]
