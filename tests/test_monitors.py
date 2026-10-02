"""Monitors (docs/monitors.md): the concept lens without labels, a difference-of-means probe with
labels, per-example reduction and detection at a matched false-positive rate. Exact checks."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import explorers as ex  # noqa: E402
from explorers import analysis, measures  # noqa: E402
from explorers.data import Examples  # noqa: E402
from explorers.execute import jacobians, serve, unembed_matrix  # noqa: E402



def tiny_neox(d=8, layers=2, vocab=17, seed=0):
    torch.manual_seed(seed)
    cfg = transformers.GPTNeoXConfig(vocab_size=vocab, hidden_size=d, num_hidden_layers=layers,
                                     num_attention_heads=2, intermediate_size=2 * d, max_position_embeddings=16)
    return ex.open(transformers.GPTNeoXForCausalLM(cfg).eval())


def random_examples(n=3, seq=6, vocab=17, seed=0):
    return Examples(tokens=np.random.default_rng(seed).integers(0, vocab, size=(n, seq)))


@pytest.mark.parametrize("layer,target", [(0, "final"), (1, "final"), (1, "penultimate")])
def test_concept_rows_equal_the_full_lens(layer, target):
    model, examples = tiny_neox(), random_examples()
    ids = (3, 11, 5)
    out = jacobians(model, examples, {(layer, 1, target), (layer, 1, target, ids)}, batch_size=2, dim_batch=2)
    full, rows = out[(layer, 1, target)], out[(layer, 1, target, ids)]
    np.testing.assert_allclose(rows, unembed_matrix(model)[list(ids)] @ full, rtol=1e-4, atol=1e-6)


def test_concept_monitor_is_the_projection_on_the_concept_rows():
    model, examples = tiny_neox(), random_examples(n=4, seq=7)
    m = measures.concept_monitor(1, [3, 11], skip_first=1, reduce="max")
    ctx = serve(model, examples, set(m.reads), batch_size=2)
    v = unembed_matrix(model)[[3, 11]] @ jacobians(model, examples, {(1, 1, "final")}, 2)[(1, 1, "final")]
    h = ctx.stream("residual", 1)[:, 1:6]                                  # lens positions 1..5
    expected = (h @ v.T).max(axis=-1).max(axis=-1)
    np.testing.assert_allclose(m(ctx).values, expected, rtol=1e-4, atol=1e-5)


def test_logit_concept_monitor_uses_the_unembedding_rows():
    model, examples = tiny_neox(), random_examples(n=4, seq=7)
    m = measures.concept_monitor(1, [3, 11], skip_first=1, lens="logit")
    ctx = serve(model, examples, set(m.reads), batch_size=2)
    h = ctx.stream("residual", 1)[:, 1:6]
    expected = (h @ unembed_matrix(model)[[3, 11]].T).max(axis=-1).max(axis=-1)
    np.testing.assert_allclose(m(ctx).values, expected, rtol=1e-5, atol=1e-6)
    assert m.name == "logit_concept_monitor_1" and not any(r.startswith("concept") for r in m.reads)


def test_probe_monitor_fits_on_fit_rows_only():
    model = tiny_neox()
    tokens = random_examples(n=6, seq=7).tokens
    split = np.array(["fit"] * 4 + ["eval"] * 2)
    a = Examples(tokens=tokens, meta={"label": np.array([1, 0, 1, 0, 1, 1]), "split": split})
    b = a.with_meta(label=np.array([1, 0, 1, 0, 0, 0]))                    # eval labels changed
    m = measures.probe_monitor(1, skip_first=1, reduce="mean")
    ctx = serve(model, a, set(m.reads), batch_size=2)
    got = m(ctx).values
    np.testing.assert_array_equal(got, m(serve(model, b, set(m.reads), batch_size=2)).values)
    h = ctx.stream("residual", 1)[:, 1:6]
    means = h.mean(axis=1)
    direction = means[[0, 2]].mean(0) - means[[1, 3]].mean(0)
    np.testing.assert_allclose(got, (h @ direction).mean(axis=1), rtol=1e-5, atol=1e-6)


def test_loss_mask_sets_the_scored_positions():
    tokens = np.zeros((2, 5), dtype=np.int64)
    mask = np.array([[0, 1, 1, 0, 0], [0, 0, 0, 0, 1]], dtype=bool)
    assert (measures.scored_positions(Examples(tokens=tokens, loss_mask=mask), 1) == mask).all()
    scores = np.array([[9.0, 1.0, 4.0, 7.0, 0.0], [5.0, 5.0, 5.0, 5.0, -2.0]])
    np.testing.assert_array_equal(measures.reduce_positions(scores, mask, "max"), [4.0, -2.0])
    np.testing.assert_array_equal(measures.reduce_positions(scores, mask, "mean"), [2.5, -2.0])
    np.testing.assert_array_equal(measures.reduce_positions(scores, mask, "last"), [4.0, -2.0])
    none = np.zeros_like(mask)
    assert np.isnan(measures.reduce_positions(scores, none, "max")).all()
    with pytest.raises(ValueError, match="unknown reduction"):
        measures.reduce_positions(scores, mask, "median")


def test_detection_at_fpr_exact():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([9, 8, 3, 1, 10, 7, 6, 5, 4, 2, 2, 0, 0, 0], dtype=float)
    assert analysis.detection_at_fpr(y, s, 0.0) == 0.0          # threshold 10: nothing above
    assert analysis.detection_at_fpr(y, s, 0.1) == 0.5          # 1 negative allowed: threshold 7 -> {9, 8}
    assert analysis.detection_at_fpr(y, s, 0.5) == 0.75         # 5 allowed: threshold 2 -> {9, 8, 3}
    assert analysis.detection_at_fpr(y, s, 1.0) == 1.0
    assert analysis.detection_at_fpr(np.ones(3), s[:3], 0.1) is None


def test_ties_never_exceed_the_false_positive_budget():
    rng = np.random.default_rng(0)
    y = rng.random(400) < 0.3
    s = rng.integers(0, 5, 400).astype(float)                    # heavy ties
    for fpr in (0.01, 0.05, 0.1, 0.3):
        tpr = analysis.detection_at_fpr(y, s, fpr)
        neg = np.sort(s[~y])[::-1]
        threshold = neg[int(np.floor(fpr * len(neg)))]
        assert np.mean(s[~y] > threshold) <= fpr + 1e-12
        assert tpr == np.mean(s[y] > threshold)


def test_both_monitors_in_an_experiment():
    model = tiny_neox()
    tokens = random_examples(n=6, seq=7).tokens
    examples = Examples(tokens=tokens, meta={"label": np.array([1, 0, 1, 0, 1, 0]),
                                              "split": np.array(["fit"] * 4 + ["eval"] * 2)})
    ds = (ex.Experiment(model, examples, batch_size=2)
          .measure(measures.concept_monitor(1, [3, 11], skip_first=1), measures.probe_monitor(1, skip_first=1))
          .compute())
    assert ds.concept_monitor_1.dims == ("model", "example") == ds.probe_monitor_1.dims
    assert list(ds.split.values) == ["fit"] * 4 + ["eval"] * 2
