"""E1 and E2: streams, read / write / trace, and experiments, on tiny random models built locally.

Every check here is exact by construction, so it holds for any weights:
- a read equals what the model computes at that site;
- patching the only site where clean and corrupt differ restores the clean output exactly;
- patching a site that cannot reach the metric changes nothing.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import explorers as ex  # noqa: E402
from explorers.methods.probes import Logistic  # noqa: E402

V, D, L, SEQ = 16, 32, 3, 6


def neox(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = transformers.GPTNeoXConfig(vocab_size=V, hidden_size=D, num_hidden_layers=L, num_attention_heads=4,
                                     intermediate_size=2 * D, max_position_embeddings=32, **kw)
    return ex.open(transformers.GPTNeoXForCausalLM(cfg))


def llama(seed=0):
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(vocab_size=V, hidden_size=D, num_hidden_layers=L, num_attention_heads=4,
                                   num_key_value_heads=4, intermediate_size=2 * D, max_position_embeddings=32)
    return ex.open(transformers.LlamaForCausalLM(cfg))


def gpt2(seed=0):
    torch.manual_seed(seed)
    cfg = transformers.GPT2Config(vocab_size=V, n_embd=D, n_layer=L, n_head=4, n_positions=32)
    return ex.open(transformers.GPT2LMHeadModel(cfg))


QWEN3_5_TEXT = dict(vocab_size=V, hidden_size=D, num_hidden_layers=4, intermediate_size=2 * D,
                    layer_types=["linear_attention", "full_attention"] * 2, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
                    linear_key_head_dim=8, linear_value_head_dim=8, max_position_embeddings=32)


def _perturb_norms(module):
    """Qwen 3.5 norms scale by 1 + weight and start at weight = 0: move every gain off 1."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "norm" in name:
                p.add_(0.5 * torch.randn_like(p))
    return module


def qwen3_5(seed=0):
    """A hybrid (linear and full attention) Qwen 3.5 text model."""
    torch.manual_seed(seed)
    return ex.open(_perturb_norms(transformers.Qwen3_5ForCausalLM(transformers.Qwen3_5TextConfig(**QWEN3_5_TEXT))))


def qwen3_5_vl(seed=0):
    """The multimodal class Qwen 3.8 ships as: the text model sits under model.language_model."""
    torch.manual_seed(seed)
    cfg = transformers.Qwen3_5Config(text_config=QWEN3_5_TEXT, vision_config=dict(
        depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=D))
    return ex.open(_perturb_norms(transformers.Qwen3_5ForConditionalGeneration(cfg)))


def tokens(n=4, seed=0):
    return np.random.default_rng(seed).integers(0, V, size=(n, SEQ))


# --- E1: streams -------------------------------------------------------------------------------

@pytest.mark.parametrize("make", [neox, llama, gpt2, qwen3_5, qwen3_5_vl])
def test_residual_reads_match_the_model(make):
    model, ids = make(), tokens()
    L = model.n_layers
    with model.trace(ids) as run:
        reads = [run.stream("residual").read(layer) for layer in range(L + 1)]
    with torch.no_grad():
        hs = model.module(input_ids=torch.as_tensor(ids), output_hidden_states=True).hidden_states
    for layer in range(L):
        torch.testing.assert_close(reads[layer].value, hs[layer])
    torch.testing.assert_close(model.norm(reads[L].value), hs[L])          # residual[L] is before the norm
    torch.testing.assert_close(model.unembed(reads[L].value), run.logits)


def test_unembed_matrix_with_a_shifted_rms_gain():
    """Qwen 3.5 norms compute x / rms(x) * (1 + weight): the readout is W diag(1 + weight) x / rms(x)."""
    from explorers.execute import unembed_matrix

    model = qwen3_5_vl()
    x = torch.randn(5, model.d_model)
    with torch.no_grad():
        logits = model.unembed(x).numpy()
    rms = torch.sqrt(x.pow(2).mean(dim=-1) + model.norm.eps).numpy()
    np.testing.assert_allclose(x.numpy() @ unembed_matrix(model).T / rms[:, None], logits, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("make,stream,layer,part", [
    (neox, "attn_out", 1, "attention"), (neox, "mlp_out", 1, "mlp"),
    (qwen3_5, "attn_out", 0, "linear_attn"), (qwen3_5, "attn_out", 1, "self_attn"), (qwen3_5_vl, "mlp_out", 2, "mlp"),
])
def test_sublayer_reads_match_hooks(make, stream, layer, part):
    model, ids = make(), tokens()
    seen = {}
    h = getattr(model.blocks[layer], part).register_forward_hook(
        lambda _m, _i, out: seen.__setitem__("x", (out[0] if isinstance(out, tuple) else out).detach()))
    with model.trace(ids) as run:
        x = run.stream(stream).read(layer, position=[2, 4])
    h.remove()
    torch.testing.assert_close(x.value, seen["x"][:, [2, 4]])


def test_write_changes_exactly_the_target():
    model, ids = neox(), tokens()
    d = torch.randn(D)
    with model.trace(ids) as base:
        before = [base.stream("residual").read(layer) for layer in range(L + 1)]
    with model.trace(ids) as run:
        resid = run.stream("residual")
        resid.write(1, position=2, fn=ex.ops.Add(d))
        after = [resid.read(layer) for layer in range(L + 1)]
    torch.testing.assert_close(after[1].value[:, 2], before[1].value[:, 2] + d)      # the read sees the write
    torch.testing.assert_close(after[1].value[:, [0, 1, 3, 4, 5]], before[1].value[:, [0, 1, 3, 4, 5]])
    torch.testing.assert_close(after[0].value, before[0].value)                     # upstream unchanged
    for layer in (2, 3):
        torch.testing.assert_close(after[layer].value[:, :2], before[layer].value[:, :2])   # causality
        assert not torch.allclose(after[layer].value[:, 2:], before[layer].value[:, 2:])


def test_zero_write_and_value_write():
    model, ids = neox(), tokens()
    with model.trace(ids) as base:
        x = base.stream("residual").read(2, position=3)
    with model.trace(ids) as run:
        run.stream("residual").write(2, position=3, fn=ex.ops.Add(torch.zeros(D)))
    torch.testing.assert_close(run.logits, base.logits)
    with model.trace(ids) as run2:
        run2.stream("residual").write(2, position=3, value=x.value)
    torch.testing.assert_close(run2.logits, base.logits)


def test_errors():
    model = neox()
    with pytest.raises(ValueError):
        model.trace(tokens()).stream("nope").read(0)
    with pytest.raises(ValueError):
        model.trace(tokens()).stream("attn_out").read(L)                 # attn_out has layers 0..L-1
    with pytest.raises(ValueError):
        model.trace(tokens()).stream("residual").write(0, fn=lambda h: h, value=0)
    run = model.trace(tokens())
    v = run.stream("residual").read(0)
    with pytest.raises(RuntimeError):
        v.value


def test_gradients_through_a_frozen_model():
    model, ids = neox(), tokens(n=2)
    model.module.requires_grad_(False)
    with model.trace(ids, grad=True) as run:
        x = run.stream("residual").read(1)
    run.logits[:, -1, 3].sum().backward()
    assert x.grad.shape == x.value.shape
    assert torch.all(x.grad.abs().sum(-1) > 0)                     # every position reaches the last logit
    # a later position cannot influence an earlier logit
    with model.trace(ids, grad=True) as run2:
        y = run2.stream("residual").read(1)
    run2.logits[:, 2, 3].sum().backward()
    assert torch.all(y.grad[:, 3:] == 0)


# --- E2: experiments and the five canonical examples -------------------------------------------------

def test_probe_reads_a_planted_feature():
    """Label = whether the first token is in a set. residual[0] at position 0 is the token's
    embedding; 16 random embeddings in 32 dimensions are linearly independent, so any labelling is
    linearly separable: the probe must reach accuracy 1."""
    model = neox()
    ids = np.stack([np.r_[t, np.random.default_rng(t).integers(0, V, SEQ - 1)] for t in range(V)] * 2)
    labels = np.isin(ids[:, 0], [1, 4, 5, 9, 12, 15])
    ds = ex.Experiment(model, ids).read("residual", layers=[0], position=0).compute()
    X = ds.residual.sel(model=model.key, layer=0).values
    probe = Logistic(l2=1e-3).fit(X, labels)
    assert ((probe.score(X) > 0) == labels).mean() == 1.0


def test_steering_toward_a_token():
    """Adding a large multiple of the unembedding direction of token t (scaled by the norm weight)
    to the final residual must raise the logit of t."""
    model, ids = neox(), tokens()
    t = 7
    u = model.module.get_output_embeddings().weight[t].detach() * model.norm.weight.detach()

    @ex.measures.measure(reads=["logits:-1"], dims=("example",))
    def logit_t(ctx):
        return ctx.logits[-1][:, t]

    base = ex.Experiment(model, ids).measure(logit_t).compute()
    steered = ex.Experiment(model, ids).write("residual", L, position=-1, fn=ex.ops.Add(u, 50.0)).measure(logit_t).compute()
    assert (steered.logit_t.values > base.logit_t.values).all()


def pairs(n=4, seed=0):
    """Clean and corrupt examples that differ only at position 0."""
    clean = tokens(n, seed)
    corrupt = clean.copy()
    corrupt[:, 0] = (clean[:, 0] + 1) % V
    return clean, corrupt


def test_activation_patching_exact_by_construction():
    model = neox()
    clean, corrupt = pairs()
    metric = ex.measures.logit_diff(correct=3, wrong=5)
    ds = ex.Experiment(model, corrupt).patch(source=clean, stream="residual", metric=metric).compute()
    p = ds.patch.sel(model=model.key)
    np.testing.assert_allclose(p.sel(layer=0, position=0), 1.0, atol=1e-4)       # the only difference
    np.testing.assert_allclose(p.sel(layer=0).isel(position=slice(1, None)), 0.0, atol=1e-4)
    np.testing.assert_allclose(p.sel(layer=L, position=SEQ - 1), 1.0, atol=1e-4)   # all the logits
    np.testing.assert_allclose(p.sel(layer=L).isel(position=slice(0, -1)), 0.0, atol=1e-4)


def test_attribution_patching():
    """Where exact patching is linear in the site, attribution patching is exact: with the final
    norm removed, the metric is linear in residual[L] at the last position."""
    model, (clean, corrupt) = neox(), pairs()
    metric = ex.measures.logit_diff(correct=3, wrong=5)
    attr = ex.Experiment(model, corrupt).patch(source=clean, metric=metric, method="attribution").compute()
    a = attr.patch.sel(model=model.key)
    np.testing.assert_allclose(a.sel(layer=0).isel(position=slice(1, None)), 0.0, atol=1e-6)   # no difference
    np.testing.assert_allclose(a.sel(layer=L).isel(position=slice(0, -1)), 0.0, atol=1e-6)    # no path

    linear = neox()
    linear.module.gpt_neox.final_layer_norm = torch.nn.Identity()
    linear.norm = linear.module.gpt_neox.final_layer_norm
    lin = ex.Experiment(linear, corrupt).patch(source=clean, metric=metric, method="attribution",
                                          layers=[L], positions=[SEQ - 1]).compute()
    np.testing.assert_allclose(lin.patch.sel(layer=L, position=SEQ - 1).values, 1.0, atol=1e-4)


def test_sae_read():
    model, ids = neox(), tokens()
    rng = np.random.default_rng(0)
    sae = ex.sae.SAE(rng.normal(size=(D, 64)), rng.normal(size=64), rng.normal(size=(64, D)), rng.normal(size=D))
    ds = ex.Experiment(model, ids).read("residual", layers=[2], position=-1).compute()
    x = ds.residual.sel(model=model.key, layer=2).values
    f = sae.encode(x)
    np.testing.assert_allclose(f, np.maximum((x - sae.b_dec) @ sae.W_enc + sae.b_enc, 0), rtol=1e-5)
    assert (f >= 0).all() and f.shape == (len(ids), 64)
    identity = ex.sae.SAE(np.eye(D), np.zeros(D), np.eye(D), np.zeros(D))
    np.testing.assert_allclose(identity.decode(identity.encode(np.abs(x))), np.abs(x), rtol=1e-6)


def test_experiment_over_checkpoints_of_a_run():
    from explorers import toy

    task = toy.MultitaskLookup(n_tasks=4, n_symbols=4)
    run = toy.train(task, steps=20, every=10)
    ds = ex.Experiment(run, task.examples()).read("residual", layers="*", position=-1).compute()
    assert list(ds.step.values) == [0, 10, 20]
    assert ds.residual.dims == ("step", "example", "layer", "d")
    assert not np.allclose(ds.residual.sel(step=0), ds.residual.sel(step=20))
