"""LoRA adapters of a prime-rl run (docs/prime.md): merging equals running the adapter unmerged."""

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from safetensors.torch import save_file  # noqa: E402

import explorers as ex  # noqa: E402
from explorers.model import merge_lora  # noqa: E402

TARGETS = ["q_proj", "v_proj", "down_proj"]


def tiny_llama(seed=0):
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return transformers.LlamaForCausalLM(config).eval()


def write_adapter(path, model, r=4, alpha=8.0, prefix="", seed=1):
    """An adapter in prime-rl's broadcast format: `<module>.lora_{A,B}.weight` and a PEFT config."""
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, layer in model.named_modules():
        if name.split(".")[-1] in TARGETS:
            tensors[f"{prefix}{name}.lora_A.weight"] = torch.randn(r, layer.in_features, generator=g) * 0.1
            tensors[f"{prefix}{name}.lora_B.weight"] = torch.randn(layer.out_features, r, generator=g) * 0.1
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": r, "lora_alpha": alpha, "target_modules": TARGETS, "modules_to_save": None}))
    return tensors


def unmerged_logits(model, tensors, scale, ids, prefix=""):
    """Run the adapter as prime-rl's layer does: base(x) + scale * B A x, via forward hooks."""
    hooks = []
    for name, layer in model.named_modules():
        a = tensors.get(f"{prefix}{name}.lora_A.weight")
        if a is not None:
            b = tensors[f"{prefix}{name}.lora_B.weight"]
            hooks.append(layer.register_forward_hook(lambda m, i, o, a=a, b=b: o + scale * (i[0] @ a.T) @ b.T))
    try:
        with torch.no_grad():
            return model(ids).logits
    finally:
        for h in hooks:
            h.remove()


@pytest.mark.parametrize("prefix", ["", "base_model.model."])
def test_merge_equals_unmerged(tmp_path, prefix):
    ids = torch.randint(0, 64, (2, 12))
    tensors = write_adapter(tmp_path / "a", tiny_llama(), prefix=prefix)
    expected = unmerged_logits(tiny_llama(), tensors, 8.0 / 4, ids, prefix)
    base = tiny_llama()
    with torch.no_grad():
        before = base(ids).logits
    merge_lora(base, tmp_path / "a")
    with torch.no_grad():
        merged = base(ids).logits
    assert (merged - before).abs().max() > 1e-3            # the adapter changes the model
    torch.testing.assert_close(merged, expected, atol=1e-5, rtol=1e-5)


def test_merge_refuses_what_it_cannot_apply(tmp_path):
    model = tiny_llama()
    save_file({"model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(4, 32)},
              str((tmp_path / "half").mkdir() or tmp_path / "half" / "adapter_model.safetensors"))
    (tmp_path / "half" / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))
    with pytest.raises(NotImplementedError):
        merge_lora(model, tmp_path / "half")
    write_adapter(tmp_path / "saved", model)
    cfg = json.loads((tmp_path / "saved" / "adapter_config.json").read_text())
    (tmp_path / "saved" / "adapter_config.json").write_text(json.dumps({**cfg, "modules_to_save": ["lm_head"]}))
    with pytest.raises(NotImplementedError):
        merge_lora(model, tmp_path / "saved")


def test_archive_and_adapters(tmp_path):
    run, dest = tmp_path / "run", tmp_path / "archive"
    model = tiny_llama()
    for step, finished in [(1, True), (2, True), (3, False)]:
        d = run / "broadcasts" / f"step_{step}"
        write_adapter(d, model, seed=step)
        if finished:
            (d / ".finished").touch()
    assert ex.archive_adapters(run, dest) == [1, 2]
    assert ex.archive_adapters(run, dest) == []            # idempotent
    (run / "broadcasts" / "step_3" / ".finished").touch()
    assert ex.archive_adapters(run, dest) == [3]

    refs = ex.adapters("base", dest)
    assert [r.step for r in refs] == [1, 2, 3]
    assert len({r.key for r in refs}) == 3 and all(r.key.startswith("base@None+") for r in refs)
    assert ex.ModelRef("base").key == "base@None"


def test_open_with_adapter(tmp_path):
    ids = torch.randint(0, 64, (1, 10))
    tiny_llama().save_pretrained(tmp_path / "base")
    tensors = write_adapter(tmp_path / "a", tiny_llama())
    expected = unmerged_logits(tiny_llama(), tensors, 2.0, ids)
    ref = ex.ModelRef(str(tmp_path / "base"), step=5, adapter=str(tmp_path / "a"))
    model = ex.open(ref.name, adapter=ref.adapter, tokenizer=None)
    with torch.no_grad():
        torch.testing.assert_close(model.module(ids).logits, expected, atol=1e-5, rtol=1e-5)
