import argparse
from pathlib import Path

from explorers.learning.activations import Cache, HFSource
from explorers.learning.checkpoints import pythia
from explorers.learning.datasets import from_jsonl, number_comparison
from explorers.learning.probes import DiffMeans, Logistic
from explorers.learning.sweep import Sweep, write_records

PROBES = {"diff_means": DiffMeans, "logistic": Logistic}


def _layers(spec: str, n_layers: int) -> list[int]:
    if spec == "all":
        return list(range(n_layers + 1))
    return [int(x) for x in spec.split(",")]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="explorers")
    sub = p.add_subparsers(dest="cmd", required=True)
    probe = sub.add_parser("probe", help="fit probes at each checkpoint and layer, write one record each")
    probe.add_argument("--model", default="pythia-70m", help="pythia-<size>, e.g. pythia-160m")
    probe.add_argument("--checkpoints", type=int, default=8, help="how many checkpoints, log-spaced")
    probe.add_argument("--dataset", default="number_comparison", help="number_comparison or a .jsonl path")
    probe.add_argument("--layers", default="all", help="'all' or comma-separated indices (0 = embeddings)")
    probe.add_argument("--probes", default="diff_means,logistic")
    probe.add_argument("--position", choices=["last", "mean"], default="last")
    probe.add_argument("--device", default="cpu")
    probe.add_argument("--dtype", default="float32")
    probe.add_argument("--cache", default="runs/cache")
    probe.add_argument("--out", default="runs/results.jsonl")
    probe.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if not args.model.startswith("pythia-"):
        p.error("only pythia-<size> models are supported for now")
    checkpoints = pythia(args.model.removeprefix("pythia-"), n=args.checkpoints)
    dataset = number_comparison(seed=args.seed) if args.dataset == "number_comparison" else from_jsonl(Path(args.dataset))
    source = HFSource(device=args.device, dtype=args.dtype, position=args.position, cache=Cache(Path(args.cache)))

    from transformers import AutoConfig
    n_layers = AutoConfig.from_pretrained(checkpoints[-1].model, revision=checkpoints[-1].revision).num_hidden_layers
    sweep = Sweep(checkpoints=checkpoints, dataset=dataset, layers=_layers(args.layers, n_layers),
                  probes=[PROBES[name]() for name in args.probes.split(",")], source=source,
                  position=args.position, seed=args.seed)

    def show(r):
        auc = "  n/a" if r.auroc is None else f"{r.auroc:.3f} [{r.ci_low:.2f}, {r.ci_high:.2f}]"
        print(f"step {r.step:>6}  layer {r.layer:>2}  {r.probe:<10}  AUROC {auc}", flush=True)
        write_records(Path(args.out), [r])

    sweep.run(on_record=show)
