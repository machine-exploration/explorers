from collections import Counter
from pathlib import Path

from explorers.episode import Episode, episode_ids, read_episodes, write_episode
from explorers.methods import HackLabel, Propagation
from explorers.metrics import rate, score_detectors
from explorers.scenario import load_scenario


def append_new(store: Path, episodes: list[Episode]) -> int:
    """Write episodes whose id is not already in store. Return count written."""
    known = episode_ids(store)
    written = 0
    for ep in episodes:
        if ep.id not in known:
            write_episode(store, ep)
            known.add(ep.id)
            written += 1
    return written


def _next_run_dir(vf_root: Path) -> Path:
    """A fresh, never-used run directory under `vf_root`: vf-eval refuses to write
    into a run dir that already holds results unless told to `--resume`."""
    vf_root.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in vf_root.iterdir() if p.is_dir()}
    i = 0
    while f"run_{i}" in existing:
        i += 1
    return vf_root / f"run_{i}"


class Exploration:
    def __init__(self, scenario_path: Path, model: str, base_url: str, episodes: int, seed: int,
                out_dir: Path, max_concurrent: int = 2):
        self.scenario = load_scenario(scenario_path)
        self.model, self.base_url, self.episodes, self.seed = model, base_url, episodes, seed
        self.max_concurrent = max_concurrent
        self.out_dir = Path(out_dir)

    def run(self) -> dict:
        from explorers.runtime.verifiers.compile import compile_scenario
        from explorers.runtime.verifiers.convert import episodes_from_run
        from explorers.runtime.verifiers.run import VERIFIERS_COMMIT, run_vf_eval

        store = self.out_dir / "episodes"
        vf_root = self.out_dir / "vf"

        # Crash recovery: a previous attempt may have produced a traces.jsonl that
        # never made it into the store (e.g. the process died right after vf-eval
        # returned). Ingest every run dir's traces once, before generating anything new.
        if vf_root.is_dir():
            for trace_file in sorted(vf_root.glob("*/traces.jsonl")):
                append_new(store, episodes_from_run(trace_file.parent, self.model, VERIFIERS_COMMIT,
                                                    fallback_scenario=self.scenario.name,
                                                    fallback_version=self.scenario.version))

        store_episodes = read_episodes(store)
        missing = self.episodes - sum(1 for e in store_episodes if e.status == "ok")
        if missing > 0:
            used_seeds = [e.seed for e in store_episodes]
            next_seed = (max(used_seeds) + 1) if used_seeds else self.seed
            seed_offset = next_seed - self.seed
            pkg, toml = compile_scenario(self.scenario, self.out_dir, self.model, self.base_url,
                                         next_seed, missing, seed_offset=seed_offset,
                                         max_concurrent=self.max_concurrent)
            run_dir = run_vf_eval(pkg, toml, _next_run_dir(vf_root))
            episodes = episodes_from_run(run_dir, self.model, VERIFIERS_COMMIT,
                                         fallback_scenario=self.scenario.name,
                                         fallback_version=self.scenario.version)
            statuses = [e.status for e in episodes]
            if episodes and all(status != "ok" for status in statuses):
                raise RuntimeError(f"every episode in {run_dir} came back non-ok: {statuses}")
            append_new(store, episodes)
        return self.report(read_episodes(store))

    @staticmethod
    def report(episodes: list[Episode]) -> dict:
        kept = [e for e in episodes if e.status == "ok"]
        labels = {e.id: HackLabel().run(e) for e in kept}
        hack = [a for anns in labels.values() for a in anns if a.name == "hack" and a.value != "excluded"]
        excluded = sum(1 for anns in labels.values() for a in anns if a.name == "hack" and a.value == "excluded")
        modified = [a for anns in labels.values() for a in anns if a.name == "tests_modified"]
        prop = [a for e in kept for a in Propagation().run(e) if a.name == "propagated"]
        rollouts = [r for e in kept for r in e.rollouts]
        return {
            "episodes_total": len(episodes),
            "episodes_excluded": dict(Counter(e.status for e in episodes if e.status != "ok")),
            "rollouts_scored": len(hack),
            "rollouts_excluded": excluded,
            "hack_rate": rate(sum(a.value == "hack" for a in hack), len(hack)),
            "tests_modified_rate": rate(sum(bool(a.value) for a in modified), len(modified)),
            "propagation_rate": rate(sum(bool(a.value) for a in prop), len(prop)),
            "board_posts_per_episode": (sum(len(e.board) for e in kept) / len(kept)) if kept else 0.0,
            "tokens_recorded_rate": rate(sum(r.tokens_recorded for r in rollouts), len(rollouts)),
            "detectors": score_detectors(kept, HackLabel(), [Propagation()], seed=0),
        }
