from collections import Counter
from pathlib import Path

from explorers.episode import Episode, episode_ids, read_episodes, write_episode
from explorers.methods import HackLabel, Propagation
from explorers.metrics import rate
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


class Exploration:
    def __init__(self, scenario_path: Path, model: str, base_url: str, episodes: int, seed: int, out_dir: Path):
        self.scenario = load_scenario(scenario_path)
        self.model, self.base_url, self.episodes, self.seed = model, base_url, episodes, seed
        self.out_dir = Path(out_dir)

    def run(self) -> dict:
        from explorers.runtime.verifiers.compile import compile_scenario
        from explorers.runtime.verifiers.convert import episodes_from_run
        from explorers.runtime.verifiers.run import VERIFIERS_COMMIT, run_vf_eval

        store = self.out_dir / "episodes"
        store_episodes = read_episodes(store)
        missing = self.episodes - len(store_episodes)
        if missing > 0:
            pkg, toml = compile_scenario(self.scenario, self.out_dir, self.model, self.base_url,
                                         self.seed + len(store_episodes), missing)
            vf_dir = run_vf_eval(pkg, toml, self.out_dir)
            for run_dir in sorted({p.parent for p in vf_dir.rglob("traces.jsonl")}):
                append_new(store, episodes_from_run(run_dir, self.model, VERIFIERS_COMMIT))
        return self.report(read_episodes(store))

    @staticmethod
    def report(episodes: list[Episode]) -> dict:
        kept = [e for e in episodes if e.status == "ok"]
        hack = [a for e in kept for a in HackLabel().run(e) if a.name == "hack" and a.value != "excluded"]
        modified = [a for e in kept for a in HackLabel().run(e) if a.name == "tests_modified"]
        prop = [a for e in kept for a in Propagation().run(e)]
        rollouts = [r for e in kept for r in e.rollouts]
        return {
            "episodes_total": len(episodes),
            "episodes_excluded": dict(Counter(e.status for e in episodes if e.status != "ok")),
            "rollouts_scored": len(hack),
            "hack_rate": rate(sum(a.value == "hack" for a in hack), len(hack)),
            "tests_modified_rate": rate(sum(bool(a.value) for a in modified), len(modified)),
            "propagation_rate": rate(sum(bool(a.value) for a in prop), len(prop)),
            "board_posts_per_episode": (sum(len(e.board) for e in kept) / len(kept)) if kept else 0.0,
            "tokens_recorded_rate": rate(sum(r.tokens_recorded for r in rollouts), len(rollouts)),
        }
