"""Append-only JSONL store: one line per episode in `<dir>/episodes.jsonl`."""

import warnings
from pathlib import Path

from explorers.populations.episode.model import Episode

FILE = "episodes.jsonl"


def write_episode(directory: Path, episode: Episode) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / FILE).open("a", encoding="utf-8") as f:
        f.write(episode.model_dump_json() + "\n")


def read_episodes(directory: Path) -> list[Episode]:
    """Parse every line as an Episode. A bad LAST line is treated as a crash mid-write:
    it is skipped with a warning. A bad line anywhere else still raises."""
    path = directory / FILE
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    last = len(lines) - 1
    episodes: list[Episode] = []
    for i, line in enumerate(lines):
        try:
            episodes.append(Episode.model_validate_json(line))
        except Exception as exc:
            if i == last:
                warnings.warn(f"skipping truncated last line in {path}: {exc}")
                break
            raise
    return episodes


def episode_ids(directory: Path) -> set[str]:
    return {e.id for e in read_episodes(directory)}
