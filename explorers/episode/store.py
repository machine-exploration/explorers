"""Append-only JSONL store: one line per episode in `<dir>/episodes.jsonl`."""

from pathlib import Path

from explorers.episode.model import Episode

FILE = "episodes.jsonl"


def write_episode(directory: Path, episode: Episode) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / FILE).open("a", encoding="utf-8") as f:
        f.write(episode.model_dump_json() + "\n")


def read_episodes(directory: Path) -> list[Episode]:
    path = directory / FILE
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [Episode.model_validate_json(line) for line in f if line.strip()]


def episode_ids(directory: Path) -> set[str]:
    return {e.id for e in read_episodes(directory)}
