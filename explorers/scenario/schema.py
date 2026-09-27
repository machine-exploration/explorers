"""Scenario format v0. Validation fails closed: anything unsafe or missing is an error."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

FORMAT = "explorers.scenario/v0"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentsSpec(_Strict):
    count: int = Field(ge=2, le=8)
    role_prompt: str


class TasksSpec(_Strict):
    directory: str
    """Relative to the scenario file; one sub-directory per task with `prompt.md`, `src/`, `tests/`."""
    assignment: Literal["same", "round_robin"] = "same"


class BoardSpec(_Strict):
    enabled: bool = True


class SandboxSpec(_Strict):
    image: str = "python:3.11-slim"
    network: Literal["none"]
    memory_gb: float = Field(2.0, gt=0)


class LimitsSpec(_Strict):
    max_rounds: int = Field(8, ge=1, le=64)


class ChecksSpec(_Strict):
    test_command: str
    test_files: list[str] = Field(min_length=1)


class Scenario(_Strict):
    format: Literal["explorers.scenario/v0"]
    name: str
    version: str
    agents: AgentsSpec
    tasks: TasksSpec
    board: BoardSpec = BoardSpec()
    sandbox: SandboxSpec
    limits: LimitsSpec = LimitsSpec()
    checks: ChecksSpec
    root: Path = Field(default=Path("."), exclude=True)

    def task_dirs(self) -> list[Path]:
        base = self.root / self.tasks.directory
        dirs = sorted(p for p in base.iterdir() if p.is_dir())
        if not dirs:
            raise ValueError(f"no task directories under {base}")
        return dirs


def load_scenario(path: Path) -> Scenario:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    scenario = Scenario.model_validate(data)
    return scenario.model_copy(update={"root": Path(path).resolve().parent})
