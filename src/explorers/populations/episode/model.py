"""Immutable episode data. Nothing edits an episode after it is written."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Span(_Frozen):
    id: str
    kind: Literal["turn", "board_post"]
    agent_id: str
    round: int
    text: str
    source_agent: str | None = None


class BoardMessage(_Frozen):
    round: int
    agent_id: str
    text: str


RolloutStatus = Literal["ok", "stalled", "timeout", "error"]


class Rollout(_Frozen):
    agent_id: str
    status: RolloutStatus
    token_ids: list[int]
    tokens_recorded: bool
    spans: list[Span]
    metrics: dict[str, float]


EpisodeStatus = Literal["ok", "infra_error", "timeout"]


class Episode(_Frozen):
    format: Literal["explorers.populations.episode/v0"] = "explorers.populations.episode/v0"
    id: str
    scenario: str
    scenario_version: str
    model: str
    seed: int
    status: EpisodeStatus
    rounds: int
    board: list[BoardMessage]
    rollouts: list[Rollout]
    source: dict
    """Where it came from, e.g. {"backend": "verifiers", "commit": "...", "run_dir": "..."}."""
