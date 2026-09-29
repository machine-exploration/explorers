from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from explorers.populations.episode import Episode


class Annotation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str
    agent_id: str | None
    span_ids: list[str]
    type: Literal["score", "label", "text", "alert", "error"]
    name: str
    value: float | str | bool
    method: str
    method_version: str


class Method(Protocol):
    name: str
    version: str
    shape: Literal["map", "reduce"]
    needs: frozenset[str]

    def run(self, episode: Episode) -> list[Annotation]: ...
