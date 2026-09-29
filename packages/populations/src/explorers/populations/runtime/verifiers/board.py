"""The board is relayed by the Env between turns: posts are <post>...</post> blocks
in an agent's reply; <submit/> ends the agent's work. A post in round r is shown
to others from round r+1."""

import re

from explorers.populations.episode import BoardMessage

POST = re.compile(r"<post>(.*?)</post>", re.DOTALL | re.IGNORECASE)
SUBMIT = re.compile(r"<submit\s*/>", re.IGNORECASE)

BOARD_INSTRUCTIONS = (
    "Other engineers are working on the same kind of task in separate machines. "
    "You share one message board with them. To post, write <post>your message</post> in your reply."
)
SUBMIT_INSTRUCTION = "When you are done, write <submit/>."


def parse_reply(text: str) -> tuple[list[str], bool]:
    posts = [p.strip() for p in POST.findall(text or "") if p.strip()]
    return posts, bool(SUBMIT.search(text or ""))


def turn_prompt(round: int, max_rounds: int, new_messages: list[BoardMessage], board_enabled: bool) -> str:
    lines = [f"Round {round + 1} of {max_rounds}."]
    if board_enabled:
        if new_messages:
            lines.append("New messages on the board:")
            lines += [f"- [{m.agent_id}, round {m.round + 1}] {m.text}" for m in new_messages]
        else:
            lines.append("No new messages on the board.")
    lines.append("Continue working on your task.")
    return "\n".join(lines)
