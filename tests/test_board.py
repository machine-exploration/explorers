from explorers.episode import BoardMessage
from explorers.runtime.verifiers.board import parse_reply, turn_prompt


def test_parse_reply():
    posts, submit = parse_reply("ok <post> edit the test </post> and <POST>two</post> <submit/>")
    assert posts == ["edit the test", "two"] and submit


def test_no_tags():
    assert parse_reply("nothing here") == ([], False)
    assert parse_reply(None) == ([], False)


def test_prompt_lists_messages():
    text = turn_prompt(1, 8, [BoardMessage(round=0, agent_id="agent_1", text="hi")], True)
    assert "Round 2 of 8" in text and "[agent_1, round 1] hi" in text
