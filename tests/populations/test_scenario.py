import pytest
from pydantic import ValidationError

from explorers.populations.scenario import load_scenario

VALID = """
format: explorers.populations.scenario/v0
name: hf-incident-mini
version: "0"
agents: {count: 3, role_prompt: "You are an engineer."}
tasks: {directory: tasks}
sandbox: {network: none}
checks: {test_command: "python -m pytest -q tests", test_files: ["tests/test_solution.py"]}
"""


def write(tmp_path, text, complete_task=True):
    t1 = tmp_path / "tasks" / "t1"
    t1.mkdir(parents=True)
    if complete_task:
        (t1 / "prompt.md").write_text("Make the tests pass.")
        (t1 / "src").mkdir()
        (t1 / "tests").mkdir()
    p = tmp_path / "scenario.yaml"
    p.write_text(text)
    return p


def test_valid(tmp_path):
    s = load_scenario(write(tmp_path, VALID))
    assert s.agents.count == 3 and s.limits.max_rounds == 8
    assert [d.name for d in s.task_dirs()] == ["t1"]


def test_network_must_be_none(tmp_path):
    with pytest.raises(ValidationError):
        load_scenario(write(tmp_path, VALID.replace("network: none", "network: bridge")))


def test_test_files_required(tmp_path):
    bad = VALID.replace(', test_files: ["tests/test_solution.py"]', "")
    with pytest.raises(ValidationError):
        load_scenario(write(tmp_path, bad))


def test_unknown_field_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load_scenario(write(tmp_path, VALID + "extra: 1\n"))


def test_root_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="root"):
        load_scenario(write(tmp_path, VALID + "root: /etc\n"))


def test_absolute_directory_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load_scenario(write(tmp_path, VALID.replace("directory: tasks", f"directory: {tmp_path / 'tasks'}")))


def test_dotdot_directory_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load_scenario(write(tmp_path, VALID.replace("directory: tasks", "directory: ../tasks")))


def test_task_dir_missing_prompt_md_rejected(tmp_path):
    s = load_scenario(write(tmp_path, VALID, complete_task=False))
    with pytest.raises(ValueError, match="prompt.md"):
        s.task_dirs()
