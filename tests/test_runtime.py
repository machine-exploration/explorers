"""The runtime contract. Every runtime must pass this suite; LocalRuntime runs it on the CPU."""

from pathlib import Path

import pytest

import explorers as ex
from explorers.runtime import Job, JobFailed, LocalRuntime, Resources

JOBS = Path(__file__).parent / "runtime_jobs.py"


def ep(name):
    return f"{JOBS}:{name}"


@pytest.fixture(params=["local"])
def runtime(request, tmp_path):
    rt = LocalRuntime(root=tmp_path / "rt")
    with rt:
        yield rt


def test_a_job_runs_and_returns_json(runtime):
    client = ex.Client(runtime, code="test")
    h = client.submit(ep("echo"), a=2, b=3)
    assert client.result(h, timeout=60) == {"sum": 5, "inputs": [2, 3]}
    assert runtime.status(h) == "done"


def test_module_entrypoints_work(runtime):
    assert ex.Client(runtime, code="test").run("json:dumps", obj=[1, 2]) == "[1, 2]"


def test_a_failing_job_reports_failed_with_its_logs(runtime):
    h = ex.Client(runtime, code="test").submit(ep("fail"))
    with pytest.raises(JobFailed, match="planned failure"):
        runtime.result(h, timeout=60)
    assert runtime.status(h) == "failed"
    assert "Traceback" in runtime.logs(h)


def test_logs_reach_the_client(runtime):
    h = ex.Client(runtime, code="test").submit(ep("talk"))
    assert runtime.result(h, timeout=60) == "ok"
    assert "hello from the job" in runtime.logs(h)


def test_cancel_stops_a_running_job(runtime):
    h = ex.Client(runtime, code="test").submit(ep("sleep"), seconds=30)
    runtime.cancel(h)
    assert runtime.status(h) == "cancelled"
    with pytest.raises(JobFailed, match="cancelled"):
        runtime.result(h, timeout=10)


def test_a_job_past_its_timeout_fails(runtime):
    h = ex.Client(runtime, code="test").submit(ep("sleep"), Resources(timeout_s=1), seconds=30)
    with pytest.raises(JobFailed, match="timeout"):
        runtime.result(h, timeout=60)


def test_jobs_write_artifacts_the_client_can_read(runtime):
    key = ex.Client(runtime, code="test").run(ep("write_artifact"), key="abc123", text="prior weights")
    assert runtime.artifacts.get(key) == b"prior weights"


def test_results_must_be_json(runtime):
    h = ex.Client(runtime, code="test").submit(ep("not_json"))
    with pytest.raises(JobFailed, match="not JSON serializable"):
        runtime.result(h, timeout=60)


def test_arguments_must_be_json():
    with pytest.raises(TypeError, match="JSON"):
        Job(ep("echo"), {"a": object()})
    with pytest.raises(ValueError, match="entrypoint"):
        Job("no_function_here")


def test_a_job_is_data_with_a_content_key():
    job = Job(ep("echo"), {"a": 1}, "abc", Resources(gpu="A100"))
    assert Job.from_json(job.to_json()) == job
    assert job.key == Job(ep("echo"), {"a": 1}, "abc", Resources(gpu="A100")).key
    assert job.key != Job(ep("echo"), {"a": 2}, "abc", Resources(gpu="A100")).key
