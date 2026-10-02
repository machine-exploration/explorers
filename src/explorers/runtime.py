"""Client and runtimes: where jobs run, behind one contract.

    client = ex.Client(ex.LocalRuntime())                    # CPU, a subprocess: checks the contract
    h = client.submit("experiments/glp-activation/train.py:main", layer=15, tokens=10_000)
    client.result(h)                                         # the job's JSON result

    client = ex.Client(ModalRuntime(gpu="A100"))             # same job, on a GPU worker (next)

A Job is data: an entrypoint (`module:function` or `path/to/file.py:function`), keyword arguments
that are JSON, the code version it runs at and the resources it asks for. Nothing else crosses the
boundary, so runtimes are swappable. `LocalRuntime` runs every job in a subprocess on the CPU,
through the same path a remote runtime takes: the job is written as JSON, the entrypoint is
imported by name, the result is written back as JSON. A job that runs locally on small inputs only
differs from its GPU run by the hardware. `tests/test_runtime.py` is the conformance suite every
runtime must pass. Inside a job, `ex.runtime.artifacts()` is the runtime's artifact store.
"""

import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATUSES = ("queued", "running", "done", "failed", "cancelled")


@dataclass(frozen=True)
class Resources:
    gpu: str | None = None        # "A100", "H100", ...; LocalRuntime runs every job on the CPU
    memory_gb: int = 8
    timeout_s: int = 3600


@dataclass(frozen=True)
class Job:
    entrypoint: str
    kwargs: dict = field(default_factory=dict)
    code: str = "unknown"
    resources: Resources = Resources()

    def __post_init__(self):
        module, sep, func = self.entrypoint.rpartition(":")
        if not sep or not module or not func.isidentifier():
            raise ValueError("entrypoint must be 'module:function' or 'path/to/file.py:function', "
                             f"got {self.entrypoint!r}")
        try:
            json.dumps(self.kwargs, allow_nan=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"job arguments must be JSON (numbers, strings, lists, dicts): {e}") from e

    def to_json(self) -> str:
        return json.dumps({"format": "explorers.job/v1", **asdict(self)}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Job":
        d = json.loads(text)
        if d.pop("format", None) != "explorers.job/v1":
            raise ValueError("not an explorers.job/v1 document")
        return cls(d["entrypoint"], d["kwargs"], d["code"], Resources(**d["resources"]))

    @property
    def key(self) -> str:
        """Content key: equal jobs get equal keys on any machine."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:32]


@dataclass(frozen=True)
class Handle:
    id: str
    job: Job


class JobFailed(RuntimeError):
    pass


class Artifacts:
    """Bytes by key: caches, checkpoints, results. One folder; a remote runtime mounts it."""

    def __init__(self, root):
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / key

    def put(self, key: str, data: bytes) -> None:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)

    def get(self, key: str) -> bytes | None:
        p = self.path(key)
        return p.read_bytes() if p.exists() else None


class Runtime(ABC):
    """Where jobs run. Every runtime passes the same conformance suite (`tests/test_runtime.py`)."""

    artifacts: Artifacts

    def start(self) -> None: ...
    def stop(self) -> None: ...

    @abstractmethod
    def submit(self, job: Job) -> Handle: ...
    @abstractmethod
    def status(self, h: Handle) -> str: ...
    @abstractmethod
    def logs(self, h: Handle) -> str: ...
    @abstractmethod
    def result(self, h: Handle, timeout: float | None = None) -> dict | list | str | int | float | None: ...
    @abstractmethod
    def cancel(self, h: Handle) -> None: ...

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


class LocalRuntime(Runtime):
    """Runs each job in a subprocess on the CPU. For checking the contract, never for real runs."""

    def __init__(self, root=".explorers/local", workdir=None):
        self.root = Path(root).resolve()
        self.workdir = Path(workdir or os.getcwd()).resolve()
        self.artifacts = Artifacts(self.root / "artifacts")
        self._procs: dict[str, tuple[subprocess.Popen, float]] = {}

    def _dir(self, h: Handle) -> Path:
        return self.root / "jobs" / h.id

    def submit(self, job: Job) -> Handle:
        h = Handle(uuid.uuid4().hex, job)
        d = self._dir(h)
        d.mkdir(parents=True)
        (d / "job.json").write_text(job.to_json(), encoding="utf-8")
        env = {**os.environ, "EXPLORERS_ARTIFACTS": str(self.artifacts.root),
               "PYTHONPATH": os.pathsep.join([str(self.workdir), os.environ.get("PYTHONPATH", "")])}
        log = open(d / "log.txt", "wb")
        proc = subprocess.Popen([sys.executable, "-m", "explorers.runtime", str(d)], cwd=self.workdir, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        log.close()
        self._procs[h.id] = (proc, time.monotonic())
        return h

    def status(self, h: Handle) -> str:
        d = self._dir(h)
        if (d / "cancelled").exists():
            return "cancelled"
        proc, started = self._procs[h.id]
        if proc.poll() is None:
            if time.monotonic() - started > h.job.resources.timeout_s:
                proc.kill()
                proc.wait()
                (d / "error.json").write_text(json.dumps({"error": f"timeout after {h.job.resources.timeout_s}s"}))
                return "failed"
            return "running"
        return "done" if (d / "result.json").exists() else "failed"

    def logs(self, h: Handle) -> str:
        p = self._dir(h) / "log.txt"
        return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

    def result(self, h: Handle, timeout: float | None = None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while (s := self.status(h)) in ("queued", "running"):
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"job {h.id} still {s}")
            time.sleep(0.05)
        d = self._dir(h)
        if s == "done":
            return json.loads((d / "result.json").read_text(encoding="utf-8"))["result"]
        if s == "cancelled":
            raise JobFailed(f"job {h.id} was cancelled")
        err = d / "error.json"
        message = json.loads(err.read_text())["error"] if err.exists() else "the job exited without a result"
        raise JobFailed(f"job {h.id} ({h.job.entrypoint}) failed: {message}\n--- logs ---\n{self.logs(h)[-4000:]}")

    def cancel(self, h: Handle) -> None:
        proc, _ = self._procs[h.id]
        if proc.poll() is None:
            proc.kill()
            proc.wait()
            (self._dir(h) / "cancelled").touch()

    def stop(self) -> None:
        for h_id in list(self._procs):
            proc, _ = self._procs[h_id]
            if proc.poll() is None:
                proc.kill()
                proc.wait()


class Client:
    """What you use locally: builds jobs and sends them to a runtime."""

    def __init__(self, runtime: Runtime, code: str | None = None):
        self.runtime = runtime
        self.code = code or _git_version()

    def submit(self, entrypoint: str, resources: Resources = Resources(), **kwargs) -> Handle:
        return self.runtime.submit(Job(entrypoint, kwargs, self.code, resources))

    def result(self, h: Handle, timeout: float | None = None):
        return self.runtime.result(h, timeout)

    def run(self, entrypoint: str, resources: Resources = Resources(), **kwargs):
        """Submit and wait for the result."""
        return self.result(self.submit(entrypoint, resources, **kwargs))


def artifacts() -> Artifacts:
    """Inside a job: the runtime's artifact store."""
    root = os.environ.get("EXPLORERS_ARTIFACTS")
    if not root:
        raise RuntimeError("not inside a job: EXPLORERS_ARTIFACTS is not set")
    return Artifacts(root)


def _git_version() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load(entrypoint: str):
    target, _, func = entrypoint.rpartition(":")
    if target.endswith(".py"):
        spec = importlib.util.spec_from_file_location(Path(target).stem.replace("-", "_"), target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, func)


def _execute(job_dir: Path) -> int:
    """The job side of the contract: read job.json, call the entrypoint, write result.json or error.json."""
    try:
        job = Job.from_json((job_dir / "job.json").read_text(encoding="utf-8"))
        value = _load(job.entrypoint)(**job.kwargs)
        text = json.dumps({"result": value}, allow_nan=False)
    except BaseException as e:  # noqa: BLE001 - every failure must reach the client
        traceback.print_exc()
        error = json.dumps({"error": f"{type(e).__name__}: {e}"})
        (job_dir / "error.json").write_text(error, encoding="utf-8")
        return 1
    tmp = job_dir / "result.json.tmp"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(job_dir / "result.json")
    return 0


if __name__ == "__main__":
    sys.exit(_execute(Path(sys.argv[1])))
