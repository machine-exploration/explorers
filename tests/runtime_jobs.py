"""Entrypoints for the runtime conformance suite (tests/test_runtime.py)."""

import time

from explorers import runtime


def echo(a, b=1):
    return {"sum": a + b, "inputs": [a, b]}


def fail():
    raise ValueError("planned failure")


def talk():
    print("hello from the job", flush=True)
    return "ok"


def sleep(seconds):
    time.sleep(seconds)
    return seconds


def not_json():
    return object()


def write_artifact(key, text):
    runtime.artifacts().put(key, text.encode())
    return key
