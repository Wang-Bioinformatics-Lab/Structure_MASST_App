"""Tests for bin/fasst_rate.py (run: python -m pytest tests/test_fasst_rate.py)."""
import time

import pytest
import requests

import bin.fasst_rate as fr


class Resp:
    def __init__(self, status, retry_after=None):
        self.status_code = status
        self.headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}


def http_error(status, retry_after=None):
    return requests.exceptions.HTTPError(response=Resp(status, retry_after))


@pytest.fixture(autouse=True)
def fresh_gate(monkeypatch):
    monkeypatch.setattr(fr, "GATE", fr._Gate())


def test_no_limit_means_no_delay():
    calls = []
    t = time.monotonic()
    for i in range(200):
        fr.submit_with_rate_limit(lambda i=i: calls.append(i) or i)
    assert len(calls) == 200
    assert time.monotonic() - t < 0.5          # no pacing without a 429


def test_retries_429_then_succeeds_and_honours_retry_after():
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] <= 2:
            raise http_error(429, retry_after=0.2)
        return "token"

    t = time.monotonic()
    assert fr.submit_with_rate_limit(flaky) == "token"
    assert state["n"] == 3
    assert time.monotonic() - t >= 0.4          # waited what the server asked twice


def test_spacing_recovers_after_successes():
    for _ in range(4):
        fr.GATE.limited(0)                      # four 429s in a row
    assert fr.GATE.interval > 0
    for _ in range(20):
        fr.submit_with_rate_limit(lambda: None)
    assert fr.GATE.interval == 0.0              # back to full speed


def test_other_errors_are_not_retried():
    state = {"n": 0}

    def broken():
        state["n"] += 1
        raise http_error(500)
    with pytest.raises(requests.exceptions.HTTPError):
        fr.submit_with_rate_limit(broken)
    assert state["n"] == 1


def test_simulated_server_limit_completes_without_failing():
    """A server allowing 20 requests per second: a 100-request burst still completes."""
    window = []

    def server():
        now = time.monotonic()
        window[:] = [x for x in window if now - x < 1.0]
        if len(window) >= 20:
            raise http_error(429)               # no Retry-After: exponential backoff
        window.append(now)
        return "ok"

    t = time.monotonic()
    results = [fr.submit_with_rate_limit(server) for _ in range(100)]
    assert results == ["ok"] * 100
    assert time.monotonic() - t < 15            # paced near the allowed rate, not stalled
