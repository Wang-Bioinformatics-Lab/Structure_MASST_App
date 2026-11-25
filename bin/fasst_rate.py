"""
Respect FASST rate limits without slowing down when there are none.

FASST answers 429 (Too Many Requests) or 503 when a client submits faster than
it allows. Without handling, the first such answer aborted a whole search.

    token = submit_with_rate_limit(fasst.query_fasst_api_usi, usi, db, host=...)

- No rate limit: no waiting at all - requests go out as fast as before.
- On 429/503: wait what the server asks for (Retry-After) or back off
  exponentially (0.5 s, 1 s, 2 s, ... capped), retry the same request, and space
  further submissions in this process a little apart.
- Every success shrinks that spacing again, so a temporary limit does not keep
  slowing the rest of the search.

The pacing is shared by every submission in the process (one search's batch of
requests), which is what a server-side limit counts.
"""
from __future__ import annotations

import email.utils
import random
import threading
import time

import requests

RETRY_STATUS = {429, 503}
MAX_TOTAL_WAIT = float(600)    # give up on one request after this many seconds of waiting
MAX_BACKOFF = 30.0             # longest single wait without a Retry-After
MAX_INTERVAL = 5.0             # widest spacing between submissions


class _Gate:
    def __init__(self):
        self.lock = threading.Lock()
        self.interval = 0.0       # spacing between submissions; 0 = no pacing
        self.next_at = 0.0        # earliest time the next submission may go out

    def wait(self):
        with self.lock:
            now = time.monotonic()
            at = max(now, self.next_at)
            self.next_at = at + self.interval
        delay = at - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def limited(self, pause: float):
        with self.lock:
            self.interval = min(MAX_INTERVAL, max(0.1, self.interval * 2 or 0.1))
            self.next_at = max(self.next_at, time.monotonic() + pause)

    def ok(self):
        with self.lock:
            if self.interval:
                self.interval *= 0.7
                if self.interval < 0.02:
                    self.interval = 0.0


GATE = _Gate()


def _retry_after(response) -> float | None:
    """Seconds from a Retry-After header (delta-seconds or HTTP date)."""
    if response is None:
        return None
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
        return max(0.0, when.timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def submit_with_rate_limit(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying on 429/503 as described above."""
    waited = 0.0
    attempt = 0
    while True:
        GATE.wait()
        try:
            result = fn(*args, **kwargs)
        except requests.exceptions.HTTPError as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if status not in RETRY_STATUS or waited >= MAX_TOTAL_WAIT:
                raise
            pause = _retry_after(response)
            if pause is None:
                pause = min(MAX_BACKOFF, 0.5 * (2 ** attempt)) * (0.75 + random.random() / 2)
            attempt += 1
            waited += pause
            print(f"[fasst] {status} from FASST, retrying in {pause:.1f}s (attempt {attempt})")
            GATE.limited(pause)
            continue
        GATE.ok()
        return result
