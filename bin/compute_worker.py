"""
Python compute worker for the StructureMASST Rust web server.

Replaces the former Celery worker (tasks.py). The Rust server starts this
process once and talks to it over stdin/stdout using JSON lines:

    request : {"id": 1, "op": "library_table", "args": {...}}
    response: {"id": 1, "ok": true, "result": ...}
              {"id": 1, "ok": false, "error": "..."}

The operations live in bin/compute_ops.py and run in process pools. Worker
processes are forked from a single-threaded forkserver that has preloaded the
ReDU table (bin/compute_preload.py), so they share it copy-on-write, and forking
never happens from this multi-threaded process. If a worker process dies
(e.g. killed for memory), its request fails with an error and the pool is
rebuilt, instead of hanging.

Anything the library code prints goes to stderr; stdout is reserved for the
protocol.
"""

import json
import multiprocessing as mp
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Keep the protocol channel private: duplicate the real stdout, then point fd 1
# (and sys.stdout) at stderr so stray prints cannot corrupt the protocol.
_PROTO_FD = os.dup(1)
os.dup2(2, 1)
sys.stdout = sys.stderr
_write_lock = threading.Lock()


def _send(obj):
    data = (json.dumps(obj, default=str) + "\n").encode("utf-8")
    with _write_lock:
        view = memoryview(data)
        while view:
            n = os.write(_PROTO_FD, view)
            view = view[n:]


class Pool:
    """A ProcessPoolExecutor that is rebuilt when a worker process dies."""

    def __init__(self, ctx, workers, max_tasks):
        self.ctx, self.workers, self.max_tasks = ctx, workers, max_tasks
        self.lock = threading.Lock()
        self.executor = self._new()

    def _new(self):
        return ProcessPoolExecutor(max_workers=self.workers, mp_context=self.ctx, max_tasks_per_child=self.max_tasks)

    def submit(self, fn, *args):
        with self.lock:
            try:
                return self.executor.submit(fn, *args)
            except BrokenProcessPool:
                self.executor = self._new()
                return self.executor.submit(fn, *args)

    def broken(self, executor):
        with self.lock:
            if self.executor is executor:
                print("[worker] a worker process died; rebuilding the pool", flush=True)
                self.executor = self._new()


def main():
    from bin.compute_ops import LIGHT_OPS, run_op

    heavy_n = int(os.environ.get("WORKER_PROCESSES", "4"))
    light_n = int(os.environ.get("LIGHT_WORKER_PROCESSES", "2"))

    ctx = mp.get_context("forkserver")
    ctx.set_forkserver_preload(["bin.compute_preload"])
    light = Pool(ctx, light_n, None)
    heavy = Pool(ctx, heavy_n, 5)  # recycle processes like celery --max-tasks-per-child=5

    # Start the forkserver (runs the preload) before announcing readiness.
    light.submit(run_op, "ping", {}).result()

    def dispatch(req):
        rid = req.get("id")
        op = req.get("op")
        args = req.get("args") or {}
        pool = light if op in LIGHT_OPS else heavy
        try:
            fut = pool.submit(run_op, op, args)
        except Exception as e:
            _send({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"})
            return
        executor = pool.executor

        def done(f):
            try:
                ok, payload = f.result()
            except BrokenProcessPool:
                pool.broken(executor)
                ok, payload = False, "A compute process died (possibly out of memory). Please try again."
            except Exception as e:
                ok, payload = False, f"{type(e).__name__}: {e}"
            if ok:
                _send({"id": rid, "ok": True, "result": payload})
            else:
                _send({"id": rid, "ok": False, "error": payload})

        fut.add_done_callback(done)

    _send({"id": 0, "ok": True, "result": "ready"})
    print(f"[INIT] compute worker ready ({heavy_n} heavy, {light_n} light processes)", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[worker] bad request line: {e}")
            continue
        dispatch(req)

    light.executor.shutdown(wait=False, cancel_futures=True)
    heavy.executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
