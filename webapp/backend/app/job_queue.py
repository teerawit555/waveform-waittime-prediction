from __future__ import annotations

import atexit
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from .config import JOB_WORKERS
from .job_store import job_store


class JobQueue:
    """Small in-process worker queue for long-running ML jobs.

    This keeps Flask request handlers fast while preventing unlimited
    background threads/subprocesses from being started under load.
    """

    def __init__(self, max_workers: int = JOB_WORKERS):
        self.max_workers = max(1, max_workers)
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="ml-job")
        self._lock = threading.Lock()
        self._futures: dict[str, Future[Any]] = {}
        self._job_types: dict[str, str] = {}

    def submit(self, job_id: str, job_type: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        future = self._executor.submit(self._run_job, job_id, fn, *args, **kwargs)
        with self._lock:
            self._futures[job_id] = future
            self._job_types[job_id] = job_type
        future.add_done_callback(lambda done: self._on_done(job_id, done))

    def stats(self) -> dict[str, Any]:
        with self._lock:
            futures = dict(self._futures)
            job_types = dict(self._job_types)

        running = [job_id for job_id, future in futures.items() if future.running()]
        queued = [job_id for job_id, future in futures.items() if not future.running() and not future.done()]
        return {
            "max_workers": self.max_workers,
            "running": running,
            "queued": queued,
            "running_count": len(running),
            "queued_count": len(queued),
            "known_jobs": [
                {"job_id": job_id, "job_type": job_types.get(job_id), "state": self._future_state(future)}
                for job_id, future in futures.items()
            ],
        }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    @staticmethod
    def _future_state(future: Future[Any]) -> str:
        if future.running():
            return "running"
        if future.cancelled():
            return "cancelled"
        if future.done():
            return "done"
        return "queued"

    def _run_job(self, job_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(job_id, *args, **kwargs)
        except Exception:
            job_store.update(
                job_id,
                status="failed",
                progress=100,
                message="Job failed before completion",
                error=traceback.format_exc(),
            )
            raise

    def _on_done(self, job_id: str, future: Future[Any]) -> None:
        if future.cancelled():
            job_store.update(job_id, status="failed", message="Job was cancelled")
        with self._lock:
            self._futures.pop(job_id, None)
            self._job_types.pop(job_id, None)


job_queue = JobQueue()
atexit.register(job_queue.shutdown)
