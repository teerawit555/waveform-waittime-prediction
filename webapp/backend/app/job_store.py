# from __future__ import annotations

# from dataclasses import dataclass, field
# from threading import Lock
# from typing import Any


# @dataclass
# class JobData:
#     job_id: str
#     job_type: str
#     status: str = "queued"
#     progress: int = 0
#     message: str = ""
#     result: dict[str, Any] | None = None
#     error: str | None = None


# class JobStore:
#     def __init__(self):
#         self._jobs: dict[str, JobData] = {}
#         self._lock = Lock()

#     def create(self, job_id: str, job_type: str):
#         with self._lock:
#             self._jobs[job_id] = JobData(job_id=job_id, job_type=job_type)

#     def update(self, job_id: str, **kwargs):
#         with self._lock:
#             job = self._jobs.get(job_id)
#             if not job:
#                 return
#             for key, value in kwargs.items():
#                 setattr(job, key, value)

#     def as_dict(self, job_id: str):
#         with self._lock:
#             job = self._jobs.get(job_id)
#             if not job:
#                 return None
#             return {
#                 "job_id": job.job_id,
#                 "job_type": job.job_type,
#                 "status": job.status,
#                 "progress": job.progress,
#                 "message": job.message,
#                 "result": job.result,
#                 "error": job.error,
#             }
#     def list_ids(self):
#         with self._lock:
#             return list(self._jobs.keys())

# job_store = JobStore()


# เปลี่ยนจาก in-memory dict → SQLite
import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent / "jobs.db"

class JobStore:
    def __init__(self):
        self._init_db()
        self._reset_stale_jobs() 

    def _reset_stale_jobs(self):
        # jobs ที่ยัง running อยู่ตอน restart = interrupted -> mark failed
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                UPDATE jobs SET status = 'failed', message = 'Server restarted'
                WHERE status IN ('running', 'queued')
            """)

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id   TEXT PRIMARY KEY,
                    job_type TEXT,
                    status   TEXT DEFAULT 'queued',
                    progress INTEGER DEFAULT 0,
                    message  TEXT DEFAULT '',
                    result   TEXT DEFAULT NULL,
                    error    TEXT DEFAULT NULL
                )
            """)

    def create(self, job_id: str, job_type: str):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, job_type) VALUES (?, ?)",
                (job_id, job_type)
            )

    def update(self, job_id: str, **kwargs):
        if "result" in kwargs:
            kwargs["result"] = json.dumps(kwargs["result"])
        fields = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [job_id]
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(f"UPDATE jobs SET {fields} WHERE job_id = ?", values)

    def as_dict(self, job_id: str):
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "job_id":   row[0],
            "job_type": row[1],
            "status":   row[2],
            "progress": row[3],
            "message":  row[4],
            "result":   json.loads(row[5]) if row[5] else None,
            "error":    row[6],
        }

    def list_ids(self):
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT job_id FROM jobs").fetchall()
        return [r[0] for r in rows]

job_store = JobStore()