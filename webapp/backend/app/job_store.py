# เปลี่ยนจาก in-memory dict to SQLite
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "jobs.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

class JobStore:
    def __init__(self):
        self._init_db()
        self._reset_stale_jobs() 

    def _reset_stale_jobs(self):
        # jobs ที่ยัง running อยู่ตอน restart = interrupted -> mark failed
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                UPDATE jobs
                SET status = 'failed', message = 'Server restarted',
                    finished_at = COALESCE(finished_at, ?)
                WHERE status IN ('running', 'queued')
            """, (_utc_now(),))

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
                    error    TEXT DEFAULT NULL,
                    created_at  TEXT DEFAULT NULL,
                    started_at  TEXT DEFAULT NULL,
                    finished_at TEXT DEFAULT NULL
                )
            """)

            # Migrate existing local databases without deleting job history.
            existing_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            for column_name in ("created_at", "started_at", "finished_at"):
                if column_name not in existing_columns:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {column_name} TEXT DEFAULT NULL")

    def create(self, job_id: str, job_type: str):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, job_type, created_at) VALUES (?, ?, ?)",
                (job_id, job_type, _utc_now())
            )

    def update(self, job_id: str, **kwargs):
        if "result" in kwargs:
            kwargs["result"] = json.dumps(kwargs["result"])

        with sqlite3.connect(DB_PATH) as conn:
            status = kwargs.get("status")
            timestamps = conn.execute(
                "SELECT started_at, finished_at FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if timestamps:
                if status == "running" and not timestamps[0]:
                    kwargs["started_at"] = _utc_now()
                if status in {"completed", "failed"} and not timestamps[1]:
                    kwargs["finished_at"] = _utc_now()

            fields = ", ".join(f"{k} = ?" for k in kwargs)
            values = list(kwargs.values()) + [job_id]
            conn.execute(f"UPDATE jobs SET {fields} WHERE job_id = ?", values)

    def as_dict(self, job_id: str):
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if not row:
            return None

        created_at = row["created_at"]
        started_at = row["started_at"]
        finished_at = row["finished_at"]
        # User-facing runtime includes any time spent waiting in the job queue.
        start_time = _parse_timestamp(created_at or started_at)
        end_time = _parse_timestamp(finished_at) or datetime.now(timezone.utc)
        elapsed_seconds = None
        if start_time is not None:
            elapsed_seconds = max(0.0, (end_time - start_time).total_seconds())

        return {
            "job_id":   row["job_id"],
            "job_type": row["job_type"],
            "status":   row["status"],
            "progress": row["progress"],
            "message":  row["message"],
            "result":   json.loads(row["result"]) if row["result"] else None,
            "error":    row["error"],
            "created_at": created_at,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed_seconds,
        }

    def list_ids(self):
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT job_id FROM jobs").fetchall()
        return [r[0] for r in rows]

job_store = JobStore()
