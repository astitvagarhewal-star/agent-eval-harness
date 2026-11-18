"""SQLite telemetry store (WAL mode) for runs, steps, tool calls, and metrics."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RunRecord:
    run_id: str
    task_id: str
    model_id: str
    status: str  # passed / failed / errored
    failure: Optional[str]
    pass1: bool
    started_at: float
    finished_at: float
    duration_sec: float
    total_steps: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    total_latency_sec: float
    final_answer: str
    final_diff: str
    trajectory_json: str


@dataclass
class StepRecord:
    run_id: str
    step: int
    timestamp: float
    model: str
    content: str
    tool_call_name: Optional[str]
    tool_call_arguments: Optional[str]
    tool_ok: Optional[bool]
    tool_output: Optional[str]
    tool_error: Optional[str]
    finish_reason: str
    latency_sec: float


@dataclass
class ToolCallRecord:
    run_id: str
    step: int
    name: str
    arguments: str
    ok: bool
    output: str
    error: Optional[str]


class TelemetryStore:
    """Thread-safe SQLite store for evaluation telemetry.

    Uses WAL mode for concurrent readers and a single writer. All runs/steps
    are persisted so the leaderboard & replay UI can query them later.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._local = threading.local()
        self._init_schema()

    # -- connection --------------------------------------------------------- #
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -- schema ------------------------------------------------------------- #
    def _init_schema(self) -> None:
        c = self._conn()
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id            TEXT PRIMARY KEY,
                task_id           TEXT NOT NULL,
                model_id          TEXT NOT NULL,
                status            TEXT NOT NULL,
                failure           TEXT,
                pass1             INTEGER NOT NULL,
                started_at        REAL NOT NULL,
                finished_at       REAL NOT NULL,
                duration_sec      REAL NOT NULL,
                total_steps       INTEGER NOT NULL,
                total_input_tokens INTEGER NOT NULL DEFAULT 0,
                total_output_tokens INTEGER NOT NULL DEFAULT 0,
                total_cost_usd    REAL NOT NULL DEFAULT 0.0,
                total_latency_sec REAL NOT NULL DEFAULT 0.0,
                final_answer      TEXT,
                final_diff        TEXT,
                trajectory_json   TEXT
            );
            CREATE TABLE IF NOT EXISTS steps (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id            TEXT NOT NULL,
                step              INTEGER NOT NULL,
                timestamp         REAL NOT NULL,
                model             TEXT NOT NULL,
                content           TEXT,
                tool_call_name    TEXT,
                tool_call_arguments TEXT,
                tool_ok           INTEGER,
                tool_output       TEXT,
                tool_error        TEXT,
                finish_reason     TEXT,
                latency_sec       REAL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS tool_calls (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id            TEXT NOT NULL,
                step              INTEGER NOT NULL,
                name              TEXT NOT NULL,
                arguments         TEXT NOT NULL,
                ok                INTEGER NOT NULL,
                output            TEXT,
                error             TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_runs_task_model ON runs(task_id, model_id);
            CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, step);
            CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id, step);
            """
        )
        c.commit()

    # -- write -------------------------------------------------------------- #
    def insert_run(self, run: RunRecord) -> None:
        c = self._conn()
        c.execute(
            """
            INSERT INTO runs (
                run_id, task_id, model_id, status, failure, pass1,
                started_at, finished_at, duration_sec, total_steps,
                total_input_tokens, total_output_tokens, total_cost_usd,
                total_latency_sec, final_answer, final_diff, trajectory_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id, run.task_id, run.model_id, run.status, run.failure,
                int(run.pass1), run.started_at, run.finished_at, run.duration_sec,
                run.total_steps, run.total_input_tokens, run.total_output_tokens,
                run.total_cost_usd, run.total_latency_sec, run.final_answer,
                run.final_diff, run.trajectory_json,
            ),
        )
        c.commit()

    def insert_step(self, step: StepRecord) -> None:
        c = self._conn()
        c.execute(
            """
            INSERT INTO steps (
                run_id, step, timestamp, model, content, tool_call_name,
                tool_call_arguments, tool_ok, tool_output, tool_error,
                finish_reason, latency_sec
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step.run_id, step.step, step.timestamp, step.model, step.content,
                step.tool_call_name, step.tool_call_arguments, int(step.tool_ok)
                if step.tool_ok is not None else None,
                step.tool_output, step.tool_error, step.finish_reason,
                step.latency_sec,
            ),
        )
        c.commit()

    def insert_tool_call(self, call: ToolCallRecord) -> None:
        c = self._conn()
        c.execute(
            """
            INSERT INTO tool_calls (run_id, step, name, arguments, ok, output, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (call.run_id, call.step, call.name, call.arguments, int(call.ok),
             call.output, call.error),
        )
        c.commit()

    def persist_trajectory(self, run_id: str, task_id: str, model_id: str, status: str,
                           failure: Optional[str], pass1: bool, started_at: float,
                           finished_at: float, duration_sec: float,
                           total_steps: int, total_input_tokens: int,
                           total_output_tokens: int, total_cost_usd: float,
                           total_latency_sec: float, final_answer: str,
                           final_diff: str, trajectory_json: str,
                           steps: list) -> None:
        self.insert_run(
            RunRecord(
                run_id=run_id,
                task_id=task_id,
                model_id=model_id,
                status=status,
                failure=failure,
                pass1=pass1,
                started_at=started_at,
                finished_at=finished_at,
                duration_sec=duration_sec,
                total_steps=total_steps,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_cost_usd=total_cost_usd,
                total_latency_sec=total_latency_sec,
                final_answer=final_answer,
                final_diff=final_diff,
                trajectory_json=trajectory_json,
            )
        )
        for s in steps:
            self.insert_step(
                StepRecord(
                    run_id=run_id,
                    step=s.get("step", 0),
                    timestamp=s.get("timestamp", 0.0),
                    model=s.get("model", ""),
                    content=s.get("content", ""),
                    tool_call_name=(s.get("tool_call") or {}).get("name")
                    if s.get("tool_call")
                    else None,
                    tool_call_arguments=(
                        json.dumps(s.get("tool_call") or {}).get("arguments", {})
                        if s.get("tool_call")
                        else None
                    ),
                    tool_ok=(s.get("tool_result") or {}).get("ok")
                    if s.get("tool_result")
                    else None,
                    tool_output=(s.get("tool_result") or {}).get("output")
                    if s.get("tool_result")
                    else None,
                    tool_error=(s.get("tool_result") or {}).get("error")
                    if s.get("tool_result")
                    else None,
                    finish_reason=s.get("finish_reason", ""),
                    latency_sec=s.get("latency_sec", 0.0),
                )
            )

    # -- read --------------------------------------------------------------- #
    def get_run(self, run_id: str) -> Optional[RunRecord]:
        c = self._conn()
        row = c.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_run(row) if row else None

    def list_runs(self, *, task_id: Optional[str] = None,
                  model_id: Optional[str] = None) -> list[RunRecord]:
        c = self._conn()
        sql = "SELECT * FROM runs"
        params: list = []
        wheres = []
        if task_id is not None:
            wheres.append("task_id = ?")
            params.append(task_id)
        if model_id is not None:
            wheres.append("model_id = ?")
            params.append(model_id)
        if wheres:
            sql += " WHERE " + " AND ".join(wheres)
        sql += " ORDER BY started_at DESC"
        return [self._row_to_run(r) for r in c.execute(sql, params).fetchall()]

    # -- pass@k ------------------------------------------------------------- #
    def pass_at_k(self, task_id: str, model_id: str, k: int) -> float:
        """Unbiased pass@k estimator over all runs for (task, model).

        pass@k = 1 - C(n - c, k) / C(n, k), where n = total attempts,
        c = number of passing attempts. Returns 0.0 when there are no runs.
        """
        runs = self.list_runs(task_id=task_id, model_id=model_id)
        n = len(runs)
        if n == 0 or k <= 0:
            return 0.0
        k = min(k, n)
        c = sum(1 for r in runs if r.pass1)
        if c == 0:
            return 0.0
        if k > n - c:
            return 1.0
        return 1.0 - self._comb(n - c, k) / self._comb(n, k)

    @staticmethod
    def _comb(n: int, k: int) -> int:
        if k < 0 or k > n:
            return 0
        return math.comb(n, k)

    # -- utils -------------------------------------------------------------- #
    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            task_id=row["task_id"],
            model_id=row["model_id"],
            status=row["status"],
            failure=row["failure"],
            pass1=bool(row["pass1"]),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            duration_sec=row["duration_sec"],
            total_steps=row["total_steps"],
            total_input_tokens=row["total_input_tokens"],
            total_output_tokens=row["total_output_tokens"],
            total_cost_usd=row["total_cost_usd"],
            total_latency_sec=row["total_latency_sec"],
            final_answer=row["final_answer"] or "",
            final_diff=row["final_diff"] or "",
            trajectory_json=row["trajectory_json"] or "{}",
        )

    def __enter__(self) -> "TelemetryStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["TelemetryStore", "RunRecord", "StepRecord", "ToolCallRecord"]
