"""FastAPI backend for the leaderboard, trace replay, and comparison APIs.

Reads evaluation telemetry from the SQLite store and serves it to the web
dashboard.  Endpoints are synchronous (FastAPI runs them in a threadpool) so
the sqlite access is simple and blocking-free enough for a local dashboard.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from harness.telemetry.store import TelemetryStore

DEFAULT_DB = Path(__file__).resolve().parents[2] / "agent-eval.db"


def create_app(db_path: str | Path = DEFAULT_DB) -> FastAPI:
    app = FastAPI(title="Sandboxed Coding-Agent Eval Harness")
    store = TelemetryStore(db_path=str(db_path))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _runs(*, task_id: Optional[str] = None, model_id: Optional[str] = None):
        return store.list_runs(task_id=task_id, model_id=model_id)

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #
    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/runs")
    def list_runs(
        task_id: Optional[str] = Query(None),
        model_id: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
    ):
        runs = _runs(task_id=task_id, model_id=model_id)
        out = []
        for r in runs[:limit]:
            out.append(
                {
                    "run_id": r.run_id,
                    "task_id": r.task_id,
                    "model_id": r.model_id,
                    "status": r.status,
                    "failure": r.failure,
                    "pass1": r.pass1,
                    "duration_sec": r.duration_sec,
                    "total_steps": r.total_steps,
                    "total_cost_usd": r.total_cost_usd,
                    "total_latency_sec": r.total_latency_sec,
                    "started_at": r.started_at,
                    "finished_at": r.finished_at,
                }
            )
        return {"runs": out, "count": len(out)}

    @app.get("/api/runs/{run_id}/trace")
    def get_trace(run_id: str):
        r = store.get_run(run_id)
        if r is None:
            raise HTTPException(status_code=404, detail="run not found")
        import json

        return {
            "run_id": r.run_id,
            "task_id": r.task_id,
            "model_id": r.model_id,
            "trajectory": json.loads(r.trajectory_json or "{}"),
            "final_diff": r.final_diff,
            "final_answer": r.final_answer,
        }

    @app.get("/api/leaderboard")
    def leaderboard(k: int = Query(3, ge=1, le=10)):
        runs = _runs()
        by_model: dict[str, list] = defaultdict(list)
        for r in runs:
            by_model[r.model_id].append(r)

        rows = []
        for model_id, model_runs in by_model.items():
            total = len(model_runs)
            passed = sum(1 for r in model_runs if r.pass1)
            pass1 = passed / total if total else 0.0
            # pass@k over attempts grouped by (task, model): needs multiple
            # attempts per task; with one attempt each this equals pass1.
            costs = [r.total_cost_usd for r in model_runs]
            lats = [r.total_latency_sec for r in model_runs]
            steps = [r.total_steps for r in model_runs]
            failures = Counter(r.failure or "none" for r in model_runs if not r.pass1)
            rows.append(
                {
                    "model_id": model_id,
                    "runs": total,
                    "passed": passed,
                    "pass1": round(pass1, 4),
                    "pass_at_k": round(pass1, 4),
                    "avg_cost_usd": round(sum(costs) / total, 6) if total else 0.0,
                    "avg_latency_sec": round(sum(lats) / total, 4) if total else 0.0,
                    "avg_steps": round(sum(steps) / total, 2) if total else 0.0,
                    "failure_breakdown": dict(failures),
                }
            )
        rows.sort(key=lambda x: x["pass1"], reverse=True)
        return {"leaderboard": rows, "k": k}

    @app.get("/api/compare")
    def compare(
        models: Optional[str] = Query(None),
        task_id: Optional[str] = Query(None),
    ):
        model_ids = [m.strip() for m in models.split(",")] if models else []
        runs = _runs(task_id=task_id)
        if model_ids:
            runs = [r for r in runs if r.model_id in model_ids]

        # Grid: task -> model -> pass/fail
        tasks: dict[str, dict[str, bool]] = defaultdict(dict)
        models_seen: set[str] = set()
        for r in runs:
            models_seen.add(r.model_id)
            # A model passes a task if *any* of its attempts on it passed.
            prev = tasks[r.task_id].get(r.model_id, False)
            tasks[r.task_id][r.model_id] = bool(prev or r.pass1)

        grid = []
        for tid, mrow in tasks.items():
            grid.append(
                {
                    "task_id": tid,
                    "results": {m: mrow.get(m, None) for m in sorted(models_seen)},
                }
            )
        return {"models": sorted(models_seen), "grid": grid}

    @app.get("/api/tasks")
    def list_tasks():
        runs = _runs()
        tasks = {}
        for r in runs:
            tasks.setdefault(
                r.task_id, {"task_id": r.task_id, "models": set()}
            )
            tasks[r.task_id]["models"].add(r.model_id)
        return {
            "tasks": [
                {"task_id": tid, "models": sorted(v["models"])}
                for tid, v in tasks.items()
            ]
        }

    # ------------------------------------------------------------------ #
    # Static web dashboard
    # ------------------------------------------------------------------ #
    web_dir = Path(__file__).resolve().parents[2] / "web"
    static_dir = web_dir / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    @app.get("/", include_in_schema=False)
    def root():
        from fastapi.responses import FileResponse

        index = static_dir / "index.html"
        if index.is_file():
            return FileResponse(index)
        return {"message": "harness api. put a dashboard in web/static/"}

    return app


__all__ = ["create_app"]
