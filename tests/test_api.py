"""API tests for the Milestone 5 FastAPI backend.

Seeds a local SQLite telemetry store and exercises the leaderboard, runs,
trace, compare, and tasks endpoints with FastAPI's TestClient (offline).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness.server import create_app
from harness.telemetry.store import TelemetryStore, RunRecord, StepRecord


def _seed(tmp_path: Path):
    db = str(tmp_path / "eval.db")
    store = TelemetryStore(db_path=db)

    now = time.time()
    # model-x: 2 runs on task-a (1 pass, 1 fail), 1 run on task-b (pass)
    store.persist_trajectory(
        run_id="run-1", task_id="task-a", model_id="model-x", status="passed",
        failure=None, pass1=True, started_at=now, finished_at=now + 1,
        duration_sec=1.0, total_steps=3, total_input_tokens=50, total_output_tokens=20,
        total_cost_usd=0.005, total_latency_sec=0.5, final_answer="done",
        final_diff="+x\n-y", trajectory_json=json.dumps({
            "steps": [
                {"step": 1, "timestamp": now, "model": "model-x", "content": "thinking",
                 "tool_call": {"name": "read_file", "arguments": {"path": "foo.py"}},
                 "tool_result": {"ok": True, "output": "x = 1\n", "error": None},
                 "finish_reason": "tool_calls", "latency_sec": 0.2},
                {"step": 2, "timestamp": now + 0.2, "model": "model-x", "content": "done",
                 "tool_call": None, "tool_result": None,
                 "finish_reason": "stop", "latency_sec": 0.1},
            ],
            "final_answer": "done",
        }),
        steps=[{
            "step": 1, "timestamp": now, "model": "model-x", "content": "thinking",
            "tool_call_name": "read_file", "tool_call_arguments": json.dumps({"path": "foo.py"}),
            "tool_ok": True, "tool_output": "x = 1\n", "tool_error": None,
            "finish_reason": "tool_calls", "latency_sec": 0.2,
        }],
    )
    store.persist_trajectory(
        run_id="run-2", task_id="task-a", model_id="model-x", status="failed",
        failure="test_regression", pass1=False, started_at=now, finished_at=now + 1,
        duration_sec=1.0, total_steps=5, total_input_tokens=80, total_output_tokens=30,
        total_cost_usd=0.008, total_latency_sec=0.8, final_answer="",
        final_diff="", trajectory_json="{}", steps=[],
    )
    store.persist_trajectory(
        run_id="run-3", task_id="task-b", model_id="model-x", status="passed",
        failure=None, pass1=True, started_at=now, finished_at=now + 1,
        duration_sec=1.0, total_steps=2, total_input_tokens=40, total_output_tokens=10,
        total_cost_usd=0.003, total_latency_sec=0.3, final_answer="done",
        final_diff="", trajectory_json="{}", steps=[],
    )
    # model-y: 1 run on task-a (fail, api_error)
    store.persist_trajectory(
        run_id="run-4", task_id="task-a", model_id="model-y", status="failed",
        failure="api_error", pass1=False, started_at=now, finished_at=now + 1,
        duration_sec=1.0, total_steps=1, total_input_tokens=10, total_output_tokens=5,
        total_cost_usd=0.0, total_latency_sec=0.1, final_answer="",
        final_diff="", trajectory_json="{}", steps=[],
    )
    store.close()
    return db


@pytest.fixture()
def client(tmp_path):
    db = _seed(tmp_path)
    app = create_app(db_path=db)
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_leaderboard(client):
    r = client.get("/api/leaderboard")
    assert r.status_code == 200
    data = r.json()
    ids = [row["model_id"] for row in data["leaderboard"]]
    assert "model-x" in ids and "model-y" in ids
    mx = next(row for row in data["leaderboard"] if row["model_id"] == "model-x")
    assert mx["runs"] == 3
    assert mx["passed"] == 2
    assert mx["pass1"] == pytest.approx(2 / 3, abs=1e-3)
    assert mx["avg_cost_usd"] == pytest.approx((0.005 + 0.008 + 0.003) / 3, abs=1e-4)


def test_runs(client):
    r = client.get("/api/runs")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 4
    by_id = {x["run_id"]: x for x in data["runs"]}
    assert by_id["run-1"]["task_id"] == "task-a"
    assert by_id["run-1"]["model_id"] == "model-x"
    assert by_id["run-1"]["pass1"] is True

    # filter by model
    r = client.get("/api/runs?model_id=model-y")
    assert r.json()["count"] == 1


def test_trace(client):
    r = client.get("/api/runs/run-1/trace")
    assert r.status_code == 200
    data = r.json()
    assert data["run_id"] == "run-1"
    assert data["model_id"] == "model-x"
    steps = data["trajectory"]["steps"]
    assert len(steps) == 2
    assert steps[0]["tool_call"]["name"] == "read_file"
    assert data["final_diff"] == "+x\n-y"

    # missing run -> 404
    r = client.get("/api/runs/nope/trace")
    assert r.status_code == 404


def test_compare(client):
    r = client.get("/api/compare?models=model-x,model-y&task=task-a")
    assert r.status_code == 200
    data = r.json()
    assert set(data["models"]) == {"model-x", "model-y"}
    task_a = next(g for g in data["grid"] if g["task_id"] == "task-a")
    assert task_a["results"]["model-x"] is True   # run-1 passed
    assert task_a["results"]["model-y"] is False  # run-4 failed


def test_tasks(client):
    r = client.get("/api/tasks")
    assert r.status_code == 200
    ids = {t["task_id"] for t in r.json()["tasks"]}
    assert ids == {"task-a", "task-b"}
