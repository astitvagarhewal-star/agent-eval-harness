"""Tests for the grader, failure taxonomy, and telemetry store (Milestone 3).

Covers: hidden-test isolation, unbiased pass@k math, and SQLite query accuracy.
Runs offline with no external services.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from benchmarks.schema import TaskSpec
from harness.agent.loop import AgentStep, Trajectory
from harness.grader import Grader, RunResult, classify_failure
from harness.sandbox import SandboxManager
from harness.telemetry import TelemetryStore, RunRecord


TMP_TASK_ID = "_telemetry_tmp_task"


def _make_tmp_task(passed: bool) -> Path:
    """Create a scratch task dir with a lib and a hidden test, return its path."""
    task_dir = Path("benchmarks") / "tasks" / TMP_TASK_ID
    if task_dir.exists():
        shutil.rmtree(task_dir)
    (task_dir / "tests").mkdir(parents=True)
    lib = task_dir / "mylib.py"
    lib.write_text("def add(a, b):\n    return a + b\n" if passed
                   else "def add(a, b):\n    return a - b\n", encoding="utf-8")
    (task_dir / "tests" / "hidden_test.py").write_text(
        "import sys\nfrom mylib import add\n\nassert add(2, 3) == 5, 'add broken'\nprint('hidden test passed')\n",
        encoding="utf-8",
    )
    return task_dir


def _remove_tmp_task() -> None:
    task_dir = Path("benchmarks") / "tasks" / TMP_TASK_ID
    shutil.rmtree(task_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Grader: hidden-test isolation & test execution
# --------------------------------------------------------------------------- #
def test_grader_restores_hidden_tests_and_passes():
    task_dir = _make_tmp_task(passed=True)
    try:
        task = TaskSpec(
            id=TMP_TASK_ID,
            prompt="fix add",
            test_cmd="PYTHONPATH=. python tests/hidden_test.py",
            hidden_tests=["tests/hidden_test.py"],
            timeout_sec=60,
        )
        with SandboxManager(max_concurrent_sandboxes=1) as mgr:
            sid = mgr.create(TMP_TASK_ID)
            # Seed the library (as the agent would have produced).
            mgr.exec(sid, "cp /dev/null mylib.py 2>/dev/null; printf 'def add(a, b):\\n    return a + b\\n' > mylib.py")
            traj = Trajectory()
            traj.steps.append(AgentStep(
                step=1, timestamp=time.time(), model="test",
                content="done", finish_reason="stop",
            ))
            grader = Grader(mgr)
            result = grader.grade(task, sid, traj)

        assert result.passed is True
        assert result.pass1 is True
        assert result.hidden_tests_restored is True
        assert result.failed_count == 0
        assert result.total >= 1
    finally:
        _remove_tmp_task()


def test_grader_detects_test_regression():
    task_dir = _make_tmp_task(passed=False)
    try:
        task = TaskSpec(
            id=TMP_TASK_ID,
            prompt="fix add",
            test_cmd="PYTHONPATH=. python tests/hidden_test.py",
            hidden_tests=["tests/hidden_test.py"],
            timeout_sec=60,
        )
        with SandboxManager(max_concurrent_sandboxes=1) as mgr:
            sid = mgr.create(TMP_TASK_ID)
            mgr.exec(sid, "printf 'def add(a, b):\\n    return a - b\\n' > mylib.py")
            traj = Trajectory()
            traj.steps.append(AgentStep(
                step=1, timestamp=time.time(), model="test",
                content="done", finish_reason="stop",
            ))
            grader = Grader(mgr)
            result = grader.grade(task, sid, traj)

        assert result.passed is False
        assert result.failed_count >= 1
        assert result.failure == "test_regression"
    finally:
        _remove_tmp_task()


def test_grader_hidden_test_isolation():
    """The agent's trajectory must not reference hidden tests; grader restores
    them only after the agent finishes."""
    task_dir = _make_tmp_task(passed=True)
    try:
        task = TaskSpec(
            id=TMP_TASK_ID,
            prompt="fix add",
            test_cmd="PYTHONPATH=. python tests/hidden_test.py",
            hidden_tests=["tests/hidden_test.py"],
            timeout_sec=60,
        )
        with SandboxManager(max_concurrent_sandboxes=1) as mgr:
            sid = mgr.create(TMP_TASK_ID)
            mgr.exec(sid, "printf 'def add(a, b):\\n    return a + b\\n' > mylib.py")
            # The agent trajectory only touches mylib.py — never hidden tests.
            traj = Trajectory()
            traj.steps.append(AgentStep(
                step=1, timestamp=time.time(), model="test",
                content="edited mylib.py",
                tool_call={"name": "apply_patch", "arguments": {"unified_diff": ""}},
                tool_result={"ok": True, "output": "applied", "error": None},
                finish_reason="tool_calls",
            ))
            traj.steps.append(AgentStep(
                step=2, timestamp=time.time(), model="test",
                content="done", finish_reason="stop",
            ))
            grader = Grader(mgr)
            result = grader.grade(task, sid, traj)

            # Hidden test file now present in the sandbox (restored by grader).
            handle = mgr._get(sid)
            repo = handle.worktree_path or handle.root
            assert (repo / "tests" / "hidden_test.py").exists()

        assert result.passed is True
        assert result.hidden_tests_restored is True
    finally:
        _remove_tmp_task()


# --------------------------------------------------------------------------- #
# Failure classifier
# --------------------------------------------------------------------------- #
def test_classifier_step_limit():
    traj = Trajectory()
    traj.failure = "step_limit_exceeded"
    result = RunResult(task_id="t", model_id="m", passed=False, pass1=False)
    assert classify_failure(traj, result) == "step_limit_exceeded"


def test_classifier_api_error():
    traj = Trajectory()
    traj.failure = "api_error"
    result = RunResult(task_id="t", model_id="m", passed=False, pass1=False)
    assert classify_failure(traj, result) == "api_error"


def test_classifier_patch_syntax_error():
    traj = Trajectory()
    traj.steps.append(AgentStep(
        step=1, timestamp=time.time(), model="m",
        content="", tool_call={"name": "apply_patch", "arguments": {}},
        tool_result={"ok": False, "output": "", "error": "patch_syntax_error: bad diff"},
        finish_reason="tool_calls",
    ))
    result = RunResult(task_id="t", model_id="m", passed=False, pass1=False,
                       failure=None)
    assert classify_failure(traj, result) == "patch_syntax_error"


# --------------------------------------------------------------------------- #
# Telemetry: pass@k math & query accuracy
# --------------------------------------------------------------------------- #
def test_pass_at_k_math(tmp_path):
    store = TelemetryStore(db_path=str(tmp_path / "db.sqlite"))
    task_id, model_id = "task-a", "model-x"
    # 5 attempts, 2 passing.
    outcomes = [True, False, False, True, False]
    for i, ok in enumerate(outcomes):
        store.insert_run(RunRecord(
            run_id=f"run-{i}", task_id=task_id, model_id=model_id,
            status="passed" if ok else "failed", failure=None if ok else "test_regression",
            pass1=ok, started_at=time.time(), finished_at=time.time() + 1,
            duration_sec=1.0, total_steps=1, total_input_tokens=10,
            total_output_tokens=5, total_cost_usd=0.0, total_latency_sec=1.0,
            final_answer="", final_diff="", trajectory_json="{}",
        ))

    assert store.pass_at_k(task_id, model_id, k=1) == pytest.approx(0.4)
    assert store.pass_at_k(task_id, model_id, k=2) == pytest.approx(0.7)
    assert store.pass_at_k(task_id, model_id, k=3) == pytest.approx(0.9)
    assert store.pass_at_k(task_id, model_id, k=4) == 1.0
    assert store.pass_at_k(task_id, model_id, k=5) == 1.0
    # No runs → 0.0.
    assert store.pass_at_k("nope", model_id, k=1) == 0.0
    store.close()


def test_sqlite_query_accuracy(tmp_path):
    db = str(tmp_path / "db.sqlite")
    store = TelemetryStore(db_path=db)
    store.insert_run(RunRecord(
        run_id="run-1", task_id="task-a", model_id="model-x",
        status="passed", failure=None, pass1=True,
        started_at=100.0, finished_at=101.0, duration_sec=1.0,
        total_steps=3, total_input_tokens=50, total_output_tokens=20,
        total_cost_usd=0.005, total_latency_sec=0.5,
        final_answer="done", final_diff="+x", trajectory_json="{}",
    ))
    store.insert_run(RunRecord(
        run_id="run-2", task_id="task-a", model_id="model-y",
        status="failed", failure="test_regression", pass1=False,
        started_at=102.0, finished_at=103.0, duration_sec=1.0,
        total_steps=2, total_input_tokens=30, total_output_tokens=10,
        total_cost_usd=0.003, total_latency_sec=0.3,
        final_answer="", final_diff="", trajectory_json="{}",
    ))

    run = store.get_run("run-1")
    assert run is not None
    assert run.task_id == "task-a"
    assert run.pass1 is True
    assert run.total_cost_usd == 0.005

    all_runs = store.list_runs()
    assert len(all_runs) == 2

    filtered = store.list_runs(model_id="model-x")
    assert len(filtered) == 1
    assert filtered[0].run_id == "run-1"

    filtered_tasks = store.list_runs(task_id="task-b")
    assert len(filtered_tasks) == 0
    store.close()


def test_run_record_roundtrip(tmp_path):
    store = TelemetryStore(db_path=str(tmp_path / "db.sqlite"))
    rec = RunRecord(
        run_id="r", task_id="t", model_id="m", status="passed", failure=None,
        pass1=True, started_at=1.0, finished_at=2.0, duration_sec=1.0,
        total_steps=1, total_input_tokens=0, total_output_tokens=0,
        total_cost_usd=0.0, total_latency_sec=0.0, final_answer="", final_diff="",
        trajectory_json="{}",
    )
    store.insert_run(rec)
    got = store.get_run("r")
    assert got.pass1 == rec.pass1
    assert got.duration_sec == rec.duration_sec
    store.close()
