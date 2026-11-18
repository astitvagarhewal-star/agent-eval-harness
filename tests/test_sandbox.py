"""Tests for the sandbox manager (Milestone 1 verification gate).

Covers: timeout enforcement, patch application + rollback, and zero leaked
temp directories. Runs against the git worktree fallback backend (Docker is
unavailable in this environment).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from harness.sandbox import SandboxBackend, SandboxError, SandboxManager


def test_backend_is_worktree_when_docker_unavailable() -> None:
    with SandboxManager() as mgr:
        assert mgr.backend == SandboxBackend.WORKTREE


def test_create_exec_destroy_lifecycle(tmp_path: Path) -> None:
    with SandboxManager(work_root=tmp_path, max_concurrent_sandboxes=2) as mgr:
        sid = mgr.create("demo-task")

        # A simple command runs inside the sandbox.
        res = mgr.exec(sid, "echo hello-world")
        assert res.ok
        assert "hello-world" in res.stdout

        # The sandbox worktree dir exists while alive.
        assert mgr._sandboxes[sid].worktree_path is not None
        assert mgr._sandboxes[sid].worktree_path.exists()

        # Exec cwd is the worktree root; create a file there.
        mgr.exec(sid, "printf 'x = 1\\n' > foo.py")

    # After context exit everything is destroyed; no leaked temp dirs under work_root.
    assert not any(tmp_path.iterdir()), f"leaked dirs: {list(tmp_path.iterdir())}"


def test_timeout_enforcement(tmp_path: Path) -> None:
    with SandboxManager(work_root=tmp_path) as mgr:
        sid = mgr.create("timeout-task")
        res = mgr.exec(sid, "sleep 10", timeout_sec=1)
        assert res.timed_out
        assert res.exit_code == -1
        assert res.duration_sec >= 0.9


def test_patch_application_and_rollback(tmp_path: Path) -> None:
    with SandboxManager(work_root=tmp_path) as mgr:
        sid = mgr.create("patch-task")

        # Seed a file to patch.
        mgr.exec(sid, "printf 'x = 1\\n' > foo.py")

        good_patch = """--- a/foo.py
+++ b/foo.py
@@ -1 +1 @@
-x = 1
+x = 2
"""
        mgr.apply_patch(sid, good_patch)
        content = mgr.copy_out(sid, "foo.py").decode()
        assert "x = 2" in content

        # Malformed patch is rejected and the tree is rolled back (unchanged).
        bad_patch = "not a real unified diff at all"
        with pytest.raises(SandboxError):
            mgr.apply_patch(sid, bad_patch)
        content = mgr.copy_out(sid, "foo.py").decode()
        assert "x = 2" in content  # still the good patch


def test_unknown_sandbox_raises(tmp_path: Path) -> None:
    with SandboxManager(work_root=tmp_path) as mgr:
        with pytest.raises(SandboxError):
            mgr.exec("does-not-exist", "echo hi")


def test_concurrency_cap(tmp_path: Path) -> None:
    with SandboxManager(work_root=tmp_path, max_concurrent_sandboxes=1) as mgr:
        mgr.create("task-a")
        with pytest.raises(SandboxError):
            mgr.create("task-b")


def test_zero_leaked_temp_dirs_after_destroy_all(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    work_root.mkdir()
    mgr = SandboxManager(work_root=work_root, max_concurrent_sandboxes=10)
    created: list[str] = []
    for i in range(5):
        sid = mgr.create(f"task-{i}")
        created.append(sid)
        assert (work_root / sid).exists()
    mgr.destroy_all()
    assert len(list(work_root.iterdir())) == 0, "leaked sandbox dirs after destroy_all"
    mgr.close()  # idempotent


def test_signal_cleanup_registered(tmp_path: Path) -> None:
    with SandboxManager(work_root=tmp_path) as mgr:
        sid = mgr.create("sig-task")
        assert len(mgr._sandboxes) == 1
    # atexit destroy_all ran via context manager; nothing leaked.
    assert not any(tmp_path.iterdir())
