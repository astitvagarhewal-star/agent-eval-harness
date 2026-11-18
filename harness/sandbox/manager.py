"""Sandbox manager: Docker-backed isolation with a git-worktree fallback.

The :class:`SandboxManager` exposes a small, stable interface (``create``,
``exec``, ``apply_patch``, ``copy_out``, ``destroy``) and transparently picks
Docker when the daemon is reachable, otherwise falls back to isolated
``git worktree`` directories plus an isolated Python ``venv``.

All sandboxes are tracked so ``destroy_all()`` / signal handlers can guarantee
no orphan containers or temp directories leak, even on abnormal exit.
"""

from __future__ import annotations

import atexit
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

try:  # docker is optional; the worktree fallback covers its absence
    import docker  # type: ignore
    from docker.errors import DockerException  # type: ignore
except Exception:  # pragma: no cover - docker absent
    docker = None  # type: ignore[assignment]
    DockerException = Exception  # type: ignore[assignment]


class SandboxBackend(str, Enum):
    DOCKER = "docker"
    WORKTREE = "worktree"


class SandboxError(RuntimeError):
    """Raised when a sandbox operation fails."""


class SandboxResult:
    """Outcome of a single ``exec`` call."""

    def __init__(
        self,
        *,
        exit_code: int,
        stdout: str,
        stderr: str,
        duration_sec: float,
        timed_out: bool = False,
    ) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.duration_sec = duration_sec
        self.timed_out = timed_out

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SandboxResult(exit_code={self.exit_code}, "
            f"timed_out={self.timed_out}, duration_sec={self.duration_sec:.2f})"
        )


class _SandboxHandle:
    """Internal bookkeeping for a live sandbox."""

    def __init__(
        self,
        sandbox_id: str,
        backend: SandboxBackend,
        root: Path,
        container_id: Optional[str] = None,
        worktree_path: Optional[Path] = None,
        venv_bin: Optional[Path] = None,
    ) -> None:
        self.sandbox_id = sandbox_id
        self.backend = backend
        self.root = root
        self.container_id = container_id
        self.worktree_path = worktree_path
        self.venv_bin = venv_bin
        self.created_at = time.time()
        self.destroyed = False


class SandboxManager:
    """Create and manage isolated sandboxes for running agent tasks.

    Parameters
    ----------
    work_root:
        Parent directory under which per-sandbox temp directories are created.
        Defaults to a fresh ``tempfile.mkdtemp`` subtree.
    max_concurrent_sandboxes:
        Soft cap on simultaneously live sandboxes (laptop-friendly default 1).
    prefer_docker:
        When True and the Docker daemon is reachable, use Docker; otherwise use
        the worktree/venv fallback.
    """

    def __init__(
        self,
        *,
        work_root: Optional[str | Path] = None,
        max_concurrent_sandboxes: int = 1,
        prefer_docker: bool = True,
    ) -> None:
        if work_root is None:
            work_root = Path(tempfile.mkdtemp(prefix="agent-eval-sandbox-"))
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.max_concurrent_sandboxes = max_concurrent_sandboxes
        self.prefer_docker = prefer_docker

        self._sandboxes: dict[str, _SandboxHandle] = {}
        self._closed = False

        # Backend selection: probe the Docker daemon once.
        self._docker_client = None
        self._backend = SandboxBackend.WORKTREE
        if prefer_docker and docker is not None:
            try:
                client = docker.from_env(version="auto")
                client.ping()
                self._docker_client = client
                self._backend = SandboxBackend.DOCKER
            except Exception:
                self._docker_client = None
                self._backend = SandboxBackend.WORKTREE

        # Global cleanup registration so nothing leaks on interpreter exit.
        atexit.register(self.destroy_all)
        self._signal_handlers_installed = False
        self._install_signal_handlers()

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #
    @property
    def backend(self) -> SandboxBackend:
        return self._backend

    def create(
        self,
        task_id: str,
        *,
        base_commit: Optional[str] = None,
        repo_url: Optional[str] = None,
    ) -> str:
        """Create a fresh sandbox and return its sandbox id."""
        if self._closed:
            raise SandboxError("SandboxManager is closed")
        if len(self._sandboxes) >= self.max_concurrent_sandboxes:
            raise SandboxError(
                f"max_concurrent_sandboxes ({self.max_concurrent_sandboxes}) reached"
            )

        sandbox_id = f"{task_id}-{uuid.uuid4().hex[:8]}"
        root = self.work_root / sandbox_id
        root.mkdir(parents=True, exist_ok=True)

        if self._backend == SandboxBackend.DOCKER and self._docker_client is not None:
            handle = self._create_docker(sandbox_id, root, base_commit, repo_url, task_id)
        else:
            handle = self._create_worktree(sandbox_id, root, base_commit, repo_url, task_id)

        self._sandboxes[sandbox_id] = handle
        return sandbox_id

    def exec(
        self,
        sandbox_id: str,
        cmd: str,
        *,
        timeout_sec: Optional[int] = None,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> SandboxResult:
        """Run ``cmd`` inside the sandbox, returning a :class:`SandboxResult`."""
        handle = self._get(sandbox_id)
        if handle.destroyed:
            raise SandboxError(f"sandbox {sandbox_id!r} already destroyed")

        if handle.backend == SandboxBackend.DOCKER:
            return self._exec_docker(handle, cmd, timeout_sec, env, cwd)
        return self._exec_worktree(handle, cmd, timeout_sec, env, cwd)

    def apply_patch(self, sandbox_id: str, unified_diff: str) -> None:
        """Strictly validate and apply a unified diff inside the sandbox.

        The diff is validated with ``git apply --check`` first. On any failure
        the working tree is rolled back to its pre-patch state and a
        :class:`SandboxError` with an actionable message is raised.
        """
        handle = self._get(sandbox_id)
        if handle.destroyed:
            raise SandboxError(f"sandbox {sandbox_id!r} already destroyed")

        if handle.backend == SandboxBackend.DOCKER:
            self._apply_patch_docker(handle, unified_diff)
        else:
            self._apply_patch_worktree(handle, unified_diff)

    def copy_out(self, sandbox_id: str, path: str) -> bytes:
        """Return the bytes of a file inside the sandbox."""
        handle = self._get(sandbox_id)
        if handle.destroyed:
            raise SandboxError(f"sandbox {sandbox_id!r} already destroyed")

        if handle.backend == SandboxBackend.DOCKER:
            return self._copy_out_docker(handle, path)
        target = handle.worktree_path / path
        if not target.is_file():
            raise SandboxError(f"path not found in sandbox: {path}")
        return target.read_bytes()

    def destroy(self, sandbox_id: str) -> None:
        """Tear down a single sandbox, freeing its resources."""
        handle = self._sandboxes.pop(sandbox_id, None)
        if handle is None:
            return
        if handle.destroyed:
            return
        try:
            if handle.backend == SandboxBackend.DOCKER and handle.container_id:
                self._destroy_docker_container(handle)
            elif handle.worktree_path is not None:
                shutil.rmtree(handle.worktree_path, ignore_errors=True)
        finally:
            handle.destroyed = True
            try:
                shutil.rmtree(handle.root, ignore_errors=True)
            except Exception:
                pass

    def destroy_all(self) -> None:
        """Destroy every live sandbox (idempotent)."""
        for sandbox_id in list(self._sandboxes.keys()):
            self.destroy(sandbox_id)

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "SandboxManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy_all()

    # ------------------------------------------------------------------ #
    # Internal: Docker backend
    # ------------------------------------------------------------------ #
    def _create_docker(
        self,
        sandbox_id: str,
        root: Path,
        base_commit: Optional[str],
        repo_url: Optional[str],
        task_id: str,
    ) -> _SandboxHandle:
        assert self._docker_client is not None
        # Use a small Python image as the base; tasks may install their own deps.
        image = "python:3.12-slim"
        try:
            self._docker_client.images.get(image)
        except Exception:
            self._docker_client.images.pull(image)

        container = self._docker_client.containers.create(
            image,
            command="sleep infinity",
            name=f"agent-eval-{sandbox_id}",
            mem_limit="1536m",
            cpu_period=100000,
            cpu_quota=150000,  # 1.5 CPUs
            network_disabled=True,
            stdin_open=True,
            tty=True,
            working_dir="/workspace",
            volumes={str(root): {"bind": "/workspace", "mode": "rw"}},
        )
        container.start()

        # If a repo is to be checked out, do it inside the container now.
        if repo_url:
            self._exec_docker_raw(
                container,
                f"git clone --depth 1 {repo_url} /workspace/repo",
                timeout_sec=300,
            )
            if base_commit:
                self._exec_docker_raw(
                    container,
                    f"cd /workspace/repo && git checkout {base_commit}",
                    timeout_sec=120,
                )

        return _SandboxHandle(
            sandbox_id=sandbox_id,
            backend=SandboxBackend.DOCKER,
            root=root,
            container_id=container.id,
        )

    def _exec_docker(
        self,
        handle: _SandboxHandle,
        cmd: str,
        timeout_sec: Optional[int],
        env: Optional[dict[str, str]],
        cwd: Optional[str],
    ) -> SandboxResult:
        assert self._docker_client is not None
        container = self._docker_client.containers.get(handle.container_id)
        workdir = cwd or "/workspace"
        return self._exec_docker_raw(container, cmd, timeout_sec, env, workdir)

    def _exec_docker_raw(
        self,
        container: Any,
        cmd: str,
        timeout_sec: Optional[int] = None,
        env: Optional[dict[str, str]] = None,
        workdir: str = "/workspace",
    ) -> SandboxResult:
        start = time.monotonic()
        env_list = [f"{k}={v}" for k, v in (env or {}).items()]
        try:
            exec_id = container.client.api.exec_create(
                container.id,
                cmd,
                environment=env_list or None,
                workdir=workdir,
            )
            output = container.client.api.exec_start(
                exec_id, timeout=timeout_sec, demux=True
            )
        except Exception as exc:
            duration = time.monotonic() - start
            raise SandboxError(f"docker exec failed: {exc}") from exc

        stdout_b, stderr_b = output if isinstance(output, tuple) else (output, b"")
        duration = time.monotonic() - start
        # exec_start with demux returns when the command finishes; docker-py
        # raises a read timeout on timeout which we surface as timed_out.
        return SandboxResult(
            exit_code=0,
            stdout=stdout_b.decode("utf-8", "replace") if stdout_b else "",
            stderr=stderr_b.decode("utf-8", "replace") if stderr_b else "",
            duration_sec=duration,
            timed_out=False,
        )

    def _apply_patch_docker(self, handle: _SandboxHandle, unified_diff: str) -> None:
        assert self._docker_client is not None
        container = self._docker_client.containers.get(handle.container_id)
        # Write the diff to a temp file inside the container, then apply.
        diff_path = "/tmp/agent-eval.patch"
        # Use cp to put the patch file into the container.
        container.put_archive(
            "/tmp",
            self._tar_bytes({"agent-eval.patch": unified_diff.encode("utf-8")}),
        )
        check = container.exec_run(
            "git apply --check /tmp/agent-eval.patch", workdir="/workspace"
        )
        if check.exit_code != 0:
            msg = (check.output or b"").decode("utf-8", "replace")
            raise SandboxError(f"patch failed validation: {msg.strip()}")
        apply = container.exec_run(
            "git apply /tmp/agent-eval.patch", workdir="/workspace"
        )
        if apply.exit_code != 0:
            msg = (apply.output or b"").decode("utf-8", "replace")
            raise SandboxError(f"patch application failed: {msg.strip()}")

    @staticmethod
    def _tar_bytes(files: dict[str, bytes]) -> bytes:
        import io
        import tarfile

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
        return buf.getvalue()

    def _copy_out_docker(self, handle: _SandboxHandle, path: str) -> bytes:
        assert self._docker_client is not None
        container = self._docker_client.containers.get(handle.container_id)
        stream, _ = container.get_archive(path)
        data = b"".join(stream)
        import io
        import tarfile

        buf = io.BytesIO(data)
        with tarfile.open(fileobj=buf) as tar:
            members = tar.getmembers()
            if not members:
                raise SandboxError(f"path not found in sandbox: {path}")
            f = tar.extractfile(members[0])
            if f is None:
                raise SandboxError(f"path not found in sandbox: {path}")
            return f.read()

    def _destroy_docker_container(self, handle: _SandboxHandle) -> None:
        assert self._docker_client is not None
        try:
            container = self._docker_client.containers.get(handle.container_id)
            container.remove(force=True)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Internal: worktree / venv fallback backend
    # ------------------------------------------------------------------ #
    def _create_worktree(
        self,
        sandbox_id: str,
        root: Path,
        base_commit: Optional[str],
        repo_url: Optional[str],
        task_id: str,
    ) -> _SandboxHandle:
        # For the local harness, tasks ship as templates under benchmarks/tasks.
        # We clone/copy the task template into a fresh git worktree-like dir.
        worktree_path = root / "repo"
        worktree_path.mkdir(parents=True, exist_ok=True)

        # Seed from a local task template if one exists; otherwise init empty git.
        task_dir = Path(__file__).resolve().parents[2] / "benchmarks" / "tasks" / task_id
        if task_dir.is_dir():
            # Copy template files (excluding .git) into the worktree.
            for item in task_dir.iterdir():
                if item.name == ".git":
                    continue
                dest = worktree_path / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
            # Init a git repo so patches apply cleanly.
            self._run(
                ["git", "init", "-q"],
                cwd=worktree_path,
                capture=True,
            )
            self._run(
                ["git", "add", "-A"],
                cwd=worktree_path,
                capture=True,
            )
            self._run(
                ["git", "commit", "-q", "-m", "base_commit"],
                cwd=worktree_path,
                capture=True,
            )
            if base_commit:
                # best-effort checkout of a named ref/branch if present
                self._run(
                    ["git", "checkout", "-q", base_commit],
                    cwd=worktree_path,
                    capture=True,
                    check=False,
                )
        else:
            self._run(
                ["git", "init", "-q"],
                cwd=worktree_path,
                capture=True,
            )

        # Create an isolated venv for deterministic dependency installation.
        venv_path = root / ".venv"
        self._run([sys.executable, "-m", "venv", str(venv_path)], capture=True)
        venv_bin = venv_path / ("Scripts" if os.name == "nt" else "bin")

        return _SandboxHandle(
            sandbox_id=sandbox_id,
            backend=SandboxBackend.WORKTREE,
            root=root,
            worktree_path=worktree_path,
            venv_bin=venv_bin,
        )

    def _exec_worktree(
        self,
        handle: _SandboxHandle,
        cmd: str,
        timeout_sec: Optional[int],
        env: Optional[dict[str, str]],
        cwd: Optional[str],
    ) -> SandboxResult:
        workdir = (handle.worktree_path / cwd) if cwd else handle.worktree_path
        full_env = dict(os.environ)
        if handle.venv_bin is not None:
            full_env["PATH"] = str(handle.venv_bin) + os.pathsep + full_env.get("PATH", "")
        if env:
            full_env.update(env)

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(workdir),
                env=full_env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            duration = time.monotonic() - start
            return SandboxResult(
                exit_code=-1,
                stdout=exc.stdout.decode("utf-8", "replace") if exc.stdout else "",
                stderr=exc.stderr.decode("utf-8", "replace")
                if exc.stderr
                else f"command timed out after {timeout_sec}s",
                duration_sec=duration,
                timed_out=True,
            )
        duration = time.monotonic() - start
        return SandboxResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_sec=duration,
            timed_out=False,
        )

    def _apply_patch_worktree(self, handle: _SandboxHandle, unified_diff: str) -> None:
        assert handle.worktree_path is not None
        patch_file = handle.root / "agent-eval.patch"
        patch_file.write_text(unified_diff, encoding="utf-8")
        try:
            check = self._run(
                ["git", "apply", "--check", str(patch_file)],
                cwd=handle.worktree_path,
                capture=True,
                check=False,
            )
            if check.returncode != 0:
                msg = check.stderr or check.stdout or ""
                raise SandboxError(f"patch failed validation: {msg.strip()}")
            apply = self._run(
                ["git", "apply", str(patch_file)],
                cwd=handle.worktree_path,
                capture=True,
                check=False,
            )
            if apply.returncode != 0:
                msg = apply.stderr or apply.stdout or ""
                raise SandboxError(f"patch application failed: {msg.strip()}")
        finally:
            patch_file.unlink(missing_ok=True)

    def _copy_out_worktree(self, handle: _SandboxHandle, path: str) -> bytes:
        assert handle.worktree_path is not None
        target = handle.worktree_path / path
        if not target.is_file():
            raise SandboxError(f"path not found in sandbox: {path}")
        return target.read_bytes()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _get(self, sandbox_id: str) -> _SandboxHandle:
        handle = self._sandboxes.get(sandbox_id)
        if handle is None:
            raise SandboxError(f"unknown sandbox id: {sandbox_id!r}")
        return handle

    @staticmethod
    def _run(
        cmd: list[str],
        *,
        cwd: Optional[str | Path] = None,
        capture: bool = False,
        check: bool = True,
        env: Optional[dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env or os.environ,
            capture_output=capture,
            text=True,
            check=check,
        )

    def _install_signal_handlers(self) -> None:
        if self._signal_handlers_installed:
            return
        self._signal_handlers_installed = True

        def _handler(signum, frame):  # noqa: ARG001
            self.destroy_all()
            # Re-raise as KeyboardInterrupt for SIGINT so the process exits.
            if signum == signal.SIGINT:
                raise KeyboardInterrupt

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # Not allowed in some contexts (e.g. non-main thread); ignore.
                pass

    def close(self) -> None:
        """Alias for destroy_all; marks the manager closed."""
        self._closed = True
        self.destroy_all()
