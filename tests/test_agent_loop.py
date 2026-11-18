"""Tests for the agent loop (Milestone 2 verification gate).

Uses a local, in-memory FastAPI mock OpenAI-compatible server so the suite runs
offline without spending API credits. Verifies tool execution, malformed-patch
recovery, step-budget enforcement, and full JSON trajectory logging.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI

from harness.agent import AgentLoop, ModelClient, ModelRegistry, ModelSpec
from harness.sandbox import SandboxManager


# --------------------------------------------------------------------------- #
# Mock OpenAI-compatible server
# --------------------------------------------------------------------------- #
def make_mock_server(scenario: str = "happy") -> FastAPI:
    app = FastAPI()
    state = {"calls": 0}

    @app.post("/chat/completions")
    async def completions(req: dict):  # noqa: ARG001
        state["calls"] += 1
        n = state["calls"]

        if scenario == "happy":
            if n == 1:
                return _tool_response("read_file", {"path": "foo.py"})
            if n == 2:
                diff = (
                    "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
                )
                return _tool_response("apply_patch", {"unified_diff": diff})
            return _final("fixed it")

        if scenario == "malformed_then_fix":
            if n == 1:
                return _tool_response("apply_patch", {"unified_diff": "not a diff"})
            if n == 2:
                diff = (
                    "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
                )
                return _tool_response("apply_patch", {"unified_diff": diff})
            return _final("fixed it")

        if scenario == "loop_forever":
            # Always requests the same tool call -> thrash detection trips.
            return _tool_response("read_file", {"path": "foo.py"})

        return _final("done")

    return app


def _tool_response(name: str, arguments: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{len(name)}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _final(content: str) -> dict:
    return {
        "choices": [
            {
                "message": {"content": content, "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    }


def _registry() -> ModelRegistry:
    registry = ModelRegistry.__new__(ModelRegistry)
    registry._models = {}
    registry.register(
        ModelSpec(
            id="mock-model",
            label="Mock",
            role="test",
            endpoint="http://mockserver",
            api_key_env="NOPE_KEY",
            pricing={"input_per_1m": 0.0, "output_per_1m": 0.0},
            rate_limit={"rpm": 1000},
        )
    )
    return registry


async def make_client(scenario: str = "happy") -> ModelClient:
    app = make_mock_server(scenario)
    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://mockserver")
    return ModelClient(registry=_registry(), http_client=http)


@pytest.fixture()
def sandbox(tmp_path):
    with SandboxManager(work_root=tmp_path, max_concurrent_sandboxes=2) as mgr:
        sid = mgr.create("agent-task")
        mgr.exec(sid, "printf 'x = 1\\n' > foo.py")
        mgr.exec(sid, "git add -A && git -c user.email=t@t -c user.name=t commit -q -m baseline")
        yield mgr, sid


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_happy_path_tool_execution(sandbox):
    mgr, sid = sandbox
    client = await make_client("happy")
    try:
        loop = AgentLoop(client, mgr, sid, max_steps=10, model_id="mock-model")
        traj = await loop.run("Fix foo.py so x equals 2.")
    finally:
        await client.aclose()

    assert traj.failure is None
    assert traj.final_answer == "fixed it"
    assert traj.total_steps >= 2
    assert "+x = 2" in traj.final_diff
    json.loads(traj.to_json())
    assert len(traj.steps) >= 2


@pytest.mark.asyncio
async def test_malformed_patch_recovery(sandbox):
    mgr, sid = sandbox
    client = await make_client("malformed_then_fix")
    try:
        loop = AgentLoop(client, mgr, sid, max_steps=10, model_id="mock-model")
        traj = await loop.run("Fix foo.py.")
    finally:
        await client.aclose()

    assert traj.failure is None
    assert traj.final_answer == "fixed it"
    assert "+x = 2" in traj.final_diff


@pytest.mark.asyncio
async def test_step_budget_enforcement(sandbox):
    mgr, sid = sandbox
    client = await make_client("happy")
    try:
        loop = AgentLoop(client, mgr, sid, max_steps=2, model_id="mock-model")
        traj = await loop.run("Fix foo.py.")
    finally:
        await client.aclose()

    assert traj.failure == "step_limit_exceeded"
    assert traj.truncated is True
    json.loads(traj.to_json())


@pytest.mark.asyncio
async def test_thrash_detection(sandbox):
    mgr, sid = sandbox
    client = await make_client("loop_forever")
    try:
        loop = AgentLoop(client, mgr, sid, max_steps=20, model_id="mock-model")
        traj = await loop.run("Fix foo.py.")
    finally:
        await client.aclose()

    assert traj.failure == "step_limit_exceeded"
    assert traj.truncated is True


@pytest.mark.asyncio
async def test_trajectory_json_logging(sandbox):
    mgr, sid = sandbox
    client = await make_client("happy")
    try:
        loop = AgentLoop(client, mgr, sid, max_steps=10, model_id="mock-model")
        traj = await loop.run("Fix foo.py.")
    finally:
        await client.aclose()

    payload = traj.to_dict()
    assert "steps" in payload
    assert "final_answer" in payload
    assert "final_diff" in payload
    assert "total_cost_usd" in payload
    assert "total_latency_sec" in payload
    for s in payload["steps"]:
        if s["tool_call"] is not None:
            assert s["tool_result"] is not None
            assert "ok" in s["tool_result"]
