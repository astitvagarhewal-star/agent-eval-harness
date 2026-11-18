"""Agent loop with loop/thrash detection and full JSON trajectory logging.

The loop drives a model through the five sandbox tools until it either
produces a final answer (no more tool calls) or hits a budget/failure bound.
Every step is recorded to a :class:`Trajectory` for replay & grading.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from harness.sandbox import SandboxError, SandboxManager
from .client import LLMResponse, ModelClient
from .tools import ToolRegistry, ToolCall, ToolResult


# OpenAI-compatible function schemas for the five core tools.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories under a path in the sandbox, up to max_depth.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default '.')."},
                    "max_depth": {"type": "integer", "description": "Max recursion depth (default 3)."},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the sandbox with optional line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path."},
                    "start_line": {"type": "integer", "description": "First line (1-based)."},
                    "end_line": {"type": "integer", "description": "Last line (inclusive)."},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_repo",
            "description": "Search the sandbox for a regex pattern across files matching a glob.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern."},
                    "glob": {"type": "string", "description": "File glob (default '*')."},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified diff to the sandbox. Validated before applying.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unified_diff": {"type": "string", "description": "Unified diff text."},
                },
                "required": ["unified_diff"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a sandboxed shell command (output truncated).",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command."},
                    "timeout_sec": {"type": "integer", "description": "Timeout in seconds."},
                    "cwd": {"type": "string", "description": "Working directory."},
                },
                "required": ["cmd"],
                "additionalProperties": False,
            },
        },
    },
]


SYSTEM_PROMPT = """You are a software-engineering agent working inside an isolated sandbox.

Your job is to modify the repository so that its hidden tests pass. You have
five tools: list_files, read_file, search_repo, apply_patch, run_command.

Work iteratively: explore the repo, find the bug, write a fix, apply it as a
unified diff, and run the tests to verify. When you are done, stop calling
tools and give a brief final answer describing the change.

Rules:
* Prefer apply_patch with a unified diff over run_command for edits.
* Validate your patch by running the repo's test command.
* Do not attempt to edit or read hidden evaluation tests.
* Stop when the tests pass or you have no productive steps left.
"""


@dataclass
class AgentStep:
    step: int
    timestamp: float
    model: str
    content: str
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    finish_reason: str = ""
    latency_sec: float = 0.0


@dataclass
class Trajectory:
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    final_diff: str = ""
    total_steps: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_sec: float = 0.0
    truncated: bool = False
    failure: Optional[str] = None  # taxonomy tag if the run failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [
                {
                    "step": s.step,
                    "timestamp": s.timestamp,
                    "model": s.model,
                    "content": s.content,
                    "tool_call": (
                        {"name": s.tool_call.name, "arguments": s.tool_call.arguments}
                        if s.tool_call else None
                    ),
                    "tool_result": (
                        {"ok": s.tool_result.ok, "output": s.tool_result.output,
                         "error": s.tool_result.error}
                        if s.tool_result else None
                    ),
                    "finish_reason": s.finish_reason,
                    "latency_sec": s.latency_sec,
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "final_diff": self.final_diff,
            "total_steps": self.total_steps,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_latency_sec": self.total_latency_sec,
            "truncated": self.truncated,
            "failure": self.failure,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class AgentLoop:
    """Runs the model through the tool loop inside a sandbox."""

    def __init__(
        self,
        client: ModelClient,
        sandbox_manager: SandboxManager,
        sandbox_id: str,
        *,
        max_steps: int = 30,
        model_id: str = "",
    ) -> None:
        self.client = client
        self.mgr = sandbox_manager
        self.sandbox_id = sandbox_id
        self.max_steps = max_steps
        self.model_id = model_id
        self.tools = ToolRegistry(sandbox_manager, sandbox_id)

    # -- main entry ------------------------------------------------------- #
    async def run(self, prompt: str) -> Trajectory:
        traj = Trajectory()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        recent_calls: list[tuple[str, str]] = []  # (name, args-json) for thrash detection

        for step in range(1, self.max_steps + 1):
            try:
                response: LLMResponse = await self.client.complete(
                    messages,
                    self.model_id,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                )
            except Exception as exc:  # api_error path
                traj.failure = "api_error"
                traj.truncated = True
                break

            traj.total_latency_sec += response.latency_sec
            traj.total_input_tokens += response.usage.input_tokens
            traj.total_output_tokens += response.usage.output_tokens
            traj.total_cost_usd += self.client.estimate_cost(
                self.model_id, response.usage.input_tokens, response.usage.output_tokens
            )

            step_rec = AgentStep(
                step=step,
                timestamp=time.time(),
                model=self.model_id,
                content=response.content or "",
                finish_reason=response.finish_reason,
                latency_sec=response.latency_sec,
            )
            traj.steps.append(step_rec)

            if not response.tool_calls:
                # Agent is done (no more tool calls).
                traj.final_answer = response.content or ""
                break

            # Execute every tool call returned this turn.
            for tc in response.tool_calls:
                name = tc.get("name", "")
                arguments = tc.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                tool_call = ToolCall(name=name, arguments=arguments)

                # Thrash detection: same exact call 3 times in a row.
                sig = (name, json.dumps(arguments, sort_keys=True))
                recent_calls.append(sig)
                if recent_calls[-3:] == [sig, sig, sig]:
                    traj.failure = "step_limit_exceeded"
                    traj.truncated = True
                    step_rec.tool_call = tool_call
                    step_rec.tool_result = ToolResult(
                        ok=False,
                        error="thrash detected: same tool call repeated 3 times",
                    )
                    traj.steps[-1] = step_rec
                    return traj

                result = self.tools.execute(name, arguments)
                step_rec.tool_call = tool_call
                step_rec.tool_result = result

                # Feed tool result back into the conversation.
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": [
                            {
                                "id": tc.get("id", f"call-{step}"),
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call-{step}"),
                        "content": result.output if result.ok else (result.error or ""),
                    }
                )

            traj.total_steps = step

        else:
            # Exhausted max_steps without a no-tool-call response.
            traj.failure = "step_limit_exceeded"
            traj.truncated = True

        # Capture final diff from the sandbox.
        try:
            diff_res = self.tools.run_command("git diff")
            if diff_res.ok:
                traj.final_diff = diff_res.output
        except Exception:  # pragma: no cover - defensive
            traj.final_diff = ""

        return traj


__all__ = [
    "AgentStep",
    "Trajectory",
    "AgentLoop",
    "TOOL_SCHEMAS",
    "SYSTEM_PROMPT",
]
