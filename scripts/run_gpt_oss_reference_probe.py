#!/usr/bin/env python3
"""Run a small deterministic GPT-OSS reference probe through SGLang.

This script launches an SGLang server for a GPT-OSS MLA checkpoint overlay,
loads a small set of local reference problems, and compares the extracted
integer answer against the CSV answer key.

It uses a simple text prompt derived from the showtime harness so we can
validate AIMO3-style math questions without the Kaggle/harmony runtime.
"""

from __future__ import annotations

import contextlib
import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import queue
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from jupyter_client import KernelManager
from openai_harmony import (
    Author,
    Conversation,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    TextContent,
    SystemContent,
    ToolNamespaceConfig,
    load_harmony_encoding,
)


SHOWTIME_SYSTEM_PROMPT = (
    "You are an elite mathematical problem solver with expertise at the International "
    "Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through "
    "rigorous mathematical reasoning.\n\n"
    "# Problem-Solving Approach:\n"
    "1. UNDERSTAND: Carefully read and rephrase the problem in your own words. "
    "Identify what is given, what needs to be found, and any constraints.\n"
    "2. EXPLORE: Consider multiple solution strategies. Think about relevant theorems, "
    "techniques, patterns, or analogous problems. Don't commit to one approach immediately.\n"
    "3. PLAN: Select the most promising approach and outline key steps before executing.\n"
    "4. EXECUTE: Work through your solution methodically. Show all reasoning steps clearly.\n"
    "5. VERIFY: Check your answer by substituting back, testing edge cases, or using "
    "alternative methods. Ensure logical consistency throughout.\n\n"
    "# Mathematical Reasoning Principles:\n"
    "- Break complex problems into smaller, manageable sub-problems\n"
    "- Look for patterns, symmetries, and special cases that provide insight\n"
    "- Use concrete examples to build intuition before generalizing\n"
    "- Consider extreme cases and boundary conditions\n"
    "- If stuck, try working backwards from the desired result\n"
    "- Be willing to restart with a different approach if needed\n\n"
    "# Verification Requirements:\n"
    "- Cross-check arithmetic and algebraic manipulations\n"
    "- Verify that your solution satisfies all problem constraints\n"
    "- Test your answer with simple cases or special values when possible\n"
    "- Ensure dimensional consistency and reasonableness of the result\n\n"
    "# Output Format:\n"
    "The final answer must be a non-negative integer between 0 and 99999.\n"
    "Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n"
    "Think step-by-step and show your complete reasoning process. Quality of reasoning "
    "is as important as the final answer."
)

SHOWTIME_TOOL_PROMPT = (
    "Use this tool to execute Python code for:\n"
    "- Complex calculations that would be error-prone by hand\n"
    "- Numerical verification of analytical results\n"
    "- Generating examples or testing conjectures\n"
    "- Visualizing problem structure when helpful\n"
    "- Brute-force verification for small cases\n\n"
    "The environment is a stateful Jupyter notebook. Code persists between executions.\n"
    "Always use print() to display results. Write clear, well-commented code.\n\n"
    "Remember: Code should support your mathematical reasoning, not replace it. "
    "Explain what you're computing and why before running code."
)

SHOWTIME_PREFERENCE_PROMPT = (
    "You have access to `math`, `numpy`, and `sympy` for:\n\n"
    "# Symbolic Computation (sympy):\n"
    "- Algebraic manipulation and simplification\n"
    "- Solving equations and systems of equations\n"
    "- Symbolic differentiation and integration\n"
    "- Number theory functions (primes, divisors, modular arithmetic)\n"
    "- Polynomial operations and factorization\n"
    "- Working with mathematical expressions symbolically\n\n"
    "# Numerical Computation (numpy):\n"
    "- Array operations and linear algebra\n"
    "- Efficient numerical calculations for large datasets\n"
    "- Matrix operations and eigenvalue problems\n"
    "- Statistical computations\n\n"
    "# Mathematical Functions (math):\n"
    "- Standard mathematical functions (trig, log, exp)\n"
    "- Constants like pi and e\n"
    "- Basic operations for single values\n\n"
    "Best Practices:\n"
    "- Use sympy for exact symbolic answers when possible\n"
    "- Use numpy for numerical verification and large-scale computation\n"
    "- Combine symbolic and numerical approaches: derive symbolically, verify numerically\n"
    "- Document your computational strategy clearly\n"
    "- Validate computational results against known cases or theoretical bounds"
)


class AIMO3Sandbox:
    def __init__(self, timeout: float):
        self._default_timeout = float(timeout)
        self._owns_kernel = False
        self._client = None
        self._km = None

        env = os.environ.copy()
        env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        env["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "0"
        env["JUPYTER_PLATFORM_DIRS"] = "1"
        env["PYTHONWARNINGS"] = "ignore"
        env["MPLBACKEND"] = "Agg"

        self._km = KernelManager()
        self._km.start_kernel(env=env, extra_arguments=["--Application.log_level=CRITICAL"])
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        self.execute(
            "import math\n"
            "import numpy\n"
            "import sympy\n"
            "import itertools\n"
            "import collections\n"
            "import mpmath\n"
            "mpmath.mp.dps = 64\n"
        )

    def _format_error(self, traceback: list[str]) -> str:
        clean_lines: list[str] = []
        for frame in traceback:
            clean_frame = re.sub(r"\x1b\[[0-9;]*m", "", frame)
            if 'File "' in clean_frame and "ipython-input" not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return "".join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:
        client = self._client
        if client is None:
            return "[ERROR] Jupyter client not available"

        effective_timeout = float(timeout or self._default_timeout)
        msg_id = client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > effective_timeout:
                self._km.interrupt_kernel()
                return f"[ERROR] Execution timed out after {effective_timeout} seconds"

            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_list = content.get("traceback", [])
                stderr_parts.append(self._format_error(traceback_list))
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status":
                if content.get("execution_state") == "idle":
                    break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)
        if stderr:
            return f"{stdout.rstrip()}\n{stderr}" if stdout else stderr
        return stdout if stdout.strip() else "[WARN] No output. Use print() to see results."

    def reset(self) -> None:
        self.execute(
            "%reset -f\n"
            "import math\n"
            "import numpy\n"
            "import sympy\n"
            "import itertools\n"
            "import collections\n"
            "import mpmath\n"
            "mpmath.mp.dps = 64\n"
        )

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()
        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception):
                self._km.cleanup_resources()

    def __del__(self) -> None:
        self.close()


class AIMO3Tool:
    def __init__(self, local_jupyter_timeout: float, tool_prompt: str, sandbox: AIMO3Sandbox | None = None):
        self._local_jupyter_timeout = float(local_jupyter_timeout)
        self._tool_prompt = str(tool_prompt)
        self._jupyter_session = sandbox
        self._init_lock = threading.Lock()
        self._execution_lock = threading.Lock()

    def _ensure_session(self) -> None:
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    def _ensure_last_print(self, code: str) -> str:
        lines = code.strip().split("\n")
        if not lines:
            return code
        last_line = lines[-1].strip()
        if "print" in last_line or "import" in last_line:
            return code
        if not last_line or last_line.startswith("#"):
            return code
        lines[-1] = "print(" + last_line + ")"
        return "\n".join(lines)

    @property
    def instruction(self) -> str:
        return self._tool_prompt

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(name="python", description=self.instruction, tools=[])

    def _make_response(self, output: str, channel: str | None = None) -> Message:
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name="python")
        message = Message(author=author, content=[content]).with_recipient("assistant")
        if channel:
            message = message.with_channel(channel)
        return message

    def process_sync_plus(self, message: Message) -> list[Message]:
        self._ensure_session()
        raw_script = message.content[0].text
        final_script = self._ensure_last_print(raw_script)

        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script)
            except TimeoutError as exc:
                output = f"[ERROR] {exc}"

        return [self._make_response(output, channel=message.channel)]


class AIMO3RuntimeHelpers:
    BOX_RE = re.compile(r"\\boxed\s*\{\s*([0-9,]+)\s*\}")
    FINAL_INT_RE = re.compile(r"final\s+answer\s+is\s*([0-9,]+)", re.IGNORECASE)

    @classmethod
    def scan_for_answer(cls, text: str) -> int | None:
        if not text:
            return None

        m = cls.BOX_RE.findall(text)
        if m:
            with contextlib.suppress(Exception):
                v = int(m[-1].replace(",", ""))
                if 0 <= v <= 99999:
                    return v

        m = cls.FINAL_INT_RE.findall(text)
        if m:
            with contextlib.suppress(Exception):
                v = int(m[-1].replace(",", ""))
                if 0 <= v <= 99999:
                    return v

        return None

    @staticmethod
    def make_partial_assistant_message(
        *,
        text: str,
        channel: str | None,
        recipient: str | None,
        content_type: str | None,
    ) -> Message:
        msg = Message.from_role_and_content(Role.ASSISTANT, text)
        if channel:
            msg = msg.with_channel(channel)
        if recipient:
            msg = msg.with_recipient(recipient)
        if content_type:
            with contextlib.suppress(Exception):
                msg = msg.with_content_type(content_type)
        return msg

    @classmethod
    def recover_harmony_messages(
        cls,
        *,
        encoding,
        prompt_ids: list[int],
        prompt_message_count: int,
        output_ids: list[int],
    ) -> tuple[list[Message], Message | None, str | None, str]:
        parser = StreamableParser(encoding, role=Role.ASSISTANT)
        for tok in prompt_ids:
            parser.process(int(tok))

        error_summary: str | None = None
        for tok in output_ids:
            try:
                parser.process(int(tok))
            except Exception as exc:
                error_summary = f"{type(exc).__name__}: {exc}"
                break

        partial_message: Message | None = None
        if parser.current_role == Role.ASSISTANT and parser.current_content:
            partial_message = cls.make_partial_assistant_message(
                text=str(parser.current_content),
                channel=getattr(parser, "current_channel", None),
                recipient=getattr(parser, "current_recipient", None),
                content_type=getattr(parser, "current_content_type", None),
            )

        raw_text = ""
        with contextlib.suppress(Exception):
            raw_text = str(encoding.decode([int(x) for x in output_ids]) or "")

        return list(parser.messages[prompt_message_count:]), partial_message, error_summary, raw_text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe GPT-OSS MLA checkpoints on AIMO3-style questions.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default="openai/gpt-oss-120b")
    parser.add_argument("--csv-path", default="/root/reference.csv")
    parser.add_argument("--question-ids", default="92ba6a,9c1c5f,a295e9")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--kv-cache-dtype", default="bfloat16")
    parser.add_argument("--mem-fraction-static", type=float, default=0.95)
    parser.add_argument("--attention-backend", default="flashmla")
    parser.add_argument(
        "--sampling-backend",
        default="pytorch",
        help="Sampling backend used by the SGLang server. Showtime uses pytorch.",
    )
    parser.add_argument(
        "--moe-runner-backend",
        default=None,
        help="Optional MoE runner backend override passed through to SGLang.",
    )
    parser.add_argument(
        "--flashinfer-mxfp4-moe-precision",
        default=None,
        choices=("default", "bf16"),
        help="Optional flashinfer_mxfp4 MoE precision override passed through to SGLang.",
    )
    parser.add_argument("--cpu-offload-gb", type=int, default=0)
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=65536,
        help="Decode window for faithful showtime-style runs; keep this large.",
    )
    parser.add_argument("--max-running-requests", type=int, default=8)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--min-p", type=float, default=0.01)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--server-timeout-s", type=int, default=900)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--skip-server", action="store_true")
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--harmony-turns",
        type=int,
        default=128,
        help="Number of Harmony tool/reasoning turns to allow in showtime-style mode.",
    )
    parser.add_argument(
        "--mode",
        choices=("harmony", "completion"),
        default="harmony",
        help="Use the showtime-style OpenAI Harmony prompt path or a raw completion prompt.",
    )
    return parser.parse_args()


def _load_reference(csv_path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("id") or "").strip()
            if qid:
                out[qid] = {
                    "problem": str(row.get("problem") or ""),
                    "answer": str(row.get("answer") or ""),
                }
    return out


def _extract_answer(text: str) -> int | None:
    if not text:
        return None
    boxed = re.findall(r"\\boxed\s*\{\s*([0-9,]+)\s*\}", text)
    if boxed:
        try:
            return int(boxed[-1].replace(",", ""))
        except Exception:
            pass
    final_int = re.findall(r"final\s+answer\s+is\s*([0-9,]+)", text, flags=re.IGNORECASE)
    if final_int:
        try:
            return int(final_int[-1].replace(",", ""))
        except Exception:
            pass
    ints = re.findall(r"(?<!\d)(\d{1,5})(?!\d)", text)
    if ints:
        try:
            return int(ints[-1])
        except Exception:
            pass
    return None


def _looks_repetitive(text: str) -> bool:
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw:
        return False

    words = raw.split()
    if len(words) >= 8:
        top_word, top_count = Counter(words).most_common(1)[0]
        if top_count / max(len(words), 1) >= 0.7:
            return True

    # Common GPT-OSS failure mode here is a single-token loop such as
    # "to to to ..." or a tiny phrase repeated until max_new_tokens.
    if re.search(r"(?:\b\w+\b(?:\s+|$)){12,}", raw):
        return True
    return False


def _build_prompt(problem: str) -> str:
    return (
        f"{SHOWTIME_SYSTEM_PROMPT}\n\n"
        "Solve the following problem. Give a concise but complete derivation, then end with "
        "the final integer answer in \\boxed{}.\n\n"
        f"Problem:\n{problem}\n"
    )


def _build_harmony_prompt(problem: str) -> tuple[list[int], Any]:
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    system_content = (
        SystemContent.new()
        .with_model_identity(SHOWTIME_SYSTEM_PROMPT)
        .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
        .with_tools(ToolNamespaceConfig(name="python", description=SHOWTIME_TOOL_PROMPT, tools=[]))
    )
    messages = [
        Message.from_role_and_content(Role.SYSTEM, system_content),
        Message.from_role_and_content(Role.USER, problem),
    ]
    conversation = Conversation.from_messages(messages)
    prompt_ids = list(encoding.render_conversation_for_completion(conversation, Role.ASSISTANT))
    return prompt_ids, encoding


def _solve_harmony_problem(
    *,
    base_url: str,
    problem: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    timeout_s: int,
    context_tokens: int,
    buffer_tokens: int,
    max_turn_output_tokens: int,
    turns: int,
) -> dict[str, Any]:
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = [int(x) for x in encoding.stop_tokens_for_assistant_actions()]
    # Honor the requested context budget. The showtime-style harness expects the
    # caller to set the true decode window, so do not clamp it here.
    safe_context_tokens = int(context_tokens)
    user_prompt = str(problem).strip()
    system_content = (
        SystemContent.new()
        .with_model_identity(SHOWTIME_SYSTEM_PROMPT)
        .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
        .with_tools(ToolNamespaceConfig(name="python", description=SHOWTIME_TOOL_PROMPT, tools=[]))
    )
    messages = [
        Message.from_role_and_content(Role.SYSTEM, system_content),
        Message.from_role_and_content(Role.USER, user_prompt),
    ]
    conversation = Conversation.from_messages(messages)
    sandbox = AIMO3Sandbox(timeout=min(6.0, float(timeout_s)))
    tool = AIMO3Tool(local_jupyter_timeout=min(6.0, float(timeout_s)), tool_prompt=SHOWTIME_TOOL_PROMPT, sandbox=sandbox)

    transcript_parts: list[str] = []
    final_answer: int | None = None
    python_calls = 0
    python_errors = 0
    total_tokens = 0
    t0 = time.time()
    error_summary: str | None = None
    finalization_requested = False
    repetition_recovery_requested = False
    use_recovery_sampling = False
    server_text = ""

    try:
        for _turn in range(int(turns)):
            if len(conversation.messages) > 12:
                conversation = Conversation.from_messages(
                    [conversation.messages[0], conversation.messages[1], *conversation.messages[-8:]]
                )
            base_prompt_ids = list(encoding.render_conversation_for_completion(conversation, Role.ASSISTANT))
            prompt_message_count = len(conversation.messages)
            turn_output_ids: list[int] = []
            turn_recover: tuple[list[Message], Message | None, str | None, str] | None = None
            last_message: Message | None = None

            while True:
                max_new_tokens = safe_context_tokens - len(base_prompt_ids) - len(turn_output_ids) - int(buffer_tokens)
                if max_new_tokens <= 0:
                    break

                chunk_tokens = min(int(max_turn_output_tokens), int(max_tokens), int(max_new_tokens))
                if chunk_tokens <= 0:
                    break

                if use_recovery_sampling:
                    effective_temperature = max(float(temperature), 0.2)
                    effective_top_p = min(float(top_p), 0.95)
                    effective_top_k = max(int(top_k), 20)
                    effective_min_p = 0.0
                else:
                    effective_temperature = float(temperature)
                    effective_top_p = float(top_p)
                    effective_top_k = int(top_k)
                    effective_min_p = float(min_p)

                gen = _request_generate(
                    base_url,
                    base_prompt_ids + turn_output_ids,
                    max_new_tokens=chunk_tokens,
                    temperature=effective_temperature,
                    top_p=effective_top_p,
                    top_k=effective_top_k,
                    min_p=effective_min_p,
                    stop_token_ids=stop_token_ids,
                    timeout_s=timeout_s,
                )
                new_output_ids = list(gen.get("output_ids") or [])
                total_tokens += len(new_output_ids)
                if not new_output_ids:
                    break

                turn_output_ids.extend(new_output_ids)
                messages, partial_message, recover_error, raw_text = AIMO3RuntimeHelpers.recover_harmony_messages(
                    encoding=encoding,
                    prompt_ids=base_prompt_ids,
                    prompt_message_count=prompt_message_count,
                    output_ids=turn_output_ids,
                )
                turn_recover = (messages, partial_message, recover_error, raw_text)
                server_text = str(
                    gen.get("text")
                    or gen.get("response")
                    or gen.get("output_text")
                    or ""
                )
                parse_candidates = list(messages)
                if partial_message is not None:
                    parse_candidates.append(partial_message)
                recovered_assistant_text = False

                for m in parse_candidates:
                    if m.author.role != Role.ASSISTANT or not m.content:
                        continue
                    text = str(m.content[0].text or "")
                    if text.strip():
                        recovered_assistant_text = True
                    ans = AIMO3RuntimeHelpers.scan_for_answer(text)
                    if ans is not None:
                        final_answer = ans
                        break
                if final_answer is None and raw_text:
                    ans = AIMO3RuntimeHelpers.scan_for_answer(raw_text)
                    if ans is not None:
                        final_answer = ans
                if final_answer is None and server_text:
                    ans = AIMO3RuntimeHelpers.scan_for_answer(server_text)
                    if ans is not None:
                        final_answer = ans
                if final_answer is not None:
                    break

                if recover_error:
                    error_summary = recover_error
                    break

                if parse_candidates:
                    last_message = parse_candidates[-1]

                # Some checkpoints collapse into a greedy repetition loop
                # instead of surfacing a boxed final answer. If that happens,
                # preserve the raw server text for diagnostics and force a
                # stronger follow-up turn rather than continuing to stream the
                # same loop forever.
                if not recovered_assistant_text and raw_text.strip():
                    transcript_parts.append(raw_text.strip())
                elif not parse_candidates and server_text.strip():
                    transcript_parts.append(server_text.strip())
                if not repetition_recovery_requested and _looks_repetitive(server_text or raw_text):
                    repetition_recovery_requested = True
                    use_recovery_sampling = True
                    conversation.messages.append(
                        Message.from_role_and_content(
                            Role.USER,
                            (
                                "Your previous response repeated the same token pattern and did not solve the problem. "
                                "Restart the solution from scratch. Use Python if helpful. "
                                "Give only the final integer answer in \\boxed{}."
                            ),
                        )
                    )
                    finalization_requested = True
                    continue

                finish_reason = (gen.get("meta_info") or {}).get("finish_reason") or (gen.get("meta") or {}).get("finish_reason") or {}
                finish_type = finish_reason.get("type") if isinstance(finish_reason, dict) else None
                if finish_type in {"stop", "abort"}:
                    break
                if len(new_output_ids) < chunk_tokens:
                    break

            if turn_recover is None:
                break

            messages, partial_message, recover_error, raw_text = turn_recover
            if messages:
                conversation.messages.extend(messages)
                for m in messages:
                    if m.author.role != Role.ASSISTANT or not m.content:
                        continue
                    text = str(m.content[0].text or "")
                    if text:
                        transcript_parts.append(text)

            if partial_message is not None:
                text = str(partial_message.content[0].text or "") if partial_message.content else ""
                if text:
                    transcript_parts.append(text)

            if not transcript_parts and server_text:
                transcript_parts.append(server_text.strip())

            if final_answer is not None:
                break

            if last_message is None and partial_message is not None:
                last_message = partial_message

            if last_message is None:
                if not finalization_requested:
                    conversation.messages.append(
                        Message.from_role_and_content(
                            Role.USER,
                            "State only the final integer answer in \\boxed{} and nothing else.",
                        )
                    )
                    finalization_requested = True
                    continue
                break

            if last_message.channel == "final":
                answer_text = str(last_message.content[0].text or "") if last_message.content else ""
                final_answer = AIMO3RuntimeHelpers.scan_for_answer(answer_text)
                break

            if last_message.recipient == "python" and last_message in messages:
                python_calls += 1
                tool_responses = tool.process_sync_plus(last_message)
                response_text = str(tool_responses[0].content[0].text or "")
                if response_text.startswith("[ERROR]") or "Traceback" in response_text or "Error:" in response_text:
                    python_errors += 1
                if response_text.strip():
                    transcript_parts.append(f"<tool_response>\n{response_text.strip()}\n</tool_response>")
                conversation.messages.extend(tool_responses)
                continue

            if recover_error:
                error_summary = recover_error
                break

            if final_answer is None and not finalization_requested:
                conversation.messages.append(
                    Message.from_role_and_content(
                        Role.USER,
                        (
                            "State only the final integer answer in \\boxed{} and nothing else. "
                            "Do not repeat prior wording."
                        ),
                    )
                )
                finalization_requested = True
                continue

        if final_answer is None:
            merged_response = "\n".join(transcript_parts)
            final_answer = AIMO3RuntimeHelpers.scan_for_answer(merged_response)

        return {
            "answer": final_answer,
            "correct": None,
            "wall_s": round(float(time.time() - t0), 4),
            "response": "\n".join(transcript_parts),
            "python_calls": int(python_calls),
            "python_errors": int(python_errors),
            "error": error_summary,
            "total_tokens": int(total_tokens),
        }
    finally:
        with contextlib.suppress(Exception):
            sandbox.reset()
        with contextlib.suppress(Exception):
            sandbox.close()


def _wait_for_server(base_url: str, timeout_s: int) -> None:
    deadline = time.time() + float(timeout_s)
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=5)
            if r.ok:
                return
        except Exception as exc:
            last_err = exc
        time.sleep(2)
    raise TimeoutError(f"SGLang server did not become ready in {timeout_s}s: {last_err!r}")


def _launch_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", env["PYTORCH_ALLOC_CONF"])
    repo_root = Path(__file__).resolve().parent.parent
    py_path = str(repo_root / "python")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = py_path if not existing else f"{py_path}:{existing}"
    wheel_cublas_lib = "/venv/main/lib/python3.12/site-packages/nvidia/cublas/lib"
    wheel_cudart_lib = "/venv/main/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
    wheel_cu13_lib = "/venv/main/lib/python3.12/site-packages/nvidia/cu13/lib"
    cuda_lib = "/usr/local/cuda/targets/x86_64-linux/lib"
    compat_lib = "/workspace/cublas_compat"
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    ld_parts = [wheel_cublas_lib, wheel_cudart_lib, wheel_cu13_lib, compat_lib, cuda_lib]
    if existing_ld:
        ld_parts.append(existing_ld)
    env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(ld_parts))

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--served-model-name",
        str(args.served_model_name or Path(args.model_path).name),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tp",
        str(args.tp_size),
        "--page-size",
        str(args.page_size),
        "--dtype",
        args.dtype,
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--attention-backend",
        args.attention_backend,
        "--cpu-offload-gb",
        str(args.cpu_offload_gb),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--max-running-requests",
        str(args.max_running_requests),
        "--sampling-backend",
        str(args.sampling_backend),
        "--trust-remote-code",
        "--tokenizer-path",
        args.tokenizer_path,
        "--tokenizer-mode",
        "auto",
    ]
    if args.moe_runner_backend:
        cmd.extend(["--moe-runner-backend", str(args.moe_runner_backend)])
    if args.flashinfer_mxfp4_moe_precision:
        cmd.extend(
            [
                "--flashinfer-mxfp4-moe-precision",
                str(args.flashinfer_mxfp4_moe_precision),
            ]
        )
    if args.disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    print("[server]", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, env=env)


def _request_completion(
    base_url: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: int,
) -> dict[str, Any]:
    payload = {
        "model": "gpt-oss-reference-probe",
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    r = requests.post(f"{base_url}/v1/completions", json=payload, timeout=int(timeout_s))
    r.raise_for_status()
    return r.json()


def _request_generate(
    base_url: str,
    input_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    stop_token_ids: list[int] | None,
    timeout_s: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_ids": [int(x) for x in input_ids],
        "sampling_params": {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "min_p": float(min_p),
        },
        "stream": False,
    }
    if stop_token_ids:
        payload["sampling_params"]["stop_token_ids"] = [int(x) for x in stop_token_ids]

    r = requests.post(f"{base_url}/generate", json=payload, timeout=int(timeout_s))
    r.raise_for_status()
    out = r.json()
    if isinstance(out, list) and out:
        out = out[0]
    return out


def main() -> None:
    args = _parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    ref = _load_reference(csv_path)
    qids = [q.strip() for q in str(args.question_ids).split(",") if q.strip()]
    if not qids:
        raise ValueError("No question ids provided")

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    proc: subprocess.Popen[str] | None = None
    results: list[dict[str, Any]] = []
    try:
        if not args.skip_server:
            proc = _launch_server(args)
            _wait_for_server(base_url, args.server_timeout_s)

        encoding = None
        stop_token_ids: list[int] | None = None
        if args.mode == "harmony":
            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            stop_token_ids = [int(x) for x in encoding.stop_tokens_for_assistant_actions()]

        for qid in qids:
            if qid not in ref:
                raise KeyError(f"Question id not found in reference CSV: {qid}")
            problem = ref[qid]["problem"]
            expected = ref[qid]["answer"]

            t0 = time.time()
            if args.mode == "harmony":
                resp = _solve_harmony_problem(
                    base_url=base_url,
                    problem=problem,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    timeout_s=args.timeout_s,
                    context_tokens=args.max_total_tokens,
                    buffer_tokens=512,
                    max_turn_output_tokens=int(args.max_tokens),
                    turns=int(args.harmony_turns),
                )
            else:
                prompt = _build_prompt(problem)
                resp = _request_completion(
                    base_url,
                    prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    timeout_s=args.timeout_s,
                )
            wall_s = time.time() - t0

            if args.mode == "harmony":
                text = str(resp.get("response") or "")
                answer = resp.get("answer")
                if answer is None:
                    answer = _extract_answer(text)
            else:
                choice = (resp.get("choices") or [{}])[0]
                text = str(choice.get("text") or "")
                answer = _extract_answer(text)
            correct = str(answer) == expected if answer is not None else False

            row = {
                "id": qid,
                "expected": expected,
                "answer": answer,
                "correct": correct,
                "wall_s": round(float(wall_s), 4),
                "response": text,
                "python_calls": resp.get("python_calls") if isinstance(resp, dict) else None,
                "python_errors": resp.get("python_errors") if isinstance(resp, dict) else None,
            }
            results.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

        summary = {
            "model_path": args.model_path,
            "correct": sum(1 for r in results if r["correct"]),
            "total": len(results),
            "results": results,
        }
        if args.out_json:
            out_path = Path(args.out_json).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    finally:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
