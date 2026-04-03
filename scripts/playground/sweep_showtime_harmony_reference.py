#!/usr/bin/env python3
"""Showtime-faithful Harmony + Python + PaCoRe sweep on local reference problems.

This follows the showtime.py execution model:
- SGLang is only the backend compute engine via /generate
- Harmony parsing stays client-side with openai_harmony
- Python tool calls execute in local Jupyter sandboxes
- PaCoRe is handled at the attempt/synthesis layer
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import queue
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    SystemContent,
    TextContent,
    ToolNamespaceConfig,
    load_harmony_encoding,
)

from sglang.test.test_utils import kill_process_tree

from dflash.bench_reference import _launch_server


SHOWTIME_SYSTEM_PROMPT = (
    "You are an elite mathematical problem solver with expertise at the "
    "International Mathematical Olympiad (IMO) level. Your goal is to find "
    "the correct answer through rigorous mathematical reasoning.\n\n"
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
    "You have access to `math`, `numpy`, and `sympy` for exact symbolic work, "
    "numerical verification, and careful arithmetic. Use them whenever they reduce error. "
    "Always end with a single boxed non-negative integer answer."
)

BOXED_RE = re.compile(r"\\boxed\s*\{\s*([0-9,]+)\s*\}")
FINAL_INT_RE = re.compile(r"final\s+answer\s+is\s*([0-9,]+)", re.IGNORECASE)


class AIMO3Template:
    def get_system_content(
        self, system_prompt: str, tool_config: ToolNamespaceConfig
    ) -> SystemContent:
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(
        self, system_prompt: str, user_prompt: str, tool_config: ToolNamespaceConfig
    ) -> list[Message]:
        system_content = self.get_system_content(system_prompt, tool_config)
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_message = Message.from_role_and_content(Role.USER, user_prompt)
        return [system_message, user_message]


class AIMO3Sandbox:
    def __init__(self, timeout: float):
        self._default_timeout = float(timeout)
        self._km: KernelManager | None = None
        self._client = None

        env = os.environ.copy()
        env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        env["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "0"
        env["JUPYTER_PLATFORM_DIRS"] = "1"
        env["PYTHONWARNINGS"] = "ignore"
        env["MPLBACKEND"] = "Agg"

        self._km = KernelManager()
        self._km.start_kernel(
            env=env, extra_arguments=["--Application.log_level=CRITICAL"]
        )
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self.execute(
            "import math\n"
            "import numpy\n"
            "import sympy\n"
            "import itertools\n"
            "import collections\n"
            "import mpmath\n"
            "mpmath.mp.dps = 64\n"
        )

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
            if time.time() - start_time > effective_timeout:
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
                stderr_parts.extend(content.get("traceback", []))
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
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
        with contextlib.suppress(Exception):
            if self._km is not None:
                self._km.shutdown_kernel(now=True)
                self._km.cleanup_resources()


class AIMO3Tool:
    def __init__(self, *, timeout: float, tool_prompt: str, sandbox: AIMO3Sandbox):
        self._timeout = float(timeout)
        self._tool_prompt = str(tool_prompt)
        self._sandbox = sandbox

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(name="python", description=self._tool_prompt, tools=[])

    def _ensure_last_print(self, code: str) -> str:
        lines = code.strip().split("\n")
        if not lines:
            return code
        last_line = lines[-1].strip()
        if not last_line or last_line.startswith("#") or "print" in last_line or "import" in last_line:
            return code
        lines[-1] = f"print({last_line})"
        return "\n".join(lines)

    def process_sync_plus(self, message: Message) -> list[Message]:
        script = self._ensure_last_print(str(message.content[0].text or ""))
        try:
            output = self._sandbox.execute(script, timeout=self._timeout)
        except Exception as exc:
            output = f"[ERROR] {type(exc).__name__}: {exc}"
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name="python")
        msg = Message(author=author, content=[content]).with_recipient("assistant")
        if message.channel:
            msg = msg.with_channel(message.channel)
        return [msg]


class SGLangGenerateClient:
    def __init__(self, base_url: str, timeout_s: float):
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = float(timeout_s)
        self.session = requests.Session()

    def health(self) -> bool:
        with contextlib.suppress(Exception):
            r = self.session.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        return False

    def generate(self, *, input_ids: list[int], sampling_params: dict[str, Any]) -> dict[str, Any]:
        r = self.session.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": [int(x) for x in input_ids],
                "sampling_params": dict(sampling_params),
                "stream": False,
            },
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        out = r.json()
        if isinstance(out, list) and out:
            out = out[0]
        return dict(out or {})


class Helpers:
    @staticmethod
    def scan_for_answer(text: str) -> int | None:
        if not text:
            return None
        boxed = BOXED_RE.findall(text)
        if boxed:
            with contextlib.suppress(Exception):
                val = int(boxed[-1].replace(",", ""))
                if 0 <= val <= 99999:
                    return val
        final = FINAL_INT_RE.findall(text)
        if final:
            with contextlib.suppress(Exception):
                val = int(final[-1].replace(",", ""))
                if 0 <= val <= 99999:
                    return val
        return None

    @staticmethod
    def make_partial_assistant_message(
        *, text: str, channel: str | None, recipient: str | None, content_type: str | None
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
    ) -> dict[str, Any]:
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

        partial_message = None
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

        return {
            "messages": list(parser.messages[prompt_message_count:]),
            "partial_message": partial_message,
            "error": error_summary,
            "raw_text": raw_text,
        }

    @staticmethod
    def parse_widths(raw: str) -> list[int]:
        vals: list[int] = []
        for part in str(raw or "").split(","):
            part = part.strip()
            if not part:
                continue
            with contextlib.suppress(Exception):
                val = int(part)
                if val > 0:
                    vals.append(val)
        return vals

    @staticmethod
    def extract_ref_text(text: str, *, max_chars: int) -> str:
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            return ""
        ref = raw.split("</think>")[-1].strip() or raw
        if int(max_chars) > 0 and len(ref) > int(max_chars):
            ref = ref[-int(max_chars):]
        return ref.strip()

    @classmethod
    def build_pacore_user_prompt(
        cls, *, problem: str, reference_responses: list[str], preference_prompt: str
    ) -> str:
        refs = [str(r or "").strip() for r in reference_responses if str(r or "").strip()]
        original_problem = f"{problem.strip()} {preference_prompt.strip()}".strip()
        if not refs:
            return original_problem
        refs_blob = "\n".join(
            f"Reference {i}: {r}" for i, r in enumerate(refs, start=1)
        )
        return (
            "You are given a problem and a list of reference responses. Your job is to analyze "
            "these references and provide your own response.\n\n"
            f"Original Problem:\n{original_problem}\n\n"
            f"Reference Responses:\n{refs_blob}\n\n"
            "Now, based on the original problem and reference responses above, provide your own "
            "comprehensive solution. Critically evaluate the references: they may be incomplete, "
            "conflicting, or wrong. Do not merely copy or majority-vote; synthesize a better solution."
        )


def _load_reference_rows(csv_path: Path, question_ids: list[str]) -> list[dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qid = str(row.get("id") or "").strip()
            if qid:
                rows[qid] = {
                    "id": qid,
                    "problem": str(row.get("problem") or "").strip(),
                    "answer": str(row.get("answer") or "").strip(),
                }
    return [rows[qid] for qid in question_ids]


def _rank_refs(
    attempts: list[dict[str, Any]],
    *,
    max_refs: int,
    max_ref_chars: int,
    max_dup_per_answer: int,
) -> list[str]:
    answers = [a.get("answer") for a in attempts if isinstance(a.get("answer"), int)]
    support = Counter(int(x) for x in answers)
    ranked = []
    for attempt in attempts:
        answer = attempt.get("answer") if isinstance(attempt.get("answer"), int) else None
        ranked.append(
            (
                int(answer is not None),
                int(support.get(int(answer), 0) if answer is not None else 0),
                int((attempt.get("python_calls") or 0) > 0),
                int(attempt.get("completion_tokens") or 0),
                -float(attempt.get("wall_s") or 0.0),
                attempt,
            )
        )
    ranked.sort(reverse=True, key=lambda item: item[:5])
    refs: list[str] = []
    seen: set[str] = set()
    per_answer: dict[int, int] = {}
    for _score in ranked:
        attempt = _score[-1]
        if len(refs) >= int(max_refs):
            break
        text = Helpers.extract_ref_text(
            str(attempt.get("transcript") or ""),
            max_chars=int(max_ref_chars),
        )
        if not text:
            continue
        norm = " ".join(text.split())
        if norm in seen:
            continue
        answer = Helpers.scan_for_answer(text)
        if answer is not None:
            if int(per_answer.get(answer, 0)) >= int(max_dup_per_answer):
                continue
            per_answer[answer] = int(per_answer.get(answer, 0)) + 1
        refs.append(text)
        seen.add(norm)
    return refs


def _majority_answer(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    hist = Counter(int(a["answer"]) for a in attempts if isinstance(a.get("answer"), int))
    if not hist:
        return {"majority_answer": None, "majority_support": 0, "answer_hist": {}}
    answer, support = hist.most_common(1)[0]
    return {
        "majority_answer": int(answer),
        "majority_support": int(support),
        "answer_hist": {str(k): int(v) for k, v in hist.most_common()},
    }


def _sampling_params(
    *,
    max_new_tokens: int,
    stop_token_ids: list[int],
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> dict[str, Any]:
    return {
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "min_p": float(min_p),
        "stop_token_ids": [int(x) for x in stop_token_ids],
    }


def _run_attempt(
    *,
    client: SGLangGenerateClient,
    template: AIMO3Template,
    encoding,
    stop_token_ids: list[int],
    sandbox_pool: queue.Queue[AIMO3Sandbox],
    system_prompt: str,
    tool_prompt: str,
    user_prompt: str,
    context_tokens: int,
    buffer_tokens: int,
    max_turn_output_tokens: int,
    turns: int,
    jupyter_timeout_s: float,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> dict[str, Any]:
    sandbox = sandbox_pool.get()
    t0 = time.time()
    python_calls = 0
    python_errors = 0
    total_tokens = 0
    completion_tokens_sum = 0
    spec_verify_ct_sum = 0
    spec_accepted_tokens_sum = 0
    transcript_parts: list[str] = []
    final_answer: int | None = None
    last_request_debug: dict[str, Any] | None = None
    dflash_meta: dict[str, Any] = {}
    error_summary: str | None = None

    try:
        local_tool = AIMO3Tool(
            timeout=float(jupyter_timeout_s), tool_prompt=tool_prompt, sandbox=sandbox
        )
        messages = template.apply_chat_template(system_prompt, user_prompt, local_tool.tool_config)
        conversation = Conversation.from_messages(messages)

        for _ in range(int(turns)):
            base_prompt_ids = list(
                encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
            )
            prompt_message_count = len(conversation.messages)
            turn_output_ids: list[int] = []
            turn_recover: dict[str, Any] | None = None
            last_message: Message | None = None

            while True:
                max_tokens = (
                    int(context_tokens)
                    - len(base_prompt_ids)
                    - len(turn_output_ids)
                    - int(buffer_tokens)
                )
                if max_tokens <= 0:
                    break
                chunk_tokens = min(int(max_turn_output_tokens), int(max_tokens))
                if chunk_tokens <= 0:
                    break

                params = _sampling_params(
                    max_new_tokens=int(chunk_tokens),
                    stop_token_ids=stop_token_ids,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    min_p=float(min_p),
                )
                last_request_debug = {
                    "conversation_messages": int(len(conversation.messages)),
                    "base_prompt_tokens": int(len(base_prompt_ids)),
                    "turn_output_tokens": int(len(turn_output_ids)),
                    "request_prompt_tokens": int(len(base_prompt_ids) + len(turn_output_ids)),
                    "max_new_tokens": int(chunk_tokens),
                    "context_tokens": int(context_tokens),
                    "buffer_tokens": int(buffer_tokens),
                }
                gen = client.generate(
                    input_ids=base_prompt_ids + turn_output_ids, sampling_params=params
                )
                meta = dict(gen.get("meta_info") or {})
                new_output_ids = [int(x) for x in (gen.get("output_ids") or [])]
                total_tokens += len(new_output_ids)
                completion_tokens_sum += int(meta.get("completion_tokens") or len(new_output_ids))
                spec_verify_ct_sum += int(meta.get("spec_verify_ct") or 0)
                spec_accepted_tokens_sum += int(meta.get("spec_accepted_tokens") or 0)
                for k, v in meta.items():
                    if k.startswith("spec_") or k.startswith("dflash_"):
                        dflash_meta[k] = v
                if not new_output_ids:
                    break

                turn_output_ids.extend(new_output_ids)
                turn_recover = Helpers.recover_harmony_messages(
                    encoding=encoding,
                    prompt_ids=base_prompt_ids,
                    prompt_message_count=prompt_message_count,
                    output_ids=turn_output_ids,
                )
                parse_candidates = list(turn_recover["messages"])
                if turn_recover["partial_message"] is not None:
                    parse_candidates.append(turn_recover["partial_message"])

                for msg in parse_candidates:
                    if msg.author.role != Role.ASSISTANT or not msg.content:
                        continue
                    final_answer = Helpers.scan_for_answer(str(msg.content[0].text or ""))
                    if final_answer is not None:
                        break
                if final_answer is None and turn_recover["raw_text"]:
                    final_answer = Helpers.scan_for_answer(turn_recover["raw_text"])
                if final_answer is not None:
                    break
                if turn_recover["error"]:
                    error_summary = str(turn_recover["error"])
                    break
                if parse_candidates:
                    last_message = parse_candidates[-1]

                finish_reason = meta.get("finish_reason") or {}
                finish_type = (
                    finish_reason.get("type")
                    if isinstance(finish_reason, dict)
                    else None
                )
                if finish_type in {"stop", "abort"} or len(new_output_ids) < int(chunk_tokens):
                    break

            if turn_recover is None:
                break

            if turn_recover["messages"]:
                conversation.messages.extend(turn_recover["messages"])
                for msg in turn_recover["messages"]:
                    if msg.author.role == Role.ASSISTANT and msg.content:
                        transcript_parts.append(str(msg.content[0].text or ""))

            if turn_recover["partial_message"] is not None and turn_recover["partial_message"].content:
                transcript_parts.append(
                    str(turn_recover["partial_message"].content[0].text or "")
                )

            if final_answer is not None:
                break

            if last_message is None and turn_recover["partial_message"] is not None:
                last_message = turn_recover["partial_message"]
            if last_message is None:
                break
            if last_message.channel == "final":
                final_answer = Helpers.scan_for_answer(
                    str(last_message.content[0].text or "") if last_message.content else ""
                )
                break
            if last_message.recipient == "python" and last_message in turn_recover["messages"]:
                python_calls += 1
                tool_responses = local_tool.process_sync_plus(last_message)
                response_text = str(tool_responses[0].content[0].text or "")
                if response_text.startswith("[ERROR]") or "Traceback" in response_text or "Error:" in response_text:
                    python_errors += 1
                if response_text.strip():
                    transcript_parts.append(
                        f"<tool_response>\n{response_text.strip()}\n</tool_response>"
                    )
                conversation.messages.extend(tool_responses)
                continue
            break
    except Exception as exc:
        python_errors += 1
        error_summary = f"{type(exc).__name__}: {exc}"
    finally:
        sandbox.reset()
        sandbox_pool.put(sandbox)

    wall_s = float(time.time() - t0)
    return {
        "answer": final_answer,
        "python_calls": int(python_calls),
        "python_errors": int(python_errors),
        "wall_s": wall_s,
        "completion_tokens": int(completion_tokens_sum),
        "spec_verify_ct": int(spec_verify_ct_sum),
        "spec_accepted_tokens": int(spec_accepted_tokens_sum),
        "spec_accept_mean": float(spec_accepted_tokens_sum / max(spec_verify_ct_sum, 1)),
        "dflash_meta": dflash_meta or None,
        "last_request_debug": last_request_debug,
        "transcript": "\n".join(part for part in transcript_parts if part),
        "error": error_summary or "",
    }


def _solve_one_problem(
    *,
    client: SGLangGenerateClient,
    template: AIMO3Template,
    encoding,
    stop_token_ids: list[int],
    sandbox_pool: queue.Queue[AIMO3Sandbox],
    row: dict[str, str],
    attempts: int,
    early_stop: int,
    full_round: bool,
    pacore_widths: list[int],
    pacore_max_refs: int,
    pacore_max_ref_chars: int,
    pacore_max_dup_per_answer: int,
    context_tokens: int,
    buffer_tokens: int,
    max_turn_output_tokens: int,
    turns: int,
    jupyter_timeout_s: float,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> dict[str, Any]:
    widths = [*pacore_widths, 1] if pacore_widths else [int(attempts)]
    refs: list[str] = []
    rounds: list[dict[str, Any]] = []
    final_attempts: list[dict[str, Any]] = []

    for round_idx, width in enumerate(widths, start=1):
        is_final = bool(pacore_widths) and round_idx == len(widths)
        user_prompt = Helpers.build_pacore_user_prompt(
            problem=row["problem"],
            reference_responses=refs,
            preference_prompt=SHOWTIME_PREFERENCE_PROMPT,
        )
        attempts_out: list[dict[str, Any]] = []
        vote_hist: Counter[int] = Counter()
        round_terminated_by = "full_width"
        with ThreadPoolExecutor(max_workers=int(width)) as executor:
            futures = [
                executor.submit(
                    _run_attempt,
                    client=client,
                    template=template,
                    encoding=encoding,
                    stop_token_ids=stop_token_ids,
                    sandbox_pool=sandbox_pool,
                    system_prompt=SHOWTIME_SYSTEM_PROMPT,
                    tool_prompt=SHOWTIME_TOOL_PROMPT,
                    user_prompt=user_prompt,
                    context_tokens=int(context_tokens),
                    buffer_tokens=int(buffer_tokens),
                    max_turn_output_tokens=int(max_turn_output_tokens),
                    turns=int(turns),
                    jupyter_timeout_s=float(jupyter_timeout_s),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    min_p=float(min_p),
                )
                for _ in range(int(width))
            ]
            for future in as_completed(futures):
                result = future.result()
                result["is_correct"] = (
                    str(result.get("answer")) == str(row["answer"])
                )
                attempts_out.append(result)
                if isinstance(result.get("answer"), int):
                    vote_hist[int(result["answer"])] += 1
                if (
                    not full_round
                    and int(early_stop) > 0
                    and vote_hist
                    and vote_hist.most_common(1)[0][1] >= int(early_stop)
                ):
                    round_terminated_by = "early_stop"
                    break

        majority = _majority_answer(attempts_out)
        rounds.append(
            {
                "round": int(round_idx),
                "stage": "final_synthesis" if is_final else f"round_{round_idx}",
                "width": int(width),
                "input_ref_count": int(len(refs)),
                "attempts_completed": int(len(attempts_out)),
                "round_terminated_by": round_terminated_by,
                "majority": majority,
                "attempts": attempts_out,
            }
        )
        if pacore_widths and not is_final:
            refs = _rank_refs(
                attempts_out,
                max_refs=int(pacore_max_refs),
                max_ref_chars=int(pacore_max_ref_chars),
                max_dup_per_answer=int(pacore_max_dup_per_answer),
            )
        final_attempts = attempts_out

    final_majority = _majority_answer(final_attempts)
    final_answer = final_majority["majority_answer"]
    return {
        "question_id": row["id"],
        "expected_answer": int(row["answer"]),
        "final_answer": final_answer,
        "is_correct": (
            int(final_answer) == int(row["answer"]) if final_answer is not None else False
        ),
        "final_majority": final_majority,
        "rounds": rounds,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--csv-path", default="/root/reference.csv")
    p.add_argument("--question-ids", default="")
    p.add_argument("--model-path", default="/workspace/offload_root/gpt-oss-120b")
    p.add_argument("--draft-model-path", default="/root/epoch_65_step_23760")
    p.add_argument("--port", type=int, default=23211)
    p.add_argument("--context-length", type=int, default=65536)
    p.add_argument("--max-running-requests", type=int, default=8)
    p.add_argument("--cuda-graph-max-bs", type=int, default=8)
    p.add_argument("--page-size", type=int, default=1)
    p.add_argument("--draft-page-size", type=int, default=1)
    p.add_argument("--mem-fraction-static", type=float, default=0.90)
    p.add_argument("--attempts", type=int, default=8)
    p.add_argument("--early-stop", type=int, default=4)
    p.add_argument("--full-round", action="store_true")
    p.add_argument("--pacore-widths", default="8")
    p.add_argument("--pacore-max-refs", type=int, default=32)
    p.add_argument("--pacore-max-ref-chars", type=int, default=2000)
    p.add_argument("--pacore-max-dup-per-answer", type=int, default=1)
    p.add_argument("--turns", type=int, default=128)
    p.add_argument("--max-turn-output-tokens", type=int, default=96)
    p.add_argument("--buffer-tokens", type=int, default=512)
    p.add_argument("--jupyter-timeout-s", type=float, default=6.0)
    p.add_argument("--timeout-s", type=float, default=1800.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--min-p", type=float, default=0.01)
    p.add_argument("--attention-backend", default="fa3")
    p.add_argument("--moe-runner-backend", default="triton_kernel")
    p.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    p.add_argument("--draft-kv-cache-dtype", default="bfloat16")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--disable-dflash", action="store_true")
    p.add_argument("--disable-piecewise-cuda-graph", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_path)
    question_ids = (
        [q.strip() for q in str(args.question_ids).split(",") if q.strip()]
        if str(args.question_ids).strip()
        else [str(row["id"]) for row in _load_reference_rows(csv_path, [r["id"] for r in csv.DictReader(csv_path.open("r", encoding="utf-8"))])]
    )
    # Re-open cleanly if we consumed iterator above.
    if not str(args.question_ids).strip():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            question_ids = [
                str(row.get("id") or "").strip()
                for row in csv.DictReader(f)
                if str(row.get("id") or "").strip()
            ]
    rows = _load_reference_rows(csv_path, question_ids)

    server = _launch_server(
        model_path=str(args.model_path),
        port=int(args.port),
        attention_backend=str(args.attention_backend),
        moe_runner_backend=str(args.moe_runner_backend),
        kv_cache_dtype=str(args.kv_cache_dtype),
        context_length=int(args.context_length),
        cuda_graph_max_bs=int(args.cuda_graph_max_bs),
        max_running_requests=int(args.max_running_requests),
        page_size=int(args.page_size),
        enable_piecewise_cuda_graph=not bool(args.disable_piecewise_cuda_graph),
        piecewise_cuda_graph_max_tokens=None if args.disable_piecewise_cuda_graph else 8192,
        disable_cuda_graph=False,
        speculative=not bool(args.disable_dflash),
        draft_model_path=str(args.draft_model_path),
        draft_attention_backend=str(args.attention_backend),
        draft_kv_cache_dtype=str(args.draft_kv_cache_dtype),
        draft_page_size=int(args.draft_page_size),
        speculative_moe_runner_backend=str(args.moe_runner_backend),
        speculative_dflash_block_size=int(args.block_size),
        mem_fraction_static=float(args.mem_fraction_static),
        tool_server=None,
    )

    client = SGLangGenerateClient(f"http://127.0.0.1:{int(args.port)}", timeout_s=float(args.timeout_s))
    template = AIMO3Template()
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = [int(x) for x in encoding.stop_tokens_for_assistant_actions()]
    sandboxes: queue.Queue[AIMO3Sandbox] = queue.Queue()
    for _ in range(max(1, int(args.max_running_requests))):
        sandboxes.put(AIMO3Sandbox(timeout=float(args.jupyter_timeout_s)))

    results: list[dict[str, Any]] = []
    try:
        for row in rows:
            result = _solve_one_problem(
                client=client,
                template=template,
                encoding=encoding,
                stop_token_ids=stop_token_ids,
                sandbox_pool=sandboxes,
                row=row,
                attempts=int(args.attempts),
                early_stop=int(args.early_stop),
                full_round=bool(args.full_round),
                pacore_widths=Helpers.parse_widths(str(args.pacore_widths)),
                pacore_max_refs=int(args.pacore_max_refs),
                pacore_max_ref_chars=int(args.pacore_max_ref_chars),
                pacore_max_dup_per_answer=int(args.pacore_max_dup_per_answer),
                context_tokens=int(args.context_length),
                buffer_tokens=int(args.buffer_tokens),
                max_turn_output_tokens=int(args.max_turn_output_tokens),
                turns=int(args.turns),
                jupyter_timeout_s=float(args.jupyter_timeout_s),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=int(args.top_k),
                min_p=float(args.min_p),
            )
            results.append(result)
            (out_dir / f"{row['id']}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print(json.dumps({"question_id": row["id"], "is_correct": result["is_correct"]}))
    finally:
        while not sandboxes.empty():
            with contextlib.suppress(Exception):
                sandboxes.get_nowait().close()
        with contextlib.suppress(Exception):
            kill_process_tree(server.pid)

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
