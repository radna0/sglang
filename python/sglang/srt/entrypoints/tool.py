# SPDX-License-Identifier: Apache-2.0
import contextlib
import logging
import os
import queue
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from openai_harmony import Author, Message, Role, TextContent

from sglang.srt.utils import print_info_once, print_warning_once

if TYPE_CHECKING:
    # Avoid circular import.
    from sglang.srt.entrypoints.context import ConversationContext

logger = logging.getLogger(__name__)


class Tool(ABC):

    @abstractmethod
    async def get_result(self, context: "ConversationContext") -> Any:
        pass


class _LocalJupyterSandbox:

    def __init__(self, timeout: float):
        self._default_timeout = float(timeout)
        self._client = None
        self._km = None

        try:
            from jupyter_client import KernelManager
        except ImportError as exc:
            raise RuntimeError(
                "jupyter_client is required for the local Harmony python tool fallback"
            ) from exc

        env = os.environ.copy()
        env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        env["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "0"
        env["JUPYTER_PLATFORM_DIRS"] = "1"
        env["PYTHONWARNINGS"] = "ignore"
        env["MPLBACKEND"] = "Agg"

        self._km = KernelManager()
        self._km.start_kernel(
            env=env,
            extra_arguments=["--Application.log_level=CRITICAL"],
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

    def _format_error(self, traceback: list[str]) -> str:
        clean_lines: list[str] = []
        for frame in traceback:
            clean_frame = re.sub(r"\x1b\[[0-9;]*m", "", frame)
            if 'File "' in clean_frame and "ipython-input" not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return "".join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:
        if self._client is None or self._km is None:
            return "[ERROR] Jupyter client not available"

        effective_timeout = float(timeout or self._default_timeout)
        msg_id = self._client.execute(
            code, store_history=True, allow_stdin=False, stop_on_error=False
        )
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        start_time = time.time()

        while True:
            if time.time() - start_time > effective_timeout:
                self._km.interrupt_kernel()
                return f"[ERROR] Execution timed out after {effective_timeout} seconds"

            try:
                msg = self._client.get_iopub_msg(timeout=1.0)
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
                stderr_parts.append(
                    self._format_error(content.get("traceback", []))
                )
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(
                        text if text.endswith("\n") else f"{text}\n"
                    )
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)
        if stderr:
            return f"{stdout.rstrip()}\n{stderr}" if stdout else stderr
        return stdout if stdout.strip() else "[WARN] No output. Use print() to see results."

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._client is not None:
                self._client.stop_channels()
        with contextlib.suppress(Exception):
            if self._km is not None:
                self._km.shutdown_kernel(now=True)
        with contextlib.suppress(Exception):
            if self._km is not None:
                self._km.cleanup_resources()


class _LocalHarmonyPythonTool(Tool):

    def __init__(self, timeout_s: float):
        self.enabled = True
        self._sandbox = _LocalJupyterSandbox(timeout=timeout_s)
        self._execution_lock = threading.Lock()

    def _ensure_last_print(self, code: str) -> str:
        lines = str(code or "").strip().split("\n")
        if not lines:
            return str(code or "")
        last_line = lines[-1].strip()
        if "print" in last_line or "import" in last_line:
            return str(code or "")
        if not last_line or last_line.startswith("#"):
            return str(code or "")
        lines[-1] = f"print({last_line})"
        return "\n".join(lines)

    def close(self) -> None:
        self._sandbox.close()

    async def get_result(self, context: "ConversationContext") -> Any:
        from sglang.srt.entrypoints.context import HarmonyContext

        assert isinstance(context, HarmonyContext)
        last_msg = context.messages[-1]
        raw_script = last_msg.content[0].text
        final_script = self._ensure_last_print(raw_script)

        with self._execution_lock:
            output = self._sandbox.execute(final_script)

        msg = Message(
            author=Author(role=Role.TOOL, name="python"),
            content=[TextContent(text=output)],
            channel=last_msg.channel,
            recipient=Role.ASSISTANT,
        )
        return [msg]


class HarmonyBrowserTool(Tool):

    def __init__(self):
        self.enabled = True
        exa_api_key = os.getenv("EXA_API_KEY")
        if not exa_api_key:
            self.enabled = False
            print_warning_once("EXA_API_KEY is not set, browsing is disabled")
            return

        try:
            from gpt_oss.tools.simple_browser import SimpleBrowserTool
            from gpt_oss.tools.simple_browser.backend import ExaBackend
        except ImportError:
            self.enabled = False
            print_warning_once("gpt_oss is not installed, browsing is disabled")
            return

        browser_backend = ExaBackend(source="web", api_key=exa_api_key)
        self.browser_tool = SimpleBrowserTool(backend=browser_backend)
        print_info_once("Browser tool initialized")

    async def get_result(self, context: "ConversationContext") -> Any:
        from sglang.srt.entrypoints.context import HarmonyContext

        assert isinstance(context, HarmonyContext)
        last_msg = context.messages[-1]
        tool_output_msgs = []
        async for msg in self.browser_tool.process(last_msg):
            tool_output_msgs.append(msg)
        return tool_output_msgs

    @property
    def tool_config(self) -> Any:
        return self.browser_tool.tool_config


class HarmonyPythonTool(Tool):

    def __init__(self):
        self.enabled = True
        self._fallback_tool = None

        try:
            from gpt_oss.tools.python_docker.docker_tool import PythonTool
        except ImportError:
            timeout_s = float(os.getenv("SGLANG_HARMONY_PYTHON_TIMEOUT_S", "30"))
            try:
                self._fallback_tool = _LocalHarmonyPythonTool(timeout_s=timeout_s)
                print_info_once(
                    "Code interpreter tool initialized with local Jupyter fallback"
                )
                return
            except Exception as exc:
                self.enabled = False
                print_warning_once(
                    f"Local Harmony python fallback failed to initialize: {exc}"
                )
                return

        self.python_tool = PythonTool()
        print_info_once("Code interpreter tool initialized")

    def close(self) -> None:
        if self._fallback_tool is not None:
            self._fallback_tool.close()

    async def get_result(self, context: "ConversationContext") -> Any:
        if self._fallback_tool is not None:
            return await self._fallback_tool.get_result(context)

        from sglang.srt.entrypoints.context import HarmonyContext

        assert isinstance(context, HarmonyContext)
        last_msg = context.messages[-1]
        tool_output_msgs = []
        async for msg in self.python_tool.process(last_msg):
            tool_output_msgs.append(msg)
        return tool_output_msgs

    @property
    def tool_config(self) -> Any:
        if self._fallback_tool is not None:
            return None
        return self.python_tool.tool_config
