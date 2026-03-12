from __future__ import annotations

"""
Lightweight tracing shim.

Some SGLang scheduler paths import tracing helpers. Upstream variants can include
optional tracing instrumentation, but text-only serving does not require it.

This module provides no-op implementations so builds remain runnable in minimal
environments (e.g., Kaggle) without pulling additional tracing dependencies.
"""

from typing import Any, Iterable


def process_tracing_init(*_args: Any, **_kwargs: Any) -> None:
    return


def trace_set_thread_info(*_args: Any, **_kwargs: Any) -> None:
    return


def trace_set_proc_propagate_context(*_args: Any, **_kwargs: Any) -> None:
    return


def trace_event_batch(*_args: Any, **_kwargs: Any) -> None:
    return


def trace_slice_start(*_args: Any, **_kwargs: Any) -> None:
    return


def trace_slice_end(*_args: Any, **_kwargs: Any) -> None:
    return


def trace_slice_batch(*_args: Any, **_kwargs: Any) -> None:
    return

