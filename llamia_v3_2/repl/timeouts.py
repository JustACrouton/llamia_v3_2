from __future__ import annotations

from contextlib import contextmanager
import signal
from typing import Iterator, Optional


class InvokeTimeout(Exception):
    """Raised when a single graph invocation exceeds the configured wall-clock timeout."""


def _alarm_handler(_signum: int, _frame: Optional[object]) -> None:
    raise InvokeTimeout("invoke exceeded timeout")


def _set_alarm(seconds: int) -> None:
    """
    Enable SIGALRM-based timeout on platforms that support it (Linux/macOS).
    No-op on Windows.
    """
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(max(1, int(seconds)))


def _clear_alarm() -> None:
    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)


@contextmanager
def invoke_timeout(seconds: int) -> Iterator[None]:
    """
    Context manager used around `app.invoke(...)`.

    Example:
        with invoke_timeout(600):
            result = app.invoke(state, config={...})
    """
    _set_alarm(seconds)
    try:
        yield
    finally:
        _clear_alarm()
