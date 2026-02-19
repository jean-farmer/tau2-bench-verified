"""
Trace callback handler for Strands Agent events.

Captures all Strands callback events as JSONL files for debugging and analysis.
"""

import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger


def _safe_serialize(obj: Any) -> Any:
    """Make an object JSON-serializable by converting non-serializable types to strings."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    return str(obj)


# All known Strands callback event key sets.
_EVENT_KEYS = {
    "data",
    "reasoningText",
    "current_tool_use",
    "complete",
    "result",
    "message",
    "force_stop",
    "init_event_loop",
    "start",
    "start_event_loop",
    "event_loop_throttled_delay",
    "tool_stream_event",
    "tool_cancel_event",
    "tool_interrupt_event",
}


def _classify_event(kwargs: dict) -> str:
    """Return a short event type label based on which kwargs keys are present."""
    matched = set(kwargs.keys()) & _EVENT_KEYS
    if matched:
        return "_".join(sorted(matched))
    return "unknown"


class StrandsTraceHandler:
    """Callable that writes every Strands callback event as a JSON line.

    Usage::

        handler = StrandsTraceHandler(path)
        handler.open()
        try:
            agent = Agent(..., callback_handler=handler)
            agent("hello")
        finally:
            handler.close()
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._file = None

    def open(self):
        """Open the JSONL file for writing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a")
        logger.info(f"Strands trace handler opened: {self.path}")

    def close(self):
        """Flush and close the JSONL file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __call__(self, **kwargs):
        """Write one event as a JSON line.  No-op if the file is not open."""
        if self._file is None:
            return
        event = {
            "event_type": _classify_event(kwargs),
            **_safe_serialize(kwargs),
        }
        self._file.write(json.dumps(event) + "\n")
        self._file.flush()


def make_trace_path(
    save_to: Optional[Path],
    task_id: str,
    trial: int,
) -> Optional[Path]:
    """Build the JSONL trace file path for a given task/trial.

    If *save_to* is ``None`` the function returns ``None``.

    The trace is placed alongside the simulation results under a ``traces/``
    sub-directory named after the run file (without extension)::

        data/simulations/traces/<run_name>/<task_id>_trial_<trial>.jsonl
    """
    if save_to is None:
        return None
    save_to = Path(save_to)
    run_name = save_to.stem
    return save_to.parent / "traces" / run_name / f"{task_id}_trial_{trial}.jsonl"
