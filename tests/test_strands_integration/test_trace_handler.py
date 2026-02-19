"""Tests for StrandsTraceHandler and make_trace_path."""

import json
from pathlib import Path

import pytest

from tau2.strands_integration.trace_handler import (
    StrandsTraceHandler,
    make_trace_path,
)


# ---------------------------------------------------------------------------
# make_trace_path
# ---------------------------------------------------------------------------


class TestMakeTracePath:
    def test_returns_none_when_save_to_is_none(self):
        assert make_trace_path(None, "task_1", 0) is None

    def test_produces_correct_path(self, tmp_path):
        save_to = tmp_path / "simulations" / "my_run.json"
        result = make_trace_path(save_to, "task_42", 3)
        expected = tmp_path / "simulations" / "traces" / "my_run" / "task_42_trial_3.jsonl"
        assert result == expected

    def test_accepts_string_save_to(self, tmp_path):
        save_to = str(tmp_path / "results" / "run.json")
        result = make_trace_path(Path(save_to), "t1", 0)
        assert result.name == "t1_trial_0.jsonl"
        assert "traces" in str(result)


# ---------------------------------------------------------------------------
# StrandsTraceHandler
# ---------------------------------------------------------------------------


class TestStrandsTraceHandler:
    def test_creates_file_on_open(self, tmp_path):
        path = tmp_path / "traces" / "test.jsonl"
        handler = StrandsTraceHandler(path)
        handler.open()
        try:
            assert path.exists()
        finally:
            handler.close()

    def test_writes_jsonl_on_call(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        handler = StrandsTraceHandler(path)
        handler.open()
        try:
            handler(data="hello world")
            handler(start=True, message="starting")
        finally:
            handler.close()

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["event_type"] == "data"
        assert first["data"] == "hello world"

        second = json.loads(lines[1])
        assert "start" in second["event_type"]
        assert second["message"] == "starting"

    def test_handles_non_serializable_objects(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        handler = StrandsTraceHandler(path)
        handler.open()
        try:
            handler(data={"nested": object()})
        finally:
            handler.close()

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        # The non-serializable object should have been converted to a string
        assert isinstance(event["data"]["nested"], str)

    def test_noop_when_not_opened(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        handler = StrandsTraceHandler(path)
        # Should not raise
        handler(data="ignored")
        assert not path.exists()

    def test_noop_after_close(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        handler = StrandsTraceHandler(path)
        handler.open()
        handler(data="first")
        handler.close()
        handler(data="second")

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_event_type_classification(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        handler = StrandsTraceHandler(path)
        handler.open()
        try:
            handler(current_tool_use={"name": "get_user"})
            handler(complete=True, result="done")
            handler(force_stop=True)
            handler(unknown_key="x")
        finally:
            handler.close()

        lines = path.read_text().strip().splitlines()
        events = [json.loads(line) for line in lines]

        assert events[0]["event_type"] == "current_tool_use"
        assert events[1]["event_type"] == "complete_result"
        assert events[2]["event_type"] == "force_stop"
        assert events[3]["event_type"] == "unknown"

    def test_flush_after_each_event(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        handler = StrandsTraceHandler(path)
        handler.open()
        try:
            handler(data="event_1")
            # Read before close — should be flushed already
            content = path.read_text()
            assert "event_1" in content
        finally:
            handler.close()
