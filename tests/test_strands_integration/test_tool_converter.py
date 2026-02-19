"""Tests for the Strands tool converter."""

import pytest

from tau2.data_model.message import ToolCall
from tau2.domains.mock.data_model import MockDB, Task, User
from tau2.domains.mock.environment import get_environment
from tau2.environment.environment import Environment

# Skip all tests if strands is not installed
strands = pytest.importorskip("strands")

from tau2.strands_integration.tool_converter import (
    ToolTracker,
    convert_tools,
)


@pytest.fixture
def mock_db() -> MockDB:
    return MockDB(
        tasks={
            "task_1": Task(
                task_id="task_1",
                title="Test task",
                description="A test task",
                status="pending",
            )
        },
        users={"user_1": User(user_id="user_1", name="Test User", tasks=["task_1"])},
    )


@pytest.fixture
def environment(mock_db: MockDB) -> Environment:
    return get_environment(mock_db)


@pytest.fixture
def tracker() -> ToolTracker:
    return ToolTracker()


def test_convert_tools_returns_correct_count(environment: Environment, tracker: ToolTracker):
    """Test that convert_tools returns one Strands tool per tau2 tool."""
    tau2_tools = environment.get_tools()
    strands_tools = convert_tools(environment=environment, tracker=tracker)
    assert len(strands_tools) == len(tau2_tools)


def test_converted_tool_names(environment: Environment, tracker: ToolTracker):
    """Test that converted tools preserve the original tool names."""
    tau2_tools = environment.get_tools()
    strands_tools = convert_tools(environment=environment, tracker=tracker)

    tau2_names = {t.name for t in tau2_tools}
    strands_names = {t.TOOL_SPEC["name"] for t in strands_tools}
    assert tau2_names == strands_names


def test_converted_tool_has_spec(environment: Environment, tracker: ToolTracker):
    """Test that each converted tool has a valid tool_spec."""
    strands_tools = convert_tools(environment=environment, tracker=tracker)
    for tool in strands_tools:
        spec = tool.TOOL_SPEC
        assert "name" in spec
        assert "description" in spec
        assert "inputSchema" in spec
        assert "json" in spec["inputSchema"]


def test_tool_execution_routes_through_environment(
    environment: Environment, tracker: ToolTracker
):
    """Test that calling a converted tool executes via environment.get_response()."""
    strands_tools = convert_tools(environment=environment, tracker=tracker)

    # Find the create_task tool
    create_task_tool = None
    for tool in strands_tools:
        if tool.TOOL_SPEC["name"] == "create_task":
            create_task_tool = tool
            break
    assert create_task_tool is not None

    # Call the tool directly (bypassing Strands agent)
    create_task_tool(
        user_id="user_1", title="New task", description="A new test task"
    )

    # Verify it was tracked
    assert tracker.call_count == 1
    tc, tm = tracker.calls[0]
    assert tc.name == "create_task"
    assert tc.arguments["user_id"] == "user_1"
    assert tc.arguments["title"] == "New task"
    assert not tm.error

    # Verify the environment state changed
    assert "task_2" in environment.tools.db.tasks
    assert environment.tools.db.tasks["task_2"].title == "New task"


def test_tool_error_tracking(environment: Environment, tracker: ToolTracker):
    """Test that tool errors are properly counted."""
    strands_tools = convert_tools(environment=environment, tracker=tracker)

    create_task_tool = None
    for tool in strands_tools:
        if tool.TOOL_SPEC["name"] == "create_task":
            create_task_tool = tool
            break

    # Call with invalid user_id to trigger an error
    create_task_tool(
        user_id="nonexistent", title="Bad task", description="Should fail"
    )

    assert tracker.error_count == 1
    assert tracker.call_count == 1
    tc, tm = tracker.calls[0]
    assert tm.error


def test_tracker_get_calls_since(environment: Environment, tracker: ToolTracker):
    """Test ToolTracker.get_calls_since() for tracking calls per turn."""
    strands_tools = convert_tools(environment=environment, tracker=tracker)

    create_task_tool = None
    for tool in strands_tools:
        if tool.TOOL_SPEC["name"] == "create_task":
            create_task_tool = tool
            break

    # Make first call
    before_count = tracker.call_count
    create_task_tool(user_id="user_1", title="Task A", description="First")
    first_batch = tracker.get_calls_since(before_count)
    assert len(first_batch) == 1

    # Make second call
    before_count = tracker.call_count
    create_task_tool(user_id="user_1", title="Task B", description="Second")
    second_batch = tracker.get_calls_since(before_count)
    assert len(second_batch) == 1

    # Total calls
    assert tracker.call_count == 2


def test_tool_call_ids_are_unique(environment: Environment, tracker: ToolTracker):
    """Test that each tool call gets a unique ID."""
    strands_tools = convert_tools(environment=environment, tracker=tracker)

    create_task_tool = None
    for tool in strands_tools:
        if tool.TOOL_SPEC["name"] == "create_task":
            create_task_tool = tool
            break

    create_task_tool(user_id="user_1", title="Task A", description="First")
    create_task_tool(user_id="user_1", title="Task B", description="Second")

    ids = [tc.id for tc, _ in tracker.calls]
    assert len(ids) == len(set(ids))  # All unique
