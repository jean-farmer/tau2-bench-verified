"""
Converts tau2 Tool objects into Strands-compatible tool functions.
"""

import inspect
import uuid

from loguru import logger

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool as Tau2Tool


class ToolTracker:
    """Tracks tool calls and results during a Strands agent turn."""

    def __init__(self):
        self.calls: list[tuple[ToolCall, ToolMessage]] = []
        self.error_count: int = 0

    def record(self, tool_call: ToolCall, tool_msg: ToolMessage):
        self.calls.append((tool_call, tool_msg))
        if tool_msg.error:
            self.error_count += 1

    def get_calls_since(self, index: int) -> list[tuple[ToolCall, ToolMessage]]:
        return self.calls[index:]

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _convert_single_tool(
    tau2_tool: Tau2Tool,
    environment: Environment,
    tracker: ToolTracker,
    requestor: str = "assistant",
):
    """Convert a single tau2 Tool into a Strands-compatible decorated tool function."""
    from strands import tool as strands_tool

    schema = tau2_tool.openai_schema
    func_schema = schema["function"]
    tool_name = func_schema["name"]
    tool_description = func_schema.get("description", tool_name)
    parameters_schema = func_schema.get("parameters", {"type": "object", "properties": {}})

    input_schema = {"json": parameters_schema}

    def wrapper(**kwargs):
        tc_id = str(uuid.uuid4())
        tool_call = ToolCall(
            id=tc_id,
            name=tool_name,
            arguments=kwargs,
            requestor=requestor,
        )
        tool_msg = environment.get_response(tool_call)
        tracker.record(tool_call, tool_msg)
        logger.debug(
            f"Strands tool '{tool_name}' called with {kwargs} -> error={tool_msg.error}"
        )
        return tool_msg.content

    # Build an explicit signature from the tool's parameter schema so that
    # Strands' @tool decorator creates a correct Pydantic validation model.
    # Without this, the decorator sees `**kwargs` and creates a model with a
    # single required "kwargs" field, causing validation errors.
    properties = parameters_schema.get("properties", {})
    required_params = set(parameters_schema.get("required", []))
    sig_params = []
    for param_name in properties:
        default = (
            inspect.Parameter.empty if param_name in required_params else None
        )
        sig_params.append(
            inspect.Parameter(
                param_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=default,
            )
        )
    wrapper.__signature__ = inspect.Signature(sig_params)

    wrapped = strands_tool(
        name=tool_name,
        description=tool_description,
        inputSchema=input_schema,
    )(wrapper)

    return wrapped


def convert_tools(
    environment: Environment,
    tracker: ToolTracker,
    requestor: str = "assistant",
) -> list:
    """Convert all tau2 environment tools into Strands-compatible tool functions.

    Args:
        environment: The tau2 environment containing tools.
        tracker: ToolTracker to record tool calls and results.
        requestor: The requestor type for tool calls ("assistant" or "user").

    Returns:
        A list of Strands-compatible decorated tool functions.
    """
    tau2_tools = environment.get_tools()
    strands_tools = []
    for tau2_tool in tau2_tools:
        strands_tool = _convert_single_tool(
            tau2_tool=tau2_tool,
            environment=environment,
            tracker=tracker,
            requestor=requestor,
        )
        strands_tools.append(strands_tool)
    return strands_tools


def convert_user_tools(
    environment: Environment,
    tracker: ToolTracker,
) -> list:
    """Convert tau2 user tools into Strands-compatible tool functions (for solo mode).

    Args:
        environment: The tau2 environment containing user tools.
        tracker: ToolTracker to record tool calls and results.

    Returns:
        A list of Strands-compatible decorated tool functions.
    """
    tau2_tools = environment.get_user_tools()
    strands_tools = []
    for tau2_tool in tau2_tools:
        strands_tool = _convert_single_tool(
            tau2_tool=tau2_tool,
            environment=environment,
            tracker=tracker,
            requestor="assistant",
        )
        strands_tools.append(strands_tool)
    return strands_tools
