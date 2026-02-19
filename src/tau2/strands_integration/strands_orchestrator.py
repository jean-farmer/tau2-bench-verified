"""
StrandsOrchestrator: Uses a Strands Agent for the agent's inference + tool execution loop,
while keeping the existing UserSimulator for user turns.
"""

import time
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import EnvFunctionCall, InitializationData, Task
from tau2.environment.environment import Environment
from tau2.strands_integration.tool_converter import (
    ToolTracker,
    convert_tools,
    convert_user_tools,
)
from tau2.user.base import BaseUser
from tau2.user.user_simulator import DummyUser, UserSimulator, UserState
from tau2.utils.llm_utils import get_cost
from tau2.utils.utils import format_time, get_now

STOP_TOKEN = "###STOP###"

# Maximum retries for transient LLM API errors (auth refresh, DNS, rate limits)
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5  # seconds

AGENT_SOLO_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
You will be provided with a ticket that contains the user's request.
You will need to plan and call the appropriate tools to solve the ticket.

You cannot communicate with the user, only make tool calls.
Stop when you consider that you have solved the ticket.
To do so, call the `done` tool. Do not include any other tool calls after calling `done`.

Always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT_SOLO = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<ticket>
{ticket}
</ticket>
""".strip()

# Maximum tool-calling rounds within a single Strands agent turn
# to prevent infinite tool loops
MAX_INTRA_TURN_TOOL_ROUNDS = 50


def _is_transient_error(exc: Exception) -> bool:
    """Check if an exception is a transient LLM API error worth retrying."""
    try:
        import litellm
    except ImportError:
        return False
    transient_types = (
        litellm.RateLimitError,
        litellm.ServiceUnavailableError,
        litellm.APIConnectionError,
        litellm.Timeout,
    )
    if isinstance(exc, transient_types):
        return True
    # AWS credential refresh failures are transient AuthenticationErrors
    if isinstance(exc, litellm.AuthenticationError):
        msg = str(exc).lower()
        if "bedrock" in msg or "unable to locate credentials" in msg:
            return True
    return False


class StrandsOrchestrator:
    """Orchestrator that uses a Strands Agent for the agent loop.

    In conversational mode:
        - Strands Agent handles: model → tool_call → env → model → ... → text_response
        - User simulator handles: user turns (same as standard Orchestrator)

    In solo mode:
        - Strands Agent handles the entire loop including a done() tool for termination.
    """

    def __init__(
        self,
        domain: str,
        environment: Environment,
        user: BaseUser,
        task: Task,
        llm_agent: str,
        llm_args_agent: Optional[dict] = None,
        max_steps: int = 100,
        max_errors: int = 10,
        seed: Optional[int] = None,
        solo_mode: bool = False,
        validate_communication: bool = False,
        callback_handler=None,
    ):
        self.domain = domain
        self.environment = environment
        self.user = user
        self.task = task
        self.llm_agent = llm_agent
        self.llm_args_agent = deepcopy(llm_args_agent) if llm_args_agent else {}
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.seed = seed
        self.solo_mode = solo_mode
        self.validate_communication = validate_communication
        self.callback_handler = callback_handler

        self.trajectory: list[Message] = []
        self.step_count = 0
        self.termination_reason: Optional[TerminationReason] = None
        self.done_flag = False

    def run(self) -> SimulationRun:
        """Run the simulation and return a SimulationRun."""
        start_time = get_now()
        start = time.perf_counter()

        if self.solo_mode:
            self._run_solo()
        else:
            self._run_conversational()

        duration = time.perf_counter() - start
        messages = self._finalize_trajectory()
        res = get_cost(messages)
        if res is None:
            agent_cost, user_cost = None, None
        else:
            agent_cost, user_cost = res

        simulation_run = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=self.task.id,
            start_time=start_time,
            end_time=get_now(),
            duration=duration,
            termination_reason=self.termination_reason.value,
            reward_info=None,
            user_cost=user_cost,
            agent_cost=agent_cost,
            messages=messages,
            seed=self.seed,
        )
        return simulation_run

    def _run_conversational(self):
        """Run conversational mode: Strands agent + user simulator."""
        from strands import Agent
        from strands.agent.conversation_manager import NullConversationManager
        from strands.models.litellm import LiteLLMModel

        # Initialize environment
        initial_state = self.task.initial_state
        initialization_data = (
            initial_state.initialization_data if initial_state else None
        )
        initialization_actions = (
            initial_state.initialization_actions if initial_state else None
        )
        message_history = (
            deepcopy(initial_state.message_history)
            if initial_state and initial_state.message_history
            else []
        )
        for msg in message_history:
            msg.turn_idx = None
        message_history = self._add_timestamps(message_history)

        self._initialize_environment(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

        # Set user seed
        if self.seed is not None:
            self.user.set_seed(self.seed)

        # Create tool tracker and Strands tools
        tracker = ToolTracker()
        strands_tools = convert_tools(
            environment=self.environment,
            tracker=tracker,
        )

        # Build system prompt (same as LLMAgent)
        system_prompt = SYSTEM_PROMPT.format(
            domain_policy=self.environment.get_policy(),
            agent_instruction=AGENT_INSTRUCTION,
        )

        # Build LiteLLM model params
        model_params = deepcopy(self.llm_args_agent)
        if self.seed is not None and "seed" not in model_params:
            model_params["seed"] = self.seed
        # Remove any stream override — the Strands SDK manages streaming
        # internally via its stream() method; overriding it to False causes
        # litellm.acompletion to return a non-iterable ModelResponse.
        model_params.pop("stream", None)

        model = LiteLLMModel(
            model_id=self.llm_agent,
            params=model_params,
        )

        agent = Agent(
            model=model,
            tools=strands_tools,
            system_prompt=system_prompt,
            callback_handler=self.callback_handler,
            conversation_manager=NullConversationManager(),
        )

        # Handle message history if present
        if message_history:
            self.trajectory = list(message_history)
            strands_history = self._tau2_to_strands_messages(message_history)
            agent.messages = strands_history

            # Determine what to do based on last message
            last_msg = message_history[-1]
            if isinstance(last_msg, AssistantMessage) and not last_msg.is_tool_call():
                # Last message was agent text → need user response
                user_state = self.user.get_init_state(
                    message_history=[
                        m
                        for m in message_history
                        if isinstance(m, (AssistantMessage, UserMessage, ToolMessage))
                        and (
                            not isinstance(m, ToolMessage)
                            or m.requestor != "assistant"
                        )
                    ]
                )
                try:
                    user_msg, user_state = self.user.generate_next_message(
                        last_msg, user_state
                    )
                except Exception as e:
                    logger.error(f"User simulator error: {e}")
                    self.done_flag = True
                    self.termination_reason = TerminationReason.USER_ERROR
                    return
                if UserSimulator.is_stop(user_msg):
                    self.done_flag = True
                    self.termination_reason = TerminationReason.USER_STOP
                    self.trajectory.append(user_msg)
                    return
                self.trajectory.append(user_msg)
                last_user_text = user_msg.content
            elif isinstance(last_msg, UserMessage) and not last_msg.is_tool_call():
                # Last message was user text → feed to agent
                user_state = self.user.get_init_state(
                    message_history=[
                        m
                        for m in message_history
                        if isinstance(m, (AssistantMessage, UserMessage, ToolMessage))
                        and (
                            not isinstance(m, ToolMessage)
                            or m.requestor != "assistant"
                        )
                    ]
                )
                last_user_text = last_msg.content
            elif isinstance(last_msg, ToolMessage):
                # History ends with a tool result — agent was mid-turn.
                user_state = self.user.get_init_state(
                    message_history=[
                        m
                        for m in message_history
                        if isinstance(m, (AssistantMessage, UserMessage, ToolMessage))
                        and (
                            not isinstance(m, ToolMessage)
                            or m.requestor != "assistant"
                        )
                    ]
                )
                # The Strands agent's message history already contains the
                # pending tool result. Feed an empty prompt so the agent
                # continues its turn in the main loop.
                last_user_text = ""
            else:
                # Unresumable state (e.g. assistant tool call with no
                # result, or user tool call). Terminate gracefully instead
                # of leaving termination_reason unset.
                logger.warning(
                    f"Cannot resume from history ending with "
                    f"{type(last_msg).__name__}; terminating."
                )
                self.done_flag = True
                self.termination_reason = TerminationReason.AGENT_ERROR
                return
        else:
            # No message history - start with default greeting
            first_message = AssistantMessage(
                role="assistant",
                content="Hi! How can I help you today?",
                cost=0.0,
                timestamp=get_now(),
            )
            self.trajectory.append(first_message)

            user_state = self.user.get_init_state()
            try:
                user_msg, user_state = self.user.generate_next_message(
                    first_message, user_state
                )
            except Exception as e:
                logger.error(f"User simulator error: {e}")
                self.done_flag = True
                self.termination_reason = TerminationReason.USER_ERROR
                return
            if UserSimulator.is_stop(user_msg):
                self.done_flag = True
                self.termination_reason = TerminationReason.USER_STOP
                self.trajectory.append(user_msg)
                return
            self.trajectory.append(user_msg)
            last_user_text = user_msg.content

        # Main conversation loop
        while not self.done_flag:
            if last_user_text is None:
                break

            # Agent turn: feed user message to Strands agent
            call_count_before = tracker.call_count
            result = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    result = agent(last_user_text)
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES and _is_transient_error(e):
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            f"Transient error (attempt {attempt + 1}/{MAX_RETRIES + 1}), "
                            f"retrying in {delay}s: {e}"
                        )
                        time.sleep(delay)
                        continue
                    logger.error(f"Strands agent error: {e}")
                    self.done_flag = True
                    self.termination_reason = TerminationReason.AGENT_ERROR
                    break
            if self.done_flag:
                break

            # Extract trajectory from the Strands turn
            try:
                agent_turn_messages = self._extract_turn_from_strands(
                    agent.messages, tracker, call_count_before
                )
            except Exception as e:
                logger.error(f"Error extracting agent turn: {e}")
                self.done_flag = True
                self.termination_reason = TerminationReason.AGENT_ERROR
                break
            self.trajectory.extend(agent_turn_messages)

            # Validate agent messages against communication protocol
            for m in agent_turn_messages:
                if isinstance(m, AssistantMessage):
                    if self._check_communication(m, "agent"):
                        break
            if self.done_flag:
                break

            # Get the agent's final text response
            agent_text = str(result)

            # Check for agent stop
            if STOP_TOKEN in (agent_text or ""):
                self.done_flag = True
                self.termination_reason = TerminationReason.AGENT_STOP
                break

            # Check error limit
            if tracker.error_count >= self.max_errors:
                self.done_flag = True
                self.termination_reason = TerminationReason.TOO_MANY_ERRORS
                break

            # Increment step count (one full agent turn = one step)
            self.step_count += 1
            if self.step_count >= self.max_steps:
                self.done_flag = True
                self.termination_reason = TerminationReason.MAX_STEPS
                break

            # User turn
            last_agent_msg = self.trajectory[-1]
            try:
                user_msg, user_state = self.user.generate_next_message(
                    last_agent_msg, user_state
                )
                user_msg.validate()
            except Exception as e:
                logger.error(f"User simulator error: {e}")
                self.done_flag = True
                self.termination_reason = TerminationReason.USER_ERROR
                break

            if UserSimulator.is_stop(user_msg):
                self.done_flag = True
                self.termination_reason = TerminationReason.USER_STOP
                self.trajectory.append(user_msg)
                break

            self.trajectory.append(user_msg)

            # Validate user message against communication protocol
            if self._check_communication(user_msg, "user"):
                break

            # Handle user tool calls (e.g., telecom domain)
            if user_msg.is_tool_call():
                try:
                    user_msg, user_state = self._handle_user_tool_calls(
                        user_msg, user_state, tracker
                    )
                except Exception as e:
                    logger.error(f"User simulator error during tool calls: {e}")
                    self.done_flag = True
                    self.termination_reason = TerminationReason.USER_ERROR
                    break
                last_user_text = user_msg.content if user_msg.content else ""
                # Feed user tool results back to Strands agent context
                # The user's tool interaction is invisible to the agent,
                # but the user's final response is what matters
            else:
                last_user_text = user_msg.content

    def _run_solo(self):
        """Run solo mode: Strands agent with done() tool for termination."""
        from strands import Agent, tool as strands_tool
        from strands.agent.conversation_manager import NullConversationManager
        from strands.models.litellm import LiteLLMModel

        # Initialize environment
        initial_state = self.task.initial_state
        initialization_data = (
            initial_state.initialization_data if initial_state else None
        )
        initialization_actions = (
            initial_state.initialization_actions if initial_state else None
        )
        message_history = (
            deepcopy(initial_state.message_history)
            if initial_state and initial_state.message_history
            else []
        )
        for msg in message_history:
            msg.turn_idx = None
        message_history = self._add_timestamps(message_history)

        self._initialize_environment(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

        # Create tool tracker and Strands tools
        tracker = ToolTracker()
        strands_tools = convert_tools(
            environment=self.environment,
            tracker=tracker,
        )

        # Add user tools if available (solo mode gets both)
        if self.environment.user_tools:
            user_strands_tools = convert_user_tools(
                environment=self.environment,
                tracker=tracker,
            )
            strands_tools.extend(user_strands_tools)

        # Add done() tool
        done_called = [False]

        @strands_tool
        def done() -> str:
            """Call this function when you are done with the task."""
            done_called[0] = True
            return STOP_TOKEN

        strands_tools.append(done)

        # Build system prompt
        agent_instruction = AGENT_SOLO_INSTRUCTION
        system_prompt = SYSTEM_PROMPT_SOLO.format(
            agent_instruction=agent_instruction,
            domain_policy=self.environment.get_policy(),
            ticket=self.task.ticket,
        )

        # Build LiteLLM model params
        model_params = deepcopy(self.llm_args_agent)
        if self.seed is not None and "seed" not in model_params:
            model_params["seed"] = self.seed
        # Remove any stream override — the Strands SDK manages streaming
        # internally via its stream() method; overriding it to False causes
        # litellm.acompletion to return a non-iterable ModelResponse.
        model_params.pop("stream", None)

        model = LiteLLMModel(
            model_id=self.llm_agent,
            params=model_params,
        )

        agent = Agent(
            model=model,
            tools=strands_tools,
            system_prompt=system_prompt,
            callback_handler=self.callback_handler,
            conversation_manager=NullConversationManager(),
        )

        # Inject message history if present
        if message_history:
            self.trajectory = list(message_history)
            strands_history = self._tau2_to_strands_messages(message_history)
            agent.messages = strands_history

        # Run the agent loop
        # In solo mode, the Strands agent loops until it produces text or calls done()
        # We handle this with a loop that re-invokes the agent if needed
        prompt = None  # First call has no user prompt in solo mode

        # For solo mode, the entire task is in the system prompt.
        # We give the agent a kick to start by sending a minimal prompt.
        if not message_history:
            prompt = "Please proceed with solving the ticket."

        call_count_before = tracker.call_count

        result = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                result = agent(prompt)
                break
            except Exception as e:
                if attempt < MAX_RETRIES and _is_transient_error(e):
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Transient error in solo mode (attempt {attempt + 1}/{MAX_RETRIES + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                    continue
                logger.error(f"Strands agent error in solo mode: {e}")
                self.done_flag = True
                self.termination_reason = TerminationReason.AGENT_ERROR
                return

        # Extract trajectory
        try:
            agent_turn_messages = self._extract_turn_from_strands(
                agent.messages, tracker, call_count_before
            )
        except Exception as e:
            logger.error(f"Error extracting agent turn in solo mode: {e}")
            self.done_flag = True
            self.termination_reason = TerminationReason.AGENT_ERROR
            return
        self.trajectory.extend(agent_turn_messages)

        # Validate agent messages against communication protocol
        for m in agent_turn_messages:
            if isinstance(m, AssistantMessage):
                if self._check_communication(m, "agent"):
                    return

        if done_called[0]:
            # Add the done() call to trajectory as a stop indicator
            # Find the last AssistantMessage and mark it with stop token
            for msg in reversed(self.trajectory):
                if isinstance(msg, AssistantMessage):
                    if msg.is_tool_call():
                        for tc in msg.tool_calls:
                            if tc.name == "done":
                                msg.content = STOP_TOKEN
                                msg.tool_calls = None
                                break
                    break
            self.termination_reason = TerminationReason.AGENT_STOP
        elif STOP_TOKEN in str(result):
            self.termination_reason = TerminationReason.AGENT_STOP
        elif tracker.error_count >= self.max_errors:
            self.termination_reason = TerminationReason.TOO_MANY_ERRORS
        else:
            self.termination_reason = TerminationReason.AGENT_ERROR

        self.done_flag = True

    def _extract_turn_from_strands(
        self,
        strands_messages: list,
        tracker: ToolTracker,
        call_count_before: int,
    ) -> list[Message]:
        """Extract tau2 trajectory messages from a Strands agent turn.

        Parses the Strands message history to reconstruct properly grouped
        tau2 messages (AssistantMessage with tool_calls + ToolMessages + final text).
        """
        new_calls = tracker.get_calls_since(call_count_before)
        if not new_calls and not strands_messages:
            return []

        trajectory = []
        call_idx = 0

        # Find the new messages added during this turn
        # We scan backwards from the end to find all messages from this turn
        # A turn starts with a user message (the prompt) and ends with the final assistant message
        # We need to find tool-calling assistant messages and the final text response

        # Process tracked tool calls into batches by parsing Strands messages
        # The Strands message list contains interleaved assistant and user (tool result) messages

        # Simple approach: just build trajectory from tracked calls
        # Group consecutive calls into batches based on Strands message structure
        if new_calls:
            # Parse Strands messages to determine batch groupings
            batches = self._determine_tool_batches(strands_messages, len(new_calls), call_count_before)

            for batch_size in batches:
                batch_calls = new_calls[call_idx : call_idx + batch_size]
                call_idx += batch_size

                if batch_calls:
                    # Create AssistantMessage with grouped tool calls.
                    # Use the earliest ToolMessage timestamp so that after
                    # _finalize_trajectory sorts by timestamp the assistant
                    # message comes before its tool results.
                    tool_calls = [tc for tc, _ in batch_calls]
                    tool_msgs = [tm for _, tm in batch_calls]
                    earliest_tool_ts = min(
                        tm.timestamp for tm in tool_msgs if tm.timestamp
                    )
                    assistant_msg = AssistantMessage(
                        role="assistant",
                        tool_calls=tool_calls,
                        timestamp=earliest_tool_ts,
                    )
                    trajectory.append(assistant_msg)
                    trajectory.extend(tool_msgs)

        # Add final text response if present
        if strands_messages:
            last_msg = strands_messages[-1]
            if last_msg.get("role") == "assistant":
                text_parts = []
                has_tool_use = False
                for block in last_msg.get("content", []):
                    if "text" in block:
                        text_parts.append(block["text"])
                    if "toolUse" in block:
                        has_tool_use = True
                if text_parts:
                    text = "".join(text_parts)
                    if has_tool_use:
                        # Mixed text + tool calls in the same Strands
                        # message.  Attach the text to the last tool-call
                        # AssistantMessage so it is preserved in the
                        # trajectory (the communication check will flag
                        # this as a protocol violation when enabled).
                        for m in reversed(trajectory):
                            if (
                                isinstance(m, AssistantMessage)
                                and m.is_tool_call()
                            ):
                                m.content = text
                                break
                    else:
                        assistant_msg = AssistantMessage(
                            role="assistant",
                            content=text,
                            timestamp=get_now(),
                        )
                        trajectory.append(assistant_msg)

        return trajectory

    def _determine_tool_batches(
        self,
        strands_messages: list,
        total_new_calls: int,
        call_count_before: int,
    ) -> list[int]:
        """Determine how tool calls are grouped into batches from Strands messages.

        Returns a list of batch sizes (how many tool calls per model response).
        """
        batches = []
        calls_accounted = 0

        # Scan Strands messages for assistant messages with toolUse blocks
        for msg in strands_messages:
            if calls_accounted >= total_new_calls:
                break
            if msg.get("role") == "assistant":
                tool_uses = [
                    block for block in msg.get("content", []) if "toolUse" in block
                ]
                if tool_uses:
                    batch_size = len(tool_uses)
                    if calls_accounted + batch_size > total_new_calls:
                        # Only count remaining calls
                        batch_size = total_new_calls - calls_accounted
                    batches.append(batch_size)
                    calls_accounted += batch_size

        # If we couldn't determine batches from messages, treat all as one batch
        if calls_accounted < total_new_calls:
            remaining = total_new_calls - calls_accounted
            batches.append(remaining)

        return batches

    def _handle_user_tool_calls(
        self,
        user_msg: UserMessage,
        user_state: UserState,
        tracker: ToolTracker,
    ) -> tuple[UserMessage, UserState]:
        """Handle user tool calls (e.g., in telecom domain).

        Executes user tool calls against the environment and feeds results
        back to the user simulator until the user produces a text response.
        """
        while user_msg.is_tool_call():
            tool_msgs = []
            for tool_call in user_msg.tool_calls:
                tool_msg = self.environment.get_response(tool_call)
                if tool_msg.error:
                    tracker.error_count += 1
                tool_msgs.append(tool_msg)
                self.trajectory.append(tool_msg)

            # Feed tool results back to user
            from tau2.data_model.message import MultiToolMessage

            if len(tool_msgs) > 1:
                env_response = MultiToolMessage(role="tool", tool_messages=tool_msgs)
            else:
                env_response = tool_msgs[0]

            user_msg, user_state = self.user.generate_next_message(
                env_response, user_state
            )
            user_msg.validate()
            self.trajectory.append(user_msg)

            if UserSimulator.is_stop(user_msg):
                self.done_flag = True
                self.termination_reason = TerminationReason.USER_STOP
                break

        return user_msg, user_state

    def _check_communication(self, msg, role: str) -> bool:
        """Validate communication protocol for a message.

        Mirrors Orchestrator._check_communication_error().  When
        ``self.validate_communication`` is False this is a no-op.

        Args:
            msg: An AssistantMessage or UserMessage to validate.
            role: ``"agent"`` or ``"user"``.

        Returns:
            True if a violation was detected (simulation terminated).
        """
        if not self.validate_communication:
            return False

        termination = (
            TerminationReason.AGENT_ERROR
            if role == "agent"
            else TerminationReason.USER_ERROR
        )

        # Empty message (no text, no tool calls)
        if not msg.is_tool_call() and not msg.has_text_content():
            logger.error(f"{role} sent an empty message: {msg}")
            self.done_flag = True
            self.termination_reason = termination
            return True

        # Mixed message (both text and tool calls)
        if msg.is_tool_call() and msg.has_text_content():
            logger.error(
                f"{role} sent both text content and tool calls: {msg}"
            )
            self.done_flag = True
            self.termination_reason = termination
            return True

        # Solo mode: agent must only use tool calls (stop token exempt)
        if (
            role == "agent"
            and self.solo_mode
            and msg.has_text_content()
            and STOP_TOKEN not in (msg.content or "")
        ):
            logger.error(f"Agent sent text in solo mode: {msg}")
            self.done_flag = True
            self.termination_reason = termination
            return True

        return False

    def _initialize_environment(
        self,
        initialization_data: Optional[InitializationData],
        initialization_actions: Optional[list[EnvFunctionCall]],
        message_history: list[Message],
    ):
        """Initialize the environment state."""
        self.environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )
        self.environment.sync_tools()

    def _finalize_trajectory(self) -> list[Message]:
        """Sort trajectory by timestamp and assign turn indices."""
        messages = sorted(
            deepcopy(self.trajectory),
            key=lambda x: x.timestamp,
        )
        for i, msg in enumerate(messages):
            msg.turn_idx = i
        return messages

    @staticmethod
    def _add_timestamps(message_history: list[Message]) -> list[Message]:
        """Add timestamps to message history for proper ordering."""
        time_offset = datetime.now() - timedelta(seconds=len(message_history))
        for i, msg in enumerate(message_history):
            msg.timestamp = format_time(time_offset + timedelta(seconds=i))
        return message_history

    @staticmethod
    def _tau2_to_strands_messages(message_history: list[Message]) -> list[dict]:
        """Convert tau2 message history to Strands message format.

        Strands uses Bedrock-style messages:
        - {"role": "user", "content": [{"text": "..."}]}
        - {"role": "assistant", "content": [{"text": "..."}, {"toolUse": {...}}]}
        - {"role": "user", "content": [{"toolResult": {...}}]}
        """
        strands_messages = []

        for msg in message_history:
            if isinstance(msg, UserMessage):
                if msg.is_tool_call():
                    # User tool calls - skip for Strands context
                    # (agent doesn't see user tool calls)
                    continue
                strands_messages.append(
                    {"role": "user", "content": [{"text": msg.content or ""}]}
                )
            elif isinstance(msg, AssistantMessage):
                content_blocks = []
                if msg.content:
                    content_blocks.append({"text": msg.content})
                if msg.is_tool_call():
                    for tc in msg.tool_calls:
                        content_blocks.append(
                            {
                                "toolUse": {
                                    "name": tc.name,
                                    "input": tc.arguments,
                                    "toolUseId": tc.id,
                                }
                            }
                        )
                if content_blocks:
                    strands_messages.append(
                        {"role": "assistant", "content": content_blocks}
                    )
            elif isinstance(msg, ToolMessage):
                if msg.requestor == "assistant":
                    # Tool result for agent
                    status = "error" if msg.error else "success"
                    strands_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "toolResult": {
                                        "toolUseId": msg.id,
                                        "status": status,
                                        "content": [{"text": msg.content or ""}],
                                    }
                                }
                            ],
                        }
                    )
                # Skip user tool results (agent doesn't see them)

        return strands_messages
