"""Integration tests for the StrandsOrchestrator."""

import pytest

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import TerminationReason

# Skip all tests if strands is not installed
strands = pytest.importorskip("strands")

from tau2.strands_integration.strands_orchestrator import StrandsOrchestrator


@pytest.fixture
def mock_environment():
    """Get a mock domain environment."""
    from tau2.domains.mock.environment import get_environment

    return get_environment()


@pytest.fixture
def mock_tasks():
    """Load mock domain tasks."""
    from tau2.run import get_tasks

    return get_tasks(task_set_name="mock")


@pytest.fixture
def mock_user():
    """Create a user simulator for mock domain."""
    from tau2.user.user_simulator import UserSimulator

    return UserSimulator


class TestStrandsOrchestratorConversational:
    """Tests for conversational mode."""

    def test_trajectory_has_correct_message_types(
        self, mock_environment, mock_tasks
    ):
        """Test that the trajectory contains proper tau2 message types."""
        task = mock_tasks[0]
        from tau2.user.user_simulator import UserSimulator

        user = UserSimulator(
            instructions=str(task.user_scenario),
            llm="gpt-4.1",
            llm_args={"temperature": 0},
        )

        orchestrator = StrandsOrchestrator(
            domain="mock",
            environment=mock_environment,
            user=user,
            task=task,
            llm_agent="gpt-4.1",
            llm_args_agent={"temperature": 0},
            max_steps=5,
            max_errors=5,
            seed=42,
        )
        simulation = orchestrator.run()

        # Verify basic trajectory structure
        assert len(simulation.messages) > 0
        assert simulation.termination_reason is not None

        # Check message types are valid tau2 types
        for msg in simulation.messages:
            assert isinstance(
                msg, (AssistantMessage, UserMessage, ToolMessage)
            ), f"Unexpected message type: {type(msg)}"

        # First message should be the agent greeting
        assert isinstance(simulation.messages[0], AssistantMessage)
        assert simulation.messages[0].content is not None

    def test_tool_calls_are_properly_paired(self, mock_environment, mock_tasks):
        """Test that each tool-calling AssistantMessage is followed by matching ToolMessages."""
        task = mock_tasks[0]
        from tau2.user.user_simulator import UserSimulator

        user = UserSimulator(
            instructions=str(task.user_scenario),
            llm="gpt-4.1",
            llm_args={"temperature": 0},
        )

        orchestrator = StrandsOrchestrator(
            domain="mock",
            environment=mock_environment,
            user=user,
            task=task,
            llm_agent="gpt-4.1",
            llm_args_agent={"temperature": 0},
            max_steps=5,
            max_errors=5,
            seed=42,
        )
        simulation = orchestrator.run()

        messages = simulation.messages
        i = 0
        while i < len(messages):
            msg = messages[i]
            if isinstance(msg, AssistantMessage) and msg.is_tool_call():
                # Next N messages should be ToolMessages
                num_tool_calls = len(msg.tool_calls)
                for j in range(num_tool_calls):
                    tool_msg = messages[i + 1 + j]
                    assert isinstance(tool_msg, ToolMessage), (
                        f"Expected ToolMessage after tool-calling AssistantMessage, "
                        f"got {type(tool_msg)} at index {i + 1 + j}"
                    )
                i += 1 + num_tool_calls
            else:
                i += 1

    def test_max_steps_terminates(self, mock_environment, mock_tasks):
        """Test that max_steps causes proper termination."""
        task = mock_tasks[0]
        from tau2.user.user_simulator import UserSimulator

        user = UserSimulator(
            instructions=str(task.user_scenario),
            llm="gpt-4.1",
            llm_args={"temperature": 0},
        )

        orchestrator = StrandsOrchestrator(
            domain="mock",
            environment=mock_environment,
            user=user,
            task=task,
            llm_agent="gpt-4.1",
            llm_args_agent={"temperature": 0},
            max_steps=1,  # Very low limit
            max_errors=5,
            seed=42,
        )
        simulation = orchestrator.run()

        assert simulation.termination_reason in {
            TerminationReason.MAX_STEPS,
            TerminationReason.AGENT_STOP,
            TerminationReason.USER_STOP,
        }
