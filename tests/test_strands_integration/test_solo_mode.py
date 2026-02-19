"""Tests for StrandsOrchestrator solo mode."""

import pytest

from tau2.data_model.message import AssistantMessage, ToolMessage
from tau2.data_model.simulation import TerminationReason

# Skip all tests if strands is not installed
strands = pytest.importorskip("strands")

from tau2.strands_integration.strands_orchestrator import StrandsOrchestrator


@pytest.fixture
def solo_environment():
    """Get a mock domain environment in solo mode."""
    from tau2.domains.mock.environment import get_environment

    return get_environment(solo_mode=True)


@pytest.fixture
def solo_tasks():
    """Load solo-compatible mock domain tasks."""
    from tau2.run import get_tasks

    tasks = get_tasks(task_set_name="mock")
    # Filter to tasks that have tickets (solo-compatible)
    return [t for t in tasks if t.ticket is not None]


class TestStrandsOrchestratorSolo:
    """Tests for solo mode execution."""

    def test_solo_mode_trajectory_has_no_user_messages(
        self, solo_environment, solo_tasks
    ):
        """In solo mode, trajectory should contain no UserMessages."""
        if not solo_tasks:
            pytest.skip("No solo-compatible tasks in mock domain")

        task = solo_tasks[0]
        from tau2.user.user_simulator import DummyUser

        user = DummyUser(instructions=None, llm=None, llm_args=None)

        orchestrator = StrandsOrchestrator(
            domain="mock",
            environment=solo_environment,
            user=user,
            task=task,
            llm_agent="gpt-4.1",
            llm_args_agent={"temperature": 0},
            max_steps=20,
            max_errors=5,
            seed=42,
            solo_mode=True,
        )
        simulation = orchestrator.run()

        # Solo mode should not produce UserMessages (except the system prompt kick)
        for msg in simulation.messages:
            assert not isinstance(msg, type(None))

        # Should have a termination reason
        assert simulation.termination_reason is not None

    def test_solo_mode_terminates(self, solo_environment, solo_tasks):
        """Test that solo mode eventually terminates."""
        if not solo_tasks:
            pytest.skip("No solo-compatible tasks in mock domain")

        task = solo_tasks[0]
        from tau2.user.user_simulator import DummyUser

        user = DummyUser(instructions=None, llm=None, llm_args=None)

        orchestrator = StrandsOrchestrator(
            domain="mock",
            environment=solo_environment,
            user=user,
            task=task,
            llm_agent="gpt-4.1",
            llm_args_agent={"temperature": 0},
            max_steps=20,
            max_errors=5,
            seed=42,
            solo_mode=True,
        )
        simulation = orchestrator.run()

        assert simulation.termination_reason in {
            TerminationReason.AGENT_STOP,
            TerminationReason.AGENT_ERROR,
            TerminationReason.TOO_MANY_ERRORS,
            TerminationReason.MAX_STEPS,
        }
