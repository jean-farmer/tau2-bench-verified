"""
Entry point for running tau2 tasks with the Strands Agent SDK.
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.registry import registry
from tau2.strands_integration.strands_orchestrator import StrandsOrchestrator
from tau2.strands_integration.trace_handler import StrandsTraceHandler, make_trace_path
from tau2.user.user_simulator import DummyUser, UserSimulator


def run_task_strands(
    domain: str,
    task: Task,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    max_steps: int = 100,
    max_errors: int = 10,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    seed: Optional[int] = None,
    enforce_communication_protocol: bool = False,
    user: str = "user_simulator",
    enable_trace: bool = False,
    save_to: Optional[Path] = None,
    trial: Optional[int] = None,
) -> SimulationRun:
    """Run a single task using the Strands Agent SDK.

    This function mirrors run.run_task() but uses StrandsOrchestrator
    instead of the standard Orchestrator.

    Args:
        domain: The domain name.
        task: The task to run.
        llm_agent: The LLM model ID for the agent.
        llm_args_agent: Additional args for the agent LLM.
        llm_user: The LLM model ID for the user simulator.
        llm_args_user: Additional args for the user LLM.
        max_steps: Maximum simulation steps.
        max_errors: Maximum tool errors before termination.
        evaluation_type: Type of evaluation to perform.
        seed: Random seed for reproducibility.
        enforce_communication_protocol: Whether to enforce protocol rules.
        user: The user implementation name.
        enable_trace: Whether to write JSONL trace files for Strands callback events.
        save_to: Path to the simulation results file (used to derive trace path).
        trial: Trial number (used in trace file naming).

    Returns:
        SimulationRun with trajectory and evaluation results.
    """
    if max_steps <= 0:
        raise ValueError("Max steps must be greater than 0")
    if max_errors <= 0:
        raise ValueError("Max errors must be greater than 0")

    logger.info(
        f"STARTING STRANDS SIMULATION: Domain: {domain}, Task: {task.id}, "
        f"Agent LLM: {llm_agent}, User: {user}"
    )

    # Determine solo mode
    solo_mode = task.ticket is not None and user == "dummy_user"

    # Create environment
    environment_constructor = registry.get_env_constructor(domain)
    environment = environment_constructor(solo_mode=solo_mode) if solo_mode else environment_constructor()

    # Create user
    try:
        user_tools = environment.get_user_tools()
    except Exception:
        user_tools = None

    UserConstructor = registry.get_user_constructor(user)
    user_instance = UserConstructor(
        tools=user_tools,
        instructions=str(task.user_scenario),
        llm=llm_user,
        llm_args=llm_args_user,
    )

    # Set up trace handler if enabled
    trace_handler = None
    if enable_trace and save_to is not None:
        trace_path = make_trace_path(save_to, task.id, trial if trial is not None else 0)
        if trace_path is not None:
            trace_handler = StrandsTraceHandler(trace_path)

    # Create and run orchestrator
    orchestrator = StrandsOrchestrator(
        domain=domain,
        environment=environment,
        user=user_instance,
        task=task,
        llm_agent=llm_agent,
        llm_args_agent=llm_args_agent,
        max_steps=max_steps,
        max_errors=max_errors,
        seed=seed,
        solo_mode=solo_mode,
        validate_communication=enforce_communication_protocol,
        callback_handler=trace_handler,
    )
    if trace_handler is not None:
        trace_handler.open()
    try:
        simulation = orchestrator.run()
    finally:
        if trace_handler is not None:
            trace_handler.close()

    # Evaluate
    reward_info = evaluate_simulation(
        domain=domain,
        task=task,
        simulation=simulation,
        evaluation_type=evaluation_type,
        solo_mode=solo_mode,
    )
    simulation.reward_info = reward_info

    logger.info(
        f"FINISHED STRANDS SIMULATION: Domain: {domain}, Task: {task.id}, "
        f"Reward: {reward_info.reward}"
    )
    return simulation
