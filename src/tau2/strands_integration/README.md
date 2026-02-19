# Strands Agent SDK Integration

This module allows running tau2 benchmark evaluations using the [Strands Agents SDK](https://github.com/strands-agents/sdk-python) instead of the built-in orchestrator. The Strands orchestrator replaces the agent's inference and tool-execution loop while keeping the existing user simulator, environment, and evaluation pipeline unchanged.

## Installation

Install tau2 with the `strands` extra:

```bash
pip install -e '.[strands]'
```

This pulls in `strands-agents` and `strands-agents-tools` on top of the base dependencies.

## Usage

Pass `--agent-framework strands` to the `tau2 run` command:

```bash
# Conversational mode (agent + user simulator)
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-5.1 --agent-framework strands

# Solo mode (ticket-based, no user simulator)
tau2 run --domain airline --agent-llm gpt-4.1 --user dummy_user --agent-framework strands

# With JSONL trace logging enabled
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-5.1 --agent-framework strands --enable-trace

# Restrict to specific tasks
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-5.1 --agent-framework strands --task-ids task_001 task_002
```

All other flags (`--num-trials`, `--max-steps`, `--seed`, `--save-to`, etc.) work the same as with the default framework.

## How It Works

### Conversational Mode

The `StrandsOrchestrator` manages a turn-based loop:

1. The agent greeting is sent, and the user simulator responds.
2. The user's message is passed to a Strands `Agent`, which runs its internal loop (model inference → tool calls → environment → model → ... → text response).
3. The agent's text response is sent back to the user simulator.
4. Repeat until a stop condition is reached (user stop, agent stop, max steps, or too many errors).

tau2 environment tools are converted into Strands-compatible tool functions via `tool_converter.py`. The environment executes all tool calls, so tool behavior is identical to the default orchestrator.

### Solo Mode

For tasks that include a `ticket` field (used with `--user dummy_user`), the agent receives the ticket in its system prompt and calls tools autonomously until it calls the `done()` tool to signal completion. No user simulator is involved.

### Model Routing

The Strands integration uses `LiteLLMModel` under the hood, so any model string supported by [LiteLLM](https://docs.litellm.ai/docs/providers) works with `--agent-llm`. Set the appropriate API key environment variables in your `.env` file as usual.

## Trace Logging

Pass `--enable-trace` to write a JSONL file for every task/trial capturing all Strands callback events (model streaming chunks, tool invocations, etc.). Trace files are saved alongside simulation results:

```
data/simulations/traces/<run_name>/<task_id>_trial_<n>.jsonl
```

## Module Structure

| File | Purpose |
|---|---|
| `run.py` | Entry point (`run_task_strands`) mirroring `tau2.run.run_task` |
| `strands_orchestrator.py` | `StrandsOrchestrator` — manages the agent/user loop using a Strands `Agent` |
| `tool_converter.py` | Converts tau2 `Tool` objects into Strands `@tool`-decorated functions |
| `trace_handler.py` | `StrandsTraceHandler` callback that writes JSONL trace files |
