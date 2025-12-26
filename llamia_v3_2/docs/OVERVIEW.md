# Llamia Architecture Overview

## Core Components

### 1. State Managementstate.py`)
- **LlamiaState**: Central data container
- Persists across execution turns
- Manages workflow routing

### 2. Node Agentsnodes/`)
- **Planner**: Task decomposition
- **Coder**: Code generation
- **Critic**: Quality assurance
- **Researcher**: Web/data gathering

### 3. Supporting Modules
- **LLM Client**: Model interactionsllm_client.py`)
- **Executor**: Safe code executionexecutor.py`)
- **Configuration**: Settingsconfig.py)

## Workflow Diagram
(Add architecture diagram here when available)
