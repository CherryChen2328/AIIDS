# AIIDS — AIIDS: Multi-AI Agent Driven Industrial Data Space Framework for Multi-Factory Wafer Fabrication Order Allocation and Scheduling

**AIIDS (Multi-AI Agent Driven Industrial Data Space Framework)** is a multi-agent system for multi-fab wafer order allocation and scheduling. It integrates three core mechanisms to enable adaptive, reliable, and efficient collaboration among specialized AI agents:

1. **Two-Stage System Prompt Mechanism**
   - Static prompt P₁ = {Role, Workflow, OutReq, ValRule}: defines agent role, coarse-grained workflow, output requirements, and industrial value rules.
   - Dynamic prompt P₂ = {SceneFeat, Workflow_f, TaskParam, CoopReq}: captures real-time scheduling scenario features, scenario-adapted workflow refinements, dynamic task parameters, and collaboration requirements.
   - Semantic fusion P = F(P₁, P₂): an LLM-based fusion function that unifies the two prompt stages into a coherent system prompt.

2. **Adaptive Hierarchical Negotiation and Communication Mechanism**
   - Global key information compression and broadcast: I_compress = LLM_compress(I^T).
   - Shared semantic tag space: directional X2Y tags with semantic matching degree S = Sim(Embed(L), Embed(L')) and threshold τ.
   - Two-phase confirmation (publish → receipt → execution confirmation), communication timeout detection, and planner-driven fallback.
   - Formalized communication process via a finite state machine M = {Q, q₀, q_f, Σ, δ}.

3. **Dynamic Tool-Chain Invocation Mechanism**
   - Tool usage experience library E = {E_desc, E_order, E_exam} stored as a RAG database with semantic vector retrieval.
   - Domain-aware retrieval: candidates are filtered by agent domain and ranked by semantic similarity.
   - Executability verification Φ(Eᵢ) with event-driven incremental updates (Sim_max < τ_e triggers autonomous tool-chain construction).

## System Architecture

```
User Request → planning_agent (Planner)
                  ├── data_agent (Data Cleaning)
                  ├── scenario_agent (Scenario Modeling)
                  └── decision_agent (Decision Optimization)
                       ↓
        Shared Semantic Tag Space (pubsub + semantic_tag_space)
```

The system executes a three-stage pipeline — **data cleaning → scenario modeling → decision optimization** — connected through semantic tags `data2scenario → scenario2decision → decision_ready`.

The decision optimization stage employs a knowledge-guided dual-population multi-objective evolutionary approach (NSGA-II combined with differential evolution), featuring knowledge-based initialization, local search, and an external Pareto archive for maintaining the non-dominated solution set.

## Repository Structure

```
AIIDS/
├── main.py                       # Entry point (planner orchestration, integrates the three mechanisms)
├── planning_agent/               # Planner agent (task decomposition, coordination, fallback)
├── data_agent/                   # Data cleaning agent (missing values / outliers / noise)
├── scenario_agent/               # Scenario modeling agent (entity-relationship extraction, simulation model generation)
├── decision_agent/               # Decision optimization agent (evolutionary solving, Pareto front)
├── pubsub_broker.py              # Message broker (semantic tag space + receipts + timeout fallback)
├── pubsub_tools.py               # Communication tools (semantic_publish/subscribe/receipt)
├── semantic_tag_space.py         # Shared semantic tag space (X2Y tags + matching degree S + threshold τ)
├── embedding_service.py          # Text embedding Embed(·) (Ollama BGE-M3 / API / character fallback)
├── global_info_compressor.py     # Global information compression I_compress = LLM_compress(I^T)
├── communication_fsm.py          # Communication finite state machine M={Q,q0,qf,Σ,δ}
├── fsm_supervisor.py             # FSM supervisor (drives state transitions in the execution loop)
├── dynamic_prompt_generator.py   # Dynamic prompt P₂ generation (SceneFeat/Workflow_f/TaskParam/CoopReq)
├── system_prompt_fusion.py       # Prompt fusion F(P₁,P₂) (LLM-based semantic normalization)
├── tool_experience_library.py    # Tool-chain experience library (retrieval / Φ verification / incremental update)
├── tool_experience_data.json     # Initial experience library data (v1.2)
├── mechanism_probe.py            # Mechanism runtime probe (emits verification reports)
├── data/case.xlsx                # Example dataset (100 orders / 10 fabs / 4 process nodes)
└── .env.example                  # Environment variable template (copy to .env and fill in your API key)
```

## Installation

### Prerequisites

- Python 3.11+
- (Recommended) A conda virtual environment:

```bash
conda create -n aiids python=3.11
conda activate aiids
```

### Install Dependencies

```bash
pip install "langchain>=1.0" langchain-openai langchain-deepseek deepagents \
            pandas openpyxl python-dotenv sentence-transformers
```

### Optional: Local Embedding Model

For the semantic matching in Mechanism ②, the system uses an embedding model with the following priority:
1. Ollama local embedding (e.g., BGE-M3) — recommended for offline use:

```bash
ollama pull BGE-M3
```

2. Or configure an LLM embedding API via environment variables (see `.env.example`).

## Quick Start

### 1. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your `DEEPSEEK_API_KEY` (required for LLM inference).

### 2. Run the Full Multi-Agent Pipeline

```bash
python main.py
```

This executes the complete pipeline: data cleaning → scenario modeling → decision optimization, coordinated through the semantic tag space.

### 3. Fast Debug Mode (Optional)

To skip the LLM-based prompt fusion stages for quicker debugging:

```bash
FAST_MODE=1 python main.py
```

### 4. Check the Results

After a run, the following artifacts are generated:
- Cleaned data file under `output/` (or the path reported by `data_agent`)
- Scenario model code (e.g., `order_scheduling_scenario.py`)
- Decision results JSON (e.g., `decision_result.json`)
- `mechanism_verification_report_*.json` — runtime verification report of the three mechanisms

## Customization

- **Dataset**: replace `data/case.xlsx` (or update the input path in `main.py`) with your own order allocation and scheduling data.
- **Base LLM**: modify `init_model()` in `main.py` to switch to another OpenAI-compatible chat model.
- **Experience Library**: edit `tool_experience_data.json` to add or refine tool-chain usage experiences.
- **Mechanism Parameters**: thresholds such as τ (tag matching) and τ_e (tool-chain matching) can be tuned via environment variables (see `.env.example`).

## License

For academic research use only.
