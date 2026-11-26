# AIIDS — Adaptive Multi-Agent Framework for Multi-Factory Wafer Order Allocation and Scheduling

**AIIDS (Adaptive Intelligent Industrial Decision System)**

> 📄 **If you use this repository, please cite:**
> *AIIDS: Multi-AI agent driven industrial data space framework for multi-factory wafer fabrication order allocation and scheduling*,
> **Journal of Manufacturing Systems**, Volume 90, February 2027, Pages 126–144.
> See [Section 7](#7-license-and-citation).

---

> ## ⚠️ Disclaimer
>
> This repository is a **workflow demonstration release**. It is intended to help readers follow the overall operation flow of AIIDS — the three-stage pipeline (data cleaning → scenario modelling → decision optimisation) and how the three core mechanisms cooperate along it.
>
> **It does not represent the complete implementation of the system.** The goal of this release is to make the system's operation flow easier to understand.

---

## 1. System Architecture

```
User request → planning_agent  (orchestration)
                  ├── data_agent      (data cleaning)
                  ├── scenario_agent  (scenario modelling)
                  └── decision_agent  (decision optimisation)
                       ↓
        Shared semantic tag space (pubsub_broker + semantic_tag_space)
```

The three stages are chained by semantic tags:
`data2scenario → scenario2decision → decision_ready`.

## 2. Core Mechanisms

### Mechanism ① — Two-Stage System Prompt
- **Static prompt** **P₁ = {Role, Workflow, OutReq, ValRule}** — constants shared across scenarios.
- **Dynamic prompt** **P₂ = {SceneFeat, Workflow_f, TaskParam, CoopReq}** — scenario-specific parameters.
- **Semantic fusion** **P = F(P₁, P₂)**: an LLM performs logical composition and semantic normalisation of the two stages.

Implementation: `dynamic_prompt_generator.py`, `system_prompt_fusion.py`

### Mechanism ② — Adaptive Hierarchical Negotiation and Communication
- **Global key-information compression broadcast**: **I_compress = LLM_compress(I^T)**
- **Shared semantic tag space**: X2Y directional tags with match score **S = Sim(Embed(L), Embed(L'))** under threshold τ.
- **Two-phase confirmation** (publish → receipt → execution confirmation), plus timeout detection and planner-side fallback dispatch.
- **Communication finite-state machine** **M = {Q, q₀, q_f, Σ, δ}**.

Implementation: `semantic_tag_space.py`, `pubsub_broker.py`, `pubsub_tools.py`, `global_info_compressor.py`, `communication_fsm.py`, `fsm_supervisor.py`

### Mechanism ③ — Dynamic Tool-Chain Invocation
- **Tool-usage experience library** **E = {E_desc, E_order, E_exam}**, retrieved by semantic vector similarity (RAG).
- **Domain-aware retrieval**: candidates are filtered by the requesting agent's domain, then ranked by semantic similarity under threshold τ_e.
- **Executability test** **Φ(Eᵢ)** with event-driven incremental ingestion: when Sim_max < τ_e the agent autonomously constructs a new chain, which is admitted only if Φ(Eᵢ) = 1.

Implementation: `tool_experience_library.py`, `tool_experience_data.json`

## 3. Repository Structure

```
AIIDS/
├── main.py                       # Entry point: planner orchestration + mechanism wiring
├── KTPO.py                       # Decision optimiser (knowledge-guided dual-population NSGA-II + DE)
├── dynamic_prompt_generator.py   # Mechanism ①: dynamic prompt P₂ generation
├── system_prompt_fusion.py       # Mechanism ①: semantic fusion F(P₁, P₂)
├── global_info_compressor.py     # Mechanism ②: global information compression and broadcast
├── semantic_tag_space.py         # Mechanism ②: shared semantic tag space (match score S + τ)
├── pubsub_broker.py              # Mechanism ②: message broker (receipts / timeout / fallback)
├── pubsub_tools.py               # Mechanism ②: communication tool set (publish / subscribe / receipt)
├── communication_fsm.py          # Mechanism ②: communication finite-state machine
├── fsm_supervisor.py             # Mechanism ②: FSM supervisor attached to the execution loop
├── tool_experience_library.py    # Mechanism ③: experience library (retrieval / Φ check / incremental update)
├── tool_experience_data.json     # Mechanism ③: initial experience-library data
├── embedding_service.py          # Embed(·): Ollama BGE-M3, with character n-gram fallback
├── model_limits.py               # Per-model output-budget clamping and thinking-mode control
├── token_counter.py              # Token-usage accounting
├── mechanism_probe.py            # Mechanism probe (emits the verification report)
├── planning_agent/               # Planner prompt
├── data_agent/                   # Data-cleaning agent prompt
├── scenario_agent/               # Scenario-modelling agent prompt
├── decision_agent/               # Decision-optimisation agent prompt
├── case/
│   └── case.xlsx                 # Minimal example case (14 orders / 3 fabs / 5 process nodes)
├── .env.example                  # Environment-variable template
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## 4. Example Case (`case/case.xlsx`)

To make the full pipeline runnable in a few minutes, the repository ships a single **minimal case**:

| Item | Scale |
|---|---|
| Orders | 14 |
| Wafer fabs | 3 (F1 / F2 / F3) |
| Process nodes | 5 (130 / 90 / 55 / 28 nm) |
| Order batch size | 500 – 1000 wafers |
| Processing time | 1 – 3 cycles (T) |
| Delivery cycle | 4 – 8 T |

The schema matches the original data:
`Order ID / Process node / Batch size (wafers) / Processing time (cycles T) / Eligible fabs / Delivery cycle (cycles T)`.

## 5. Quick Start

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Configure environment variables (at minimum DEEPSEEK_API_KEY)
cp .env.example .env
#    then edit .env and fill in your API key

# 3) Optional: local embedding model (used by Mechanism ②)
#    Ollama + BGE-M3; if unavailable, the pipeline falls back to a
#    character n-gram embedding and still runs end to end

# 4) Run
python main.py
```

Outputs written to the working directory:

| File | Description |
|---|---|
| `mechanism_verification_report_*.json` | Mechanism verification report (Φ results, semantic match details, FSM states) |
| `token_usage_report_*.json` | Per-agent token-usage statistics |
| `decision_result.json` | Decision result (Pareto front + recommended solution) |
| `scenario_model.py` | Simulation model generated by the scenario agent |

## 6. Requirements

| Dependency | Purpose | Required |
|---|---|---|
| Python 3.10+ | — | ✅ |
| DeepSeek (or any OpenAI-compatible) API | Multi-agent reasoning | ✅ |
| `deepagents` / `langchain` | Agent framework | ✅ |
| Ollama + BGE-M3 | Semantic embedding | Recommended (falls back gracefully) |

## 7. License and Citation

**Copyright © 2027. All rights reserved.**

The code, data and documentation in this repository are provided for **academic research and teaching** purposes.

> **⚠️ Condition of use: any work that uses the code, data or methods of this repository — including papers, theses, technical reports and open-source projects — must explicitly cite the following paper.**

### Reference

> **AIIDS: Multi-AI agent driven industrial data space framework for multi-factory wafer fabrication order allocation and scheduling**
> *Journal of Manufacturing Systems*, Volume 90, February 2027, Pages 126–144.

### BibTeX

```bibtex
@article{aiids2027,
  title   = {AIIDS: Multi-AI agent driven industrial data space framework for
             multi-factory wafer fabrication order allocation and scheduling},
  journal = {Journal of Manufacturing Systems},
  volume  = {90},
  pages   = {126--144},
  year    = {2027},
  month   = {February}
}
```

Commercial use requires prior written permission from the authors.
