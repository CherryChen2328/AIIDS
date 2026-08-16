# AIIDS — 自适应多智能体晶圆调度系统

**AIIDS (Adaptive Intelligent Industrial Decision System)**：面向多工厂晶圆订单分配与调度的自适应多智能体系统，集成了三大创新机制：

1. **机制① 两阶段系统提示词（Two-Stage System Prompt Mechanism）**
   - 静态提示词 P₁ = {Role, Workflow, OutReq, ValRule}（角色/流程/输出约束/价值准则）
   - 动态提示词 P₂ = {SceneFeat, Workflow_f, TaskParam, CoopReq}（场景特征/流程细化/任务参数/协作要求）
   - 语义融合 P = F(P₁, P₂)：LLM 对两阶段提示词进行逻辑拼接与语义归一化

2. **机制② 自适应分层协商与通信（Adaptive Hierarchical Negotiation and Communication Mechanism）**
   - 全局关键信息压缩广播：I_compress = LLM_compress(I^T)
   - 共享语义标签空间：X2Y 定向标签 + 语义匹配度 S = Sim(Embed(L), Embed(L'))，阈值 τ
   - 两阶段确认机制（发布 → 回执 → 执行确认）+ 通信超时检测 + 规划智能体异常回退
   - 通信过程有限状态机形式化 M = {Q, q₀, q_f, Σ, δ}

3. **机制③ 动态工具链调用（Dynamic Tool-Chain Invocation Mechanism）**
   - 工具使用经验库 E = {E_desc, E_order, E_exam}（RAG 存储，语义向量检索）
   - 域感知检索：按 agent 域过滤 + 语义匹配度选择工具链
   - 可执行性判定 Φ(Eᵢ) + 事件驱动增量入库（Sim_max < τ_e → 自主构建新链）

## 系统架构

```
用户需求 → planning_agent（规划智能体）
              ├── data_agent（数据清洗智能体）
              ├── scenario_agent（场景建模智能体）
              └── decision_agent（决策优化智能体）
                   ↓
          共享语义标签空间（pubsub + semantic_tag_space）
```

三阶段流水线：**数据清洗 → 场景建模 → 决策优化**，通过语义标签 `data2scenario → scenario2decision → decision_ready` 无缝衔接。

## 目录结构

```
AIIDS/
├── main.py                       # 主入口（规划智能体编排，集成三大机制）
├── planning_agent/               # 规划智能体（任务分解、进度协调、异常回退）
├── data_agent/                   # 数据清洗智能体（缺失值/异常/噪声处理）
├── scenario_agent/               # 场景建模智能体（实体关系提取、仿真模型生成）
├── decision_agent/               # 决策优化智能体（进化算法求解、Pareto 前沿）
├── pubsub_broker.py              # 消息代理（语义标签空间 + 回执 + 超时回退）
├── pubsub_tools.py               # 通信工具集（semantic_publish/subscribe/receipt）
├── semantic_tag_space.py         # 共享语义标签空间（X2Y 标签 + 匹配度 S + τ 阈值）
├── embedding_service.py          # 文本向量化 Embed(·)（Ollama BGE-M3 / API / 字符回退）
├── global_info_compressor.py     # 全局信息压缩 I_compress = LLM_compress(I^T)
├── communication_fsm.py          # 通信有限状态机 M={Q,q0,qf,Σ,δ}
├── fsm_supervisor.py             # FSM 监督器（接入执行循环，驱动状态转移）
├── dynamic_prompt_generator.py   # 动态提示词 P₂ 生成（SceneFeat/Workflow_f/TaskParam/CoopReq）
├── system_prompt_fusion.py       # 提示词融合 F(P₁,P₂)（LLM 语义归一化）
├── tool_experience_library.py    # 工具链经验库（检索/Φ验证/增量更新）
├── tool_experience_data.json     # 初始经验库数据（v1.2，含 KTPO 方法论与合批规则）
├── mechanism_probe.py            # 机制运行探针（输出《机制运行验证报告》）
├── KTPO.py                       # 对照算法：知识引导双种群进化（NSGA-II+DE）
├── run_ktpo_case.py              # KTPO 对照实验脚本
├── data/case.xlsx                # 实验数据（100订单/10工厂/4工艺节点）
└── .env.example                  # 环境变量模板（复制为 .env 并填入 API key）
```

## 快速开始

### 环境要求
- Python 3.11+（建议 conda 虚拟环境）
- `deepagents`、`langchain`（1.0+）、`langchain-openai`、`pandas`、`openpyxl`、`python-dotenv`
- 可选：本地 Ollama + BGE-M3 嵌入模型（机制②语义匹配，也可配置 LLM 嵌入 API）

### 运行

```bash
# 1. 配置环境变量
cp .env.example .env   # 填入 DEEPSEEK_API_KEY

# 2. 运行多智能体全流程（完整机制验证模式）
python main.py

# 3. 快速调试模式（跳过 LLM 提示词融合，速度更快）
FAST_MODE=1 python main.py
```

### 对照实验（KTPO 进化算法）

```bash
python run_ktpo_case.py   # 用 data/case.xlsx 运行 KTPO，输出 Pareto 前沿
```

## 实验结果

| 方法 | Makespan (T) | 准时交付率 OTD | 总拖期 |
|---|---|---|---|
| LLM 贪心基线 | 25 | 28% | 487 |
| **AIIDS（机制③经验注入）** | **11** | **76%** | **69** |
| KTPO 原生（对照） | 13 | 76% | — |

## 引用

本项目的三大机制设计对应论文：

> Two-Stage System Prompt Mechanism / Adaptive Hierarchical Negotiation and Communication Mechanism / Dynamic Tool-Chain Invocation Mechanism

## 许可

仅供学术研究使用。
