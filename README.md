# AIIDS — 自适应多智能体晶圆订单分配与调度系统

**AIIDS (Adaptive Intelligent Industrial Decision System)**

> ## ⚠️ 声明 / Disclaimer
>
> 本仓库是一个 **流程演示版本（workflow demonstration release）**，目的是让读者更清晰地理解 AIIDS 的**整体运转流程**——即「数据清洗 → 场景建模 → 决策优化」三阶段闭环，以及三大核心机制如何在这条流水线上协同工作。
>
> **本仓库不代表系统的完整实现。** 以下内容**未包含**在内：
>
> - 大规模实验框架与消融实验（`run_ablation.py`、`run_batch.py` 及 60 轮次的原始记录）
> - 参数标定流程（L9 正交实验、阈值网格搜索）与统计显著性分析脚本
> - Transformer 规则选择对照基线及其训练权重
> - 多规模算例集（本仓库仅提供 **1 个最小案例**，见 `case/`）
> - 工业部署组件、数据后处理与可视化脚本
>
> 论文中报告的完整实验结果、参数取值与统计结论，**请以论文为准**。本仓库的目标是「**可读**」——让读者看懂流程与机制，而非「可复现论文中的全部数字」。
>
> ---
>
> This repository is a **workflow demonstration release** intended to make the end-to-end operation of AIIDS — the three-stage pipeline (data cleaning → scenario modelling → decision optimisation) and how the three mechanisms cooperate along it — easier to follow. **It does not represent the complete implementation of the system.** Full experiment harnesses, parameter-calibration and statistical-analysis scripts, comparison baselines, additional case sets and deployment components are not included. Please refer to the paper for the complete results.

---

## 1. 系统架构

```
用户需求 → planning_agent（规划智能体）
              ├── data_agent     （数据清洗智能体）
              ├── scenario_agent （场景建模智能体）
              └── decision_agent （决策优化智能体）
                   ↓
         共享语义标签空间（pubsub_broker + semantic_tag_space）
```

三阶段流水线通过语义标签 `data2scenario → scenario2decision → decision_ready` 衔接。

## 2. 三大核心机制

### 机制① 两阶段系统提示词
- 静态提示词 **P₁ = {Role, Workflow, OutReq, ValRule}**（跨场景常量）
- 动态提示词 **P₂ = {SceneFeat, Workflow_f, TaskParam, CoopReq}**（场景相关参数）
- 语义融合 **P = F(P₁, P₂)**：由 LLM 完成逻辑拼接与语义归一化

实现：`dynamic_prompt_generator.py`、`system_prompt_fusion.py`

### 机制② 自适应分层协商与通信
- 全局关键信息压缩广播：**I_compress = LLM_compress(I^T)**
- 共享语义标签空间：X2Y 定向标签 + 语义匹配度 **S = Sim(Embed(L), Embed(L'))**，阈值 τ
- 两阶段确认（发布 → 回执 → 执行确认）+ 超时检测 + 规划智能体异常回退
- 通信有限状态机 **M = {Q, q₀, q_f, Σ, δ}**

实现：`pubsub_broker.py`、`pubsub_tools.py`、`semantic_tag_space.py`、`global_info_compressor.py`、`communication_fsm.py`、`fsm_supervisor.py`

### 机制③ 动态工具链调用
- 工具使用经验库 **E = {E_desc, E_order, E_exam}**（语义向量检索，RAG）
- 域感知检索：按 agent 域过滤 + 语义匹配度选择工具链，阈值 τ_e
- 可执行性判定 **Φ(Eᵢ)** + 事件驱动增量入库（Sim_max < τ_e 时自主构建新链）

实现：`tool_experience_library.py`、`tool_experience_data.json`

## 3. 目录结构

```
AIIDS/
├── main.py                       # 主入口：规划智能体编排 + 三大机制装配
├── KTPO.py                       # 决策优化算法（知识引导双种群 NSGA-II + DE）
├── dynamic_prompt_generator.py   # 机制①：动态提示词 P₂ 生成
├── system_prompt_fusion.py       # 机制①：F(P₁, P₂) 语义融合
├── global_info_compressor.py     # 机制②：全局信息压缩广播
├── semantic_tag_space.py         # 机制②：共享语义标签空间（匹配度 S + τ）
├── pubsub_broker.py              # 机制②：消息代理（回执 / 超时 / 异常回退）
├── pubsub_tools.py               # 机制②：通信工具集（publish/subscribe/receipt）
├── communication_fsm.py          # 机制②：通信有限状态机
├── fsm_supervisor.py             # 机制②：FSM 监督器（接入执行循环）
├── tool_experience_library.py    # 机制③：经验库（检索 / Φ 验证 / 增量更新）
├── tool_experience_data.json     # 机制③：初始经验库数据
├── embedding_service.py          # Embed(·)：Ollama BGE-M3，失败时字符 n-gram 回退
├── model_limits.py               # 按模型钳制输出预算 / 关闭思考模式
├── token_counter.py              # Token 用量统计
├── mechanism_probe.py            # 机制运行探针（输出验证报告）
├── planning_agent/               # 规划智能体提示词
├── data_agent/                   # 数据清洗智能体提示词
├── scenario_agent/               # 场景建模智能体提示词
├── decision_agent/               # 决策优化智能体提示词
├── case/
│   └── case.xlsx                 # 最小示例案例（14 订单 / 3 工厂 / 5 工艺节点）
├── .env.example                  # 环境变量模板
├── requirements.txt
└── README.md
```

## 4. 示例案例（`case/case.xlsx`）

为便于快速跑通全流程，仓库只附一个**最小案例**：

| 项目 | 规模 |
|---|---|
| 订单数 | 14 |
| 晶圆厂 | 3（F1 / F2 / F3） |
| 工艺节点 | 5 类（130 / 90 / 55 / 28 nm） |
| 订单批量 | 500 ~ 1000 片 |
| 处理时间 | 1 ~ 3 周期 T |
| 交付周期 | 4 ~ 8 T |

字段与原始数据一致：`订单编号 / 工艺节点 / 订单批量（片）/ 处理时间（周期T）/ 可处理工厂 / 交付时间（周期T）`。

## 5. 快速开始

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 配置环境变量（至少填 DEEPSEEK_API_KEY）
cp .env.example .env
#    编辑 .env，填入你的 API key

# 3) 可选：本地嵌入模型（机制② 语义匹配用）
#    Ollama + BGE-M3；未启动时自动回退到字符 n-gram，不影响流程跑通

# 4) 运行
python main.py
```

运行后会在当前目录生成：

| 文件 | 说明 |
|---|---|
| `mechanism_verification_report_*.json` | 机制运行验证报告（Φ 判定、语义匹配明细、FSM 状态） |
| `token_usage_report_*.json` | 各智能体 Token 用量统计 |
| `decision_result.json` | 决策结果（Pareto 前沿 + 推荐方案） |
| `scenario_model.py` | 场景智能体生成的仿真模型 |

## 6. 运行前置条件

| 依赖 | 用途 | 必需 |
|---|---|---|
| Python 3.10+ | — | ✅ |
| DeepSeek（或兼容 OpenAI 协议的）API | 多智能体推理 | ✅ |
| `deepagents` / `langchain` | 智能体框架 | ✅ |
| Ollama + BGE-M3 | 语义向量化 | 建议（缺失时降级） |

## 7. 许可与引用

如本仓库对您的研究有帮助，请引用对应论文。完整实验数据与统计结果请以论文为准。
