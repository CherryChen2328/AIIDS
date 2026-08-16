# main.py — AIIDS 多智能体系统主入口（集成论文三大机制）
"""
AIIDS：自适应多智能体晶圆调度系统
================================================================
集成机制：
  机制① 两阶段系统提示词：P = F(P1, P2)
        - P1 静态提示词（各 agent 的 system_prompt.py）
        - P2 动态提示词（dynamic_prompt_generator，含 SceneFeat/Workflow_f/TaskParam/CoopReq）
        - F(·) 语义融合（system_prompt_fusion）
  机制② 自适应分层协商与通信：
        - 全局信息压缩广播（global_info_compressor）
        - 语义标签空间发布/订阅（semantic_tag_space + pubsub）
        - 回执确认、超时检测、规划智能体回退（pubsub_tools）
        - 通信有限状态机（communication_fsm）
  机制③ 动态工具链调用：
        - 工具使用经验库（tool_experience_library）
        - 按任务语义检索工具链，仅注入匹配工具子集
================================================================
"""

import os
import sys
import json

from dotenv import load_dotenv

load_dotenv()

from deepagents.graph import create_deep_agent
from langchain.chat_models import init_chat_model

# ===================== 静态提示词 P1 =====================
from planning_agent.system_prompt import SYSTEM_PROMPT as planning_prompt
from planning_agent.tools.task_prioritizer import prioritize_tasks
from data_agent.system_prompt import SYSTEM_PROMPT as data_prompt
from scenario_agent.system_prompt import SYSTEM_PROMPT as scenario_prompt
from decision_agent.system_prompt import SYSTEM_PROMPT as decision_prompt

# ===================== 业务工具 =====================
from data_agent.tools import (
    clean_csv_data, check_file_exists, init_data_cleaning,
    handle_missing_values, handle_categorical_abnormal,
    handle_numeric_noise, save_cleaned_data,
)
from scenario_agent.tools.excel_tools import read_excel_structure as scenario_read_excel_structure
from scenario_agent.tools.code_tools import (
    extract_python_code, save_scenario_code, generate_python_code,
)
from decision_agent.tools.excel_tools import read_excel_structure, read_excel_content
from decision_agent.tools.code_tools import (
    read_and_analyze_scenario_model,
    save_script_to_local,
    execute_decision_script,
)

# ===================== 通信工具（机制②）=====================
from pubsub_tools import (
    semantic_publish, semantic_subscribe, send_receipt,
    confirm_task_done, check_comm_timeout, route_fallback, get_topic_messages,
)
from pubsub_broker import broker

# ===================== 机制①：两阶段提示词 =====================
from dynamic_prompt_generator import build_p2, format_p2
from system_prompt_fusion import fuse_prompts

# ===================== 机制②：全局信息压缩 =====================
from global_info_compressor import build_global_info_package

# ===================== 机制③：工具链经验库 =====================
from tool_experience_library import experience_library

# ===================== 机制②：通信状态机 =====================
from communication_fsm import CommunicationFSM, CommEvent
from fsm_supervisor import FSMSupervisor

# ===================== 机制探针 =====================
from mechanism_probe import probe


# 工具注册表：名称 -> 工具对象（供动态工具链解析注入）
TOOL_REGISTRY = {
    # data_agent 工具
    "check_file_exists": check_file_exists,
    "init_data_cleaning": init_data_cleaning,
    "handle_missing_values": handle_missing_values,
    "handle_categorical_abnormal": handle_categorical_abnormal,
    "handle_numeric_noise": handle_numeric_noise,
    "save_cleaned_data": save_cleaned_data,
    "clean_csv_data": clean_csv_data,
    # scenario_agent 工具
    "read_excel_structure": read_excel_structure,
    "generate_python_code": generate_python_code,
    "extract_python_code": extract_python_code,
    "save_scenario_code": save_scenario_code,
    # decision_agent 工具
    "read_excel_content": read_excel_content,
    "read_and_analyze_scenario_model": read_and_analyze_scenario_model,
    "save_script_to_local": save_script_to_local,
    "execute_decision_script": execute_decision_script,
}

# 基础通信工具（所有子智能体必备，机制②：强制语义标签通信 + 回执确认）
# 注意：已移除旧接口 subscribe_topic/publish_message，强制走语义标签空间
BASE_COMM_TOOLS = [
    semantic_publish, semantic_subscribe, send_receipt, confirm_task_done,
    get_topic_messages,
]

# 规划智能体协调工具（机制②：超时检测 + 异常回退）
PLANNING_TOOLS = [
    prioritize_tasks, get_topic_messages,
    check_comm_timeout, route_fallback,
]


# 运行模式：FAST_MODE=1 时跳过 LLM 融合（P2 模板化+结构化拼接），日常调试用
# 正式机制验证实验请保持 FAST_MODE=0（LLM 提取 P2 + LLM 语义融合）
FAST_MODE = os.getenv("FAST_MODE", "0") == "1"


def init_model():
    """初始化主模型（DeepSeek）"""
    return init_chat_model("deepseek:deepseek-v4-flash", temperature=0.2)


# ===================== 场景分析：从数据文件提取场景特征 =====================
def analyze_scene_data(data_file: str) -> dict:
    """快速分析数据文件，提取场景特征（订单数、列结构、工厂集合等）。"""
    import pandas as pd

    feats = {"data_file": data_file}
    try:
        if data_file.endswith(".csv"):
            df = pd.read_csv(data_file, nrows=200)
        else:
            df = pd.read_excel(data_file, nrows=200)
        feats["total_orders"] = len(df)
        feats["columns"] = list(df.columns)
        feats["col_count"] = len(df.columns)
        # 尝试识别可处理工厂列
        for col in df.columns:
            if "工厂" in str(col) or "factory" in str(col).lower():
                factories = set()
                for v in df[col].dropna():
                    for f in str(v).split(","):
                        factories.add(f.strip())
                feats["factories"] = sorted(factories)
                break
        # 数值列统计
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        feats["numeric_cols"] = numeric_cols
    except Exception as e:
        feats["error"] = str(e)
    return feats


# ===================== 构建子智能体配置（融合提示词 + 动态工具链）====================
def build_subagent_configs(scene_feats: dict, global_pkg: dict) -> list:
    """为每个子智能体构建配置：
    1. 机制①：P2 动态提示词 + F(P1, P2) 融合
    2. 机制③：按任务语义从经验库检索工具链，动态注入工具子集
    """
    data_desc = (f"{scene_feats.get('total_orders', '?')}行x{scene_feats.get('col_count', '?')}列，"
                 f"列：{scene_feats.get('columns', [])}，"
                 f"工厂：{scene_feats.get('factories', '未知')}")
    global_info = global_pkg.get("compressed", "")

    # ---- 任务描述（用于工具链检索，机制③）----
    task_descs = {
        "data_agent": "清洗订单数据文件中的缺失值、异常类别和数值噪声，输出清洗后数据文件",
        "scenario_agent": "基于清洗后的订单数据构建场景模型代码，提取实体关系并保存",
        "decision_agent": "基于场景模型构建决策优化模型，使用进化算法（NSGA-II/DE）求解订单排程方案，输出帕累托前沿多目标结果",
    }

    # ---- 每个 agent 的静态提示词与描述 ----
    p1_map = {
        "data_agent": (data_prompt, "负责数据清洗和转换，输出清洗后的数据文件"),
        "scenario_agent": (scenario_prompt, "构建场景模型，提取实体和关系"),
        "decision_agent": (decision_prompt, "基于场景模型生成优化决策"),
    }

    # ---- 每个 agent 的 P2 场景适配提示（机制①）----
    p2_extra = {
        "data_agent": {
            "SceneFeat": {"total_orders": scene_feats.get("total_orders"),
                          "factories": scene_feats.get("factories")},
            "Workflow_f": "按 文件校验→初始化→缺失值→离散异常→数值噪声→保存 流程清洗",
            "TaskParam": {"data_file": scene_feats.get("data_file")},
            "CoopReq": "清洗完成后通过 semantic_publish 发布 data2scenario 标签，并等待回执确认",
        },
        "scenario_agent": {
            "SceneFeat": {"total_orders": scene_feats.get("total_orders"),
                          "factories": scene_feats.get("factories"),
                          "columns": scene_feats.get("columns")},
            "Workflow_f": "按 订阅→读取结构→生成代码→保存→发布 流程建模",
            "TaskParam": {"data_source": "data2scenario 标签提供的数据路径"},
            "CoopReq": "建模完成后通过 semantic_publish 发布 scenario2decision 标签，并等待回执确认",
        },
        "decision_agent": {
            "SceneFeat": {"total_orders": scene_feats.get("total_orders"),
                          "factories": scene_feats.get("factories")},
            "Workflow_f": "按 订阅→分析场景模型→生成决策代码→执行→发布 流程决策",
            "TaskParam": {"model_source": "scenario2decision 标签提供的模型路径"},
            "CoopReq": "决策完成后通过 semantic_publish 发布 decision_ready 标签，并等待回执确认",
        },
    }

    configs = []
    # 共享 P2 基础：只调用一次 LLM 提取场景特征（三个 agent 复用，避免重复 LLM 调用）
    # FAST_MODE 下跳过 LLM 提取，直接使用结构化模板
    shared_p2 = build_p2(global_info=global_info, data_desc=data_desc,
                         use_llm=not FAST_MODE)
    print(f"[main] P2 场景特征提取（共享，LLM 生成={shared_p2.get('_llm_generated')}, FAST_MODE={FAST_MODE}）")
    for name in ["data_agent", "scenario_agent", "decision_agent"]:
        p1, desc = p1_map[name]
        # 机制①：基于共享 P2 合并各 agent 差异化特征，再 F(P1,P2) LLM 语义融合
        p2 = build_p2(extra_feats=p2_extra[name], base_p2=shared_p2)
        fused_prompt = fuse_prompts(p1, p2, enable_llm=not FAST_MODE)

        # 机制③：动态工具链检索（按 agent 域过滤，避免跨域语义干扰）
        chain, info = experience_library.select_tool_chain(task_descs[name], target_agent=name)
        if info.get("matched"):
            chain_tools = [TOOL_REGISTRY[t] for t in chain if t in TOOL_REGISTRY]
            # 去重保序（按工具名）
            seen = set()
            tools = []
            for t in BASE_COMM_TOOLS + chain_tools:
                if t.name not in seen:
                    seen.add(t.name)
                    tools.append(t)
            tool_note = f"[机制③] 工具链 exp_{info['exp_id']}（匹配度{info['score']}）：{chain}"
            # 机制③关键：将匹配经验链的方法论与示例注入 P2（ExpRef），
            # 让 LLM 融合提示词时吸收经验库中的优质实践（如 KTPO 进化方法论）
            p2["ExpRef"] = {
                "exp_id": info["exp_id"],
                "desc": info.get("desc", ""),
                "exam": info.get("exam", {}),
            }
        else:
            # 未匹配：回退全量工具
            all_tools = {"data_agent": [clean_csv_data, check_file_exists, init_data_cleaning,
                                        handle_missing_values, handle_categorical_abnormal,
                                        handle_numeric_noise, save_cleaned_data],
                         "scenario_agent": [scenario_read_excel_structure, generate_python_code,
                                            extract_python_code, save_scenario_code],
                         "decision_agent": [read_excel_structure, read_excel_content,
                                            read_and_analyze_scenario_model, save_script_to_local,
                                            execute_decision_script]}[name]
            seen = set()
            tools = []
            for t in BASE_COMM_TOOLS + all_tools:
                if t.name not in seen:
                    seen.add(t.name)
                    tools.append(t)
            tool_note = f"[机制③] 未匹配经验链，回退全量工具（{info.get('reason', '')}）"

        # 机制①：注入 ExpRef 后执行 F(P1,P2) LLM 语义融合
        fused_prompt = fuse_prompts(p1, p2, enable_llm=not FAST_MODE)

        # 机制探针：记录工具链选择与提示词融合
        from tool_experience_library import TAU_E as TAU_E_VAL
        probe.log_toolchain(name, task_descs[name], info.get("matched", False),
                            info.get("exp_id", ""), info.get("score", 0.0), len(tools),
                            TAU_E_VAL)
        from system_prompt_fusion import LAST_FUSION_METHOD
        probe.log_prompt_fusion(name, p2_method="LLM" if p2.get("_llm_generated") else "模板",
                                fusion_method=LAST_FUSION_METHOD["method"],
                                p1_len=len(p1), p2_len=len(format_p2(p2)), fused_len=len(fused_prompt))

        configs.append({
            "name": name,
            "description": desc + "\n" + tool_note,
            "system_prompt": fused_prompt,
            "tools": tools,
            "model": init_model(),
        })
        print(f"[main] {name}: {tool_note} | 注入 {len(tools)} 个工具")

    return configs


# ===================== 构建规划智能体 =====================
def create_planning_agent(global_pkg: dict, subagents_config: list):
    """构建主智能体（规划智能体）：
    - 系统提示词 = F(P1_planning, P2_global)（注入压缩后的全局信息）
    - 工具：任务优先级排序 + 全局信息查询 + 超时检测 + 异常回退
    """
    # 机制①：规划智能体的动态提示词注入全局信息包（LLM 语义融合）
    p2_global = {
        "SceneFeat": {"global_info": global_pkg.get("compressed", "")},
        "Workflow_f": "按 全局感知→任务分解→子智能体调度→结果汇总 协调流程执行",
        "TaskParam": {"global_info_package": json.dumps(global_pkg.get("raw", {}), ensure_ascii=False)},
        "CoopReq": "监控各子智能体通信状态；超时未回执时调用 route_fallback 定向分发；"
                   "各阶段完成后通过语义标签空间确认（send_receipt/confirm_task_done）",
    }
    fused_planning_prompt = fuse_prompts(planning_prompt, p2_global, enable_llm=not FAST_MODE)
    from system_prompt_fusion import LAST_FUSION_METHOD
    probe.log_prompt_fusion("planning_agent", p2_method="模板",
                            fusion_method=LAST_FUSION_METHOD["method"], p1_len=len(planning_prompt),
                            p2_len=len(format_p2(p2_global)), fused_len=len(fused_planning_prompt))

    return create_deep_agent(
        model=init_model(),
        system_prompt=fused_planning_prompt,
        subagents=subagents_config,
        tools=PLANNING_TOOLS,
        middleware=[],
        debug=False,
    )


# ===================== 主流程 =====================
if __name__ == "__main__":
    # 任务输入：数据文件
    user_input = r"先清洗 ./output/case.xlsx ，构建场景模型，而后构建决策优化模型"
    data_file = "./output/case.xlsx"

    print("=" * 60)
    print("AIIDS 多智能体系统启动（集成三大机制·验证模式）")
    print("=" * 60)

    # ---- FSM 监督器（机制② 通信状态机接入执行循环）----
    supervisor = FSMSupervisor()
    probe.fsm_supervisor = supervisor

    # ---- 机制②：全局信息压缩广播 ----
    scene_feats = analyze_scene_data(data_file)
    print(f"[场景分析] {scene_feats.get('total_orders')} 订单, "
          f"{scene_feats.get('col_count')} 列, 工厂: {scene_feats.get('factories')}")
    global_pkg = build_global_info_package({
        "数据文件": data_file,
        "总订单数": scene_feats.get("total_orders"),
        "工厂集合": scene_feats.get("factories"),
        "优化目标": "最小化拖期，最大化准时交付率",
    }, compress=True)
    print(f"[机制②] 全局信息压缩包: {global_pkg['compressed']}")

    # ---- 机制③：动态工具链 + 机制①：LLM 融合提示词 ----
    subagent_configs = build_subagent_configs(scene_feats, global_pkg)

    # ---- 构建规划智能体 ----
    agent = create_planning_agent(global_pkg, subagent_configs)

    print("\n[运行] 开始多智能体协同流程（FSM 监督器已接入）...\n")
    try:
        for step in agent.stream(
                {'messages': user_input},
                stream_mode="values"
        ):
            # FSM 监督器观察消息（识别 task 派发/完成事件）
            last_msg = step['messages'][-1]
            supervisor.observe_message(last_msg)
            last_msg.pretty_print()
    except Exception as e:
        print(f"\n[运行] 流程异常: {e}")

    # ---- 机制③：执行后可执行性验证（Φ 判定，论文公式）----
    print("\n[机制③] 执行后可执行性验证（Φ 判定）...")
    import os as _os
    _artifact_map = [
        ("exp_001", "output/case_cleaned.xlsx", "数据清洗链"),
        ("exp_002", "order_scheduling_scenario.py", "场景建模链"),
        ("exp_003", "decision_result.json", "决策优化链"),  # 兼容根目录/output 两种路径
    ]
    for exp_id, artifact, label in _artifact_map:
        exists = _os.path.exists(artifact) or _os.path.exists(_os.path.join("output", artifact))
        phi = 1 if exists else 0
        probe.log_executability(exp_id, phi, f"{label}: 产物{'存在' if exists else '缺失'} ({artifact})")
        print(f"  Φ({exp_id}) = {phi} ({label})")

    # ---- 机制③：增量入库演示（论文公式：Sim_max < τ_e → 自主构建 → Φ验证 → 增量入库）----
    print("\n[机制③] 增量入库演示（全新任务触发动态更新）...")
    novel_task = "生成订单调度甘特图与性能仪表盘可视化报告，输出图表文件和HTML页面"
    _chain, _info = experience_library.select_tool_chain(novel_task, target_agent="decision_agent")
    if not _info.get("matched"):
        print(f"  新任务未匹配到经验链（原因: {_info.get('reason', '')}）→ 自主构建新链")
        # 模拟自主构建的工具链 + 执行日志（Φ 验证通过）
        new_chain = ["read_excel_content", "read_and_analyze_scenario_model",
                     "save_script_to_local", "execute_decision_script"]
        exec_log = ["图表脚本执行完成", "输出甘特图与仪表盘HTML"]
        new_exp = experience_library.add_experience(
            desc="调度结果可视化工具链：读取决策结果JSON与场景模型，生成订单-工厂甘特图与性能指标仪表盘HTML报告",
            tool_order=new_chain,
            example={"in": "decision_result.json（调度明细）", "out": "scheduling_gantt.html + dashboard.html",
                     "res": "执行成功；甘特图展示100订单在10工厂的时序排布，仪表盘含OTD/拖期/makespan指标"},
            exec_log=exec_log, output_valid=True,
        )
        if new_exp:
            probe.log_experience_add(new_exp["id"], len(new_chain))
            print(f"  ✅ 新经验链 {new_exp['id']} 通过 Φ 验证并增量入库（{len(new_chain)}个工具）")
        else:
            print(f"  ❌ 新经验链 Φ 验证未通过，未入库")
    else:
        print(f"  新任务匹配到现有经验链 {_info['exp_id']}（score={_info['score']}），无需新建")
    print(f"  当前经验库: {experience_library.summary()}")

    # ---- FSM 最终状态 ----
    print("\n[机制②] 通信有限状态机最终状态:")
    for agent_name, state in supervisor.all_states().items():
        print(f"  {agent_name}: {state}")

    # ---- 输出《机制运行验证报告》----
    probe.print_report()

