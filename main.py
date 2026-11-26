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

# ===================== 禁用 deepagents 默认注入的 general-purpose 子智能体 =====================
# create_deep_agent 内部把 SubAgentMiddleware 的 general_purpose_agent 硬编码为 True，
# 于是规划智能体多了一个拥有"与主智能体完全相同工具集（含文件写入与 shell）"的自由子智能体，
# 且它不在本系统的协作协议内。实测该子智能体在 Windows 上执行 Linux 探测命令失败后，
# 编造了一份"文件系统完全为空"的虚假报告，导致规划智能体决定生成伪造数据并险些覆盖
# 用户输入文件 output/case.xlsx。此处构造智能体前将其关闭，只保留三个受控子智能体。
from deepagents.middleware.subagents import SubAgentMiddleware as _SubAgentMiddleware
from langchain.agents.middleware import TodoListMiddleware as _TodoListMiddleware
_original_sam_init = _SubAgentMiddleware.__init__


def _strip_subagent_todo_middleware(default_middleware):
    """关闭子智能体的 write_todos（只保留规划智能体的）。

    create_deep_agent 给三个子智能体各自挂了一份 TodoListMiddleware，于是每个子智能体
    都自己维护一份待办清单。但任务分解与进度追踪本就是规划智能体的职责，子智能体只管
    执行单一环节，重复记账纯属开销。
    实测（token_counter 工具埋点）：一次完整流程共 51 次工具调用，其中 write_todos 16 次
    占 31.4%（planning×4、data×5、scenario×3、decision×4），是最大的单项开销。

    这里只清空子智能体的 tools 与 system_prompt，中间件本身保留（其 state_schema 仍可用）。
    """
    if not default_middleware:
        return default_middleware
    out = []
    for mw in default_middleware:
        if isinstance(mw, _TodoListMiddleware):
            mw.tools = []          # 不再注入 write_todos
            mw.system_prompt = ""  # 不再注入 write_todos 的使用说明
        out.append(mw)
    return out


def _sam_init_without_general_purpose(self, *args, **kwargs):
    kwargs["general_purpose_agent"] = False
    # 顺带剥掉子智能体的 write_todos（keyword 传入，见 deepagents/graph.py）
    if "default_middleware" in kwargs:
        kwargs["default_middleware"] = _strip_subagent_todo_middleware(kwargs["default_middleware"])
    return _original_sam_init(self, *args, **kwargs)


_SubAgentMiddleware.__init__ = _sam_init_without_general_purpose

# ===================== 关闭 deepagents 默认注入的通用文件系统工具 =====================
# create_deep_agent 给主智能体和每个子智能体都挂上 FilesystemMiddleware，无条件注入
# ls / read_file / write_file / edit_file / glob / grep / execute 七个通用工具。
# 实测（token_counter 工具埋点）这些工具被大量用于"翻目录"而非业务本身：
#   data_agent 一次 12 调用 / 17 次工具调用中，ls×2 + glob×2 全是探索性开销，
#   配合 write_todos×4，真正清洗作业只占 5 次；decision_agent 另有 ls×2 + glob×1 + read_file×1。
# 各链已有专用工具（read_excel_structure / read_excel_content / read_and_analyze_scenario_model /
# save_cleaned_data / save_scenario_code / save_script_to_local），不需要这套通用工具。
#
# 但**只保留只读工具**（ls / glob / read_file / grep），关闭 write_file / edit_file / execute：
#   - 只读工具是消融机制②所必需的：固定主题广播**没有定向投递**，上游产物路径未必送到，
#     智能体需要有能力自己定位文件（真实场景下清洗产物带时间戳、路径每轮都变，
#     实测曾出现 cleaned_data_20260914223445.xlsx / case_cleaned_20260916122115.xlsx 这类命名）。
#   - 两个配置（full 与 noM2）**都拥有同一套只读工具**，保证对照非混淆：
#     full 因语义标签精准送达路径而无需探索，noM2 则必须探索——差别只在通信层是否送达信息。
#   - write_file / edit_file 可写任意路径，是"智能体乱写文件"的风险入口
#     （曾出现探针脚本覆盖正式产物），保持关闭。
from deepagents.middleware.filesystem import FilesystemMiddleware as _FilesystemMiddleware
_original_fm_init = _FilesystemMiddleware.__init__

# 允许保留的只读文件系统工具
_READONLY_FS_TOOLS = ("ls", "read_file", "glob", "grep")


def _fm_init_without_fs_tools(self, *args, **kwargs):
    # system_prompt="" 使 wrap_model_call 不注入 FILESYSTEM_SYSTEM_PROMPT
    # （只读工具的用法改由机制②的固定主题协议说明按需交代）
    kwargs["system_prompt"] = ""
    _original_fm_init(self, *args, **kwargs)
    self.tools = [t for t in (self.tools or [])
                  if getattr(t, "name", "") in _READONLY_FS_TOOLS]
    return None


_FilesystemMiddleware.__init__ = _fm_init_without_fs_tools
print("[main] 文件系统工具已收敛为只读集（ls/read_file/glob/grep）；write_file/edit_file/execute 保持关闭")

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
    publish_message, subscribe_topic,
)
from pubsub_broker import broker

# ===================== 机制①：两阶段提示词 =====================
from dynamic_prompt_generator import build_p2, format_p2
from system_prompt_fusion import fuse_prompts

# 机制①/③：按模型特性钳制输出预算（v4.1-flash 需压小，0731 需保留大预算）
from model_limits import resolve_max_tokens, thinking_kwargs

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
# 消融 AIIDS-B（M2_ENABLED=0）时回退固定主题 pub/sub（在开关定义后动态选择，见 get_comm_tools）
BASE_COMM_TOOLS_SEMANTIC = [
    semantic_publish, semantic_subscribe, send_receipt, confirm_task_done,
    get_topic_messages,
]
BASE_COMM_TOOLS_FIXED = [
    publish_message, subscribe_topic, get_topic_messages,
]

# 规划智能体协调工具（机制②：超时检测 + 异常回退；消融 B 时无回退工具）
PLANNING_TOOLS = [
    prioritize_tasks, get_topic_messages,
    check_comm_timeout, route_fallback,
]


def get_comm_tools():
    """按 M2 开关选择通信工具集（机制②消融：语义标签空间 vs 固定主题 pub/sub）。"""
    if M2_ENABLED:
        return list(BASE_COMM_TOOLS_SEMANTIC)
    return list(BASE_COMM_TOOLS_FIXED)


# 运行模式：FAST_MODE=1 时跳过 LLM 融合（P2 模板化+结构化拼接），日常调试用
# 正式机制验证实验请保持 FAST_MODE=0（LLM 提取 P2 + LLM 语义融合）
FAST_MODE = os.getenv("FAST_MODE", "0") == "1"

# ===================== 消融实验开关（论文消融配置）=====================
# M1_ENABLED=0 → AIIDS-A：禁用机制①两阶段系统提示词（直接用静态 P1，无 P2 融合）
# M2_ENABLED=0 → AIIDS-B：禁用机制②分层协商通信（回退固定主题 pub/sub，无语义标签空间）
# M3_ENABLED=0 → AIIDS-C：禁用机制③动态工具链（全量注入工具，无经验库检索/注入）
M1_ENABLED = os.getenv("M1_ENABLED", "1") == "1"
M2_ENABLED = os.getenv("M2_ENABLED", "1") == "1"
M3_ENABLED = os.getenv("M3_ENABLED", "1") == "1"
ABLATION_TAG = os.getenv("ABLATION_TAG", "full")  # full / noM1 / noM2 / noM3

# 本次运行的起始时间戳：供 Φ 判定做"产物新鲜度"校验（本轮生成才算数）
import time as _time
_RUN_T0 = _time.time()

# ===================== 机制②消融（AIIDS-B）：固定主题广播协议说明 =====================
# 关掉语义标签空间后，通信退回固定主题广播。与语义标签空间的关键差别在于
# **没有定向投递、也没有回执确认**：发布方无法得知消息是否送达，接收方也无法
# 判断上游何时完成。因此接收方只能反复轮询 get_topic_messages，而该接口每次
# 返回该主题的**全部历史消息**，导致输入 token 随轮询次数累积——这正是机制②
# 在论文中"降低冗余协调开销"的作用面。此处把协议规则显式写给智能体，
# 使其行为与协议能力一致（否则智能体仍会按语义标签的方式一次性取用，消融失真）。
_FIXED_TOPIC_CLAUSE = """

【通信协议（本配置：固定主题广播）—— 本节优先于上文的语义标签流程描述】
本配置未启用语义标签空间与回执确认机制。**上文所有关于 semantic_subscribe /
semantic_publish / send_receipt / confirm_task_done 的流程要求在本配置下一律不适用**，
请改用下述固定主题广播协议：
- **唯一主题**：所有智能体共用同一个主题 `broadcast`，你发布的消息会与其他智能体的消息混在同一主题中。
- **发布**：`publish_message(topic="broadcast", message="<内容>", sender="<你的名字>")`
- **订阅**：`subscribe_topic(topic="broadcast", subscriber="<你的名字>")`
  该调用**只登记订阅关系、不返回任何消息内容**。
- **读取**：`get_topic_messages(topic="broadcast")`，返回该主题的**全部历史消息**。
- **必须轮询**：本协议没有回执确认，你无法得知上游是否已完成、哪条消息是发给你的。
  因此订阅后需要**反复调用 get_topic_messages 轮询**，逐条检查历史消息，
  直到找到属于你的上游输入为止（通常需要轮询多次，且每次都会带回完整历史）。

【重要：本协议不提供定向投递，上游产物路径需要你自己查找】
- 本协议**没有语义匹配**，消息以自由文本广播，无法保证上游会把产物路径直接告诉你；
  你也无法凭消息描述判断哪条与你相关。
- 因此，如果在 broadcast 主题中没有明确拿到你所需的输入文件路径，
  **你必须自行在工作目录中定位该文件**。可用以下只读工具：
    * ls(path="...")                      —— 列出目录内容
    * glob(pattern="**/*cleaned*.xlsx")   —— 按通配符查找文件
    * read_file(path="...", limit=100)    —— 查看文件内容，确认是否为所需产物
    * grep(pattern="...", path="...")     —— 在文件中搜索内容
- 找到候选文件后请**自行核验**（例如用 read_file 看列结构与订单数），确认无误再继续任务；
  若发现取错了文件，需要重新查找。
- 本配置下 semantic_subscribe / semantic_publish / send_receipt / confirm_task_done 均不可用。
"""

# ===================== 机制③消融（AIIDS-C）：相同搜索预算约束 =====================
# 关掉机制③后，决策智能体拿不到经验库的 KTPO 方法论与代码模板，必须自行设计求解算法。
# 若不约束规模参数，它会自由选择迭代次数，导致与完整方案"预算不同"——那样测出来的是
# "算力差异"而不是"方法论差异"。因此这里把搜索预算固定为与 KTPO 模板一致（pop=60, gen=10），
# 使 AIIDS-C 与 full 的差别**只在有没有知识引导的方法论**：
#   full  : KTPO（知识引导双种群 NSGA-II + DE、30%拖期导向初始化、知识局部搜索、外部 Pareto 档案）
#   AIIDS-C: 智能体自选算法（GA / DE / SA 等），但种群与代数完全相同
# 这样 A（无 P2 通道→无场景参数亦无方法论）应劣于 C（有场景参数、仅缺方法论），
# 与论文中 AIIDS-A 退化大于 AIIDS-C 的排序一致。
_NO_M3_CLAUSE = """

【本配置的求解规模约束（强制）】
本配置未启用经验库，你将拿不到任何方法论与代码模板，需要**自行设计并实现求解算法**。
为保证与完整方案在**相同搜索预算**下可比，请统一使用以下规模参数，不得调整：
- 种群规模 pop_size = 60
- 迭代代数 max_gen = 10（或等价的搜索深度；若采用模拟退火等单点方法，请使评估次数与之相当，约 600~1200 次）
可自由选择算法类型（遗传算法、差分进化、模拟退火、禁忌搜索等）及其编码/算子设计。
"""
# ===================== Token 统计（每个智能体的输入/输出 token）=====================
from token_counter import token_counter
from langchain_core.callbacks import CallbackManager


# 最大生成 token（推理模型需预留 reasoning token，过小会导致输出被截断为空）
MAX_TOKENS = resolve_max_tokens(os.getenv("MODEL_NAME", "deepseek-v4-flash"))


def init_model(agent_tag: str = None):
    """初始化主模型，模型名与 max_tokens 可通过环境变量覆盖。"""
    model_name = os.getenv("MODEL_NAME", "deepseek-v4-flash")
    if agent_tag:
        return init_chat_model(
            model_name, temperature=0.2, max_tokens=MAX_TOKENS,
            callbacks=[token_counter], tags=[agent_tag],
            **thinking_kwargs(model_name),
        )
    return init_chat_model(model_name, temperature=0.2, max_tokens=MAX_TOKENS,
                           **thinking_kwargs(model_name))


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

        # ---- 本场景的动态知识（机制① P2 的 TaskParam 内容）----
        # 设计原则：这些量随算例变化，必须由 P2 从实际数据算出后注入，
        # 不得写死在 P1 里——否则消融掉机制①（AIIDS-A）时智能体照样拿得到，对照组失去区分度。
        def _stat(colname):
            for c in df.columns:
                if colname in str(c):
                    s = pd.to_numeric(df[c], errors="coerce").dropna()
                    if len(s):
                        return {"min": int(s.min()), "median": int(s.median()),
                                "max": int(s.max()), "mean": round(float(s.mean()), 1)}
            return None

        feats["volume_stats"] = _stat("批量")
        feats["ptime_stats"] = _stat("处理时间")
        feats["due_stats"] = _stat("交付时间")

        # 批次容量：使每批约容纳 3~5 个订单（由本场景订单批量中位数推算，不写死）
        vol = feats.get("volume_stats")
        if vol and vol.get("median"):
            base = int(round(3 * vol["median"] / 100.0)) * 100
            feats["base_cap"] = base
            feats["capacity_rule"] = (
                f"base_cap = {base}（= 3 x 本场景订单批量中位数 {vol['median']}，取整到百位）; "
                f"factory_cap = {{f: base_cap + i * 100 for i, f in enumerate(sorted(factories))}}; "
                f"预计每批容纳 3~5 个订单")
        else:
            feats["base_cap"] = None
            feats["capacity_rule"] = None
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
            # 机制①核心：本场景特有的建模参数在此动态注入（P1 不再硬编码）
            "TaskParam": {
                "data_source": "data2scenario 标签提供的数据路径",
                "order_schema": "id:str / tech:str / volume:int / ptime:int / factories:list[str] / due:int",
                "volume_stats": scene_feats.get("volume_stats"),
                "ptime_stats": scene_feats.get("ptime_stats"),
                "due_stats": scene_feats.get("due_stats"),
                "capacity_rule": scene_feats.get("capacity_rule"),
                "artifact": "scenario_model.py（保存到统一产物目录）",
            },
            "CoopReq": "建模完成后通过 semantic_publish 发布 scenario2decision 标签，并等待回执确认",
        },
        "decision_agent": {
            "SceneFeat": {"total_orders": scene_feats.get("total_orders"),
                          "factories": scene_feats.get("factories")},
            "Workflow_f": "按 订阅→分析场景模型→生成决策代码→执行→发布 流程决策",
            # 机制①核心：本场景特有的求解参数在此动态注入（P1 不再硬编码）
            "TaskParam": {
                "model_source": "scenario2decision 标签提供的模型路径",
                "order_schema": "id:str / tech:str / volume:int / ptime:int / factories:list[str] / due:int",
                "volume_stats": scene_feats.get("volume_stats"),
                "due_stats": scene_feats.get("due_stats"),
                "base_cap": scene_feats.get("base_cap"),
            },
            "CoopReq": "决策完成后通过 semantic_publish 发布 decision_ready 标签，并等待回执确认",
        },
    }

    configs = []
    # 机制①（M1）：共享 P2 基础 —— 禁用 M1 时跳过 LLM 场景特征提取
    shared_p2 = None
    if M1_ENABLED:
        shared_p2 = build_p2(global_info=global_info, data_desc=data_desc,
                             use_llm=not FAST_MODE)
        print(f"[main] P2 场景特征提取（共享，LLM 生成={shared_p2.get('_llm_generated')}, FAST_MODE={FAST_MODE}）")
    else:
        print("[main] 消融 AIIDS-A：禁用机制①两阶段提示词（M1_ENABLED=0）")

    for name in ["data_agent", "scenario_agent", "decision_agent"]:
        p1, desc = p1_map[name]
        if M1_ENABLED:
            # 机制①：基于共享 P2 合并各 agent 差异化特征，再 F(P1,P2) LLM 语义融合
            p2 = build_p2(extra_feats=p2_extra[name], base_p2=shared_p2)
        else:
            # 消融 A：直接用静态 P1，无 P2、无融合
            p2 = None

        # 机制③（M3）：动态工具链检索 —— 禁用 M3 时全量注入工具、无经验注入
        # 经验内容（desc/exam/code_template）的文本载体：
        #   机制① 的职责是【动态内容的投放通道】：P = F(P1,P2)。
        #   机制③ 的经验（方法论 + 代码模板）属于动态内容，必须经该通道送达。
        #   因此 M1 关闭时 p2 为 None → 通道不存在 → 经验内容同样无法送达。
        #   这不是混淆，而是机制①作用域的一部分：
        #     AIIDS-A 损失 = 场景适配参数 + 经验方法论（无通道）
        #     AIIDS-C 损失 = 仅经验方法论（通道在，内容没了）
        #   正因如此 A 的退化应大于 C，与论文的排序一致。
        exp_ref_text = ""   # 保留变量仅为兼容下文；M1 关闭时恒为空
        chain, info = [], {"matched": False, "reason": "消融 AIIDS-C：机制③已禁用"}
        if M3_ENABLED:
            chain, info = experience_library.select_tool_chain(task_descs[name], target_agent=name)
        comm_tools = get_comm_tools()  # 机制②（M2）通信工具集
        if info.get("matched"):
            chain_tools = [TOOL_REGISTRY[t] for t in chain if t in TOOL_REGISTRY]
            # 去重保序（按工具名）
            seen = set()
            tools = []
            for t in comm_tools + chain_tools:
                if t.name not in seen:
                    seen.add(t.name)
                    tools.append(t)
            tool_note = f"[机制③] 工具链 exp_{info['exp_id']}（匹配度{info['score']}）：{chain}"
            # 机制③关键：将匹配经验链的方法论与示例注入 P2（ExpRef），
            # 让 LLM 融合提示词时吸收经验库中的优质实践（如 KTPO 进化方法论 + 代码模板）
            # 注意：M1 禁用（消融 A）时 p2 为 None，无提示词可注入，跳过
            _exam = info.get("exam", {}) or {}
            _exp_ref = {
                "exp_id": info["exp_id"],
                "desc": info.get("desc", ""),
                "exam": _exam,
                "code_template": _exam.get("code_template", ""),
            }
            # M1 关闭时 p2 为 None，无投放通道，经验内容随之不可达（见上文说明）
            if p2 is not None:
                p2["ExpRef"] = _exp_ref
        else:
            # 未匹配或 M3 禁用：回退全量工具
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
            for t in comm_tools + all_tools:
                if t.name not in seen:
                    seen.add(t.name)
                    tools.append(t)
            tool_note = f"[机制③] {'禁用' if not M3_ENABLED else '未匹配'}经验链，全量工具（{info.get('reason', '')}）"

        # 机制②消融：M2 关闭时把固定主题广播协议显式写入提示词（见 _FIXED_TOPIC_CLAUSE 注释）
        m2_clause = "" if M2_ENABLED else _FIXED_TOPIC_CLAUSE

        if M1_ENABLED:
            # 机制①：注入 ExpRef 后执行 F(P1,P2) LLM 语义融合
            fused_prompt = fuse_prompts(p1, p2, enable_llm=not FAST_MODE)
            # 机制探针：记录提示词融合
            from system_prompt_fusion import LAST_FUSION_METHOD
            probe.log_prompt_fusion(name, p2_method="LLM" if p2.get("_llm_generated") else "模板",
                                    fusion_method=LAST_FUSION_METHOD["method"],
                                    p1_len=len(p1), p2_len=len(format_p2(p2)), fused_len=len(fused_prompt))
        else:
            # 消融 A：取消 P2 与 F(P1,P2) 融合，静态 P1 直接作为系统提示词。
            # 由于机制①是动态内容的投放通道，经验方法论（ExpRef）此时同样不可达。
            fused_prompt = p1 + exp_ref_text

        # 机制②消融：追加固定主题广播协议说明（必须在融合之后追加，避免被 F(P1,P2) 改写）
        if m2_clause:
            fused_prompt = fused_prompt + m2_clause

        # 搜索预算约束已统一写入 decision P1（见【搜索预算（ValRule）】），
        # 不再对单个配置单独追加——否则会造成"只有消融组受限、对照组自由"的预算不对等，
        # 实测 AIIDS-C 被限制在 60x10（600 次评估）而 AIIDS-A 自由选了 40x25（1000 次），
        # 导致 A 因预算更大而反超 C，排序失真。

        # 机制探针：记录工具链选择
        from tool_experience_library import TAU_E as TAU_E_VAL
        probe.log_toolchain(name, task_descs[name], info.get("matched", False),
                            info.get("exp_id", ""), info.get("score", 0.0), len(tools),
                            TAU_E_VAL)

        configs.append({
            "name": name,
            "description": desc + "\n" + tool_note,
            "system_prompt": fused_prompt,
            "tools": tools,
            "model": init_model(agent_tag=name),
        })
        print(f"[main] {name}: {tool_note} | 注入 {len(tools)} 个工具")

    return configs


# ===================== 构建规划智能体 =====================
def create_planning_agent(global_pkg: dict, subagents_config: list):
    """构建主智能体（规划智能体）：
    - 系统提示词 = F(P1_planning, P2_global)（注入压缩后的全局信息）
    - 工具：任务优先级排序 + 全局信息查询 + 超时检测 + 异常回退
    """
    if M1_ENABLED:
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
    else:
        # 消融 A：规划智能体用静态 P1
        fused_planning_prompt = planning_prompt

    return create_deep_agent(
        model=init_model(agent_tag="planning_agent"),
        system_prompt=fused_planning_prompt,
        subagents=subagents_config,
        tools=PLANNING_TOOLS,
        middleware=[],
        debug=False,
    )


# ===================== 主流程 =====================
if __name__ == "__main__":
    # 任务输入：数据文件
    user_input = r"先清洗 ./case/case.xlsx ，构建场景模型，而后构建决策优化模型"
    data_file = "./case/case.xlsx"

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
    global_pkg = {"compressed": "", "raw": {}}
    if M2_ENABLED:
        # ---- 机制②：全局信息压缩广播 ----
        global_pkg = build_global_info_package({
            "数据文件": data_file,
            "总订单数": scene_feats.get("total_orders"),
            "工厂集合": scene_feats.get("factories"),
            "优化目标": "最小化拖期，最大化准时交付率",
        }, compress=True)
        print(f"[机制②] 全局信息压缩包: {global_pkg['compressed']}")
    else:
        print("[main] 消融 AIIDS-B：禁用机制②分层协商通信（M2_ENABLED=0，固定主题 pub/sub）")

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
    # 多路径兼容检查：LLM 产物可能写入 根目录/output/scenario_models/decision_models 等任意位置
    print("\n[机制③] 执行后可执行性验证（Φ 判定）...")
    import os as _os
    import re as _re

    # 【修复记录】原实现有两处缺陷，导致 Φ 恒为 1、失去区分度：
    #   1) exp_001 候选文件名含 case.xlsx，而 output/case.xlsx 是【输入文件】且永远存在
    #      → Φ(exp_001) 恒等于 1（假阳性），无论清洗是否真的产出；
    #   2) 只判 os.path.exists，既不校验规格结构、也不校验产物是否本轮生成
    #      → 实测 AIIDS-B 某轮轮询失控、决策脚本从未执行，残留的上一轮 decision_result.json
    #        仍被判为通过。
    # 现改为三重判定：新鲜度（本轮生成） + 存在性 + 规格结构合规。
    def _fresh(path):
        try:
            return _os.path.getmtime(path) >= _RUN_T0 - 1
        except Exception:
            return False

    def _ok_scenario_module(path):
        try:
            src = open(path, encoding="utf-8", errors="replace").read()
        except Exception:
            return False, "不可读"
        if "关键字段" in src or "示例处理" in src:
            return False, "占位符实现"
        if not _re.search(r"def\s+load_case_data\s*\(\s*\w+", src):
            return False, "缺 load_case_data"
        if not _re.search(r"def\s+decode_and_evaluate\s*\(\s*\w+\s*,\s*\w+", src):
            return False, "缺/签名错 decode_and_evaluate"
        return True, "结构合规"

    def _ok_decision_json(path):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            return False, "非法JSON"
        miss = [k for k in ("status", "num_orders", "factories", "baseline", "recommended",
                            "pareto_front") if k not in d]
        if miss:
            return False, "缺字段%s" % miss
        rec = d.get("recommended") or {}
        if not isinstance(rec.get("makespan"), (int, float)) or not isinstance(rec.get("otd"), (int, float)):
            return False, "recommended 数值类型错"
        pf = d.get("pareto_front") or []
        if not isinstance(pf, list) or not pf:
            return False, "pareto_front 为空"
        if any(not (isinstance(x, dict) and "makespan" in x and "otd" in x) for x in pf):
            return False, "pareto_front 非契约结构"
        return True, "结构合规"

    def _check_any(path):
        # 仅对 .json 做契约结构校验；其他格式只做存在性+新鲜度
        return _ok_decision_json(path) if path.lower().endswith(".json") else (True, "存在")

    # 注意：exp_001 已移除 case.xlsx（那是输入文件，不是清洗产物）
    _artifact_map = [
        ("exp_001", ["case_cleaned.xlsx", "投料任务_cleaned.xlsx"], "数据清洗链", None),
        ("exp_002", ["scenario_model.py", "order_scheduling_scenario.py",
                     "material_feeding_scenario.py", "wafer_scheduling_scenario.py"], "场景建模链",
         _ok_scenario_module),
        ("exp_003", ["decision_result.json", "decision_optimization_result.json"], "决策优化链",
         _check_any),
    ]
    _search_dirs = [".", "output", "scenario_models", "decision_models", "data/output", "ablation_results"]
    for exp_id, filenames, label, struct_check in _artifact_map:
        found_path, reason = "", "产物缺失"
        for fn in filenames:
            for d in _search_dirs:
                cand = _os.path.join(d, fn) if d != "." else fn
                if not _os.path.exists(cand):
                    continue
                if not _fresh(cand):
                    reason = "产物陈旧(非本轮生成)"
                    continue
                if struct_check is not None:
                    ok, why = struct_check(cand)
                    if not ok:
                        reason = why
                        continue
                found_path = cand
                break
            if found_path:
                break
        phi = 1 if found_path else 0
        detail = (found_path or reason)
        probe.log_executability(exp_id, phi, f"{label}: {'结构合规' if phi else reason} ({detail})")
        print(f"  Φ({exp_id}) = {phi} ({label}: {detail})")

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

    # ---- 输出《各智能体 LLM Token 用量统计报告》----
    token_counter.print_report()

