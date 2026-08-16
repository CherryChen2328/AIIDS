# dynamic_prompt_generator.py
"""两阶段系统提示词机制 - 第二阶段动态提示词生成（论文机制①）。

P2 = {SceneFeat, Workflow_f, TaskParam, CoopReq}
- SceneFeat: 当前多工厂系统的调度场景特征（订单量、工厂数、工艺分布、产能状态）
- Workflow_f: 场景适配的流程细化（针对当前场景的粗粒度流程调整建议）
- TaskParam: 动态任务程序与约束（批量、交期、产能上限、优先级规则）
- CoopReq: 协作交互要求（与谁协作、发布/订阅哪些主题、回执确认要求）

由规划智能体（planning_agent）在第二阶段根据实时场景生成，并注入子智能体。
"""

import json
import os
import time
from typing import Dict, Optional

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

P2_MODEL = os.getenv("P2_MODEL", "deepseek:deepseek-v4-flash")

# 场景特征提取模板：从数据文件/全局信息中提取 P2 所需的场景特征
# 注意：JSON 示例中的花括号用 {{ }} 转义（避免与 .format() 冲突）
SCENE_FEAT_EXTRACT_PROMPT = """你是多工厂晶圆调度系统的场景分析师。请根据以下数据描述与全局信息，提取当前调度场景的关键特征。

请严格按 JSON 格式输出，包含以下字段：
{{
  "SceneFeat": {{
    "total_orders": 订单总数,
    "factories": 工厂列表或数量,
    "process_nodes": 工艺节点分布,
    "total_volume": 总订单量,
    "capacity_status": 产能状态简述
  }},
  "TaskParam": {{
    "batch_range": 订单批量范围,
    "due_range": 交付周期范围,
    "capacity_limits": 产能约束,
    "priority_rules": 优先级规则
  }},
  "Workflow_f": "针对该场景的执行流程细化建议（1-2句话）",
  "CoopReq": "协作要求（需要与哪些智能体协作、传递什么信息，1-2句话）"
}}

全局信息（压缩后）：
{global_info}

数据描述：
{data_desc}

只输出 JSON，不要输出其他内容。
"""


def _get_llm():
    return init_chat_model(P2_MODEL, temperature=0.2)


def extract_scene_features(global_info: str, data_desc: str) -> Dict:
    """LLM 提取场景特征（SceneFeat / TaskParam / Workflow_f / CoopReq）。

    Args:
        global_info: 压缩后的全局信息包（I_compress）
        data_desc: 数据文件描述（如 "100行x6列，订单/工艺/工厂/交期"）

    Returns:
        P2 字典；失败时返回降级默认值。
    """
    try:
        llm = _get_llm()
        prompt = SCENE_FEAT_EXTRACT_PROMPT.format(global_info=global_info or "无", data_desc=data_desc or "无")
        resp = llm.invoke(prompt)
        text = (resp.content or "").strip()
        # 提取 JSON（兼容模型输出带 ```json 包裹）
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        p2 = json.loads(text)
        return p2
    except Exception as e:
        print(f"[dynamic_prompt_generator] 场景特征提取失败，使用降级默认值: {e}")
        return {
            "SceneFeat": {"total_orders": None, "factories": None, "total_volume": None},
            "TaskParam": {"priority_rules": "交期紧迫度优先"},
            "Workflow_f": "按数据清洗→场景建模→决策优化标准流程执行",
            "CoopReq": "通过语义标签空间协作，发布/订阅对应主题并回执确认",
        }


def build_p2(global_info: str = "", data_desc: str = "",
             extra_feats: Optional[Dict] = None, use_llm: bool = True,
             base_p2: Optional[Dict] = None) -> Dict:
    """构建完整的动态提示词 P2 = {SceneFeat, Workflow_f, TaskParam, CoopReq}。

    Args:
        global_info: 压缩后的全局信息（可选，来自全局信息包）
        data_desc: 数据描述（可选）
        extra_feats: 外部已知特征（可选，如 {"total_orders": 100, "factories": 10}）
        use_llm: 是否调用 LLM 提取（False 时仅用 extra_feats 构建）
        base_p2: 已提取的共享 P2 基础（多 agent 场景下复用，避免重复 LLM 调用）

    Returns:
        P2 字典（含 created_at 时间戳）
    """
    if base_p2 is not None:
        # 复用共享 P2 基础（深拷贝，避免污染共享对象）
        import copy
        p2 = copy.deepcopy(base_p2)
    elif use_llm:
        p2 = extract_scene_features(global_info, data_desc)
        # 标记是否为 LLM 成功生成（extract_scene_features 失败时返回降级默认值，无法区分，
        # 这里通过检查关键字段是否被填充来判断；更稳妥：extract 内部标记）
        p2["_llm_generated"] = p2.get("SceneFeat", {}).get("total_orders") is not None
    else:
        p2 = {
            "SceneFeat": {},
            "TaskParam": {},
            "Workflow_f": "按数据清洗→场景建模→决策优化标准流程执行",
            "CoopReq": "通过语义标签空间协作，发布/订阅对应主题并回执确认",
        }
        p2["_llm_generated"] = False
    # 合并外部已知特征（LLM 结果优先保留，外部特征补充缺失项）
    if extra_feats:
        for k, v in extra_feats.items():
            if k == "SceneFeat":
                p2.setdefault("SceneFeat", {}).update({kk: vv for kk, vv in v.items() if kk not in p2.get("SceneFeat", {})})
            elif k not in p2 or p2[k] in (None, ""):
                p2[k] = v
    p2["created_at"] = time.time()
    return p2


def format_p2(p2: Dict) -> str:
    """将 P2 字典格式化为注入提示词的文本段。"""
    lines = []
    if p2.get("SceneFeat"):
        lines.append("【当前场景特征 SceneFeat】")
        for k, v in p2["SceneFeat"].items():
            if v is not None and str(v) != "":
                lines.append(f"- {k}: {v}")
    if p2.get("Workflow_f"):
        lines.append(f"【场景适配流程 Workflow_f】{p2['Workflow_f']}")
    if p2.get("TaskParam"):
        lines.append("【动态任务参数与约束 TaskParam】")
        for k, v in p2["TaskParam"].items():
            if v is not None and str(v) != "":
                lines.append(f"- {k}: {v}")
    if p2.get("CoopReq"):
        lines.append(f"【协作要求 CoopReq】{p2['CoopReq']}")
    if p2.get("ExpRef"):
        lines.append("【工具链经验参考 ExpRef（机制③经验库）】")
        exp_ref = p2["ExpRef"]
        lines.append(f"- 经验链: {exp_ref.get('exp_id', '')}")
        lines.append(f"- 方法论: {exp_ref.get('desc', '')}")
        if exp_ref.get("exam"):
            exam = exp_ref["exam"]
            lines.append(f"- 输入示例: {exam.get('in', '')}")
            lines.append(f"- 输出示例: {exam.get('out', '')}")
            lines.append(f"- 效果反馈: {exam.get('res', '')}")
        lines.append("- 重要：请遵循经验链的方法论与关键设计（如个体编码、解码器复用、算法组件），提升方案质量")
    return "\n".join(lines)


if __name__ == "__main__":
    p2 = build_p2(
        global_info="总订单交付周期72小时；高优先级订单优先保障；产能利用率目标90%。",
        data_desc="100行x6列：订单编号/工艺节点/订单批量/处理时间/可处理工厂/交付时间",
        extra_feats={"SceneFeat": {"total_orders": 100, "factories": 10}},
    )
    print(json.dumps(p2, ensure_ascii=False, indent=2))
    print("---格式化---")
    print(format_p2(p2))
