# system_prompt_fusion.py
"""两阶段系统提示词机制 - 语义融合函数（论文机制①）。

核心公式：P = F(P1, P2)
- P1：初始静态提示词 = {Role, Workflow, OutReq, ValRule}
- P2：动态提示词 = {SceneFeat, Workflow_f, TaskParam, CoopReq}
- F(·)：语义融合函数——通过 LLM 对两阶段提示词进行逻辑拼接与语义归一化，
  避免指令冲突，形成统一结构化表达，作为智能体的最终系统提示词。

工程实现采用"结构化拼接 + LLM 归一化"：
1. 结构化拼接：P1 为主体，P2 以动态适配区块注入（保证 P1 的稳定基线不被破坏）
2. LLM 归一化（可选开关）：当 enable_llm_fusion=True 时，调用 LLM 生成融合后的
   统一系统提示词（消除指令冲突、冗余与矛盾）
"""

import os
from typing import Optional

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

from dynamic_prompt_generator import format_p2

load_dotenv()

FUSION_MODEL = os.getenv("FUSION_MODEL", "deepseek:deepseek-v4-flash")

# 最近一次融合实际使用的方法（"llm" / "structural"），供机制探针读取
LAST_FUSION_METHOD = {"method": "structural"}


def _get_llm():
    return init_chat_model(FUSION_MODEL, temperature=0.1)


# LLM 归一化模板
FUSION_PROMPT_TEMPLATE = """你是提示词工程专家。请将"静态基础提示词"与"动态场景适配提示词"融合为一份统一、无冲突、结构清晰的智能体系统提示词。

要求：
1. 以静态基础提示词为主体框架，保留其全部核心规范（角色、流程、输出要求、价值准则）
2. 将动态场景适配内容（场景特征、任务参数、流程细化、协作要求）自然融入对应部分
3. 消除两段提示词之间的指令冲突与冗余表述
4. 输出为可直接作为系统提示词使用的完整文本，不要额外解释

===== 静态基础提示词 P1（角色/流程/输出要求/价值准则）=====
{p1}

===== 动态场景适配提示词 P2（场景特征/任务参数/流程细化/协作要求）=====
{p2}

===== 融合后的系统提示词 =====
"""


def structural_fusion(p1: str, p2: dict) -> str:
    """结构化拼接融合（不调用 LLM，稳定快速）。

    以 P1 为主体，将 P2 的格式化文本作为"动态场景适配区块"追加。
    """
    p2_text = format_p2(p2)
    if not p2_text:
        return p1
    return f"{p1}\n\n# ===== 动态场景适配（本任务注入）=====\n{p2_text}"


def llm_fusion(p1: str, p2: dict) -> str:
    """LLM 语义融合（论文 F(·) 的完整实现：逻辑拼接 + 语义归一化）。"""
    p2_text = format_p2(p2)
    try:
        llm = _get_llm()
        prompt = FUSION_PROMPT_TEMPLATE.format(p1=p1, p2=p2_text)
        resp = llm.invoke(prompt)
        fused = (resp.content or "").strip()
        if fused:
            LAST_FUSION_METHOD["method"] = "llm"
            return fused
        LAST_FUSION_METHOD["method"] = "structural"
        return structural_fusion(p1, p2)
    except Exception as e:
        print(f"[system_prompt_fusion] LLM 融合失败，降级为结构化拼接: {e}")
        LAST_FUSION_METHOD["method"] = "structural"
        return structural_fusion(p1, p2)


def fuse_prompts(p1: str, p2: dict, enable_llm: bool = True) -> str:
    """两阶段提示词融合入口：P = F(P1, P2)。

    Args:
        p1: 静态提示词（各 agent 的 SYSTEM_PROMPT）
        p2: 动态提示词字典（dynamic_prompt_generator.build_p2 的输出）
        enable_llm: True 用 LLM 归一化融合；False 用结构化拼接（更快、零成本）

    Returns:
        融合后的最终系统提示词 P
    """
    if enable_llm:
        return llm_fusion(p1, p2)
    LAST_FUSION_METHOD["method"] = "structural"
    return structural_fusion(p1, p2)


if __name__ == "__main__":
    # 自测：用 data_agent 的静态提示词 + 一个模拟 P2
    import sys
    sys.path.insert(0, ".")
    from data_agent.system_prompt import SYSTEM_PROMPT as P1

    P2 = {
        "SceneFeat": {"total_orders": 100, "factories": 10, "capacity_status": "偏紧张"},
        "Workflow_f": "按标准三阶段流程执行",
        "TaskParam": {"priority_rules": "交期紧迫度优先", "due_range": "3-9周期"},
        "CoopReq": "通过语义标签空间协作，发布 cleaned_data_ready 并回执确认",
    }
    # 结构化融合（快速验证）
    fused = fuse_prompts(P1, P2, enable_llm=False)
    print("=== 结构化融合结果（末尾 600 字）===")
    print(fused[-600:])
