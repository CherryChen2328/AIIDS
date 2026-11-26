# global_info_compressor.py
"""全局关键信息压缩与发布（论文机制②：(1) Global Key Information Compression and Release）。

核心公式：I_compress = LLM_compress(I^T)
- I^T = {I_1^T, ..., I_P^T}：原始全局文本信息集合（总订单量、整体产能约束、主要优化目标等）
- I_compress：压缩后的简洁语义摘要，统一分发给所有 agent

设计目标：
- 保留一致的协作方向（所有 agent 看到同一份全局信息包）
- 缓解过长上下文导致的冗余开销
"""

import os
import json
import time
from typing import List, Optional

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

from model_limits import resolve_max_tokens, thinking_kwargs

load_dotenv()

# ===================== configuration =====================
COMPRESS_MODEL = os.getenv("COMPRESS_MODEL", "deepseek:deepseek-v4-flash")
COMPRESS_MAX_WORDS = int(os.getenv("COMPRESS_MAX_WORDS", "60"))  # target length


MAX_TOKENS = resolve_max_tokens(COMPRESS_MODEL)


def _get_llm():
    """懒加载 LLM 用于语义压缩。"""
    try:
        from token_counter import token_counter
        return init_chat_model(COMPRESS_MODEL, temperature=0.1, max_tokens=MAX_TOKENS,
                               callbacks=[token_counter], tags=["planning_agent"],
                               **thinking_kwargs(COMPRESS_MODEL))
    except Exception:
        return init_chat_model(COMPRESS_MODEL, temperature=0.1, max_tokens=MAX_TOKENS,
                               **thinking_kwargs(COMPRESS_MODEL))


COMPRESS_PROMPT_TEMPLATE = """你是多工厂晶圆调度系统的全局信息压缩器。请将以下全局核心信息压缩为简洁的语义摘要。

要求：
1. 只保留关键数值、约束和目标，删除修饰性文字
2. 输出为一段简短的中文摘要，不超过 {max_words} 字
3. 保留全部数值与约束条件（不遗漏任何数字）
4. 不添加原文没有的信息

原始全局信息：
{info_text}

压缩后的摘要：
"""


def compress_global_info(info_items: List[str], max_words: int = COMPRESS_MAX_WORDS) -> str:
    """LLM 语义压缩全局信息（论文公式 I_compress = LLM_compress(I^T)）。

    Args:
        info_items: 原始全局信息列表（如 ["总订单量100单", "产能利用率目标90%", ...]）
        max_words: 压缩目标字数

    Returns:
        压缩后的语义摘要；压缩失败时返回原文拼接（降级保底）。
    """
    if not info_items:
        return ""
    info_text = "\n".join(f"- {item}" for item in info_items)
    try:
        llm = _get_llm()
        prompt = COMPRESS_PROMPT_TEMPLATE.format(info_text=info_text, max_words=max_words)
        resp = llm.invoke(prompt)
        compressed = (resp.content or "").strip()
        # strip stray quotes or markers
        compressed = compressed.strip('"').strip("'").strip()
        return compressed if compressed else info_text
    except Exception as e:
        print(f"[global_info_compressor] LLM 压缩失败，降级为原文拼接: {e}")
        return info_text


def build_global_info_package(scene_feats: dict, compress: bool = True) -> dict:
    """构建标准化的全局信息包（统一分发给所有 agent）。

    Args:
        scene_feats: 场景特征字典，如 {
            "total_orders": 100, "total_factories": 10,
            "optimization_objective": "最小化拖期", "capacity_target": "利用率>=90%",
            "priority_rules": "高优先级订单优先"
        }
        compress: 是否启用 LLM 语义压缩

    Returns:
        {
            "raw": {...},                    # raw structured features
            "compressed": "...",             # compressed global information I_compress
            "created_at": ts
        }
    """
    # raw information set I^T
    info_items = []
    for k, v in scene_feats.items():
        if v is not None and str(v) != "":
            info_items.append(f"{k}: {v}")

    if compress:
        compressed = compress_global_info(info_items)
    else:
        compressed = "; ".join(info_items)

    return {
        "raw": scene_feats,
        "compressed": compressed,
        "created_at": time.time(),
    }


if __name__ == "__main__":
    # self-test: minimal example
    scene = {
        "总订单交付周期": "72小时",
        "优先级规则": "高优先级订单优先保障交付",
        "全局产能利用率目标": "90%",
    }
    pkg = build_global_info_package(scene)
    print("原始:", scene)
    print("压缩后:", pkg["compressed"])
