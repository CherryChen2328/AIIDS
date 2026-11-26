# -*- coding: utf-8 -*-
"""按模型特性解析 LLM 输出预算 MAX_TOKENS。

背景（2026-09-16 实测，同一段 5322 字符的 F(P1,P2) 融合提示词）：

  funhpc deepseek-v4.1-flash @ max_tokens=262144 -> 115.2s，输出 9050 tokens
  funhpc deepseek-v4.1-flash @ max_tokens=65536  ->  24.5s，输出 3716 tokens
  funhpc deepseek-v4-flash-0731 @ max_tokens=262144 -> 25.9s，输出 2508 tokens

结论：两个模型对输出预算的需求**相反**。

1. deepseek-v4.1-flash 是推理模型。给它 262144 的输出预算、再让它"输出完整的融合
   系统提示词"，它会生成超长的推理链 + 超长正文。全流程有 4 次融合调用，叠加起来
   启动阶段可以几十分钟零输出，看起来像"卡死"（实测 20 分钟无任何输出）。
   => 必须把输出预算压到 65536。

2. deepseek-v4-flash-0731 相反。把预算压到 65536 会让它的输出被截断，触发反复重写：
   实测全流程 10.55min / 78 次调用 / 输入 87.6 万 token
         ->    14.63min / 105 次调用 / 输入 175.0 万 token（决策智能体调用数 26 -> 46）。
   => 必须保留 262144。

因此不要手工在 .env 里为所有模型填同一个值，交给本模块按模型名自动钳制。
"""
import os

# 每个模型的输出预算上限（不在表内的模型不限制）
MODEL_MAX_OUTPUT = {
    "deepseek-v4.1-flash": 65536,
    # 第三方 funhpc 上的模型实测输出上限（2026-09-16）：
    #   qwen3.7-plus 请求 max_tokens=262144 时稳定返回 HTTP 400「请求参数错误」，
    #   131072 可用；这里取 65536（本流程融合输出仅约 6000 token，余量充足且更稳）。
    #   kimi-k3 在 262144 下可用，不钳制。
    "qwen3.7-plus": 65536,
    #   glm-5.3：本框架下【不可用】，钳制参数救不了，保留此条目仅为记录。
    #   2026-09-17 实测（funhpc，长提示词 = 真实融合提示词体量）：
    #     输入  200  字符, mt=65536 -> OK, completion=693,  content=195 字符
    #     输入  800  字符, mt=65536 -> OK, completion=844,  content=282 字符
    #     输入 2000  字符, mt=65536 -> OK, completion=900,  content=265 字符
    #     输入 3500  字符, mt=65536 -> finish=length, completion=4096, content=""
    #     输入 5300  字符, mt=65536 -> finish=length, completion=4096, content=""
    #     输入 5300  字符, mt=32768 -> finish=length, completion=4096, content=""(6 字符)
    #     输入 5300  字符, mt=262144 -> HTTP 400 {'code':100501,'msg':'请求参数错误'}
    #   即：输入超过约 2000~3500 字符后，输出预算被烧到 4096 后中止且 content 返回空串。
    #   usage 里 reasoning_tokens 报 0，因此不是"思考链没关掉"；reasoning_effort=none 与
    #   enable_thinking=false 两种关思考写法结果完全相同（thinking={type:disabled} 直接 400）。
    #   本框架每次调用都携带 2 万字符级系统提示词，故 glm-5.3 下每个调用都返回空，
    #   融合步骤必然降级、智能体无法产出脚本 —— 不是本项目的 bug，是中转站的模型限制。
    "glm-5.3": 65536,
    #   glm-5.2：实测 65536 可用（输入 5300 字符 → 输出 5809 字符，finish=stop，58.8s）；
    #   262144 未验证，故按已验证的安全值钳制。
    "glm-5.2": 65536,
}

DEFAULT_MAX_TOKENS = 262144

_printed = set()


# ===================== 思考模式（推理链）开关 =====================
# THINKING_MODE=on（默认）保留模型原生推理链；=off 尝试关闭，以缩短端到端时延。
#
# 实测（3 道调度推理题各重复 5 次，temperature=0.2）：
#   官网 deepseek-flash + reasoning_effort="none"
#       -> 15/15 正确（与开思考持平）；单次 1.3s -> 0.7s，推理 token 归零，答案内容不变
#   funhpc deepseek-v4-flash-0731 + enable_thinking=false
#       -> 12/15 正确（合批题 3/5、产能题 4/5）—— 质量明显退化
# 结论：只对"关掉后仍无损失"的模型启用；0731 保留推理链。
#
# 接线方式：必须用 extra_body 透传（用 model_kwargs 会触发 langchain UserWarning）。
_MODEL_THINKING_OFF = {
    "deepseek-flash": {"reasoning_effort": "none"},
    # 0731 的开关参数名不同（enable_thinking）。合成推理题上关思考会掉正确率
    # （15/15 -> 12/15），但真实生成类任务（F1,P2 融合）几乎无损（5112 -> 4956 字、27.2s -> 20.6s）。
    # 因此保留启用，以完整流程的 Φ/OTD 作为最终判据；若质量崩了再退回按 agent 粒度控制。
    "deepseek-v4-flash-0731": {"enable_thinking": False},
    # funhpc v4.1-flash 关思考同样无损（15/15 持平），三个参数名都可用
    # （reasoning_effort=none / thinking={type:disabled} / enable_thinking=false），这里取与官网一致的那个
    "deepseek-v4.1-flash": {"reasoning_effort": "none"},
    # 第三方 funhpc 上的其他模型（2026-09-16 实测均支持 reasoning_effort="none" 关闭推理链）
    #   kimi-k3       关思考后 1.0~1.3s
    #   qwen3.7-plus  关思考后 1.2~1.3s
    #   glm-5.3       关思考后 5.3~13.1s（明显慢于前两者）
    "kimi-k3": {"reasoning_effort": "none"},
    "qwen3.7-plus": {"reasoning_effort": "none"},
    "glm-5.3": {"reasoning_effort": "none"},
    # glm-5.2：2026-09-17 实测【可用】，与 5.3 有质的区别（5.3 在长输入下返回空串）。
    #   关思考实测有效：短提示 3.6s(reasoning_effort=none) vs 10.9s(无参数)。
    "glm-5.2": {"reasoning_effort": "none"},
}


def thinking_off_supported(model_name: str = None) -> bool:
    """该模型关闭思考后是否仍能保持正确率。"""
    return bare_model_name(model_name or os.getenv("MODEL_NAME") or "") in _MODEL_THINKING_OFF


def thinking_kwargs(model_name: str = None) -> dict:
    """返回传给 init_chat_model 的额外关键字（关闭思考）；不适用时返回 {}。"""
    if os.getenv("THINKING_MODE", "on").strip().lower() not in ("off", "0", "false", "no", "disabled"):
        return {}
    name = bare_model_name(model_name) or bare_model_name(
        os.getenv("MODEL_NAME") or os.getenv("FUSION_MODEL") or "")
    kw = _MODEL_THINKING_OFF.get(name)
    if not kw:
        return {}
    key = ("thinking_off", name)
    if key not in _printed:
        _printed.add(key)
        print(f"[model_limits] 已关闭 {name} 的思考模式（THINKING_MODE=off）：{kw}")
    return {"extra_body": dict(kw)}


def bare_model_name(name: str) -> str:
    """去掉 langchain 的 provider 前缀，例如 deepseek:deepseek-v4.1-flash -> deepseek-v4.1-flash。"""
    return (name or "").split(":")[-1].strip()


def resolve_max_tokens(model_name: str = None, requested: int = None) -> int:
    """返回该模型实际可用的输出预算。

    Args:
        model_name: 模型名（可带 provider 前缀）；缺省时依次看 MODEL_NAME / FUSION_MODEL 环境变量。
        requested:  期望值；缺省时读环境变量 MAX_TOKENS。
    """
    if requested is None:
        try:
            requested = int(os.getenv("MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
        except (TypeError, ValueError):
            requested = DEFAULT_MAX_TOKENS

    name = bare_model_name(model_name) or bare_model_name(
        os.getenv("MODEL_NAME") or os.getenv("FUSION_MODEL") or "")

    cap = MODEL_MAX_OUTPUT.get(name)
    if cap and requested > cap:
        key = (name, requested, cap)
        if key not in _printed:
            _printed.add(key)
            print(f"[model_limits] {name} 输出预算 {requested} 超出该模型安全值，"
                  f"已钳制为 {cap}（原因见 model_limits.py 模块注释）")
        return cap
    return requested
