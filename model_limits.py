# -*- coding: utf-8 -*-
"""Resolve the per-model output budget (MAX_TOKENS) and thinking-mode switches.

Backbones differ widely in how much output budget they need for the same prompt:
some spend it on a long reasoning chain, others truncate and start rewriting when
the budget is too small. This module keeps the per-model caps in one place instead
of asking the user to hard-code a single value in .env.
"""
import os

# Per-model output-budget caps. Models absent from this table are not capped.
MODEL_MAX_OUTPUT = {
    "deepseek-v4.1-flash": 65536,
    # Backbones served through a third-party relay: a smaller cap is safer.
    "qwen3.7-plus": 65536,
    "glm-5.3": 65536,
    "glm-5.2": 65536,
}

DEFAULT_MAX_TOKENS = 262144

_printed = set()


# ===================== thinking mode =====================
# THINKING_MODE=on keeps the model's native reasoning chain; "off" asks the backend
# to skip it so that end-to-end latency drops.
#
# The switch is passed through extra_body (using model_kwargs triggers a langchain
# UserWarning). Only models that stay accurate with reasoning disabled are listed.
_MODEL_THINKING_OFF = {
    "deepseek-flash": {"reasoning_effort": "none"},
    # 0731 exposes a differently named parameter.
    "deepseek-v4-flash-0731": {"enable_thinking": False},
    "deepseek-v4.1-flash": {"reasoning_effort": "none"},
    "kimi-k3": {"reasoning_effort": "none"},
    "qwen3.7-plus": {"reasoning_effort": "none"},
    "glm-5.3": {"reasoning_effort": "none"},
    "glm-5.2": {"reasoning_effort": "none"},
}


def thinking_off_supported(model_name: str = None) -> bool:
    """Whether the model stays accurate with reasoning disabled."""
    return bare_model_name(model_name or os.getenv("MODEL_NAME") or "") in _MODEL_THINKING_OFF


def thinking_kwargs(model_name: str = None) -> dict:
    """Extra keyword arguments for init_chat_model that disable reasoning."""
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
        print(f"[model_limits] thinking mode disabled for {name} (THINKING_MODE=off): {kw}")
    return {"extra_body": dict(kw)}


def bare_model_name(name: str) -> str:
    """Strip the langchain provider prefix, e.g. deepseek:model-x -> model-x."""
    return (name or "").split(":")[-1].strip()


def resolve_max_tokens(model_name: str = None, requested: int = None) -> int:
    """Return the output budget actually usable by this model.

    Args:
        model_name: model name, optionally with a provider prefix.
        requested: desired value; defaults to the MAX_TOKENS environment variable.
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
            print(f"[model_limits] {name} output budget {requested} exceeds the safe "
                  f"value for this model; clamped to {cap}")
        return cap
    return requested
