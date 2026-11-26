# token_counter.py
"""每个智能体的 LLM 输入/输出 Token 统计器。

通过 LangChain CallbackHandler 拦截所有 LLM 调用：
- 用 run tags 区分智能体（planning_agent / data_agent / scenario_agent / decision_agent）
- 从 usage_metadata 读取精确的 input_tokens / output_tokens（DeepSeek/OpenAI 兼容 API 均返回）
- 汇总每个智能体的：调用次数、输入 token、输出 token、总 token、平均每次调用

挂载方式（main.py）：
    from langchain_core.callbacks import BaseCallbackHandler
    config = {"callbacks": [token_counter], "tags": ["data_agent"]}
    model.with_config(config)
"""

import json
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler


class TokenCounter(BaseCallbackHandler):
    """按智能体统计 LLM 调用的 token 用量。"""

    def __init__(self):
        # agent -> {calls, input_tokens, output_tokens, total_tokens, durations, details}
        self.stats: Dict[str, dict] = defaultdict(lambda: {
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "durations": [], "details": [],
            "tool_counts": Counter(),   # per-tool call count for this agent
        })
        self._start_times: Dict[str, float] = {}

    def _agent_of(self, tags: Optional[List[str]]) -> str:
        """从 run tags 识别智能体；无 tags 归为 unknown。"""
        if tags:
            for t in tags:
                if t in ("planning_agent", "data_agent", "scenario_agent", "decision_agent"):
                    return t
        return "unknown"

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None,
                            tags=None, metadata=None, **kwargs) -> None:
        agent = self._agent_of(list(tags or []))
        self._start_times[run_id] = (agent, time.time())

    def on_llm_end(self, response, *, run_id, parent_run_id=None, tags=None, **kwargs) -> None:
        if run_id not in self._start_times:
            return
        agent, t0 = self._start_times.pop(run_id)
        duration = time.time() - t0

        # read exact counts from usage_metadata
        input_tokens = output_tokens = total_tokens = None
        try:
            usage = getattr(response, "llm_output", None) or {}
            # langchain 1.0: AIMessage.usage_metadata
            if hasattr(response, "generations") and response.generations:
                msg = response.generations[0][0].message
                um = getattr(msg, "usage_metadata", None)
                if um:
                    input_tokens = um.get("input_tokens") or um.get("prompt_tokens")
                    output_tokens = um.get("output_tokens") or um.get("completion_tokens")
                    total_tokens = um.get("total_tokens")
            if input_tokens is None:
                input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            if output_tokens is None:
                output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            if total_tokens is None and (input_tokens is not None and output_tokens is not None):
                total_tokens = input_tokens + output_tokens
        except Exception:
            pass

        # estimate when usage metadata is unavailable
        if input_tokens is None or output_tokens is None:
            try:
                prompts = kwargs.get("prompts") or []
                prompt_chars = sum(len(p) for p in prompts)
                input_tokens = max(1, prompt_chars // 2) if input_tokens is None else input_tokens
                gen_text = ""
                if hasattr(response, "generations"):
                    gen_text = "".join(g.text for g in response.generations[0])
                output_tokens = max(1, len(gen_text) // 2) if output_tokens is None else output_tokens
                total_tokens = input_tokens + output_tokens
            except Exception:
                input_tokens = input_tokens or 0
                output_tokens = output_tokens or 0
                total_tokens = total_tokens or (input_tokens + output_tokens)

        # record which tools were requested and the text / reasoning sizes
        # used to locate repeated lightweight round trips
        tools_called, text_chars, reasoning_chars = [], 0, 0
        try:
            msg = response.generations[0][0].message
            for tc in (getattr(msg, "tool_calls", None) or []):
                nm = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if nm:
                    tools_called.append(nm)
            if not tools_called:
                for tc in (getattr(msg, "additional_kwargs", {}) or {}).get("tool_calls", []) or []:
                    nm = (tc.get("function") or {}).get("name")
                    if nm:
                        tools_called.append(nm)
            content = getattr(msg, "content", "")
            text_chars = len(content) if isinstance(content, str) else len(str(content or ""))
            reasoning = (getattr(msg, "additional_kwargs", {}) or {}).get("reasoning_content") or ""
            reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
        except Exception:
            pass

        s = self.stats[agent]
        s["calls"] += 1
        s["input_tokens"] += input_tokens or 0
        s["output_tokens"] += output_tokens or 0
        s["total_tokens"] += total_tokens or 0
        s["durations"].append(round(duration, 2))
        s["tool_counts"].update(tools_called)
        s["details"].append({
            "call": s["calls"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "duration_s": round(duration, 2),
            "tools": tools_called,
            "text_chars": text_chars,
            "reasoning_chars": reasoning_chars,
        })

    # ---------------- report ----------------
    def report(self) -> dict:
        rep = {}
        for agent, s in sorted(self.stats.items()):
            calls = s["calls"]
            rep[agent] = {
                "调用次数": calls,
                "输入Token总计": s["input_tokens"],
                "输出Token总计": s["output_tokens"],
                "Token总计": s["total_tokens"],
                "平均每次输入Token": round(s["input_tokens"] / calls, 1) if calls else 0,
                "平均每次输出Token": round(s["output_tokens"] / calls, 1) if calls else 0,
                "平均耗时(s)": round(sum(s["durations"]) / calls, 2) if calls else 0,
                "总耗时(s)": round(sum(s["durations"]), 2),
                "工具调用统计": dict(s["tool_counts"].most_common()),
                "调用明细": s["details"],
            }
        return rep

    def print_report(self) -> None:
        print("\n" + "=" * 70)
        print("《各智能体 LLM Token 用量统计报告》")
        print("=" * 70)
        rep = self.report()
        header = f"{'智能体':<16}{'调用':>6}{'输入tok':>12}{'输出tok':>12}{'总计tok':>12}{'耗时s':>10}"
        print(header)
        print("-" * 70)
        for agent, s in rep.items():
            print(f"{agent:<16}{s['调用次数']:>6}{s['输入Token总计']:>12}{s['输出Token总计']:>12}"
                  f"{s['Token总计']:>12}{s['总耗时(s)']:>10}")
        print("-" * 70)
        # tool-call composition per agent
        for agent, s in rep.items():
            tc = s.get("工具调用统计") or {}
            if not tc:
                continue
            total = sum(tc.values())
            print(f"  [{agent}] 工具调用 {total} 次: " +
                  "  ".join(f"{k}×{v}" for k, v in tc.items()))
        print("=" * 70)
        total_in = sum(s["输入Token总计"] for s in rep.values())
        total_out = sum(s["输出Token总计"] for s in rep.values())
        total_calls = sum(s["调用次数"] for s in rep.values())
        print(f"{'合计':<16}{total_calls:>6}{total_in:>12}{total_out:>12}{total_in + total_out:>12}")
        print("=" * 70)
        # save report
        report_path = f"token_usage_report_{time.strftime('%Y%m%d%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"报告已保存: {report_path}")


# global singleton
token_counter = TokenCounter()
