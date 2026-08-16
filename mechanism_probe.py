# mechanism_probe.py
"""机制运行探针：记录三大机制在真实场景中的触发情况，输出《机制运行验证报告》。

记录内容：
- 机制①：P2 生成方式（LLM/模板）、F(P1,P2) 融合方式（LLM/结构化）、提示词长度
- 机制②：语义发布/订阅调用、匹配度 S 分数、回执/确认次数、超时检测、回退分发
- 机制③：工具链匹配（exp_id、score、注入工具数）、Φ 验证、增量更新
- 通信 FSM：状态转移序列

验证报告：report() 输出结构化 JSON（供论文实验记录）。
"""

import json
import time
from typing import Dict, List


class MechanismProbe:
    """全局机制探针（单例）。"""

    def __init__(self):
        self.start_time = time.time()
        self.events: List[dict] = []          # 所有机制事件流水
        # 机制② 计数器
        self.semantic_publishes: List[dict] = []
        self.semantic_subscribes: List[dict] = []   # 含匹配度 S
        self.receipts: List[dict] = []
        self.confirms: List[dict] = []
        self.timeout_checks: List[dict] = []
        self.fallbacks: List[dict] = []
        # 机制③ 计数器
        self.toolchain_matches: List[dict] = []
        self.executability_checks: List[dict] = []
        self.experience_adds: List[dict] = []
        # 机制① 计数器
        self.prompt_fusions: List[dict] = []
        # FSM 转移序列
        self.fsm_transitions: List[dict] = []
        self.fsm_illegal: List[dict] = []
        # FSM 监督器（由 main.py 注入，实现探针->状态机联动）
        self.fsm_supervisor = None
        # 发布记录：tag -> sender（用于回执/确认时定位发送方）
        self._publish_senders: Dict[str, str] = {}

    # ---------------- 通用记录 ----------------
    def _log(self, mechanism: str, event: str, detail: dict) -> None:
        self.events.append({
            "t": round(time.time() - self.start_time, 2),
            "mechanism": mechanism,
            "event": event,
            **detail,
        })

    # ---------------- 机制②：通信 ----------------
    def log_publish(self, tag: str, sender: str, description: str) -> None:
        self.semantic_publishes.append({"tag": tag, "sender": sender, "description": description[:60]})
        self._publish_senders[tag] = sender
        self._log("M2", "semantic_publish", {"tag": tag, "sender": sender})
        # FSM 联动：发送方发布结果 → 结果反馈事件
        if self.fsm_supervisor:
            self.fsm_supervisor.fire(sender, "result_feedback", f"发布标签 {tag}")

    def log_subscribe(self, agent: str, tag_query: str, matched: List[str],
                      scores: Dict[str, float], threshold: float) -> None:
        self.semantic_subscribes.append({
            "agent": agent, "tag_query": tag_query,
            "matched": matched, "scores": scores, "threshold": threshold,
        })
        self._log("M2", "semantic_subscribe", {"agent": agent, "tag_query": tag_query,
                                               "matched": matched, "scores": scores})

    def log_receipt(self, tag: str, agent: str, ok: bool) -> None:
        self.receipts.append({"tag": tag, "agent": agent, "ok": ok})
        self._log("M2", "send_receipt", {"tag": tag, "agent": agent, "ok": ok})
        # FSM 联动：接收方返回回执 → 回执事件（发送方进入等待确认状态）
        if self.fsm_supervisor and ok:
            sender = self._publish_senders.get(tag)
            if sender:
                self.fsm_supervisor.fire(sender, "receipt", f"标签 {tag} 收到回执")

    def log_confirm(self, tag: str, agent: str, ok: bool) -> None:
        self.confirms.append({"tag": tag, "agent": agent, "ok": ok})
        self._log("M2", "confirm_task_done", {"tag": tag, "agent": agent, "ok": ok})
        # FSM 联动：发送方确认任务完成 → 执行确认事件（进入终止态）
        if self.fsm_supervisor and ok:
            sender = self._publish_senders.get(tag)
            if sender:
                self.fsm_supervisor.fire(sender, "exec_confirm", f"标签 {tag} 确认执行完成")

    def log_timeout_check(self, n_timeout: int) -> None:
        self.timeout_checks.append({"timeout_count": n_timeout})
        self._log("M2", "check_comm_timeout", {"timeout_count": n_timeout})

    def log_fallback(self, tag: str, target: str, ok: bool) -> None:
        self.fallbacks.append({"tag": tag, "target": target, "ok": ok})
        self._log("M2", "route_fallback", {"tag": tag, "target": target, "ok": ok})

    # ---------------- 机制③：工具链 ----------------
    def log_toolchain(self, agent: str, task_desc: str, matched: bool,
                      exp_id: str, score: float, n_tools: int, threshold: float) -> None:
        self.toolchain_matches.append({
            "agent": agent, "task_desc": task_desc[:50], "matched": matched,
            "exp_id": exp_id, "score": score, "n_tools": n_tools, "threshold": threshold,
        })
        self._log("M3", "toolchain_select", {"agent": agent, "matched": matched,
                                             "exp_id": exp_id, "score": score, "n_tools": n_tools})

    def log_executability(self, exp_id: str, phi: int, log_sample: str) -> None:
        self.executability_checks.append({"exp_id": exp_id, "Phi": phi, "log_sample": log_sample[:80]})
        self._log("M3", "executability_check", {"exp_id": exp_id, "Phi": phi})

    def log_experience_add(self, exp_id: str, n_tools: int) -> None:
        self.experience_adds.append({"exp_id": exp_id, "n_tools": n_tools})
        self._log("M3", "experience_add", {"exp_id": exp_id, "n_tools": n_tools})

    # ---------------- 机制①：提示词 ----------------
    def log_prompt_fusion(self, agent: str, p2_method: str, fusion_method: str,
                          p1_len: int, p2_len: int, fused_len: int) -> None:
        self.prompt_fusions.append({
            "agent": agent, "p2_method": p2_method, "fusion_method": fusion_method,
            "p1_len": p1_len, "p2_len": p2_len, "fused_len": fused_len,
        })
        self._log("M1", "prompt_fusion", {"agent": agent, "p2_method": p2_method,
                                          "fusion_method": fusion_method, "fused_len": fused_len})

    # ---------------- FSM ----------------
    def log_fsm(self, agent: str, from_state: str, to_state: str, event: str, valid: bool) -> None:
        record = {"agent": agent, "from": from_state, "to": to_state, "event": event, "valid": valid}
        if valid:
            self.fsm_transitions.append(record)
        else:
            self.fsm_illegal.append(record)
        self._log("M2-FSM", event, record)

    # ---------------- 验证报告 ----------------
    def report(self) -> dict:
        """生成《机制运行验证报告》（结构化 JSON，供论文实验记录）。"""
        # 计算语义订阅匹配统计
        sub_scores = [s for sub in self.semantic_subscribes for s in sub.get("scores", {}).values()]
        return {
            "运行时长(s)": round(time.time() - self.start_time, 2),
            "机制① 两阶段系统提示词": {
                "融合次数": len(self.prompt_fusions),
                "P2生成方式": [f["p2_method"] for f in self.prompt_fusions],
                "融合方式": [f["fusion_method"] for f in self.prompt_fusions],
                "平均融合后提示词长度": round(sum(f["fused_len"] for f in self.prompt_fusions) / len(self.prompt_fusions), 1) if self.prompt_fusions else 0,
            },
            "机制② 分层协商通信": {
                "语义发布次数": len(self.semantic_publishes),
                "发布标签": [p["tag"] for p in self.semantic_publishes],
                "语义订阅次数": len(self.semantic_subscribes),
                "订阅匹配明细": self.semantic_subscribes,
                "匹配度S样本": [round(s, 4) for s in sub_scores[:10]],
                "匹配度S均值": round(sum(sub_scores) / len(sub_scores), 4) if sub_scores else 0,
                "回执次数": len(self.receipts),
                "执行确认次数": len(self.confirms),
                "超时检测次数": len(self.timeout_checks),
                "超时单元总数": sum(t["timeout_count"] for t in self.timeout_checks),
                "回退分发次数": len(self.fallbacks),
            },
            "机制③ 动态工具链": {
                "工具链选择次数": len(self.toolchain_matches),
                "匹配明细": self.toolchain_matches,
                "可执行性验证次数": len(self.executability_checks),
                "验证结果Phi": [e["Phi"] for e in self.executability_checks],
                "经验增量入库次数": len(self.experience_adds),
            },
            "通信有限状态机": {
                "有效转移数": len(self.fsm_transitions),
                "非法转移数": len(self.fsm_illegal),
                "非法转移明细": [f"{t['agent']}: {t['from']}--{t['event']}-->{t['to']}" for t in self.fsm_illegal],
                "转移序列": [f"{t['agent']}: {t['from']}--{t['event']}-->{t['to']}" for t in self.fsm_transitions],
            },
        }

    def print_report(self) -> None:
        """控制台输出验证报告。"""
        print("\n" + "=" * 70)
        print("《机制运行验证报告》")
        print("=" * 70)
        rep = self.report()
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        print("=" * 70)
        # 保存报告文件
        report_path = f"mechanism_verification_report_{time.strftime('%Y%m%d%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"报告已保存: {report_path}")


# 全局探针单例
probe = MechanismProbe()
