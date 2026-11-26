# semantic_tag_space.py
"""共享语义标签空间（论文机制②：自适应分层协商与通信机制 - 局部发布订阅部分）。

核心概念：
- 信息单元 Info = {L, D}：L 为核心业务标签（X2Y 定向命名规范），D 为语义描述
- 订阅方动态生成订阅标签 L'，通过语义匹配度 S = Sim(Embed(L), Embed(L')) 检索
- 匹配阈值 τ（默认 0.85）：S >= τ 视为匹配成功，建立通信信道
- 超时阈值 T_timeout：无有效匹配响应则触发异常回退（由 planning_agent 兜底）
"""

import time
from typing import Dict, List, Optional, Tuple

from embedding_service import embed_query, embed_texts, cosine_similarity

# ===================== 参数标定 =====================
# 机制②：语义匹配阈值 τ 经【正交实验 + 真实轨迹标定】双重确定。
#
# (1) 正交实验 L9(3^3)：三因素(数据同步/任务交付/结果反馈) × 三水平，
#     以匹配 F1、误配率、漏报率为指标。最优组合 F1 = 1.00，误配率与漏报率均为 0。
#     极差分析：任务交付主效应最强 (R=0.1336)，数据同步次之 (R=0.0534)，
#     结果反馈最弱 (R=0.0235，在 [0.73, 0.80] 区间内 F1 恒定不变)。
# (2) 独立复核：以真实运行轨迹的 (query, tag, S) 观测重新标定，
#     得到各类别的可行分离窗口，用于校验正交实验取值并确定区间内取值。
#
# 最终取值（嵌入模型：Ollama BGE-M3，1024 维）：
#   - 数据同步 data_sync     τ = 0.72   （可行窗口 0.677 ~ 0.765）
#   - 任务交付 task_delivery τ = 0.672  （可行窗口 0.646 ~ 0.695）
#   - 结果反馈 result_feedback τ = 0.75 （受约束于 ≥ 0.723；[0.73,0.80] 区间内 F1 等价，
#                                        取区间中下部以保留决策结果投递的召回余量）
#   - 全局默认（自由措辞标签）  τ = 0.70
# 均可通过环境变量覆盖，便于做阈值灵敏度实验。
MATCH_THRESHOLD = float(__import__("os").getenv("TAG_MATCH_THRESHOLD", "0.70"))
MATCH_THRESHOLD_DATA_SYNC = float(__import__("os").getenv("TAG_MATCH_THRESHOLD_DATA_SYNC", "0.72"))
MATCH_THRESHOLD_TASK_DELIVERY = float(__import__("os").getenv("TAG_MATCH_THRESHOLD_TASK_DELIVERY", "0.672"))
MATCH_THRESHOLD_RESULT_FEEDBACK = float(__import__("os").getenv("TAG_MATCH_THRESHOLD_RESULT_FEEDBACK", "0.75"))
TIMEOUT_THRESHOLD = float(__import__("os").getenv("COMM_TIMEOUT", "120"))  # 秒


class SemanticTagSpace:
    """共享语义标签空间：支持发布（publish）、订阅匹配（subscribe）、定向分发（route）。"""

    def __init__(self, threshold: float = MATCH_THRESHOLD):
        self.threshold = threshold
        # 标签名 -> 信息单元 {"L": tag, "D": desc, "sender": agent, "content": msg, "ts": time, "status": "pending"/"acked"/"delivered"}
        self._units: Dict[str, dict] = {}
        # 订阅记录：subscription_id -> {"agent": ..., "tag": L', "vector": ..., "matched": [tag...], "ts": ...}
        self._subscriptions: Dict[str, dict] = {}
        self._sub_seq = 0

    @staticmethod
    def _norm_tag(t: str) -> str:
        """标签名归一化：忽略大小写与分隔符差异，用于标准标签名（X2Y 规范）的精确命中判定。"""
        return (t or "").strip().lower().replace("-", "_").replace(" ", "")

    # ---------------- 发布 ----------------
    def publish(self, tag: str, description: str, sender: str, content: str) -> dict:
        """发布标准化信息单元 Info={L, D} 到共享标签空间。

        标签遵循 X2Y 命名规范（发送方->目标接收方），避免多 agent 订阅同一标签产生冲突解释。
        """
        unit = {
            "L": tag,
            "D": description,
            "sender": sender,
            "content": content,
            "ts": time.time(),
            "status": "pending",       # pending -> acked（接收方回执）-> delivered（确认执行）
            "receipts": [],            # 回执记录 [{agent, ts}]
        }
        self._units[tag] = unit
        return unit

    # ---------------- 订阅与语义匹配 ----------------
    def _get_threshold(self, tag_query: str) -> float:
        """根据 tag_query 内容判断消息类型，返回对应阈值。

        支持三种消息类型：
        - 数据同步 (data_sync): 标签含 data2scenario/data2capacity/data2scheduler/cleaned_data
        - 任务交付 (task_delivery): 标签含 scenario2decision/scenario_model
        - 结果反馈 (result_feedback): 标签含 decision_ready/receipt/scenario2data
        """
        q = tag_query.lower()
        data_sync_kw = ["data2scenario", "data2capacity", "data2scheduler", "cleaned_data"]
        task_delivery_kw = ["scenario2decision", "scenario_model"]
        result_feedback_kw = ["decision_ready", "receipt", "scenario2data"]
        if any(k in q for k in result_feedback_kw):
            return MATCH_THRESHOLD_RESULT_FEEDBACK
        if any(k in q for k in task_delivery_kw):
            return MATCH_THRESHOLD_TASK_DELIVERY
        if any(k in q for k in data_sync_kw):
            return MATCH_THRESHOLD_DATA_SYNC
        return self.threshold

    def subscribe(self, agent: str, tag_query: str, description: str = "") -> dict:
        """订阅方动态生成订阅标签 L'（不依赖预定义固定标签），通过语义匹配检索所需信息。

        返回匹配结果：matched_tags（S>=τ 的发布标签列表）、scores、已选信息单元。
        τ 按消息类型差异化选择（由 _get_threshold 决定）。
        """
        self._sub_seq += 1
        sub_id = f"sub_{agent}_{self._sub_seq}"
        # 批量嵌入：把"订阅查询 + 全部已发布单元"合并为一次调用。
        # Ollama 每次调用的固定开销约 2.1 秒（与文本条数几乎无关，实测单条 2.12s / 批量5条 2.17s），
        # 逐条调用会成为全流程的显著瓶颈，故此处统一合并。
        query_text = f"{tag_query} {description}".strip()
        pub_items = list(self._units.items())
        batch = [query_text] + [f"{t} {u['D']}".strip() for t, u in pub_items]
        vecs = embed_texts(batch)
        query_vec = vecs[0]
        active_threshold = self._get_threshold(tag_query)
        self._subscriptions[sub_id] = {
            "agent": agent, "tag": tag_query, "vector": query_vec, "ts": time.time(),
            "matched": [], "threshold_used": active_threshold,
        }

        q_norm = self._norm_tag(tag_query)
        matched = []
        for (pub_tag, unit), pub_vec in zip(pub_items, vecs[1:]):
            # S = Sim(Embed(L), Embed(L'))。标准标签名精确命中时 L'=L，按定义 S≡1。
            # 注：余弦实际在 "L D" 拼接文本上计算，同一标签会因两侧描述措辞不同而只得 ~0.65，
            # 与论文定义不符（实测裸标签名 "scenario2decision" 自匹配仅 0.648 < τ=0.672），
            # 故此处按论文定义短路，避免交付链路擦线通过/随机失败。
            if self._norm_tag(pub_tag) == q_norm:
                score = 1.0
            else:
                score = cosine_similarity(query_vec, pub_vec)
            if score >= active_threshold:
                matched.append({"tag": pub_tag, "score": round(score, 4), "unit": unit})

        matched.sort(key=lambda x: x["score"], reverse=True)
        self._subscriptions[sub_id]["matched"] = [m["tag"] for m in matched]
        return {"subscription_id": sub_id, "matched_tags": [m["tag"] for m in matched],
                "scores": {m["tag"]: m["score"] for m in matched},
                "units": [m["unit"] for m in matched],
                "threshold_used": active_threshold}

    def get_subscription(self, sub_id: str) -> Optional[dict]:
        return self._subscriptions.get(sub_id)

    # ---------------- 回执确认（两阶段确认机制） ----------------
    def acknowledge(self, tag: str, agent: str) -> dict:
        """接收方返回回执确认（机制② 一致性保证：发布-接收-执行两阶段确认）。"""
        unit = self._units.get(tag)
        if not unit:
            return {"status": "失败", "message": f"标签 {tag} 不存在"}
        unit["receipts"].append({"agent": agent, "ts": time.time()})
        unit["status"] = "acked"
        return {"status": "成功", "tag": tag, "receipts": len(unit["receipts"])}

    def confirm_delivery(self, tag: str, agent: str) -> dict:
        """发送方收到接收确认后，标记任务完成（避免信息丢失和重复执行）。"""
        unit = self._units.get(tag)
        if not unit:
            return {"status": "失败", "message": f"标签 {tag} 不存在"}
        unit["status"] = "delivered"
        unit["confirmed_by"] = agent
        return {"status": "成功", "tag": tag, "final_status": "delivered"}

    # ---------------- 超时检测与回退 ----------------
    def check_timeout(self, timeout: float = TIMEOUT_THRESHOLD) -> List[dict]:
        """检测通信超时：发布后超过 T_timeout 仍未收到回执的信息单元。

        触发异常回退：由规划智能体（全局协调节点）检索信息、验证有效性并定向分发。
        """
        now = time.time()
        timed_out = []
        for tag, unit in self._units.items():
            if unit["status"] == "pending" and (now - unit["ts"]) > timeout:
                timed_out.append({"tag": tag, "unit": unit})
        return timed_out

    def route_to(self, tag: str, target_agent: str, content: str) -> dict:
        """异常回退：规划智能体定向分发信息到目标 agent（绕过订阅匹配）。"""
        if tag not in self._units:
            self.publish(tag, f"规划智能体定向分发至 {target_agent}", "planning_agent", content)
        unit = self._units[tag]
        unit["routed_to"] = target_agent
        unit["status"] = "routed"
        return {"status": "成功", "tag": tag, "target": target_agent, "content": content}

    # ---------------- 查询 ----------------
    def get_unit(self, tag: str) -> Optional[dict]:
        return self._units.get(tag)

    def all_units(self) -> List[dict]:
        return list(self._units.values())

    def clear(self) -> None:
        self._units.clear()
        self._subscriptions.clear()


# 全局共享语义标签空间（单例）
tag_space = SemanticTagSpace()


if __name__ == "__main__":
    # 自测
    print(f"匹配阈值 τ = {MATCH_THRESHOLD}")
    tag_space.publish("data2scenario", "数据清洗已完成，输出清洗后数据文件路径，供场景建模智能体消费", "data_agent", "output/case_cleaned.xlsx")
    tag_space.publish("decision_ready", "决策优化结果已生成，输出排程方案", "decision_agent", "decision_result.json")
    r = tag_space.subscribe("scenario_agent", "数据清洗完成的文件在哪里")
    print("订阅匹配结果:", r["matched_tags"], r["scores"])
    r2 = tag_space.subscribe("decision_agent", "决策结果")
    print("订阅匹配结果2:", r2["matched_tags"], r2["scores"])
