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

from embedding_service import embed_query, cosine_similarity

# ===================== 参数标定 =====================
# 论文机制②：匹配阈值 τ 需经初步实验标定。
# 原论文在典型调度场景下标定为 0.85（对应其嵌入模型）。
# 本项目使用 Ollama BGE-M3（1024维）实测标定：
#   相关对（同一业务语义）相似度 0.64~0.80，不相关对 0.50~0.53
#   取中间安全阈值 τ = 0.62，兼顾通信准确率与覆盖率。
MATCH_THRESHOLD = float(__import__("os").getenv("TAG_MATCH_THRESHOLD", "0.62"))
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
    def subscribe(self, agent: str, tag_query: str, description: str = "") -> dict:
        """订阅方动态生成订阅标签 L'（不依赖预定义固定标签），通过语义匹配检索所需信息。

        返回匹配结果：matched_tags（S>=τ 的发布标签列表）、scores、已选信息单元。
        """
        self._sub_seq += 1
        sub_id = f"sub_{agent}_{self._sub_seq}"
        query_vec = embed_query(f"{tag_query} {description}".strip())
        self._subscriptions[sub_id] = {
            "agent": agent, "tag": tag_query, "vector": query_vec, "ts": time.time(),
            "matched": [],
        }

        matched = []
        for pub_tag, unit in self._units.items():
            pub_vec = embed_query(f"{pub_tag} {unit['D']}".strip())
            score = cosine_similarity(query_vec, pub_vec)
            if score >= self.threshold:
                matched.append({"tag": pub_tag, "score": round(score, 4), "unit": unit})

        matched.sort(key=lambda x: x["score"], reverse=True)
        self._subscriptions[sub_id]["matched"] = [m["tag"] for m in matched]
        return {"subscription_id": sub_id, "matched_tags": [m["tag"] for m in matched],
                "scores": {m["tag"]: m["score"] for m in matched},
                "units": [m["unit"] for m in matched]}

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
