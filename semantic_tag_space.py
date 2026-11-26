# semantic_tag_space.py
"""Shared semantic tag space (Mechanism 2: adaptive hierarchical negotiation and
communication; local publish/subscribe layer).

Concepts:
- Information unit Info = {L, D}: L is the business tag (X2Y directional naming),
  D is the semantic description.
- A subscriber dynamically generates a query tag L' and retrieves information by
  semantic match S = Sim(Embed(L), Embed(L')).
- Match threshold tau: S >= tau is treated as a match and a channel is established.
- Timeout T_timeout: an unacknowledged unit triggers fallback dispatch by the planner.
"""

import time
from typing import Dict, List, Optional, Tuple

from embedding_service import embed_query, embed_texts, cosine_similarity

# ------------------------------------------------------------------ thresholds
# Every threshold below can be overridden through an environment variable.
MATCH_THRESHOLD = float(__import__("os").getenv("TAG_MATCH_THRESHOLD", "0.70"))
MATCH_THRESHOLD_DATA_SYNC = float(__import__("os").getenv("TAG_MATCH_THRESHOLD_DATA_SYNC", "0.72"))
MATCH_THRESHOLD_TASK_DELIVERY = float(__import__("os").getenv("TAG_MATCH_THRESHOLD_TASK_DELIVERY", "0.672"))
MATCH_THRESHOLD_RESULT_FEEDBACK = float(__import__("os").getenv("TAG_MATCH_THRESHOLD_RESULT_FEEDBACK", "0.75"))
TIMEOUT_THRESHOLD = float(__import__("os").getenv("COMM_TIMEOUT", "120"))  # seconds


class SemanticTagSpace:
    """Shared semantic tag space: publish, subscribe/match and direct routing."""

    def __init__(self, threshold: float = MATCH_THRESHOLD):
        self.threshold = threshold
        # tag -> information unit {L, D, sender, content, ts, status}
        self._units: Dict[str, dict] = {}
        # subscription records: subscription_id -> {agent, tag, vector, matched, ts}
        self._subscriptions: Dict[str, dict] = {}
        self._sub_seq = 0

    @staticmethod
    def _norm_tag(t: str) -> str:
        """Normalise a tag name for exact matching of standard X2Y tags."""
        return (t or "").strip().lower().replace("-", "_").replace(" ", "")

    # ---------------- publish ----------------
    def publish(self, tag: str, description: str, sender: str, content: str) -> dict:
        """Publish a standardised information unit Info = {L, D} to the shared space.

        Tags follow the X2Y convention (sender -> receiver) so that several agents
        subscribing to the same tag do not produce conflicting interpretations.
        """
        unit = {
            "L": tag,
            "D": description,
            "sender": sender,
            "content": content,
            "ts": time.time(),
            "status": "pending",       # pending -> acked -> delivered
            "receipts": [],            # receipt log [{agent, ts}]
        }
        self._units[tag] = unit
        return unit

    # ---------------- subscribe and semantic matching ----------------
    def _get_threshold(self, tag_query: str) -> float:
        """Return the threshold that applies to the message type implied by tag_query.

        Three message types are recognised:
        - data sync      : data2scenario / data2capacity / data2scheduler / cleaned_data
        - task delivery  : scenario2decision / scenario_model
        - result feedback: decision_ready / receipt / scenario2data
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
        """Retrieve information by semantic match against a dynamically generated L'.

        Returns the matched tags (S >= tau), their scores and the matched units.
        tau is selected per message type by _get_threshold.
        """
        self._sub_seq += 1
        sub_id = f"sub_{agent}_{self._sub_seq}"
        # Embed the query and every published unit in a single batch call: the fixed
        # per-call overhead of the embedding backend dominates the per-text cost, so
        # issuing one call per text would become a bottleneck.
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
            # S = Sim(Embed(L), Embed(L')). When the standard tag names match exactly
            # then L' = L and S is 1 by definition, so take that shortcut instead of
            # computing the cosine over the concatenated "L D" text.
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

    # ---------------- receipt confirmation (two-phase) ----------------
    def acknowledge(self, tag: str, agent: str) -> dict:
        """Receiver acknowledges a unit (phase 1 of the two-phase confirmation)."""
        unit = self._units.get(tag)
        if not unit:
            return {"status": "失败", "message": f"标签 {tag} 不存在"}
        unit["receipts"].append({"agent": agent, "ts": time.time()})
        unit["status"] = "acked"
        return {"status": "成功", "tag": tag, "receipts": len(unit["receipts"])}

    def confirm_delivery(self, tag: str, agent: str) -> dict:
        """Sender marks the unit delivered after the acknowledgement arrives."""
        unit = self._units.get(tag)
        if not unit:
            return {"status": "失败", "message": f"标签 {tag} 不存在"}
        unit["status"] = "delivered"
        unit["confirmed_by"] = agent
        return {"status": "成功", "tag": tag, "final_status": "delivered"}

    # ---------------- timeout detection and fallback ----------------
    def check_timeout(self, timeout: float = TIMEOUT_THRESHOLD) -> List[dict]:
        """Return units still pending for longer than T_timeout since publication.

        The planner then re-issues the information by direct routing.
        """
        now = time.time()
        timed_out = []
        for tag, unit in self._units.items():
            if unit["status"] == "pending" and (now - unit["ts"]) > timeout:
                timed_out.append({"tag": tag, "unit": unit})
        return timed_out

    def route_to(self, tag: str, target_agent: str, content: str) -> dict:
        """Fallback: the planner routes information directly to a target agent."""
        if tag not in self._units:
            self.publish(tag, f"规划智能体定向分发至 {target_agent}", "planning_agent", content)
        unit = self._units[tag]
        unit["routed_to"] = target_agent
        unit["status"] = "routed"
        return {"status": "成功", "tag": tag, "target": target_agent, "content": content}

    # ---------------- queries ----------------
    def get_unit(self, tag: str) -> Optional[dict]:
        return self._units.get(tag)

    def all_units(self) -> List[dict]:
        return list(self._units.values())

    def clear(self) -> None:
        self._units.clear()
        self._subscriptions.clear()


# Global shared tag space (singleton)
tag_space = SemanticTagSpace()


if __name__ == "__main__":
    # self-test
    print(f"匹配阈值 τ = {MATCH_THRESHOLD}")
    tag_space.publish("data2scenario", "数据清洗已完成，输出清洗后数据文件路径，供场景建模智能体消费", "data_agent", "output/case_cleaned.xlsx")
    tag_space.publish("decision_ready", "决策优化结果已生成，输出排程方案", "decision_agent", "decision_result.json")
    r = tag_space.subscribe("scenario_agent", "数据清洗完成的文件在哪里")
    print("订阅匹配结果:", r["matched_tags"], r["scores"])
    r2 = tag_space.subscribe("decision_agent", "决策结果")
    print("订阅匹配结果2:", r2["matched_tags"], r2["scores"])
