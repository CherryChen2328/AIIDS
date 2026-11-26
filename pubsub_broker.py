# pubsub_broker.py
"""自适应分层协商与通信机制 - 消息代理（论文机制②）。

基于共享语义标签空间（semantic_tag_space）的消息代理：
- 发布：向语义标签空间发布 Info={L, D}（X2Y 定向标签 + 语义描述）
- 订阅：动态生成订阅标签 L'，通过语义匹配度 S = Sim(Embed(L), Embed(L')) 检索
- 回执：两阶段确认机制（发布 → 接收回执 → 执行确认）
- 超时：检测未回执信息单元，触发异常回退（由规划智能体定向分发）
- 兼容旧接口：publish(topic, message) 保留，映射到语义标签空间
"""

import time
import threading
from typing import Any, Callable, Dict, List, Optional

from semantic_tag_space import SemanticTagSpace, tag_space as _default_tag_space


class PubSubBroker:
    """轻量级消息代理：语义标签空间 + 订阅回调分发。"""

    def __init__(self, tag_space: Optional[SemanticTagSpace] = None):
        self.tag_space = tag_space or _default_tag_space
        # 主题名 -> 订阅者回调列表（兼容旧接口）
        self.subscribers: Dict[str, List[Callable]] = {}
        # 主题名 -> 历史消息（兼容旧接口）
        self.topic_messages: Dict[str, List[Any]] = {}
        # 回执记录：tag -> 已回执 agent 集合
        self._receipts: Dict[str, set] = {}
        self._lock = threading.Lock()

    # ================= 兼容旧接口：publish / subscribe / get_latest_message =================
    def publish(self, topic: str, message: Any) -> None:
        """向主题发布消息（兼容旧接口）。

        内部升级：在语义标签空间登记信息单元（标签=主题名，语义描述=发送者说明）。
        """
        with self._lock:
            if topic not in self.topic_messages:
                self.topic_messages[topic] = []
            self.topic_messages[topic].append(message)

            sender = message.get("sender", "unknown") if isinstance(message, dict) else "unknown"
            content = message.get("content", message) if isinstance(message, dict) else message
            desc = message.get("description", f"主题 {topic} 的消息") if isinstance(message, dict) else f"主题 {topic} 的消息"
            # 登记到语义标签空间（标签即主题名，语义描述用于向量匹配）
            self.tag_space.publish(topic, desc, sender, str(content))

        # 分发消息给所有订阅者
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    callback(message)
                except Exception as e:
                    print(f"[pubsub] 回调执行失败: {e}")

    def subscribe(self, topic: str, callback: Callable) -> None:
        """订阅主题，注册回调函数（兼容旧接口）。"""
        with self._lock:
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            self.subscribers[topic].append(callback)

    def get_latest_message(self, topic: str) -> Any:
        """获取主题的最新消息。"""
        msgs = self.topic_messages.get(topic, [])
        return msgs[-1] if msgs else None

    def get_all_messages(self, topic: str) -> List[Any]:
        return self.topic_messages.get(topic, [])

    # ================= 语义标签空间接口（机制②新增）=================
    def publish_unit(self, tag: str, description: str, sender: str, content: str) -> dict:
        """发布标准化信息单元 Info={L, D} 到共享语义标签空间。"""
        unit = self.tag_space.publish(tag, description, sender, content)
        # 同步到旧接口的历史消息
        with self._lock:
            if tag not in self.topic_messages:
                self.topic_messages[tag] = []
            self.topic_messages[tag].append({"sender": sender, "content": content, "description": description})
        return unit

    def subscribe_semantic(self, agent: str, tag_query: str, description: str = "") -> dict:
        """语义订阅：动态生成订阅标签 L'，通过匹配度 S 检索信息单元。"""
        result = self.tag_space.subscribe(agent, tag_query, description)
        return result

    def acknowledge(self, tag: str, agent: str) -> dict:
        """回执确认：接收方收到信息后返回回执（两阶段确认-阶段1）。"""
        result = self.tag_space.acknowledge(tag, agent)
        with self._lock:
            self._receipts.setdefault(tag, set()).add(agent)
        return result

    def confirm_delivery(self, tag: str, agent: str) -> dict:
        """执行确认：发送方收到回执后标记任务完成（两阶段确认-阶段2）。"""
        return self.tag_space.confirm_delivery(tag, agent)

    def check_timeout_units(self, timeout: float = 120) -> List[dict]:
        """检测通信超时（发布后超时未回执），供规划智能体执行回退。"""
        return self.tag_space.check_timeout(timeout)

    def route_fallback(self, tag: str, target_agent: str, content: str) -> dict:
        """异常回退：规划智能体作为全局协调节点，定向分发信息到目标 agent。"""
        return self.tag_space.route_to(tag, target_agent, content)

    def semantic_search(self, query: str, top_k: int = 5) -> List[dict]:
        """在共享标签空间中按语义检索所有信息单元（供规划智能体回退/协调使用）。"""
        from embedding_service import embed_query, cosine_similarity
        q_vec = embed_query(query)
        scored = []
        for unit in self.tag_space.all_units():
            tag = unit.get("L", "")
            unit_dict = dict(unit)
            unit_vec = embed_query(f"{tag} {unit.get('D', '')}".strip())
            s = cosine_similarity(q_vec, unit_vec)
            scored.append({"tag": tag, "score": round(s, 4), "unit": unit_dict})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# 实例化全局消息代理（单例）
broker = PubSubBroker()
