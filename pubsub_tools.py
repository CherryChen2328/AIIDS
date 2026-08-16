# pubsub_tools.py
"""多智能体通信工具集（论文机制②：自适应分层协商与通信机制）。

工具清单：
- publish_message: 发布消息（兼容旧接口，登记到语义标签空间）
- subscribe_topic: 订阅主题（兼容旧接口）
- get_topic_messages: 获取主题历史消息
- semantic_publish: 发布标准化信息单元 Info={L, D}（X2Y 标签 + 语义描述）
- semantic_subscribe: 语义订阅（动态标签 L' + 匹配度 S 检索）
- send_receipt: 回执确认（两阶段确认-阶段1）
- confirm_task_done: 执行确认（两阶段确认-阶段2）
- check_comm_timeout: 通信超时检测（触发异常回退）
- route_fallback: 异常回退定向分发（规划智能体兜底）
"""

from typing import Any, Optional
from langchain.tools import tool
from pubsub_broker import broker
from mechanism_probe import probe


# ================= 兼容旧接口 =================
@tool
def publish_message(topic: str, message: str, sender: str) -> str:
    """
    向指定主题发布消息，通知订阅者（兼容旧接口）

    参数:
        topic: 消息主题（字符串，如"cleaned_data_ready"）
        message: 消息内容（字符串或JSON格式字符串）
        sender: 发送者名称（标识哪个智能体发送的消息）

    返回:
        发布结果（成功/失败）
    """
    try:
        full_message = {"sender": sender, "content": message}
        broker.publish(topic, full_message)
        return f"成功向主题「{topic}」发布消息（发送者：{sender}）"
    except Exception as e:
        return f"发布失败：{str(e)}"


@tool
def subscribe_topic(topic: str, subscriber: str) -> str:
    """
    订阅指定主题，接收该主题的所有新消息（兼容旧接口）

    参数:
        topic: 消息主题（字符串，如"scenario_model_updated"）
        subscriber: 订阅者名称（标识哪个智能体订阅）

    返回:
        订阅结果及最新消息（如有）
    """
    def on_message_received(message: Any):
        print(f"\n【{subscriber}】收到主题「{topic}」的消息：{message}")

    try:
        broker.subscribe(topic, on_message_received)
        latest_msg = broker.get_latest_message(topic)
        latest_info = f"最新消息：{latest_msg}" if latest_msg else "暂无历史消息"
        return f"成功订阅主题「{topic}」（订阅者：{subscriber}）。{latest_info}"
    except Exception as e:
        return f"订阅失败：{str(e)}"


@tool
def get_topic_messages(topic: str) -> str:
    """获取指定主题的所有历史消息"""
    messages = broker.get_all_messages(topic)
    return f"主题「{topic}」的历史消息：{messages}"


# ================= 机制②新增：语义标签空间通信 =================
@tool
def semantic_publish(tag: str, description: str, message: str, sender: str) -> str:
    """
    向共享语义标签空间发布标准化信息单元 Info={L, D}（机制②）

    参数:
        tag: 核心业务标签，遵循 X2Y 定向命名规范（X=发送方缩写, Y=目标接收方缩写），
             如 "data2scenario"（数据智能体→场景智能体）
        description: 语义描述 D（简要说明标签含义、信息内容、适用场景）
        message: 信息内容
        sender: 发送者名称

    返回:
        发布结果
    """
    try:
        broker.publish_unit(tag, description, sender, message)
        probe.log_publish(tag, sender, description)
        return f"已发布信息单元到共享标签空间：标签「{tag}」（发送者：{sender}）"
    except Exception as e:
        return f"发布失败：{str(e)}"


@tool
def semantic_subscribe(agent: str, tag_query: str, description: str = "") -> str:
    """
    语义订阅：动态生成订阅标签 L'，通过语义匹配度 S 检索所需信息（机制②）

    参数:
        agent: 订阅者名称
        tag_query: 订阅标签 L'（按自身业务需求动态生成，无需预定义固定标签）
        description: 订阅需求补充描述（提升匹配精度）

    返回:
        匹配到的信息单元列表（含匹配度 S）与内容
    """
    try:
        result = broker.subscribe_semantic(agent, tag_query, description)
        matched = result.get("matched_tags", [])
        probe.log_subscribe(agent, tag_query, matched, result.get("scores", {}), broker.tag_space.threshold)
        if not matched:
            return f"语义订阅未匹配到信息（订阅标签：{tag_query}，阈值 τ={broker.tag_space.threshold}）。" \
                   f"如持续未匹配将触发超时回退机制，由规划智能体定向分发。"
        lines = [f"匹配到 {len(matched)} 条信息："]
        for tag in matched:
            unit = broker.tag_space.get_unit(tag)
            score = result["scores"].get(tag, 0)
            lines.append(f"- 标签「{tag}」匹配度 S={score:.4f}，内容：{unit['content'] if unit else 'N/A'}")
        return "\n".join(lines)
    except Exception as e:
        return f"语义订阅失败：{str(e)}"


@tool
def send_receipt(tag: str, agent: str) -> str:
    """
    回执确认：接收方收到信息后返回回执（机制② 两阶段确认-阶段1）

    参数:
        tag: 已接收信息的标签
        agent: 接收方（回执方）名称

    返回:
        回执结果
    """
    try:
        result = broker.acknowledge(tag, agent)
        ok = result.get("status") == "成功"
        probe.log_receipt(tag, agent, ok)
        if ok:
            return f"已向「{tag}」返回回执（接收方：{agent}）"
        return f"回执失败：{result.get('message')}"
    except Exception as e:
        probe.log_receipt(tag, agent, False)
        return f"回执失败：{str(e)}"


@tool
def confirm_task_done(tag: str, agent: str) -> str:
    """
    执行确认：发送方收到回执后标记任务完成（机制② 两阶段确认-阶段2）

    参数:
        tag: 信息标签
        agent: 确认方名称

    返回:
        确认结果
    """
    try:
        result = broker.confirm_delivery(tag, agent)
        ok = result.get("status") == "成功"
        probe.log_confirm(tag, agent, ok)
        if ok:
            return f"「{tag}」已标记为完成（确认方：{agent}），避免重复执行"
        return f"确认失败：{result.get('message')}"
    except Exception as e:
        probe.log_confirm(tag, agent, False)
        return f"确认失败：{str(e)}"


@tool
def check_comm_timeout() -> str:
    """
    通信超时检测：检查共享标签空间中超时未回执的信息单元（机制② 异常回退触发条件）

    返回:
        超时信息单元列表；无超时返回正常状态
    """
    try:
        timed_out = broker.check_timeout_units()
        probe.log_timeout_check(len(timed_out))
        if not timed_out:
            return "通信状态正常：无超时未回执的信息单元"
        lines = [f"发现 {len(timed_out)} 个通信超时信息单元："]
        for item in timed_out:
            unit = item["unit"]
            lines.append(f"- 标签「{item['tag']}」发送者：{unit['sender']}，发布后未收到回执")
        lines.append("建议：由规划智能体执行回退机制（route_fallback），定向分发信息。")
        return "\n".join(lines)
    except Exception as e:
        return f"超时检测失败：{str(e)}"


@tool
def route_fallback(tag: str, target_agent: str, message: str) -> str:
    """
    异常回退：规划智能体作为全局协调节点，从共享标签空间定向分发信息到目标智能体（机制②）

    参数:
        tag: 信息标签
        target_agent: 目标接收智能体名称
        message: 要分发的信息内容

    返回:
        回退分发结果
    """
    try:
        result = broker.route_fallback(tag, target_agent, message)
        ok = result.get("status") == "成功"
        probe.log_fallback(tag, target_agent, ok)
        if ok:
            return f"回退分发成功：标签「{tag}」已定向分发至 {target_agent}"
        return f"回退分发失败：{result.get('message')}"
    except Exception as e:
        probe.log_fallback(tag, target_agent, False)
        return f"回退分发失败：{str(e)}"
