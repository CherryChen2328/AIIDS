# fsm_supervisor.py
"""通信有限状态机监督器：接入多智能体实际执行循环（论文机制② (3)）。

职责：
- 为每个子智能体维护一个 CommunicationFSM 实例
- 在真实执行过程中，由机制探针（mechanism_probe）触发状态事件：
    * semantic_publish（发布结果）        -> RESULT_FEEDBACK 结果反馈
    * send_receipt（收到回执）            -> RECEIPT 回执
    * confirm_task_done（确认执行完成）   -> EXEC_CONFIRM 执行确认
    * planning task 派发（由 main 循环注入）-> TASK_ASSIGN 任务分配
- 记录全部状态转移序列，供《机制运行验证报告》输出
"""

from communication_fsm import CommunicationFSM, CommEvent, AgentState

AGENT_NAMES = ["data_agent", "scenario_agent", "decision_agent", "planning_agent"]


class FSMSupervisor:
    def __init__(self, agent_names: list = None):
        self.fsms = {name: CommunicationFSM(name) for name in (agent_names or AGENT_NAMES)}
        self._pending_tasks = {}  # tool_call_id -> agent_name（planning 派发的 task）

    def fire(self, agent: str, event: str, detail: str = ""):
        """触发状态事件（供机制探针调用）。"""
        fsm = self.fsms.get(agent)
        if fsm is None:
            return
        try:
            ev = CommEvent(event)
        except ValueError:
            return
        prev = fsm.state
        next_state, valid = fsm.trigger(ev, detail)
        # 记录转移（valid 与否都记录；自循环合法转移也记录）
        from mechanism_probe import probe
        probe.log_fsm(agent, prev.value, next_state.value, event, valid)

    # ---------------- 主循环挂钩：识别 planning 的 task 派发 ----------------
    def observe_message(self, message) -> None:
        """观察 agent.stream 中的每条消息，识别任务分配/完成事件。"""
        msg_type = getattr(message, "type", "")
        # AIMessage 中的 tool_calls（planning 调用 task 工具）
        tool_calls = getattr(message, "tool_calls", None) or []
        has_tool_calls = len(tool_calls) > 0

        # planning_agent 状态驱动：
        if msg_type == "ai" and has_tool_calls:
            # planning 开始执行（write_todos/task 等）→ 任务接收 → 执行中
            if self.fsms["planning_agent"].state.value == "idle":
                self.fire("planning_agent", "task_assign", "planning 开始协调")
            if self.fsms["planning_agent"].state.value == "task_receiving":
                self.fire("planning_agent", "state_sync", "planning 确认任务清单")
        elif msg_type == "ai" and not has_tool_calls:
            # planning 输出最终汇总（无工具调用）→ 提交 → 完成
            if self.fsms["planning_agent"].state.value in ("executing", "task_receiving"):
                self.fire("planning_agent", "result_feedback", "planning 汇总完成")
            if self.fsms["planning_agent"].state.value == "submitting":
                self.fire("planning_agent", "exec_confirm", "planning 任务终止")

        # 子智能体 task 派发/完成事件
        for tc in tool_calls:
            name = tc.get("name", "")
            if name == "task":
                args = tc.get("args", {}) or {}
                sub_type = args.get("subagent_type") or args.get("subagent") or ""
                if sub_type in self.fsms:
                    self._pending_tasks[tc.get("id")] = sub_type
                    self.fire(sub_type, "task_assign", f"planning 派发任务: {args.get('description', '')[:50]}")
                    # task 工具同步执行：派发后自动进入执行态（task_receiving -> executing）
                    if self.fsms[sub_type].state.value == "task_receiving":
                        self.fire(sub_type, "state_sync", "任务接收确认，进入执行")
        # ToolMessage（task 工具返回 → 子智能体执行完成）
        if msg_type == "tool" and getattr(message, "name", "") == "task":
            sub_type = self._pending_tasks.pop(getattr(message, "tool_call_id", ""), None)
            if sub_type:
                self.fire(sub_type, "result_feedback", "task 工具返回（子智能体执行完成）")

    def get_state(self, agent: str) -> str:
        fsm = self.fsms.get(agent)
        return fsm.state.value if fsm else "unknown"

    def all_states(self) -> dict:
        return {name: fsm.state.value for name, fsm in self.fsms.items()}

    def transition_summary(self) -> list:
        from mechanism_probe import probe
        return probe.fsm_transitions
