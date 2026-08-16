# communication_fsm.py
"""多智能体通信过程形式化：有限状态机（论文机制②：(3) Formal Specification）。

状态机五元组 M = {Q, q0, qf, Σ, δ}：
- Q   = {idle 空闲, task_receiving 接收任务, executing 执行中, exception 异常处理, submitting 结果提交}
- q0  = idle（初始空闲状态）
- qf  = done（任务完成终止状态）
- Σ   = 触发事件：task_assign 任务分配 / state_sync 状态同步 / result_feedback 结果反馈 /
         comm_timeout 通信超时 / fallback_instr 回退指令 / receipt 回执 / exec_confirm 执行确认
- δ   : Q × Σ → Q 状态转移函数

作用：明确通信触发条件与状态转移规则，保证协作过程的有序性与一致性。
"""

from enum import Enum
from typing import Callable, Dict, Optional


class AgentState(str, Enum):
    """Q：智能体通信状态集。"""
    IDLE = "idle"                    # 空闲
    TASK_RECEIVING = "task_receiving"  # 任务接收
    EXECUTING = "executing"          # 任务执行
    EXCEPTION = "exception"          # 异常处理
    SUBMITTING = "submitting"        # 结果提交
    DONE = "done"                    # 终止


class CommEvent(str, Enum):
    """Σ：触发事件集（三类核心通信消息 + 异常事件）。"""
    TASK_ASSIGN = "task_assign"        # 任务分配
    STATE_SYNC = "state_sync"          # 状态同步
    RESULT_FEEDBACK = "result_feedback"  # 结果反馈
    COMM_TIMEOUT = "comm_timeout"      # 通信超时（异常）
    FALLBACK_INSTR = "fallback_instr"  # 回退指令（异常）
    RECEIPT = "receipt"                # 回执确认（机制② 两阶段确认）
    EXEC_CONFIRM = "exec_confirm"      # 执行确认（机制② 两阶段确认）


# δ：状态转移函数（Q × Σ → Q）
TRANSITION_TABLE: Dict[AgentState, Dict[CommEvent, AgentState]] = {
    AgentState.IDLE: {
        CommEvent.TASK_ASSIGN: AgentState.TASK_RECEIVING,   # 空闲 + 任务分配 → 任务接收
        CommEvent.STATE_SYNC: AgentState.IDLE,              # 空闲 + 状态同步 → 空闲
    },
    AgentState.TASK_RECEIVING: {
        CommEvent.TASK_ASSIGN: AgentState.TASK_RECEIVING,   # 任务分配确认 → 仍接收
        CommEvent.STATE_SYNC: AgentState.EXECUTING,         # 确认就绪 → 执行
        CommEvent.RECEIPT: AgentState.TASK_RECEIVING,       # 回执 → 仍接收
        CommEvent.RESULT_FEEDBACK: AgentState.SUBMITTING,   # 边接收边提交（异步场景）
    },
    AgentState.EXECUTING: {
        CommEvent.RESULT_FEEDBACK: AgentState.SUBMITTING,   # 执行完成反馈 → 提交
        CommEvent.STATE_SYNC: AgentState.EXECUTING,         # 中间状态同步 → 继续执行
        CommEvent.RECEIPT: AgentState.EXECUTING,            # 执行中收到回执 → 继续执行
        CommEvent.TASK_ASSIGN: AgentState.EXECUTING,        # 执行中追加任务 → 继续执行
        CommEvent.COMM_TIMEOUT: AgentState.EXCEPTION,       # 通信超时 → 异常处理
    },
    AgentState.EXCEPTION: {
        CommEvent.FALLBACK_INSTR: AgentState.EXECUTING,     # 收到回退指令 → 恢复执行
        CommEvent.TASK_ASSIGN: AgentState.TASK_RECEIVING,   # 重新分配 → 重新接收
        CommEvent.RECEIPT: AgentState.EXCEPTION,            # 回执 → 仍异常
    },
    AgentState.SUBMITTING: {
        CommEvent.RECEIPT: AgentState.SUBMITTING,           # 发送方回执 → 等待确认
        CommEvent.RESULT_FEEDBACK: AgentState.SUBMITTING,   # 发布多个标签（重复结果反馈，自循环）
        CommEvent.EXEC_CONFIRM: AgentState.DONE,            # 接收方执行确认 → 终止
        CommEvent.COMM_TIMEOUT: AgentState.EXCEPTION,       # 超时未确认 → 异常
    },
    AgentState.DONE: {},                                    # 终止态无转移
}


class CommunicationFSM:
    """单个智能体的通信有限状态机。"""

    def __init__(self, agent_name: str, on_transition: Optional[Callable[[str, AgentState, AgentState, CommEvent], None]] = None):
        self.agent_name = agent_name
        self.state: AgentState = AgentState.IDLE             # q0 = idle
        self.history: list = []
        self._on_transition = on_transition

    @property
    def is_terminal(self) -> bool:
        return self.state == AgentState.DONE                # qf = done

    def trigger(self, event: CommEvent, detail: str = "") -> tuple:
        """触发事件，执行状态转移 δ(q, σ) → q'。

        Returns:
            (next_state, valid)：valid=True 表示事件在转移表中存在（含自循环），
            valid=False 表示事件在当前状态无定义（非法转移，保持原状态）。
        """
        prev = self.state
        table = TRANSITION_TABLE.get(self.state, {})
        if event in table:
            next_state = table[event]
            valid = True
        else:
            next_state = prev
            valid = False

        self.state = next_state
        record = {"from": prev.value, "event": event.value, "to": next_state.value,
                  "valid": valid, "detail": detail}
        self.history.append(record)
        if self._on_transition:
            self._on_transition(self.agent_name, prev, next_state, event)
        return next_state, valid

    def reset(self) -> None:
        """任务完成后重置到空闲态（供下一轮任务复用）。"""
        self.state = AgentState.IDLE

    def summary(self) -> str:
        """输出状态机运行摘要（供日志/论文实验记录）。"""
        valid = [h for h in self.history if h["valid"]]
        invalid = [h for h in self.history if not h["valid"]]
        return (f"[{self.agent_name}] 当前状态={self.state.value}, "
                f"有效转移={len(valid)}, 非法转移={len(invalid)}")


if __name__ == "__main__":
    # 自测：data_agent 完整通信生命周期
    fsm = CommunicationFSM("data_agent")
    print("初始:", fsm.state.value)
    fsm.trigger(CommEvent.TASK_ASSIGN, "planning_agent 分配清洗任务")     # idle -> task_receiving
    fsm.trigger(CommEvent.STATE_SYNC, "确认接收")                          # task_receiving -> executing
    fsm.trigger(CommEvent.RESULT_FEEDBACK, "清洗完成发布 cleaned_data_ready")  # executing -> submitting
    fsm.trigger(CommEvent.RECEIPT, "scenario_agent 回执")                  # submitting -> submitting
    fsm.trigger(CommEvent.EXEC_CONFIRM, "scenario_agent 确认执行")         # submitting -> done
    print("最终:", fsm.state.value, "| 终止:", fsm.is_terminal)
    print(fsm.summary())
