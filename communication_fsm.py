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
    IDLE = "idle"                    # idle
    TASK_RECEIVING = "task_receiving"  # receiving a task
    EXECUTING = "executing"          # executing
    EXCEPTION = "exception"          # exception handling
    SUBMITTING = "submitting"        # submitting results
    DONE = "done"                    # terminal


class CommEvent(str, Enum):
    """Σ：触发事件集（三类核心通信消息 + 异常事件）。"""
    TASK_ASSIGN = "task_assign"        # task assignment
    STATE_SYNC = "state_sync"          # state synchronisation
    RESULT_FEEDBACK = "result_feedback"  # result feedback
    COMM_TIMEOUT = "comm_timeout"      # communication timeout (exception)
    FALLBACK_INSTR = "fallback_instr"  # fallback instruction (exception)
    RECEIPT = "receipt"                # receipt (phase 1)
    EXEC_CONFIRM = "exec_confirm"      # execution confirmation (phase 2)


# delta: transition function (Q x Sigma -> Q)
TRANSITION_TABLE: Dict[AgentState, Dict[CommEvent, AgentState]] = {
    AgentState.IDLE: {
        CommEvent.TASK_ASSIGN: AgentState.TASK_RECEIVING,   # idle + task_assign -> receiving
        CommEvent.STATE_SYNC: AgentState.IDLE,              # idle + state_sync -> idle
    },
    AgentState.TASK_RECEIVING: {
        CommEvent.TASK_ASSIGN: AgentState.TASK_RECEIVING,   # task_assign while receiving -> receiving
        CommEvent.STATE_SYNC: AgentState.EXECUTING,         # state_sync -> executing
        CommEvent.RECEIPT: AgentState.TASK_RECEIVING,       # receipt -> receiving
        CommEvent.RESULT_FEEDBACK: AgentState.SUBMITTING,   # submit while still receiving (async)
    },
    AgentState.EXECUTING: {
        CommEvent.RESULT_FEEDBACK: AgentState.SUBMITTING,   # result_feedback -> submitting
        CommEvent.STATE_SYNC: AgentState.EXECUTING,         # state_sync -> executing
        CommEvent.RECEIPT: AgentState.EXECUTING,            # receipt -> executing
        CommEvent.TASK_ASSIGN: AgentState.EXECUTING,        # task_assign -> executing
        CommEvent.COMM_TIMEOUT: AgentState.EXCEPTION,       # communication timeout -> exception
    },
    AgentState.EXCEPTION: {
        CommEvent.FALLBACK_INSTR: AgentState.EXECUTING,     # fallback instruction -> resume executing
        CommEvent.TASK_ASSIGN: AgentState.TASK_RECEIVING,   # reassignment -> receiving again
        CommEvent.RECEIPT: AgentState.EXCEPTION,            # receipt -> still exception
    },
    AgentState.SUBMITTING: {
        CommEvent.RECEIPT: AgentState.SUBMITTING,           # sender receipt -> awaiting confirmation
        CommEvent.RESULT_FEEDBACK: AgentState.SUBMITTING,   # several tags published (self-loop)
        CommEvent.EXEC_CONFIRM: AgentState.DONE,            # execution confirmation -> done
        CommEvent.COMM_TIMEOUT: AgentState.EXCEPTION,       # timeout -> exception
    },
    AgentState.DONE: {},                                    # terminal state: no transitions
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
    # self-test: full communication lifecycle of data_agent
    fsm = CommunicationFSM("data_agent")
    print("初始:", fsm.state.value)
    fsm.trigger(CommEvent.TASK_ASSIGN, "planning_agent assigns the cleaning task")     # idle -> task_receiving
    fsm.trigger(CommEvent.STATE_SYNC, "acknowledge receipt")                          # task_receiving -> executing
    fsm.trigger(CommEvent.RESULT_FEEDBACK, "cleaning done, publish cleaned_data_ready")  # executing -> submitting
    fsm.trigger(CommEvent.RECEIPT, "scenario_agent receipt")                  # submitting -> submitting
    fsm.trigger(CommEvent.EXEC_CONFIRM, "scenario_agent confirms execution")         # submitting -> done
    print("最终:", fsm.state.value, "| 终止:", fsm.is_terminal)
    print(fsm.summary())
