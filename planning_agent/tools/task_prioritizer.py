from langchain.tools import tool


@tool
def prioritize_tasks(tasks: list[str]) -> str:
    """
    对任务列表进行优先级排序（1-最高，N-最低）

    参数:
        tasks: 任务列表（字符串数组，如["数据清洗", "场景建模"]）

    返回:
        排序后的任务列表及优先级说明
    """
    prioritized = sorted(enumerate(tasks, 1), key=lambda x: x[0])
    return "优先级排序结果：\n" + "\n".join([f"{i}. {task}" for i, task in prioritized])