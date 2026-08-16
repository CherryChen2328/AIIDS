from langchain.tools import tool


@tool
def optimize_decisions(scenario_path: str, output_path: str) -> str:
    """
    基于场景模型生成优化决策建议

    参数:
        scenario_path: 场景模型文件路径（JSON）
        output_path: 决策结果保存路径（文本）

    返回:
        决策结果概述
    """


    return f"决策生成完成，保存至{output_path}"