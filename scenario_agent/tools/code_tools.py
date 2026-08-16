# scenario_agent/tools/code_tools.py
from langchain.tools import tool
import re
import json
from datetime import datetime
import os
from pathlib import Path

# 同一目录下的相对导入
from .excel_tools import scenario_context


@tool
def generate_python_code(business_scenario: str, requirements: str) -> str:
    """
    根据业务场景和需求描述生成初始Python代码框架
    参数:
        business_scenario: 业务场景名称（如"投料任务决策"、"产能分析"）
        requirements: 功能需求描述（如"需要计算投料优先级、检查设备产能"）
    返回:
        生成的代码框架（字符串）
    """
    try:
        if not business_scenario or not requirements:
            return "错误：业务场景和需求描述不能为空"

        # 生成基础代码框架（实际应用中可集成大模型生成逻辑）
        class_name = "".join([s.capitalize() for s in business_scenario.split()]) + "Scenario"
        code_framework = f"""import pandas as pd
import numpy as np

class {class_name}:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self._load_data()
        self._preprocess_data()

    def _load_data(self):
        \"\"\"加载数据文件\"\"\"
        if self.data_path.endswith('.csv'):
            return pd.read_csv(self.data_path)
        else:
            return pd.read_excel(self.data_path)

    def _preprocess_data(self):
        \"\"\"数据预处理（处理缺失值、标准化等）\"\"\"
        self.data = self.data.dropna(subset=['关键字段'])  # 示例处理
        return self.data

    def analyze(self):
        \"\"\"核心分析逻辑\"\"\"
        # 根据需求实现: {requirements}
        result = self._calculate_key_indicators()
        return result

    def _calculate_key_indicators(self):
        \"\"\"计算关键指标\"\"\"
        # 实现具体计算逻辑
        return {{}}

# 示例调用
if __name__ == "__main__":
    scenario = {class_name}("数据文件路径")
    print(scenario.analyze())
"""
        # 存储到全局上下文
        scenario_context.generated_code = code_framework
        return f"代码框架生成成功（{class_name}）：\n{code_framework}"
    except Exception as e:
        return f"代码生成失败：{str(e)}"


@tool
def extract_python_code(raw_content: str) -> str:
    """从智能体响应中提取Python代码"""
    try:
        if not raw_content or raw_content.strip() == "":
            return "错误：输入内容为空"

        # 优先匹配```python代码块
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, raw_content, re.DOTALL)
        if matches:
            code = matches[0].strip()
            scenario_context.generated_code = code
            return code

        # 直接识别纯代码
        if raw_content.strip().startswith(("class ", "def ", "import ", "#!/usr/bin/env python")):
            scenario_context.generated_code = raw_content.strip()
            return raw_content.strip()

        return "错误：未提取到有效Python代码"
    except Exception as e:
        return f"提取代码失败：{str(e)}"


@tool
def save_scenario_code(filename: str = None) -> str:
    """
    保存大模型生成的场景模型代码到文件（默认自动生成带时间戳的文件名）
    参数:
        filename: 自定义文件名（可选，无需.py后缀）
    返回:
        保存结果（路径+文件大小）
    """
    try:
        # 检查大模型是否已生成代码
        if scenario_context.generated_code is None or scenario_context.generated_code.strip() == "":
            return "错误：全局上下文中无生成的场景模型代码（请先让大模型生成代码）"

        # 构建输出目录（兼容Windows）
        output_dir = Path(".")
        # output_dir.mkdir(exist_ok=True)

        # 自动生成文件名（优先自定义，否则按业务场景+时间戳）
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            # 根据Excel路径匹配业务关键词（贴合投料任务场景）
            keywords = "scenario_model"
            if scenario_context.excel_path:
                if "投料" in scenario_context.excel_path:
                    keywords = "material_feeding_scenario"
                elif "产能" in scenario_context.excel_path:
                    keywords = "capacity_analysis_scenario"
                elif "决策" in scenario_context.excel_path:
                    keywords = "decision_optimization_scenario"
            filename = f"{keywords}_{timestamp}"

        # 补全.py后缀
        if not filename.endswith(".py"):
            filename += ".py"
        file_path = output_dir / filename

        # 保存大模型生成的代码
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(scenario_context.generated_code)

        # 返回保存结果（绝对路径）
        abs_path = file_path.resolve()
        file_size = file_path.stat().st_size
        return f"✅ 场景模型代码保存成功：\n路径：{abs_path}\n大小：{file_size} 字节"

    except Exception as e:
        return f"❌ 保存代码失败：{str(e)}"