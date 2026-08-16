# scenario_agent/tools/excel_tools.py
from langchain.tools import tool
import pandas as pd
import json
import os
from pathlib import Path

# ===================== 全局上下文定义（关键修正）=====================
class ScenarioContext:
    """场景模型构建上下文（存储Excel信息/生成的代码）"""
    def __init__(self):
        self.excel_path: str = None          # 数据文件路径
        self.excel_structure: dict = None    # Excel/CSV结构
        self.excel_content: list = None      # Excel/CSV内容
        self.generated_code: str = None      # 大模型生成的代码

# 初始化全局上下文实例（必须实例化，否则导入后无法使用）
scenario_context = ScenarioContext()

# ===================== 原有工具逻辑（保留，仅修正路径兼容）=====================
@tool
def read_excel_structure(file_path: str) -> str:
    """
    读取Excel/CSV文件的结构信息（列名、数据类型、行数、非空值、示例值）
    参数:
        file_path: 文件绝对路径（支持.xlsx/.csv）
    返回:
        JSON格式的结构信息
    """
    try:
        # 兼容Windows路径
        file_path = file_path.replace("\\", "/")
        if not os.path.exists(file_path):
            return f"错误：文件 {file_path} 不存在"

        # 兼容CSV/Excel
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, encoding="utf-8")
        else:
            df = pd.read_excel(file_path)

        structure = {
            "file_path": file_path,
            "file_type": "CSV" if file_path.endswith(".csv") else "Excel",
            "total_rows": len(df),
            "columns": [],
            "data_types": df.dtypes.astype(str).to_dict(),
            "non_null_counts": df.notnull().sum().to_dict()
        }

        for col in df.columns:
            structure["columns"].append({
                "name": col,
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notnull().sum()),
                "null_count": int(df[col].isnull().sum()),
                "sample_values": df[col].dropna().head(5).tolist()
            })

        # 更新全局上下文
        scenario_context.excel_path = file_path
        scenario_context.excel_structure = structure

        return json.dumps(structure, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"读取文件结构失败：{str(e)}"

@tool
def read_excel_content(file_path: str, rows: int = 10) -> str:
    """
    读取Excel/CSV文件的前N行内容，用于理解数据格式
    参数:
        file_path: 文件绝对路径
        rows: 读取行数（默认10行）
    返回:
        JSON格式的内容信息
    """
    try:
        file_path = file_path.replace("\\", "/")
        if not os.path.exists(file_path):
            return f"错误：文件 {file_path} 不存在"

        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, encoding="utf-8").head(rows)
        else:
            df = pd.read_excel(file_path).head(rows)

        content = {
            "file_path": file_path,
            "rows_read": rows,
            "content": df.to_dict("records"),
            "columns": df.columns.tolist()
        }

        scenario_context.excel_content = content
        return json.dumps(content, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"读取文件内容失败：{str(e)}"