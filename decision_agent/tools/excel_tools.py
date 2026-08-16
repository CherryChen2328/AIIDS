from langchain.tools import tool
import pandas as pd
import json
import os
from pathlib import Path
from typing import Optional  # 补充类型注解


# ===================== 全局上下文定义（优化版）=====================
class ScenarioContext:
    """场景/决策模型构建上下文（兼容场景模型 + 完整适配决策模型）"""

    def __init__(self):
        # ========== 基础数据字段（场景/决策模型共用） ==========
        self.excel_path: Optional[str] = None  # 清洗后数据路径（CSV/Excel）
        self.excel_structure: Optional[dict] = None  # 数据结构信息（read_excel_structure输出）
        self.excel_content: Optional[list] = None  # 数据内容预览（read_excel_content输出）

        # ========== 场景模型相关字段（决策模型输入） ==========
        self.scenario_model_path: Optional[str] = None  # 场景模型Python脚本路径
        self.scenario_model_analysis: Optional[dict] = None  # 场景模型分析结果（read_and_analyze_scenario_model输出）

        # ========== 决策模型生成/执行字段 ==========
        # self.generated_code: Optional[str] = None  # 生成的决策模型代码（generate_decision_code输出）
        self.decision_script_path: Optional[str] = None  # 保存的决策脚本路径（save_scenario_code输出）
        self.execution_result: Optional[dict] = None  # 决策脚本执行结果（execute_decision_script输出）
        self.error_message: Optional[str] = None  # 执行/生成错误信息（容错）


# 初始化全局上下文实例（单例，全工具共享）
scenario_context = ScenarioContext()


# ===================== 原有工具逻辑（完全保留，仅路径兼容）=====================
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