from langchain.tools import tool
import re
import json
import subprocess
from datetime import datetime
import os
import sys
from pathlib import Path

# 同一目录下的相对导入
from .excel_tools import scenario_context

# 新增：默认配置（替代test_config）
DEFAULT_CONFIG = {
    "default_save_dir": "decision_models"  # 统一保存目录
}


@tool
def read_and_analyze_scenario_model(scenario_model_path: str) -> str:
    """
    极简提取场景模型的核心可用信息（供AI自主编写代码）
    """
    try:
        scenario_model_path = scenario_model_path.replace("\\", "/")
        if not os.path.exists(scenario_model_path):
            return f"❌ 场景模型文件不存在：{scenario_model_path}"

        # 读取脚本内容
        with open(scenario_model_path, "r", encoding="utf-8") as f:
            code = f.read()

        # 提取所有公开可调用函数（无类、无self、非私有）
        func_pattern = r"def\s+([a-zA-Z0-9_]+)\s*\((.*?)\):"
        func_matches = re.findall(func_pattern, code)
        usable_functions = []
        for func_name, func_args in func_matches:
            if not func_name.startswith("_"):  # 仅保留公开函数
                args = [a.strip() for a in func_args.split(",") if a.strip() and not a.startswith("self")]
                args_str = ", ".join(args)
                call_example = f"{func_name}({args_str})"
                usable_functions.append({
                    "name": func_name,
                    "args": args_str,
                    "call_example": call_example
                })

        # 极简分析结果（仅传递关键信息）
        analysis_result = {
            "usable_functions": usable_functions,
            "file_path": scenario_model_path,
            "tip": "函数可直接import调用，无需定义类"
        }
        scenario_context.scenario_model_analysis = analysis_result

        # 输出极简的函数信息（供AI参考）
        func_info = "\n  - ".join(
            [f"{f['name']}({f['args']})" for f in usable_functions]) if usable_functions else "无公开可调用函数"
        return f"""✅ 场景模型核心可用信息：
1. 脚本路径：{scenario_model_path}
2. 公开可调用函数：
  - {func_info}
3. 使用说明：先创建类实例，然后调用相关函数
4. 并基于分析结果编写代码，无需markdown包裹，仅返回纯代码字符串，要简洁明了，不要注释
"""

    except Exception as e:
        err_msg = f"❌ 分析场景模型失败：{str(e)}"
        scenario_context.error_message = err_msg
        return err_msg


# @tool
# def generate_decision_code(business_scenario: str, requirements: str) -> str:
#     """
#     传递决策代码编写的核心信息，让AI完全自主编写适配的极简代码
#     前置：先调用read_and_analyze_scenario_model分析场景模型
#     参数:
#         business_scenario: 业务场景（如投料任务决策）
#         requirements: 具体业务需求（需明确、可落地）
#     返回:
#         代码编写指引（包含所有关键信息，无固定模板）
#     """
#     try:
#         # 校验前置条件
#         if not scenario_context.scenario_model_analysis:
#             return "❌ 前置依赖：请先调用read_and_analyze_scenario_model分析场景模型！"
#         if not business_scenario or not requirements:
#             return "❌ 业务场景和需求不能为空，请明确描述！"
#
#         # 提取所有关键信息（供AI自主编写代码使用）
#         model_path = scenario_context.scenario_model_path  # 场景模型路径
#         data_path = scenario_context.excel_path or "./output/投料任务.xlsx"  # 数据路径
#         usable_funcs = scenario_context.scenario_model_analysis["usable_functions"]  # 场景模型可用函数
#         data_fields = scenario_context.excel_structure["columns"] if scenario_context.excel_structure else []  # 数据字段
#         data_field_names = [f["name"] for f in data_fields] if data_fields else []  # 仅提取字段名
#         guide = f"""
# # ========== 决策代码编写核心信息（请完全自主编写极简可运行代码） ==========
# 1. 场景模型信息：
#    - 脚本路径：{model_path}
#    - 可用公开函数（直接调用，无需类）：{[f['name'] for f in usable_funcs] if usable_funcs else '无'}
#    - 函数调用示例参考：{[f['call_example'] for f in usable_funcs] if usable_funcs else '无'}
#
# 2. 数据信息：
#    - 数据文件路径：{data_path}
#    - 数据包含字段：{data_field_names if data_field_names else '请自行读取数据后查看'}
#    - 数据类型：Excel（请用pd.read_excel加载，指定engine="openpyxl"）
#
# 3. 核心业务需求：
#    - 业务场景：{business_scenario}
#    - 具体要求：{requirements}
#
# 4. 强制编写要求（必须遵守）：
#    ✅ 极简原则：仅保留核心逻辑，无冗余代码、无无用注释、无固定模板
#    ✅ 适配性：必须基于实际数据字段和场景模型函数编写，禁止使用不存在的字段/函数
#    ✅ 可运行：代码可直接执行，包含数据加载、核心逻辑、结果输出全流程
#    ✅ 结果输出：结果保存成excel文件
#    ✅ 异常处理：极简异常捕获，仅打印错误信息即可
#
# """
#         return guide
#
#     except Exception as e:
#         err_msg = f"❌ 生成代码编写指引失败：{str(e)}"
#         scenario_context.error_message = err_msg
#         return err_msg


@tool
def execute_decision_script(script_path: str) -> str:
    """
    执行保存的决策模型脚本（独立子进程，避免污染主环境）
    参数:
        script_path: 决策脚本绝对路径（必填）
    返回:
        JSON格式的执行结果（输出/错误/返回码）
    """
    try:
        # 校验脚本路径
        if not script_path or not os.path.exists(script_path):
            return json.dumps({
                "status": "failed",
                "error": f"决策脚本不存在：{script_path}",
                "suggestion": "检查路径是否正确或重新保存脚本"
            }, ensure_ascii=False)

        # 执行脚本（独立子进程+超时保护）
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            encoding="utf-8",
            timeout=30  # 防止脚本卡死
        )

        # 结构化执行结果
        execution_result = {
            "status": "success" if result.returncode == 0 else "failed",
            "script_path": script_path,
            "return_code": result.returncode,
            "stdout": result.stdout,  # 脚本正常输出
            "stderr": result.stderr,  # 脚本错误输出
            "result_file": "decision_result.json" if os.path.exists("decision_result.json") else None
        }

        # 更新上下文
        scenario_context.execution_result = execution_result
        scenario_context.error_message = result.stderr if result.returncode != 0 else None

        return json.dumps(execution_result, ensure_ascii=False, indent=2)

    except subprocess.TimeoutExpired:
        return json.dumps({
            "status": "failed",
            "error": "脚本执行超时（30秒）",
            "suggestion": "检查脚本是否有死循环或数据量过大"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "failed",
            "error": f"执行脚本失败：{str(e)}",
            "suggestion": "检查脚本语法或Python环境依赖"
        }, ensure_ascii=False)


@tool
def save_script_to_local(filename: str, code_content: str) -> str:
    """
    核心功能：将生成的纯Python代码保存到本地文件
    参数：
    - filename: 保存文件名（自动补全.py后缀）
    - code_content: 纯Python代码字符串（无markdown包裹）
    输出：保存成功/失败提示（含绝对路径）
    """
    try:
        # 校验输入
        if not code_content.strip():
            return "❌ 代码内容不能为空！"

        # 自动补全.py后缀
        if not filename.endswith(".py"):
            filename += ".py"

        # 处理保存目录
        # save_dir = DEFAULT_CONFIG["default_save_dir"]
        save_dir = "."
        os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

        # 构建保存路径
        save_path = os.path.abspath(os.path.join(save_dir, filename))

        # # 避免覆盖已有文件
        # if os.path.exists(save_path):
        #     return f"⚠️  文件已存在！{save_path}\n请更换文件名后重新保存"

        # 写入文件
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(code_content)

        # 更新上下文（保留路径供后续执行参考）
        scenario_context.decision_script_path = save_path

        return f"✅ 脚本保存成功！路径：{save_path}\n文件大小：{len(code_content)} 字符"
    except Exception as e:
        return f"❌ 保存失败：{str(e)}"
