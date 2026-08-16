from langchain.tools import tool
import pandas as pd
import numpy as np
from scipy import stats
import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ===================== 全局上下文（适配多智能体框架）=====================
class DataCleaningContext:
    """数据清洗全局上下文（替代ToolRuntime，适配多智能体）"""

    def __init__(self):
        self.file_path: str = None  # 原始文件路径
        self.file_type: str = None  # 文件类型（csv/excel）
        self.raw_df: pd.DataFrame = None  # 原始数据
        self.cleaning_df: pd.DataFrame = None  # 清洗中数据
        self.cleaned_df: pd.DataFrame = None  # 清洗后数据
        self.numeric_cols: list = []  # 数值型特征列
        self.cat_cols: list = []  # 离散型特征列
        self.cleaning_log: list = []  # 清洗日志
        self.cleaned_file_path: str = None  # 清洗后文件路径


# 初始化全局上下文
data_context = DataCleaningContext()


# ===================== 核心工具（适配多智能体+时序清洗）=====================
@tool
def check_file_exists(file_path: str) -> str:
    """
    校验文件存在性及格式（多智能体协作第一步）
    参数:
        file_path: 文件绝对/相对路径（支持CSV/Excel）
    返回:
        JSON格式的文件校验结果
    """
    try:
        # 适配Windows路径
        file_path = file_path.replace("\\", "/")
        if not os.path.exists(file_path):
            return json.dumps({
                "status": "失败",
                "message": f"文件不存在：{file_path}",
                "suggestion": "检查文件路径或上传文件"
            }, ensure_ascii=False)

        # 识别文件类型
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in [".csv", ".xlsx", ".xls"]:
            return json.dumps({
                "status": "失败",
                "message": f"不支持的文件格式：{file_ext}（仅支持CSV/Excel）",
                "suggestion": "转换为CSV/XLSX格式"
            }, ensure_ascii=False)

        # 更新上下文
        data_context.file_path = file_path
        data_context.file_type = "csv" if file_ext == ".csv" else "excel"
        data_context.cleaning_log.append(f"文件校验通过：{file_path}（类型：{data_context.file_type}）")

        return json.dumps({
            "status": "成功",
            "file_info": {
                "path": file_path,
                "type": data_context.file_type,
                "size": os.path.getsize(file_path) / 1024,  # KB
                "next_step": "调用init_data_cleaning初始化清洗"
            }
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "status": "失败",
            "message": f"文件校验出错：{str(e)}"
        }, ensure_ascii=False)


@tool
def init_data_cleaning() -> str:
    """
    初始化数据清洗上下文（必选第二步）
    功能：加载数据、自动识别特征类型
    无任何强制列要求（移除ID列硬性约束）
    """
    if not data_context.file_path:
        return json.dumps({
            "status": "失败",
            "message": "请先调用check_file_exists校验文件",
            "next_step": "执行check_file_exists"
        }, ensure_ascii=False)

    try:
        # 加载数据
        if data_context.file_type == "csv":
            df = pd.read_csv(data_context.file_path, encoding="utf-8")
        else:
            df = pd.read_excel(data_context.file_path, engine="openpyxl")

        # 时间戳处理（可选，仅当存在时执行）
        time_range = "无时间戳列"
        if "时间戳" in df.columns:
            # 标准化时间戳
            df["时间戳"] = pd.to_datetime(df["时间戳"], errors="coerce")
            if df["时间戳"].isna().sum() > 0:
                data_context.cleaning_log.append(
                    f"警告：{df['时间戳'].isna().sum()}条时间戳格式错误（已转换为NaT）"
                )
            # 计算时间范围（过滤NaT值）
            valid_timestamps = df["时间戳"].dropna()
            if not valid_timestamps.empty:
                time_range = f"{valid_timestamps.min()} ~ {valid_timestamps.max()}"

        # 自动识别特征类型（移除ID列跳过逻辑）
        numeric_cols = []
        cat_cols = []
        for col in df.columns:
            if col == "时间戳":  # 仅跳过时间戳列
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                df[col] = df[col].astype("object").fillna("")
                cat_cols.append(col)

        # 更新上下文
        data_context.raw_df = df.copy()
        data_context.cleaning_df = df.copy()
        data_context.numeric_cols = numeric_cols
        data_context.cat_cols = cat_cols
        data_context.cleaning_log.append(
            f"清洗初始化完成：{len(df)}条记录，数值列{len(numeric_cols)}个，离散列{len(cat_cols)}个"
        )

        # 构建返回信息（动态包含时间范围）
        data_info = {
            "total_rows": len(df),
            "numeric_cols": numeric_cols,
            "cat_cols": cat_cols
        }
        if "时间戳" in df.columns:
            data_info["time_range"] = time_range

        return json.dumps({
            "status": "成功",
            "data_info": data_info,
            "next_step": "调用handle_missing_values/handle_categorical_abnormal/handle_numeric_noise"
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "status": "失败",
            "message": f"初始化清洗出错：{str(e)}"
        }, ensure_ascii=False)


@tool
def handle_missing_values(method: str = "ffill", group_col: str = None) -> str:
    """
    处理数值型特征缺失值（支持分组/全局填充）
    参数:
        method: 填充方法（ffill/bfill/interpolate/mean）
        group_col: 分组列（可选，无则全局填充）
    """
    if data_context.cleaning_df is None:
        return json.dumps({
            "status": "失败",
            "message": "请先调用init_data_cleaning初始化数据",
            "next_step": "执行init_data_cleaning"
        }, ensure_ascii=False)

    if not data_context.numeric_cols:
        return json.dumps({
            "status": "成功",
            "message": "无数值型特征列，无需处理缺失值"
        }, ensure_ascii=False)

    try:
        df = data_context.cleaning_df.copy()
        log = []

        for col in data_context.numeric_cols:
            missing_count = df[col].isna().sum()
            if missing_count == 0:
                log.append(f"{col}：无缺失值")
                continue

            # 填充逻辑：有分组列则分组填充，无则全局填充
            if group_col and group_col in df.columns:
                if method in ["ffill", "bfill"]:
                    df[col] = df.groupby(group_col)[col].transform(method)
                elif method == "interpolate":
                    df[col] = df.groupby(group_col)[col].transform(lambda x: x.interpolate("linear"))
                elif method == "mean":
                    df[col] = df.groupby(group_col)[col].transform(lambda x: x.fillna(x.mean()))
                log.append(f"{col}：按{group_col}分组处理{missing_count}个缺失值（方法：{method}）")
            else:
                if method in ["ffill", "bfill"]:
                    df[col] = df[col].ffill() if method == "ffill" else df[col].bfill()
                elif method == "interpolate":
                    df[col] = df[col].interpolate("linear")
                elif method == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                log.append(f"{col}：全局处理{missing_count}个缺失值（方法：{method}）")

        # 更新上下文
        data_context.cleaning_df = df.copy()
        data_context.cleaning_log.append(f"缺失值处理完成：{'; '.join(log)}")

        return json.dumps({
            "status": "成功",
            "processing_log": log,
            "next_step": "处理离散型异常值/数值型噪声或保存数据"
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "status": "失败",
            "message": f"处理缺失值出错：{str(e)}"
        }, ensure_ascii=False)


@tool
def handle_categorical_abnormal(fill_strategy: str = "mode", fill_value: str = "异常") -> str:
    """
    处理离散型特征异常值
    参数:
        fill_strategy: 处理策略（mode/众数 | specific/指定值 | new_category/新增类别）
        fill_value: specific策略的填充值（默认"异常"）
    """
    if data_context.cleaning_df is None:
        return json.dumps({
            "status": "失败",
            "message": "请先调用init_data_cleaning初始化数据"
        }, ensure_ascii=False)

    if not data_context.cat_cols:
        return json.dumps({
            "status": "成功",
            "message": "无离散型特征列，无需处理异常值"
        }, ensure_ascii=False)

    try:
        df = data_context.cleaning_df.copy()
        log = []

        for col in data_context.cat_cols:
            # 多值/自由文本列保护：唯一值数量过多时跳过异常清洗
            # （如"可处理工厂"的 F2,F6,F10 组合列，类别数远超合理离散列范围）
            unique_count = df[col].nunique(dropna=True)
            if unique_count > max(20, int(len(df) * 0.2)):
                log.append(f"{col}：{unique_count}个唯一值（多值/自由文本列），跳过异常清洗")
                continue

            # 枚举类列保护：唯一值较少时，每个类别都是合法业务值，不做高频替换
            # （如"工艺节点"4种工艺、"可处理工厂"4种组合，替换会破坏数据）
            if unique_count <= 10:
                log.append(f"{col}：{unique_count}个枚举类别，全部视为合法，跳过异常清洗")
                continue

            # 识别合理类别（频率前3）
            valid_cats = df[col].value_counts().nlargest(3).index.tolist()
            valid_cats = [cat for cat in valid_cats if cat != ""]
            if not valid_cats:
                log.append(f"{col}：无有效类别，跳过")
                continue

            # 标记异常值
            abnormal_mask = ~df[col].isin(valid_cats) | (df[col] == "")
            abnormal_count = abnormal_mask.sum()
            if abnormal_count == 0:
                log.append(f"{col}：无异常值")
                continue

            # 处理异常值
            if fill_strategy == "mode":
                mode_val = df.loc[~abnormal_mask, col].mode()[0]
                df.loc[abnormal_mask, col] = mode_val
                log.append(f"{col}：{abnormal_count}个异常值→众数（{mode_val}）")
            elif fill_strategy == "specific":
                df.loc[abnormal_mask, col] = fill_value
                log.append(f"{col}：{abnormal_count}个异常值→指定值（{fill_value}）")
            elif fill_strategy == "new_category":
                df.loc[abnormal_mask, col] = "其他"
                log.append(f"{col}：{abnormal_count}个异常值→新增类别（其他）")
            else:
                return json.dumps({
                    "status": "失败",
                    "message": f"不支持的策略：{fill_strategy}（可选：mode/specific/new_category）"
                }, ensure_ascii=False)

        # 更新上下文
        data_context.cleaning_df = df.copy()
        data_context.cleaning_log.append(f"离散型异常值处理完成：{'; '.join(log)}")

        return json.dumps({
            "status": "成功",
            "processing_log": log,
            "next_step": "处理数值型噪声或保存数据"
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "status": "失败",
            "message": f"处理离散型异常值出错：{str(e)}"
        }, ensure_ascii=False)


@tool
def handle_numeric_noise(method: str = "rolling_iqr", window: int = 5, threshold: float = 1.5, group_col: str = None) -> str:
    """
    处理数值型特征噪声（支持分组/全局时序检测）
    参数:
        method: 检测方法（rolling_iqr/滑动IQR | rolling_zscore/滑动Z-score | global_iqr/全局IQR | global_zscore/全局Z-score）
        window: 滑动窗口大小（仅rolling_*方法生效）
        threshold: 异常阈值（默认1.5）
        group_col: 分组列（可选，无则全局处理）
    """
    if data_context.cleaning_df is None:
        return json.dumps({
            "status": "失败",
            "message": "请先调用init_data_cleaning初始化数据"
        }, ensure_ascii=False)

    if not data_context.numeric_cols:
        return json.dumps({
            "status": "成功",
            "message": "无数值型特征列，无需处理噪声"
        }, ensure_ascii=False)

    try:
        df = data_context.cleaning_df.copy()
        log = []

        # 通用噪声处理函数
        def process_noise(series, method, window, threshold):
            series_clean = series.copy()
            if len(series.dropna()) < 10:
                return series_clean, 0

            # 滑动窗口检测
            if method.startswith("rolling"):
                window_series = series.rolling(window=window, center=True, min_periods=1)
                if method == "rolling_iqr":
                    q1 = window_series.quantile(0.25)
                    q3 = window_series.quantile(0.75)
                    iqr = q3 - q1
                    mask = (series < q1 - threshold * iqr) | (series > q3 + threshold * iqr)
                else:  # rolling_zscore
                    mean = window_series.mean()
                    std = window_series.std().replace(0, 0.001)
                    mask = np.abs((series - mean) / std) > threshold
            # 全局检测
            else:
                if method == "global_iqr":
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    mask = (series < q1 - threshold * iqr) | (series > q3 + threshold * iqr)
                else:  # global_zscore
                    # 处理空标准差
                    std_val = series.std()
                    if std_val == 0:
                        std_val = 0.001
                    z = (series - series.mean()) / std_val
                    mask = np.abs(z) > threshold

            # 插值替换异常值
            abnormal_count = mask.sum()
            if abnormal_count > 0:
                series_interp = series.interpolate("linear")
                series_clean.loc[mask] = series_interp.loc[mask]
            return series_clean, abnormal_count

        # 处理每个数值列
        for col in data_context.numeric_cols:
            # 离散业务值保护：唯一值较少时视为离散档位（如订单批量 600~1000），不做噪声修正
            # （IQR/Z-score 噪声检测针对连续时序值，对离散档位会误杀真实业务值）
            unique_count = df[col].nunique(dropna=True)
            if unique_count <= 10:
                log.append(f"{col}：{unique_count}个离散档位，跳过噪声修正")
                continue

            total_abnormal = 0
            # 有分组列则分组处理
            if group_col and group_col in df.columns:
                for _, group in df.groupby(group_col):
                    series = group[col].copy()
                    cleaned_series, cnt = process_noise(series, method, window, threshold)
                    df.loc[group.index, col] = cleaned_series
                    total_abnormal += cnt
                log.append(f"{col}：按{group_col}分组修正{total_abnormal}个噪声值（方法：{method}）")
            # 无分组列则全局处理
            else:
                series = df[col].copy()
                cleaned_series, total_abnormal = process_noise(series, method, window, threshold)
                df[col] = cleaned_series
                log.append(f"{col}：全局修正{total_abnormal}个噪声值（方法：{method}）")

        # 更新上下文（标记为最终清洗结果）
        data_context.cleaning_df = df.copy()
        data_context.cleaned_df = df.copy()
        data_context.cleaning_log.append(f"数值型噪声处理完成：{'; '.join(log)}")

        return json.dumps({
            "status": "成功",
            "processing_log": log,
            "next_step": "调用save_cleaned_data保存并发布消息"
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "status": "失败",
            "message": f"处理数值型噪声出错：{str(e)}"
        }, ensure_ascii=False)


@tool
def save_cleaned_data(filename: str = None) -> str:
    """
    保存清洗后的数据（自动生成带时间戳的文件名，支持CSV/Excel）
    参数:
        filename: 自定义文件名（可选，无需后缀）
    """
    if data_context.cleaned_df is None:
        return json.dumps({
            "status": "失败",
            "message": "无清洗后数据，请先完成缺失值/异常值/噪声处理",
            "next_step": "执行数据清洗操作"
        }, ensure_ascii=False)

    try:
        # 创建输出目录（适配Windows）
        output_dir = Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if not filename:
            filename = f"cleaned_data_{timestamp}"

        # 确定文件格式（和原始文件一致）
        file_ext = ".csv" if data_context.file_type == "csv" else ".xlsx"
        file_path = output_dir / f"{filename}{file_ext}"

        # 保存数据
        if data_context.file_type == "csv":
            data_context.cleaned_df.to_csv(file_path, index=False, encoding="utf-8")
        else:
            data_context.cleaned_df.to_excel(file_path, index=False, engine="openpyxl")

        # 更新上下文
        data_context.cleaned_file_path = str(file_path.resolve())
        data_context.cleaning_log.append(f"清洗后数据已保存：{data_context.cleaned_file_path}")

        return json.dumps({
            "status": "成功",
            "file_info": {
                "path": data_context.cleaned_file_path,
                "rows": len(data_context.cleaned_df),
                "size": os.path.getsize(file_path) / 1024  # KB
            },
            "next_step": "调用semantic_publish发布data2scenario语义标签"
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "status": "失败",
            "message": f"保存数据出错：{str(e)}"
        }, ensure_ascii=False)


# ===================== 兼容旧工具名（保证多智能体框架不报错）=====================
@tool
def clean_csv_data(file_path: str, output_path: str = None) -> str:
    """
    兼容旧版本的快捷清洗工具（一键完成校验+初始化+全量清洗+保存）
    参数:
        file_path: 原始文件路径
        output_path: 输出文件路径（可选）
    """
    # 链式调用核心工具（@tool 装饰后为 StructuredTool，需通过 .func() 调用原函数）
    check_result = check_file_exists.func(file_path)
    if "失败" in check_result:
        return check_result

    init_result = init_data_cleaning.func()
    if "失败" in init_result:
        return init_result

    # 全量清洗（全局模式，无分组）
    handle_missing_values.func()
    handle_categorical_abnormal.func()
    handle_numeric_noise.func()

    # 保存数据
    if output_path:
        filename = Path(output_path).stem
        save_result = save_cleaned_data.func(filename)
    else:
        save_result = save_cleaned_data.func()

    return save_result