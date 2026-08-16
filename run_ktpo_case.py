# run_ktpo_case.py
"""用 case.xlsx（100订单/10工厂）运行 KTPO 算法，与多智能体决策结果对比。

KTPO = 知识引导双种群进化（NSGA-II + DE），目标：最小化 Makespan + 最大化 OTD。
"""
import os
import sys
import time
import numpy as np
import pandas as pd

from KTPO import load_case_data, solve_case_ktpo

if __name__ == "__main__":
    case_file = "./data/case.xlsx"
    t0 = time.time()

    # 加载数据
    orders, factory_cap, tech_to_factories = load_case_data(case_file)
    print(f"案例: {case_file}")
    print(f"订单数量: {len(orders)}, 工厂数量: {len(factory_cap)}, 工艺节点: {len(tech_to_factories)}")
    print(f"工厂产能: {factory_cap}")
    print(f"工艺->工厂: { {k: sorted(v) for k, v in tech_to_factories.items()} }")

    # 运行 KTPO
    print(f"\n开始 KTPO 求解 (pop_size=60, max_gen=50) ...")
    pareto_obj, pareto_ind = solve_case_ktpo(
        orders, factory_cap,
        pop_size=60, max_gen=50,
    )
    elapsed = time.time() - t0

    # 结果
    print(f"\n{'=' * 60}")
    print(f"KTPO 求解完成，耗时 {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"Pareto 解数量: {len(pareto_obj)}")

    df = pd.DataFrame(pareto_obj, columns=["最大完工时间(T)", "准时交付率(%)"])
    df.insert(0, "Pareto解编号", range(1, len(df) + 1))
    print("\nPareto 前沿:")
    print(df.to_string(index=False))

    best_otd_idx = np.argmax(pareto_obj[:, 1])
    best_mk_idx = np.argmin(pareto_obj[:, 0])
    print(f"\n最优准时交付率: {pareto_obj[best_otd_idx, 1]}% (对应 Makespan: {pareto_obj[best_otd_idx, 0]}T)")
    print(f"最优Makespan: {pareto_obj[best_mk_idx, 0]}T (对应准时交付率: {pareto_obj[best_mk_idx, 1]}%)")

    # 保存
    out = "output/ktpo_case_result.xlsx"
    df.to_excel(out, index=False)
    print(f"\n结果已保存: {out}")
