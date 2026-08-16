# tool_experience_library.py
"""动态工具链调用机制 - 工具使用经验库（论文机制③）。

核心结构（论文公式）：
- 经验库 E = {E_1, ..., E_M}
- 单条经验 E_i = {E_i^desc, E_i^order, E_i^exam}
  - E_i^desc: 经验链语义描述（任务类型/场景特征/核心目标）
  - E_i^order: 工具调用序列 {T_i1, ..., T_iK}
  - E_i^exam: 使用示例 {In_i 输入样本, Out_i 输出样本, Res_i 执行效果反馈}
- 语义向量：e_i = Embed(E_i^desc)（用于相似度检索）

动态调用流程：
1. 任务描述 → 向量化 → 与经验库各 desc 向量计算相似度 Sim
2. 最大相似度 Sim_max >= τ_e：复用匹配经验链的工具序列
3. Sim_max < τ_e：自主构建新工具链 → 执行后按可执行性 Φ(E_i) 验证 → 增量入库

可执行性判定（论文公式）：
Φ(E_i) = 1 完整执行无异常且输出符合调度任务语义规范
Φ(E_i) = 0 执行失败或输出无效
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

from embedding_service import embed_query, cosine_similarity

# 匹配阈值 τ_e：任务与经验链匹配度低于此值则自主构建新链
TAU_E = float(os.getenv("TOOL_MATCH_THRESHOLD", "0.55"))
EXPERIENCE_DATA_PATH = os.path.join(os.path.dirname(__file__), "tool_experience_data.json")


class ToolExperienceLibrary:
    """工具使用经验库：加载、检索、验证、增量更新。"""

    def __init__(self, data_path: str = EXPERIENCE_DATA_PATH):
        self.data_path = data_path
        self.experiences: List[Dict] = []
        self._desc_vectors: Dict[str, List[float]] = {}
        self._load()

    # ---------------- 加载与持久化 ----------------
    def _load(self) -> None:
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.experiences = data.get("experiences", [])
        for exp in self.experiences:
            self._ensure_vector(exp)

    def _save(self) -> None:
        data = {"version": "1.0", "experiences": self.experiences}
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _ensure_vector(self, exp: Dict) -> List[float]:
        exp_id = exp["id"]
        if exp_id not in self._desc_vectors:
            self._desc_vectors[exp_id] = embed_query(exp["desc"])
        return self._desc_vectors[exp_id]

    # ---------------- 检索（语义匹配） ----------------
    def retrieve(self, task_description: str, top_k: int = 1, target_agent: str = None) -> List[Dict]:
        """按任务描述语义检索经验链（e_i = Embed(E_i^desc)，Sim = 余弦相似度）。

        Args:
            task_description: 任务描述
            top_k: 返回前 k 条
            target_agent: 按经验链所属 agent 域过滤（如 "decision_agent"），
                          避免跨域语义干扰（建模链与决策链描述高度重叠时，域过滤更准确）

        Returns:
            [{exp, score}] 按相似度降序
        """
        candidates = self.experiences
        if target_agent:
            candidates = [e for e in self.experiences if e.get("target") == target_agent]
            # 域内无经验时回退全库（保证有候选）
            if not candidates:
                candidates = self.experiences
        if not candidates:
            return []
        task_vec = embed_query(task_description)
        scored = []
        for exp in candidates:
            vec = self._ensure_vector(exp)
            s = cosine_similarity(task_vec, vec)
            scored.append({"exp": exp, "score": round(s, 4)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def select_tool_chain(self, task_description: str, target_agent: str = None) -> Tuple[List[str], Dict]:
        """为任务选择工具链（动态工具链调用的核心入口）。

        Args:
            task_description: 任务描述
            target_agent: 按 agent 域过滤候选经验链（机制③的域感知检索）

        Returns:
            (tool_names, info)
            - Sim_max >= τ_e：复用匹配经验链的工具序列
            - Sim_max <  τ_e：返回空列表（触发自主构建新链）
        """
        results = self.retrieve(task_description, top_k=1, target_agent=target_agent)
        if not results:
            return [], {"matched": False, "reason": "经验库为空"}
        best = results[0]
        if best["score"] >= TAU_E:
            return best["exp"]["order"], {
                "matched": True,
                "exp_id": best["exp"]["id"],
                "score": best["score"],
                "desc": best["exp"]["desc"],
                "exam": best["exp"]["exam"],
            }
        return [], {
            "matched": False,
            "reason": f"最大匹配度 {best['score']} < 阈值 τ_e={TAU_E}，需自主构建新工具链",
            "best_exp_id": best["exp"]["id"],
        }

    # ---------------- 可执行性验证（论文 Φ(E_i)） ----------------
    @staticmethod
    def verify_executability(exec_log: List[str], output_valid: bool = True) -> int:
        """可执行性判定函数 Φ(E_i)。

        Args:
            exec_log: 执行过程日志（若含异常/失败关键词 → Φ=0）
            output_valid: 输出是否符合调度任务语义规范

        Returns:
            1 = 完整执行无异常且输出有效；0 = 执行失败或输出无效
        """
        fail_keywords = ["异常", "失败", "错误", "error", "exception", "Traceback", "不存在"]
        for line in exec_log:
            if any(kw.lower() in str(line).lower() for kw in fail_keywords):
                return 0
        return 1 if output_valid else 0

    # ---------------- 增量更新（事件驱动） ----------------
    def add_experience(self, desc: str, tool_order: List[str],
                       example: Dict, exec_log: List[str], output_valid: bool = True) -> Optional[Dict]:
        """新增经验链：先按 Φ(E_i) 验证可执行性，通过后封装入库并计算向量。

        Args:
            desc: E_i^desc 语义描述
            tool_order: E_i^order 工具调用序列
            example: E_i^exam = {In_i, Out_i, Res_i}
            exec_log: 执行过程日志（用于 Φ 验证）
            output_valid: 输出是否有效

        Returns:
            入库的经验链；验证不通过返回 None
        """
        if self.verify_executability(exec_log, output_valid) != 1:
            return None

        exp = {
            "id": f"exp_{len(self.experiences) + 1:03d}",
            "desc": desc,
            "order": tool_order,
            "exam": example,
            "created_at": time.time(),
        }
        self.experiences.append(exp)
        self._ensure_vector(exp)
        self._save()  # 持久化（事件驱动增量更新，无周期训练开销）
        return exp

    def summary(self) -> str:
        return f"经验库共 {len(self.experiences)} 条工具链：" + \
               ", ".join(f"{e['id']}({len(e['order'])}个工具)" for e in self.experiences)


# 全局单例
experience_library = ToolExperienceLibrary()


if __name__ == "__main__":
    print(f"匹配阈值 τ_e = {TAU_E}")
    print(experience_library.summary())
    for task in ["清洗订单数据文件中的缺失值和噪声", "基于清洗数据构建场景模型代码", "构建决策优化模型求解排程方案"]:
        chain, info = experience_library.select_tool_chain(task)
        print(f"\n任务: {task}")
        print(f"  匹配: {info.get('matched')} score={info.get('score')} exp={info.get('exp_id')}")
        print(f"  工具链: {chain}")
