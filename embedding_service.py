# embedding_service.py
"""文本语义向量化服务（论文公式 Embed(·) 与 Sim(·)）。

可插拔三层实现：
1. Ollama 本地嵌入模型（默认 BGE-M3，1024 维，多语言，离线可用）
2. 失败/不可用时回退：字符级 n-gram 特征向量的余弦相似度（零依赖纯 Python）

对外统一接口：
    embed_texts(texts) -> list[list[float]]
    embed_query(text) -> list[float]
    cosine_similarity(vec_a, vec_b) -> float  # 论文公式 Sim(·)
"""

import os
import urllib.request
import json
import time
from typing import List

# ===================== 配置 =====================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "BGE-M3:latest")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "10"))

_ollama_cache = {}  # text -> vector 简单缓存
_ollama_available = None  # None=未探测, True/False=探测结果


def _ollama_embed(texts: List[str]) -> List[List[float]] | None:
    """通过 Ollama 本地嵌入模型计算向量；不可用返回 None。"""
    global _ollama_available
    if _ollama_available is False:
        return None

    # 命中缓存
    uncached = [t for t in texts if t not in _ollama_cache]
    if uncached:
        try:
            body = json.dumps({"model": OLLAMA_EMBED_MODEL, "input": uncached}).encode()
            req = urllib.request.Request(
                f"{OLLAMA_BASE_URL}/api/embed",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as r:
                data = json.loads(r.read())
            embs = data.get("embeddings") or data.get("data", [])
            for t, v in zip(uncached, embs):
                if isinstance(v, dict):
                    v = v.get("embedding", [])
                _ollama_cache[t] = list(v)
            _ollama_available = True
        except Exception:
            _ollama_available = False
            return None

    return [_ollama_cache[t] for t in texts]


# ===================== 字符级 n-gram 回退实现（零依赖）=====================
def _char_ngrams(text: str, n: int = 2) -> dict:
    """字符级 n-gram 词袋（小写、去空白）。"""
    norm = "".join(text.lower().split())
    grams = {}
    if len(norm) < n:
        grams[norm] = grams.get(norm, 0) + 1
        return grams
    for i in range(len(norm) - n + 1):
        g = norm[i:i + n]
        grams[g] = grams.get(g, 0) + 1
    return grams


def _fallback_embed(text: str) -> List[float]:
    """字符级 n-gram 频率向量（回退方案）。"""
    grams = _char_ngrams(text)
    if not grams:
        return []
    total = sum(grams.values())
    # 按固定字母表排序保证维度一致（ASCII 可打印字符的二元组）
    keys = sorted(grams.keys())
    return [grams[k] / total for k in keys]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """批量文本向量化：Ollama BGE-M3 优先，回退字符 n-gram。"""
    if not texts:
        return []
    vectors = _ollama_embed(texts)
    if vectors is not None:
        return vectors
    return [_fallback_embed(t) for t in texts]


def embed_query(text: str) -> List[float]:
    """单条文本向量化。"""
    return embed_texts([text])[0]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """余弦相似度（论文公式 Sim(·)），返回 [0,1] 区间。

    兼容不同长度向量（回退方案维度可能不同）：
    取两向量共同维度计算；若共同维度为 0 返回 0。
    """
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        # 维度不同：用交集索引计算（仅回退模式可能发生）
        d_a = {i: v for i, v in enumerate(vec_a)}
        d_b = {i: v for i, v in enumerate(vec_b)}
        common = set(d_a.keys()) & set(d_b.keys())
        if not common:
            return 0.0
        a = [d_a[i] for i in common]
        b = [d_b[i] for i in common]
    else:
        a, b = vec_a, vec_b
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


if __name__ == "__main__":
    t0 = time.time()
    v1 = embed_query("cleaned_data_ready 数据清洗完成通知")
    v2 = embed_query("data2scenario 清洗后数据已就绪")
    v3 = embed_query("decision_ready 决策结果已生成")
    v4 = embed_query("decision_ready")
    print(f"embedding dim: {len(v1)} (耗时 {time.time() - t0:.2f}s)")
    print(f"sim(清洗完成, 数据就绪) = {cosine_similarity(v1, v2):.4f}")
    print(f"sim(清洗完成, 决策结果) = {cosine_similarity(v1, v3):.4f}")
    print(f"sim(清洗完成, 决策结果短标签) = {cosine_similarity(v1, v4):.4f}")
