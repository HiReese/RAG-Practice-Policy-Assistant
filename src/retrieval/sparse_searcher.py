from pathlib import Path
import re

from src.config import settings
from src.retrieval.base import BaseSearcher
from src.schemas import ChunkRecord, SearchResult



STOPWORDS = {"the", "is", "and", "of", "in", "to", "a", "for", "on", "with", "by"}

# 领域同义词
SYNONYMS = {
    "svc": "service",
    "k8s": "kubernetes",
}

# 领域短语: 合并按"长度优先"匹配,长短语会自动先命中
PHRASES = {
    ("service", "level", "objective"): "slo",
    ("architecture", "decision", "record"): "adr",
    ("api", "key"): "api-key",
    ("top", "k"): "top-k",
}


# 评估结论:spaCy lemma 方案已弃用
# 词形还原/词干化:默认不做,只在"纯稀疏检索"或学术对标 TREC 时做
# 词形泛化在高阶场景由其他手段覆盖,BM25 端加了只增复杂度。

def tokenize(text: str) -> list[str]:
    """light and necessary tokenize for BM25 sparse search, without heavy spaCy or nltk."""

    # "web-app"→ ["web-app"]
    tokens = re.findall(r"[a-z0-9_]+(?:-[a-z0-9_]+)*", text.lower())
    # SYNONYMS: svc -> service
    tokens = [SYNONYMS.get(t, t) for t in tokens]
    # PHRASES
    merged: list[str] = []
    i = 0
    while i < len(tokens):
        for phrase, canonical in sorted(PHRASES.items(), key=lambda kv: -len(kv[0])):
            if tuple(tokens[i:i + len(phrase)]) == phrase:
                merged.append(canonical)
                i += len(phrase)
                break
        else:
            merged.append(tokens[i])
            i += 1
    # STOPWORDS
    return [t for t in merged if t not in STOPWORDS and len(t) > 1]


class Bm25SparseSearcher(BaseSearcher):
    """sparse searcher based on in-memory BM25"""

    def __init__(self, chunks_jsonl_path: str = settings.SPARSE_SEARCH_BASE_FILE):
        self.chunk_records: list[ChunkRecord] = []
        self.bm25 = None
        self._build_index(chunks_jsonl_path)
        # TODO: 未来可考虑用 sqlite + FTS5 做持久化,但目前内存够用,且 BM25 算法本身是 O(n) 的,不适合大规模数据。
        # TODO: "每次请求 new 实例"——单例化/生命周期管理 : FastAPI 生命周期:应用启动时建一次,关停时清理

    def _build_index(self, path: str):
        """build BM25 index from local jsonl file"""

        from rank_bm25 import BM25Okapi  # 惰性导入:模块可轻量 import,便于测 tokenize

        corpus = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunk = ChunkRecord.model_validate_json(line)
                self.chunk_records.append(chunk)
                corpus.append(tokenize(chunk.child_text))

        self.bm25 = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_type_filter: str | None = None,
    ) -> list[SearchResult]:
        """BM25 检索:分词 query → 打分 → (按 source 过滤)→ 返回 top_k。"""
        query_tokens = tokenize(query)
        if not query_tokens or self.bm25 is None:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # 先过滤再排序取 top_k(过滤后取,避免"先截断再过滤"导致少返回)
        scored = [
            (float(scores[i]), self.chunk_records[i]) for i in range(len(scores))
        ]
        if source_type_filter:
            scored = [(s, c) for s, c in scored if c.source in source_type_filter]

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(
                chunk_id=c.chunk_id,
                child_text=c.child_text,
                parent_text=c.parent_text,
                doc_id=c.doc_id,
                score=s,
                metadata=c.metadata,
            )
            for s, c in scored[:top_k]
        ]
