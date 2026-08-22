
from src.config import DATA_DIR, settings
from src.schemas import EvalSample, SearchResult
from src.retrieval.base import BaseSearcher
from src.retrieval.dense_searcher import DenseSearcher
from src.retrieval.sparse_searcher import Bm25SparseSearcher
import math


def load_eval_dataset(eval_file_jsonl_path: list, eval_source_types_filter: list) -> list[EvalSample]:
    samples = []
    for file in eval_file_jsonl_path:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    e = EvalSample.model_validate_json(line)
                    e.expected_doc_ids = [d.split('_')[1] for d in e.expected_doc_ids]
                    if e.source_types in eval_source_types_filter:
                        samples.append(e)
    return samples


def retrieve(
    searcher: BaseSearcher,
    samples: list[EvalSample],
    top_k: int = 5,
) -> list[list[str]]:
    """returns top_k retrieved doc_ids for each sample, easier for later eval metric calculation"""
    return [[r.doc_id for r in searcher.search(s.question, top_k=top_k)] for s in samples]


# ---------------------------------------------------------------------------
# eval metrics: hit, recall, mrr, ndcg, @top_k
# ---------------------------------------------------------------------------
def _hit(expected: list[str], retrieved: list[str]) -> float:
    """Hit(二值):只要召回里命中任意一个预期 doc_id 就是 1,否则 0。"""
    return 1.0 if any(d in expected for d in retrieved) else 0.0


def _recall(expected: list[str], retrieved: list[str]) -> float:
    """Recall@k:命中的预期 doc 数 / 预期 doc 总数。预期为空时未定义,按 0 计。"""
    if not expected:
        return 0.0
    return sum(1 for d in retrieved if d in expected) / len(expected)


def _mrr(expected: list[str], retrieved: list[str]) -> float:
    """MRR:第一个命中结果的倒数排名(rank 从 1 开始);没命中为 0。"""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def _ndcg(expected: list[str], retrieved: list[str]) -> float:
    """nDCG@k(二元相关性):rel_i = 1 当且仅当 retrieved[i] 在 expected 中。

    DCG  = Σ rel_i / log2(i+2)
    IDCG = 理想排序(所有相关文档排最前)下的 DCG
    nDCG = DCG / IDCG —— 相关文档排得越靠前,分越高。
    """
    dcg = sum(
        1.0 / math.log2(i + 2) for i, doc_id in enumerate(retrieved) if doc_id in expected
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected), len(retrieved))))
    return dcg / ideal if ideal > 0 else 0.0


def compute_metrics(
    expected_ids: list[list[str]],
    retrieved_ids: list[list[str]],
    top_k: int,
) -> dict[str, float]:
    """计算检索层指标,返回 dict。所有指标都在 retrieved[:top_k] 上计算。

    注意:expected 为空的样本会把 recall 平均拉低(保守选择),
    想跳过它们,调用前先过滤即可。
    """
    if len(expected_ids) != len(retrieved_ids):
        raise ValueError("expected_ids 与 retrieved_ids 数量必须一致")

    n = len(expected_ids)
    if n == 0:
        return {"top_k": top_k, "hit_rate": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}

    truncated = [ret[:top_k] for ret in retrieved_ids]
    return {
        "top_k": top_k,
        "hit_rate": sum(_hit(e, r) for e, r in zip(expected_ids, truncated)) / n,
        "recall": sum(_recall(e, r) for e, r in zip(expected_ids, truncated)) / n,
        "mrr": sum(_mrr(e, r) for e, r in zip(expected_ids, truncated)) / n,
        "ndcg": sum(_ndcg(e, r) for e, r in zip(expected_ids, truncated)) / n,
    }


# ---------------------------------------------------------------------------
# ③ 编排:组装 + 返回指标 dict
# ---------------------------------------------------------------------------
def evaluate_searcher(
    searcher: BaseSearcher,
    eval_dataset_path: list,
    eval_source_types_filter: list,
    top_k: int = settings.SEARCH_TOP_K,
    verbose: bool = True,
) -> dict[str, float]:
    """完整流程:加载数据 → 检索 → 算指标 → (可选)打印 → 返回指标。"""

    samples = load_eval_dataset(eval_dataset_path, eval_source_types_filter)
    print(f"Total {len(samples)} samples for evaluation on {eval_source_types_filter}")
    retrieved_ids = retrieve(searcher, samples, top_k)
    metrics = compute_metrics(
        [s.expected_doc_ids for s in samples], retrieved_ids, top_k
    )

    if verbose:
        for sample, rec in zip(samples, retrieved_ids):
            hit = any(d in sample.expected_doc_ids for d in rec)
            icon = "✅" if hit else "❌"
            print(f"{icon} Q-ID: {sample.question_id} | {sample.question[:50]}...")
            if not hit:
                print(f"   Expected: {sample.expected_doc_ids}")
                print(f"   Retrieved: {rec}")

    print("\n" + "=" * 50)
    print(f"📈 Retrieval Metrics Summary")
    print(f"  searcher: {searcher.__class__.__name__} (collection={getattr(searcher, 'collection_name', '')}, embedder={getattr(searcher, 'embedder', type(None)).__class__.__name__})") 
    print("=" * 50)
    for key, value in metrics.items():
        if key == "top_k":
            continue
        print(f"• {key:>8} @ {metrics['top_k']} : {value:.4f}")
    print("=" * 50)

    return metrics


if __name__ == "__main__":

    eval_dataset_path = [
        DATA_DIR / "EnterpriseRAG-Bench/extra_questions.jsonl",
        DATA_DIR / "EnterpriseRAG-Bench/questions.jsonl"
    ]
    eval_source_types_filter = [["confluence"]]

    metrics = evaluate_searcher(
        Bm25SparseSearcher(), 
        eval_dataset_path, eval_source_types_filter, 
        settings.SEARCH_TOP_K
    )

