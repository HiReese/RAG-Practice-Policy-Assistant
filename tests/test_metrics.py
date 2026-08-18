"""指标函数的单元测试

1. 指标函数为什么能测:它们不碰检索器/向量库/文件,喂 (预期, 召回) 出数字。
2. pytest.approx:浮点数断言别用 ==,用近似比较,避免 0.1+0.2 类精度问题。

运行(项目根目录):
    pytest tests/test_metrics.py -v
"""
import math

import pytest

from src.evaluation.eval_retrieval import _hit, _mrr, _ndcg, _recall, compute_metrics


def test_hit_命中任意预期即为1():
    assert _hit(["A"], ["B", "A"]) == 1.0
    assert _hit(["A"], ["B", "C"]) == 0.0


def test_mrr_取第一个命中位置的倒数():
    assert _mrr(["A"], ["B", "A", "C"]) == pytest.approx(0.5)  # rank=2 -> 1/2
    assert _mrr(["A"], ["B", "C"]) == 0.0                     # 没命中


def test_recall_命中预期数除以预期总数():
    assert _recall(["A", "B"], ["A", "C", "D"]) == pytest.approx(0.5)  # 命中 1/2


def test_ndcg_相关文档越靠前分越高():
    assert _ndcg(["A"], ["A", "B", "C"]) == pytest.approx(1.0)          # 第 0 位,满
    assert _ndcg(["A"], ["B", "C", "A"]) == pytest.approx(1.0 / math.log2(4))  # 第 2 位,衰减


def test_compute_metrics_多query求平均():
    metrics = compute_metrics(
        expected_ids=[["A"], ["X"]],
        retrieved_ids=[["A", "B"], ["C", "D"]],   # query1 命中, query2 全miss
        top_k=5,
    )
    assert metrics["hit_rate"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx(0.5)   # (1.0 + 0.0) / 2
    assert metrics["recall"] == pytest.approx(0.5)


def test_compute_metrics_输入长度不一致要报错():
    with pytest.raises(ValueError, match="数量必须一致"):
        compute_metrics(expected_ids=[["A"]], retrieved_ids=[["A"], ["B"]])
