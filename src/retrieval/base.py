# src/retrieval/base.py
from abc import ABC, abstractmethod
from typing import List, Optional
from src.schemas import SearchResult


# 表明这是接口/基类，不是直接可实例化的具体类。
class BaseSearcher(ABC):
    """所有 Searcher 的抽象基类"""

    # 标记方法为必须在子类中实现。
    # 若子类没有实现所有抽象方法，尝试实例化会抛出 TypeError，提醒你实现接口。
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        source_type_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        pass
