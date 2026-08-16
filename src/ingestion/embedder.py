# src/ingestion/embedder.py
"""
Embedder 策略层:统一接口,支持多种 embedding 后端。
- 本地: fastembed(免费,无需 key)
- OpenAI 兼容 API: 同一个 OpenAI client,换 base_url + key 即可调多家

used by indexer.py, not for running independently.
"""
from abc import ABC, abstractmethod
from typing import List
from src.config import EMBED_PROVIDERS
import logging
logger = logging.getLogger(__name__) 


class BaseEmbedder(ABC):
    """Embedding 后端统一接口:embed() + vector_size。"""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        ...

    @property
    @abstractmethod
    def vector_size(self) -> int:
        ...


class FastEmbedder(BaseEmbedder):
    """免费本地模型(fastembed,无需 PyTorch / API key)"""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding  # 惰性导入

        print(f"📦 加载本地 Embedding 模型: {model_name}...")
        self.model = TextEmbedding(model_name=model_name)
        self._model_name = model_name
        self._vector_size = self._detect_vector_size()

    def _detect_vector_size(self) -> int:
        """读模型自带的 embedding_size;读不到就实际嵌入一次取长度。"""
        dim = getattr(self.model, "embedding_size", None)
        if isinstance(dim, int) and dim > 0:
            return dim
        return len(next(self.model.embed(["dimension probe"])))

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self.model.embed(texts)]

    @property
    def vector_size(self) -> int:
        return self._vector_size


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI 兼容协议的 embedding 后端:openai / gemini
    """
    def __init__(
        self,
        provider: str = "openai",
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):

        from openai import OpenAI
        self.provider = provider
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._vector_size: int | None = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in resp.data]

    @property
    def vector_size(self) -> int:
        """惰性探测:第一次用到才嵌入一次,取向量真实长度(API 自带,不内置映射表)。"""
        if self._vector_size is None:
            probe = self.client.embeddings.create(
                model=self.model_name, input=["dimension probe"]
            )
            self._vector_size = len(probe.data[0].embedding)
        return self._vector_size


def get_embedder(
        provider: str, 
        model_name: str, 
        api_key: str = None,
        base_url: str = None, 
    ) -> BaseEmbedder:
    """工厂函数:根据 provider 返回对应的 Embedder 实例。"""
    if provider == "fastembed":
        return FastEmbedder(
            model_name=model_name
        )
    if provider in EMBED_PROVIDERS:
        return OpenAIEmbedder(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )
    raise ValueError(
        f"不支持的 provider: {provider}"
    )
