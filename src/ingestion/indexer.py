# src/ingestion/indexer.py (ingestor)
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from src.config import settings, DATA_DIR
from src.ingestion.schemas import ChunkRecord
from src.ingestion.embedder import get_embedder, BaseEmbedder
import logging
logger = logging.getLogger(__name__) 

import hashlib

def stable_point_id(chunk_id: str) -> int:
    return int.from_bytes(
        hashlib.md5(chunk_id.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )


class QdrantIngestor:

    def __init__(
        self,
        embedder: BaseEmbedder,
        qdrant_url: str = settings.QDRANT_URL,
        collection_name: str = settings.QDRANT_COLLECTION_NAME,
    ):
        self.embedder = embedder
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
       

    def ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            logger.info(f"🛠️ [Qdrant] 创建 Collection '{self.collection_name}', 维度: {self.embedder.vector_size}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedder.vector_size, distance=Distance.COSINE
                ),
            )

    def upsert_chunks(self, chunks: List[ChunkRecord]):
        if not chunks:
            return

        texts = [c.child_text for c in chunks]
        embeddings = self.embedder.embed(texts)

        points = []
        for chunk, vector in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=stable_point_id(chunk.chunk_id),
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "child_text": chunk.child_text,
                        "parent_text": chunk.parent_text,
                        "doc_id": chunk.doc_id,
                        "source": chunk.source,
                        "title": chunk.title,
                        **chunk.metadata,
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)


def run_indexing_job(
    chunks_jsonl: str,
    collection_name: str = settings.QDRANT_COLLECTION_NAME,
    batch_size: int = 100,
):
    """独立执行的 Indexing 任务：读 ChunkRecord -> Embed -> 存 Qdrant"""
    # 动态根据 settings 加载对应的 Embedding 模型 (fastembed / openai)
    embedder = get_embedder(
        provider=settings.EMBEDDING_PROVIDER,
        model_name=settings.EMBEDDING_MODEL_NAME,
        api_key=settings.EMBEDDING_API_KEY,
    )

    ingestor = QdrantIngestor(embedder=embedder, collection_name=collection_name)
    ingestor.ensure_collection()

    chunk_buffer: List[ChunkRecord] = []
    total_indexed = 0

    logger.info(f"🚀 [Indexer] 开始向量化并写入 Qdrant: Provider={settings.EMBEDDING_PROVIDER}, Collection={collection_name}")

    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            chunk = ChunkRecord.model_validate_json(line)
            chunk_buffer.append(chunk)

            if len(chunk_buffer) >= batch_size:
                ingestor.upsert_chunks(chunk_buffer)
                total_indexed += len(chunk_buffer)
                logger.info(f"进度：已向量化并写入 {total_indexed} 条...")
                chunk_buffer.clear()

    if chunk_buffer:
        ingestor.upsert_chunks(chunk_buffer)
        total_indexed += len(chunk_buffer)

    logger.info(f"🎉 [Indexer] 全部完成！共写入 {total_indexed} 条向量至 Qdrant Collection '{collection_name}'")


def main():
    chunks_jsonl: str = DATA_DIR / "processed" / "chunked_confluence.jsonl"
    run_indexing_job(chunks_jsonl)


if __name__ == "__main__":
    main()
    