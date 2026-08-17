
from typing import Optional, List
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.retrieval.base import BaseSearcher
from src.schemas import SearchResult
from src.ingestion.embedder import BaseEmbedder, get_embedder
from src.config import settings


class DenseSearcher(BaseSearcher):
    """基于 Qdrant 的 Dense 向量检索器"""

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        qdrant_url: str = settings.QDRANT_URL,
        collection_name: str = settings.QDRANT_COLLECTION_NAME,
    ):
        # initialize embedder
        self.embedder = embedder or get_embedder(
                provider=settings.EMBEDDING_PROVIDER,
                model_name=settings.EMBEDDING_MODEL_NAME,
                api_key=settings.EMBEDDING_API_KEY,
                base_url=settings.EMBEDDING_BASE_URL,
        )
        # initialize qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name


    def search(
            self,
            query: str,
            top_k: int = 5,
            source_type_filter: Optional[str] = None,
        ) -> List[SearchResult]:
        """handles a single query for each call, returns with all payload"""

        if not query.strip():
            return []
        
        query_vector = self.embedder.embed([query])[0]

        query_filter = None
        if source_type_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source_type",
                        match=MatchValue(value=source_type_filter),
                    )
                ]
            )

        hits = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        ).model_dump()
        
        results = []
        for hit in hits['points']:
            payload = hit.get("payload") or {}
            results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", ""),
                    child_text=payload.get("child_text", ""),
                    parent_text=payload.get("parent_text", ""),
                    doc_id=payload.get("doc_id", ""),
                    score=hit.get("score"),
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k not in ["child_text", "parent_text", "chunk_id", "doc_id"]
                    },
                )
            )
        return results


if __name__ == "__main__":
    test_question = {
        "question_id": "qst_0011", 
        "question_type": "basic", 
        "source_types": ["confluence"], 
        "question": "How long is the validity period for the telemetry driven runbook author certification from the operator training bootcamp?", 
        "expected_doc_ids": ["dsid_46a4cb87db414e769f2df86f01626948"], 
        "gold_answer": "The \"Telemetry-Driven Runbook Author (Redwood)\" certification is valid for 18 months.", 
        "answer_facts": ["The Telemetry-Driven Runbook Author (Redwood) certification is valid for 18 months."]
    }
    expected_doc_ids = test_question.get("expected_doc_ids")

    searcher = DenseSearcher()
    result = searcher.search(test_question["question"])
    print(result[0].doc_id)
    print(result[0].child_text)
    