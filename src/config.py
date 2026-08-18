import os
from pathlib import Path


def _find_project_root() -> Path:
    """向上查找包含 pyproject.toml 的目录作为项目根。所有路径都以 PROJECT_ROOT 为锚点。"""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"找不到项目根(未在 {current} 及其上级目录中找到 pyproject.toml)")

PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"


# ----------------------------------------------
# RAG settings for experiments
# ----------------------------------------------
from dotenv import load_dotenv
if os.getenv("ENVIRONMENT") != "production":
    load_dotenv(PROJECT_ROOT / ".env")

EMBED_PROVIDERS = {
    "fastembed": (
        "",
        None,
        ["BAAI/bge-small-en-v1.5", "BAAI/bge-small-zh-v1.5", "sentence-transformers/all-MiniLM-L6-v2"]
    ),
    "openai": (
        "OPENAI_API_KEY", 
        None, # set the url as None so as to use the default
        ["text-embedding-3-small"]
    ),
    "gemini": (
        "GEMINI_API_KEY", 
        "https://generativelanguage.googleapis.com/v1beta/openai/", 
        ["gemini-embedding-2"], # 使用 output_dimensionality 参数控制输出嵌入向量的大小
    )
}

class Settings():
    """hyperparam settings for the RAG system"""

    # Experinemts: chunking
    PARENT_CHUNK_SIZE: int = 6000
    PARENT_CHUNK_OVERLAP: int = 400
    CHILD_CHUNK_SIZE: int = 1500
    CHILD_CHUNK_OVERLAP: int = 200
    SEPARATORS: list = ["\n\n", "\n", ". ", " "]

    # Experinemts: embedding
    EMBEDDING_PROVIDER: str = list(EMBED_PROVIDERS.keys())[0]
    EMBEDDING_MODEL_NAME: str = EMBED_PROVIDERS[EMBEDDING_PROVIDER][2][0]
    EMBEDDING_API_KEY: str = os.getenv(EMBED_PROVIDERS[EMBEDDING_PROVIDER][0]) or "** NO API KEY **"
    EMBEDDING_BASE_URL: str = None

    # vector database
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "confluence_policies"

    # sparse search
    SPARSE_SEARCH_BASE_FILE: str = DATA_DIR / "processed" / "chunked_confluence.jsonl"

    SEARCH_TOP_K: int = 5


settings = Settings()
