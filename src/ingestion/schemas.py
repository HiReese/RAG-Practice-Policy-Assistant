"""schemas.py —— ingestion 阶段的数据契约(schema)。

设计原则:
- "可缺失"字段一律给默认值,向后兼容三方调用:
- 用内置 dict[str, Any](Python >= 3.10),不用旧式 typing.Dict / Optional。
"""
from typing import Any
from pydantic import BaseModel, Field


class ParsedDocument(BaseModel):
    doc_id: str = Field(description="unique doc id, e.g. dsid_xxxx")
    source: str = Field(description="source of the document, e.g. confluence / jira / slack")
    title: str = Field(description="title of the document")
    content: str = Field(description="content of the document")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="etra document metadata, optional. (e.g. file_path / updated_at ...);",
    )


class ChunkRecord(BaseModel):
    """chunker.py 输出:parent-child 切分后的一条子块记录。"""

    chunk_id: str = Field(description="unique child chunk id, e.g. {doc_id}_p{p}_c{c}")
    child_text: str = Field(description="子块文本 —— 向量检索单元")
    parent_text: str = Field(description="子块所属父块 —— LLM 上下文")
    doc_id: str = Field(description="")
    source: str = Field(description="")
    title: str = Field(default="", description="所属文档标题(检索结果展示用)")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="结构化元数据:section 面包屑 / p_idx / c_idx 等,用于 Qdrant 过滤与展示",
    )
