import pytest
from pydantic import ValidationError

from src.schemas import ChunkRecord, ParsedDocument


def test_parsed_document_缺必填字段报错():
    """doc_id/source/title/content 是必填,少一个就拒绝,不传脏数据。"""
    with pytest.raises(ValidationError, match="content"):
        ParsedDocument(doc_id="1", source="confluence", title="t")  # 缺 content


def test_chunk_record_缺source报错():
    """source 是必填(文档来源必须明确),缺了拒绝。"""
    with pytest.raises(ValidationError, match="source"):
        ChunkRecord(chunk_id="a_p0_c0", child_text="c", parent_text="p", doc_id="1")


def test_chunk_record_title_metadata_有默认值():
    """title/metadata 允许缺省 —— 向后兼容;source 必须显式给。"""
    chunk = ChunkRecord(
        chunk_id="a_p0_c0", child_text="c", parent_text="p", doc_id="1", source="confluence"
    )
    assert chunk.title == ""
    assert chunk.metadata == {}


def test_chunk_record_json_往返():
    """model_dump_json 写出的行,必须能被 model_validate_json 原样读回。"""
    chunk = ChunkRecord(
        chunk_id="a_p0_c0", child_text="child", parent_text="parent",
        doc_id="1", source="confluence", metadata={"section": "Overview"},
    )
    restored = ChunkRecord.model_validate_json(chunk.model_dump_json())
    assert restored == chunk
