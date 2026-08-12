from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.schemas import ParsedDocument, ChunkRecord
from pathlib import Path

from src.config import DATA_DIR, settings
import logging
logger = logging.getLogger(__name__)


class TextChunker:

    def __init__(
            self, 
            parent_chunk_size: int = settings.PARENT_CHUNK_SIZE,
            parent_chunk_overlap: int = settings.PARENT_CHUNK_OVERLAP,
            child_chunk_size: int = settings.CHILD_CHUNK_SIZE,
            child_chunk_overlap: int = settings.CHILD_CHUNK_OVERLAP

    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap
            )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,  chunk_overlap= child_chunk_overlap
        )

    def chunk_document(self, doc: ParsedDocument) -> list[ChunkRecord]:
        chunks = []

        parent_texts = self.parent_splitter.split_text(doc.content)
        for p_idx, p_text in enumerate(parent_texts):
            child_texts = self.child_splitter.split_text(p_text)
            for c_idx, c_text in enumerate(child_texts):
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"{doc.doc_id}_p{p_idx}_c{c_idx}",
                        child_text=c_text,
                        parent_text=p_text,
                        doc_id=doc.doc_id,
                        source=doc.source,
                        title=doc.title,
                        metadata=doc.metadata,
                        )
                )
        return chunks

def run_chunking_job(
        input_file: Path, 
        output_file: Path
):
    chunker = TextChunker()
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    logger.info(f"🚀 [Chunker] 开始切分文档: {input_file} -> {output_file}")
    logger.info(f"⚙️ 当前 Chunk 参数: Child={settings.CHILD_CHUNK_SIZE}, Parent={settings.PARENT_CHUNK_SIZE}")

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            if not line.strip():
                continue

            doc = ParsedDocument.model_validate_json(line)
            chunks = chunker.chunk_document(doc)

            # 流式逐条追加写入 chunks.jsonl
            for chunk in chunks:
                f_out.write(chunk.model_dump_json() + "\n")
                total_chunks += 1

    logger.info(f"✅ [Chunker] 完成！共生成并发落盘 {total_chunks} 条 ChunkRecord 到 {output_file}")


def main():
    logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        )
    
    input_jsonl = DATA_DIR / "processed" / "parsed_confluence.jsonl"
    output_jsonl = DATA_DIR / "processed" / "chunked_confluence.jsonl"
    run_chunking_job(input_jsonl, output_jsonl)


if __name__ == "__main__":
    # python -m src.ingestion.chunker
    main()