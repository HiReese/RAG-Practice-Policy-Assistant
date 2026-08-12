from pathlib import Path
from typing import Literal
import re
from src.config import DATA_DIR
from src.ingestion.schemas import ParsedDocument
import logging
logger = logging.getLogger(__name__)


def parse_confluence_file(file_path: Path, source: str) -> ParsedDocument:
    """读取单个 Confluence txt/md 文件并提取 metadata"""
    file_name = file_path.stem
    text = open(file_path, encoding="utf-8").read().strip()
    m = re.match(r"dsid_([0-9a-f]+)__(.+)", file_name)
    doc_id, title = (m.group(1), m.group(2)) if m else ("", file_name)
    return ParsedDocument(doc_id=doc_id, title=title, content=text, source=source)


def run_parser(input_dir: Path, output_file: Path, source: str, write_mode: Literal['w', 'a'] = 'w'):
    """parse files in {input_dir} and write the parsed document into {output_file} one by one"""
    if write_mode not in ('w', 'a'):
        raise ValueError(f"write_mode 只能是 'w' 或 'a'，收到: {write_mode!r}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"🚀 [Parser] 开始解析目录: {input_dir}")
    parsed_docs = 0
    with open(output_file, write_mode, encoding='utf-8') as f_out:
        for file in input_dir.glob("**/*"):
            if file.is_file() and file.suffix in [".txt", ".md"]:
                parsed_doc = parse_confluence_file(file, source)
                # 逐行写入 JSONL
                f_out.write(parsed_doc.model_dump_json() + "\n")
                parsed_docs+=1
    logger.info(f"✅ [Parser] 解析完成！成功保存 {parsed_docs} 份文档至 {output_file}")


def main() -> None:
    """CLI 入口"""
    # 不加 basicConfig,logger.info 默认不显示(默认级别是 WARNING)。
    # 本地想看得细:level 改成 logging.DEBUG。FileHandler/JSON 等按需再加。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    source = "confluence"
    output_file = DATA_DIR / "processed" / f"parsed_{source}.jsonl"
    run_parser(DATA_DIR / "EnterpriseRAG-Bench" / "confluence", output_file, source)
    run_parser(DATA_DIR / "EnterpriseRAG-Bench" / "confluence 2", output_file, source, 'a')


if __name__ == "__main__":
    main()
