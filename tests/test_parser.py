"""pytest 不是另一个东西,它就是"帮你跑这些检验的工具"。对错由你的 assert 决定。

单元测试:
1. 测"纯函数":parse_confluence_file 输入 Path、输出 ParsedDocument
2. tmp_path fixture:测"读写文件"时,用 pytest 内置的临时目录。
   每个测试独享一个临时目录,用完自动清理 —— 不会污染你真实的 data/ 目录。
3. 断言"输出文件内容":读回 run_parser 写的 JSONL,逐行解析断言。

运行(在项目根目录):
    pytest tests/test_parser.py -v        # 只跑这个文件
    pytest -v                             # 跑全部测试
"""
import json

from src.ingestion.parser import parse_confluence_file, run_parser


def test_parse_confluence_file_提取doc_id_title_source(tmp_path):
    """文件名里的 dsid_xxx 和标题必须被正确拆出来(这决定后续对账能否命中)。

    真实文件命名格式: dsid_<文档id>__<标题>.txt
    """
    file_path = tmp_path / "dsid_0dd4dcc98daa4a81896c79c0ca232a30__service-guide.txt"
    file_path.write_text("Overview\n\nSome SOP content", encoding="utf-8")

    doc = parse_confluence_file(file_path, source="confluence")

    assert doc.doc_id == "0dd4dcc98daa4a81896c79c0ca232a30"
    assert doc.title == "service-guide"
    assert doc.source == "confluence"
    assert doc.content == "Overview\n\nSome SOP content"  # 注意 .strip() 已去掉首尾空白


def test_run_parser_每个文件输出一行jsonl(tmp_path):
    """run_parser 是文件系统 IO:用 tmp_path 造输入目录,再断言输出文件。

    关键点:测试里绝不能碰真实的 data/ 目录 —— tmp_path 就是为此存在的。
    """
    input_dir = tmp_path / "docs"
    input_dir.mkdir()

    (input_dir / "dsid_aaa__doc-a.txt").write_text("content A", encoding="utf-8")
    (input_dir / "dsid_bbb__doc-b.txt").write_text("content B", encoding="utf-8")
    (input_dir / "notes.pdf").write_text("not a txt", encoding="utf-8")  # 非 txt/md,应被跳过

    output_file = tmp_path / "out" / "parsed.jsonl"
    run_parser(input_dir, output_file, source="confluence")

    # 读回输出文件,逐行解析
    lines = [json.loads(l) for l in output_file.read_text(encoding="utf-8").splitlines()]

    assert len(lines) == 2               # pdf 被过滤,只剩 2 个 txt
    assert lines[0]["doc_id"] == "aaa"
    assert lines[0]["title"] == "doc-a"
    assert lines[1]["doc_id"] == "bbb"
    assert lines[1]["source"] == "confluence"
