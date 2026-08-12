# 🏢 Enterprise Confluence Policy Assistant (Production-Ready RAG)
tag: question answering, text retrieval

A lightweight, production-grade RAG service designed to answer enterprise policy questions from internal Confluence docs with zero hallucinations and exact citations.

EnterpriseRAG-Bench provides the first publicly accessible dataset focused entirely on company internal data.

Data Sources
Source	Documents	Description
Jira	6,120	Support tickets, internal and customer facing
Confluence	5,189	Wikis, runbooks, and structured documentation



![Architecture](./docs/architecture.png)

┌─── [Confluence Docs] ───► Confluence Collection (SOP / Policy)
[EnterpriseRAG-Bench]─┤
                     └─── [Jira Tickets] ──────► Jira Collection (Issues / Troubleshooting)
                                                       │
                                                       ▼
[User Query] ───► [FastAPI Endpoint] ───► [Hybrid Search + Reranker] ───► [Structured LLM Output]



## ✨ Key Features & Engineering Decisions
- **Parent-Child Chunking**: Indexed on 256-token child chunks for dense vector precision, retrieving 1024-token parent contexts for full LLM understanding.
- **Pydantic Guardrails**: Enforces structured JSON outputs and explicit refusal mechanisms (`is_policy_found=False`) when context is insufficient.
- **Production Boilerplate**: Built with FastAPI, Qdrant (Dockerized), and modular API layers for seamless scale-up to Jira/Slack datasets.

## Technical intro


## Next Steps/Features

Confluence 对应“企业政策/SOP/内部知识库”：Confluence 包含大量带有层级目录、表格、操作步骤（How-to）的标准化 Wiki 文档，是典型的企业级 Standard Operating Procedures (SOP) 检索场景。
2. Confluence 切分策略（Chunking）注意 Markdown/HTML 标签
Confluence 数据通常导出为带有格式的文本或 HTML/Markdown：
不要直接用简单的 RecursiveCharacterTextSplitter。
建议按 Heading（标题级次 #, ##） 切分，或者保留其 Markdown 中的表格结构。否则，一旦 Sop 中的操作步骤或规则表格被切断，LLM 就容易产生幻觉。

Jira 对应“智能客服/售后/排障”
1. 善用 Jira 的 Metadata 进行 Filtering（数据检索加分项）
Jira 数据非常适合练习 Metadata Filtering + Vector Search



## 🚀 Quick Start
1. **Start Qdrant Vector DB**
   ```bash
   docker compose up -d