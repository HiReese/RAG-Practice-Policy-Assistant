# 🏢 Enterprise Confluence Policy Assistant (Production-Ready RAG)
tag: question answering, text retrieval

A lightweight, production-grade RAG service designed to answer enterprise policy questions from internal Confluence docs with zero hallucinations and exact citations.

## Data source
EnterpriseRAG-Bench provides the first publicly accessible dataset focused entirely on company internal data. Two types of data sources used / to be used in this project include:
- Confluence	5,189	Wikis, runbooks, and structured documentation
- Jira	6,120	Support tickets, internal and customer facing

## Architecture
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
- Evaluation
- Clear Tracking and Citation


## Next Steps/Features
- TODO: add jira data for customer support


## 🚀 Quick Start
1. **Start Qdrant Vector DB**
   ```bash
   docker compose up -d