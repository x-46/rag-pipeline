# rag-pipeline

A RAG pipeline that ingests PDFs, Markdown files, and source code into a hybrid vector store and answers questions via multi-query retrieval, cross-encoder reranking, and parent-child context expansion.

Built with Qdrant, MongoDB, LangChain, Unstructured / Docling, FastEmbed BM25, and a HuggingFace cross-encoder. Models (LLM + embeddings) connect via any OpenAI-compatible API endpoint.

## Docs

- [Setup](docs/setup.md) - installation, config, all arguments, Docker
- [Architecture](docs/architecture.md) - repo structure, what each part does
- [Ingestion](docs/ingestion.md) - how ingestion works, how to extend it
- [RAG pipeline](docs/rag.md) - the retrieval algorithm and design decisions
- [Dev reference](docs/dev.md) - day-to-day development commands
- [Development Notebook](rag.ipynb) - the notebook used to develop and test the pipeline