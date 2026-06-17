# Architecture

This is a monorepo with one shared library and three services.

```
packages/
  rag_core/       all logic: ingestion, retrieval pipeline, storage, config

services/
  rag_api/        FastAPI service - answers questions        (Docker container)
  ingestion/      one-shot CLI - populates the stores        (Docker container)
  test_client/    manual test client                         (local only)
```

`rag_api` and `ingestion` both depend on `rag_core`. Neither depends on the other. `rag_core` has no dependency on the services.

## rag_core

The shared library. Everything lives here so it can be used by both services without duplication.

- `ingestion/` - document model (`RagDocument`, `ParentSection`), loader backends, chunker
- `pipeline.py` - orchestrates the full query -> retrieval -> answer flow
- `query_rewrite.py` - LLM-based multi-query expansion
- `reranker.py` - cross-encoder reranking and RRF fusion
- `qdrant.py` - Qdrant read/write (vector store)
- `mongodb.py` - MongoDB read/write (parent sections, metrics)
- `config.py` - all settings via pydantic-settings, loaded from `.env`

## rag_api

FastAPI app. Exposes `POST /v1/chat/completions` and `GET /v1/models` in the OpenAI format. At startup it initializes the `RagPipeline` once and keeps it in memory. Each request is stateless - the pipeline holds no request-specific state. The response includes a `sources` field with the file paths of the retrieved sections.

## ingestion

One-shot script. Reads documents from configured directories, chunks them, groups them into parent sections, then writes chunks to Qdrant and parent sections to MongoDB. Drops and recreates both collections on every run. See [ingestion.md](ingestion.md) for how it works and how to extend it.

## Data stores

The MongoDB stores the full parent sections. Qdrant stores child chunks with dense and BM25 sparse embeddings, these chunks link back to via `parent_id` to their parent sections. Both collections share the name `rag_collection` by default. A separate `request_metrics` collection in MongoDB stores per-request metrics.

All config comes from a `.env` file. See [setup.md](setup.md) for the full reference.
