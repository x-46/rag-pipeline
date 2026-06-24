# Dev Reference

## Install and update

```bash
uv sync --all-packages                    # initial install
uv lock && uv sync --all-packages         # after changing any pyproject.toml
```

## Run locally

```bash
uv run rag-api        # API on :8000, hot reload enabled
uv run ingestion      # one-shot ingestion
```

## Test ingestion (dry run)

No Qdrant or MongoDB required.

```bash
uv run ingestion-test --markdown /path/to/docs
uv run ingestion-test --pdf /path/to/pdfs
uv run ingestion-test --code /path/to/src --ext py,ts
```

Prints a preview of each section and chunk.

## Add a package

```bash
uv add <package> --package rag-core       # shared core
uv add <package> --package rag-api        # API only
uv add <package> --package ingestion      # ingestion service only
```

After any `pyproject.toml` change run `uv lock` to update the lockfile before building Docker images.

## rag-core dependency layout

`rag-core` dependencies are split into a base layer and an optional `ingestion` extra to keep the API image lean. The base layer contains everything the API needs: LLM, retrieval, reranker, Qdrant, MongoDB. The `ingestion` extra adds the document-loading stack: docling, unstructured, tokenizer, text splitters.

The ingestion service declares `rag-core[ingestion]` as its dependency and gets both layers. The API declares `rag-core` and only gets the base. The document-loading packages are never installed in the API image.
