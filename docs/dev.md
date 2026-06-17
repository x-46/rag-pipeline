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

## Install a package

```bash
uv pip install <package-name> --package rag_core  # or --package rag-api, etc.
```
