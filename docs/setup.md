# Setup

## What you need

- Python 3.13+, [uv](https://docs.astral.sh/uv/), Docker + Compose
- An OpenAI-compatible endpoint for the LLM and one for embeddings

## Install

```bash
uv sync --all-packages
```

## Configuration

Create a `.env` file at the repo root. All services pick it up automatically.

```bash
# LLM
LLM_MODEL_BASE_URL=http://localhost:11434/v1
LLM_MODEL_NAME=openai-gpt-oss-120b
LLM_MODEL_API_KEY=placeholder

# Embedding model
EMBEDDING_MODEL_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL_NAME=nomic-embed-text
EMBEDDING_MODEL_API_KEY=placeholder

# Databases - these defaults match docker compose
QDRANT_URL=http://localhost:6333
MONGO_URI=mongodb://localhost:27017
MONGO_DB=rag
```

## Start the databases

```bash
docker compose up -d qdrant mongodb
```

## Run ingestion

Set the paths to your data. Any path you leave empty is skipped.

```bash
INGEST_PDF_BASE_PATH=/path/to/pdfs \
INGEST_MARKDOWN_BASE_PATH=/path/to/markdown \
INGEST_CODE_BASE_PATH=/path/to/source \
INGEST_CODE_EXTENSIONS=".py" \
uv run ingestion
```

**Ingestion environment variables:**

| Variable | Default | Description |
|---|---|---|
| `INGEST_LOADER` | `unstructured` | Loader backend: `unstructured` or `docling` |
| `INGEST_PDF_BASE_PATH` | - | Root directory scanned for `**/*.pdf` |
| `INGEST_MARKDOWN_BASE_PATH` | - | Root directory scanned for `**/*.md` |
| `INGEST_CODE_BASE_PATH` | - | Root directory scanned for code files |
| `INGEST_CODE_EXTENSIONS` | - | Comma-separated file extensions, e.g. `.py,.ts` |

Ingestion is destructive - it drops and recreates both the Qdrant collection and the MongoDB collection on every run.

## Start the API

```bash
uv run rag-api
```

API available at `http://localhost:8000`. Quick test:

```bash
uv run test-client
```

## Docker

`rag_api` and `ingestion` each have a `Dockerfile`. The build context must be the repo root.

**Build:**

```bash
docker build -f services/rag_api/Dockerfile -t rag-api:dev .
docker build -f services/ingestion/Dockerfile -t ingestion:dev .
```

**Start databases first:**

```bash
docker compose up -d qdrant mongodb
```

**Run the API:**

```bash
docker run --rm \
  --env-file .env \
  --network rag-pipeline-net \
  -e MONGO_URI=mongodb://mongodb:27017 \
  -e QDRANT_URL=http://qdrant:6333 \
  -p 8000:8000 \
  rag-api:dev
```

**Run ingestion** (mount your data as read-only volumes):

```bash
docker run --rm \
  --env-file .env \
  --network rag-pipeline-net \
  -v "/path/to/pdfs:/data/pdfs:ro" \
  -v "/path/to/markdown:/data/md:ro" \
  -v "/path/to/source:/data/code:ro" \
  -e MONGO_URI=mongodb://mongodb:27017 \
  -e QDRANT_URL=http://qdrant:6333 \
  -e INGEST_PDF_BASE_PATH=/data/pdfs \
  -e INGEST_MARKDOWN_BASE_PATH=/data/md \
  -e INGEST_CODE_BASE_PATH=/data/code \
  -e INGEST_CODE_EXTENSIONS=".py" \
  ingestion:dev
```

`--env-file .env` loads the model variables. Database URLs are overridden with `-e` because inside the container the Compose service names (`mongodb`, `qdrant`) are used as hostnames instead of `localhost`.
