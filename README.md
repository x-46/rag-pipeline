# rag-pipeline

## Dev Instructions
### Lokales Setup

1. Alle Pakete im Workspace installieren:

```bash
uv sync --all-packages
```

2. Optional: Dependencies nach Aenderungen im Lockfile aktualisieren:

```bash
uv lock
uv sync --all-packages
```

### Services lokal starten

API:

```bash
uv run rag-api
```

Ingestion:

```bash
uv run ingestion
```

### Docker: Images bauen

API:

```bash
docker build -f services/rag_api/Dockerfile -t rag-api:dev .
```

Ingestion:

```bash
docker build -f services/ingestion/Dockerfile -t ingestion:dev .
```

### Docker: Container starten

Beide Services benoetigen aktuell diese Environment-Variablen aus `rag_core.config.Settings`:

- EMBEDDING_MODEL_BASE_URL
- EMBEDDING_MODEL_NAME
- EMBEDDING_MODEL_API_KEY
- LLM_MODEL_BASE_URL
- LLM_MODEL_NAME
- LLM_MODEL_API_KEY

Beispiel (API):

```bash
docker run --rm \
	-e EMBEDDING_MODEL_BASE_URL=http://example.com \
	-e EMBEDDING_MODEL_NAME=test-embed \
	-e EMBEDDING_MODEL_API_KEY=dummy \
	-e LLM_MODEL_BASE_URL=http://example.com \
	-e LLM_MODEL_NAME=test-llm \
	-e LLM_MODEL_API_KEY=dummy \
	rag-api:dev
```