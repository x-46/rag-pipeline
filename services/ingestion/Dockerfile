FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	UV_COMPILE_BYTECODE=1 \
	UV_LINK_MODE=copy

WORKDIR /app

# Runtime libs required by OpenCV/unstructured PDF processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
	libglib2.0-0 \
	libgl1 \
	libsm6 \
	libxext6 \
	libxrender1 \
	libxcb1 \
	tesseract-ocr \
	poppler-utils \
	&& rm -rf /var/lib/apt/lists/*

# Copy lockfile and dependency metadata first for better layer caching.
COPY pyproject.toml uv.lock ./
COPY packages/rag_core/README.md packages/rag_core/README.md
COPY packages/rag_core/pyproject.toml packages/rag_core/pyproject.toml
COPY services/ingestion/README.md services/ingestion/README.md
COPY services/ingestion/pyproject.toml services/ingestion/pyproject.toml

# Install only third-party dependencies in the first layer (no dev dependencies).
RUN uv sync --frozen --no-dev --package ingestion --no-install-workspace

# Copy source code after dependency install to speed up rebuilds.
COPY packages/rag_core/src packages/rag_core/src
COPY services/ingestion/src services/ingestion/src

# Re-sync to install local workspace packages from copied sources.
RUN uv sync --frozen --no-dev --package ingestion

CMD ["uv", "run", "--frozen", "ingestion"]
