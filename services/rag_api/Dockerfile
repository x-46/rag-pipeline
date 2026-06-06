FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	UV_COMPILE_BYTECODE=1 \
	UV_LINK_MODE=copy

WORKDIR /app

# Copy lockfile and dependency metadata first for better layer caching.
COPY pyproject.toml uv.lock ./
COPY packages/rag_core/README.md packages/rag_core/README.md
COPY packages/rag_core/pyproject.toml packages/rag_core/pyproject.toml
COPY services/rag_api/README.md services/rag_api/README.md
COPY services/rag_api/pyproject.toml services/rag_api/pyproject.toml

# Install only third-party dependencies in the first layer (no dev dependencies).
RUN uv sync --frozen --no-dev --package rag-api --no-install-workspace

# Copy source code after dependency install to speed up rebuilds.
COPY packages/rag_core/src packages/rag_core/src
COPY services/rag_api/src services/rag_api/src

# Re-sync to install local workspace packages from copied sources.
RUN uv sync --frozen --no-dev --package rag-api

CMD ["uv", "run", "--frozen", "rag-api"]
