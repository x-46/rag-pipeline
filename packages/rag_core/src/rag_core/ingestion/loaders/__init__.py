from __future__ import annotations

from rag_core.ingestion.loaders.base import DocumentLoader


def get_loader(
    backend: str,
    pdf_base_path: str = "",
    markdown_base_path: str = "",
    code_base_path: str = "",
    code_extensions: list[str] | None = None,
) -> DocumentLoader:
    """Return a loader for the requested backend.

    Args:
        backend:            ``"unstructured"`` or ``"docling"``.
        pdf_base_path:      Root directory scanned for ``*.pdf`` files.
        markdown_base_path: Root directory scanned for ``*.md`` files.
        code_base_path:     Root directory scanned for code files.
        code_extensions:    Extensions (without leading dot) to treat as code,
                            e.g. ``["py", "ts"]``.

    Raises:
        ValueError: If ``backend`` is not a recognised value.
    """
    exts = code_extensions or []

    if backend == "unstructured":
        from rag_core.ingestion.loaders.unstructured import UnstructuredLoader  # noqa: PLC0415

        return UnstructuredLoader(
            pdf_base_path=pdf_base_path,
            markdown_base_path=markdown_base_path,
            code_base_path=code_base_path,
            code_extensions=exts,
        )

    if backend == "docling":
        from rag_core.ingestion.loaders.docling import DoclingLoader  # noqa: PLC0415

        return DoclingLoader(
            pdf_base_path=pdf_base_path,
            markdown_base_path=markdown_base_path,
            code_base_path=code_base_path,
            code_extensions=exts,
        )

    raise ValueError(
        f"Unknown loader backend: {backend!r}. Supported values: 'unstructured', 'docling'."
    )