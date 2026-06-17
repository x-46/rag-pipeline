import os

from rag_core.config import settings
from rag_core.ingestion import chunk_sections, get_loader
from rag_core.mongodb import recreate_mongo_collection_with_parent_elements
from rag_core.qdrant import recreate_collection_with_rag_chunks


def main() -> None:
    """Load documents, build chunks, and write them to Qdrant and MongoDB.

    Environment variables
    ---------------------
    INGEST_LOADER            Loader backend: ``"unstructured"`` (default) or ``"docling"``.
    INGEST_PDF_BASE_PATH     Root directory scanned for PDF files.
    INGEST_MARKDOWN_BASE_PATH Root directory scanned for Markdown files.
    INGEST_CODE_BASE_PATH    Root directory scanned for code files.
    INGEST_CODE_EXTENSIONS   Comma-separated file extensions (no dot), e.g. ``"py,ts"``.
    """
    loader_backend = os.getenv("INGEST_LOADER", "unstructured")
    pdf_base_path = os.getenv("INGEST_PDF_BASE_PATH", "")
    markdown_base_path = os.getenv("INGEST_MARKDOWN_BASE_PATH", "")
    code_base_path = os.getenv("INGEST_CODE_BASE_PATH", "")
    code_extensions_raw = os.getenv("INGEST_CODE_EXTENSIONS", "")
    code_extensions = [e.strip().lstrip(".") for e in code_extensions_raw.split(",") if e.strip()]

    print(f"Collection:     {settings.collection_name}")
    print(f"Qdrant URL:     {settings.qdrant_url}")
    print(f"Mongo:          {settings.mongo_db}")
    print(f"Loader backend: {loader_backend}")

    loader = get_loader(
        backend=loader_backend,
        pdf_base_path=pdf_base_path,
        markdown_base_path=markdown_base_path,
        code_base_path=code_base_path,
        code_extensions=code_extensions,
    )

    sections = loader.load()
    print(f"Loaded {len(sections)} sections")
    if not sections:
        raise ValueError(
            "No documents loaded. Set INGEST_MARKDOWN_BASE_PATH, INGEST_PDF_BASE_PATH"
            " and/or INGEST_CODE_BASE_PATH."
        )

    print("Chunking sections ...")
    chunks = chunk_sections(sections)
    print(f"Chunks: {len(chunks)}")

    print("Writing child chunks to Qdrant ...")
    recreate_collection_with_rag_chunks(chunks)

    print("Writing parent sections to MongoDB ...")
    recreate_mongo_collection_with_parent_elements(
        [section.to_mongo_dict() for section in sections]
    )

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
