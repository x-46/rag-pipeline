import os

from rag_core.config import settings
from rag_core.loader import build_parent_child_chunks, load_files
from rag_core.mongodb import recreate_mongo_collection_with_parent_elements
from rag_core.qdrant import recreate_collection_with_documents


def main() -> None:
	pdf_base_path = os.getenv("INGEST_PDF_BASE_PATH", "")
	markdown_base_path = os.getenv("INGEST_MARKDOWN_BASE_PATH", "")
	code_base_path = os.getenv("INGEST_CODE_BASE_PATH", "")
	code_extensions_raw = os.getenv("INGEST_CODE_EXTENSIONS", "")
	code_extensions = [e.strip().lstrip(".") for e in code_extensions_raw.split(",") if e.strip()]

	print(f"Collection: {settings.collection_name}")
	print(f"Qdrant URL: {settings.qdrant_url}")
	print(f"Mongo: {settings.mongo_db}")

	docs = load_files(
		pdf_base_path=pdf_base_path,
		markdown_base_path=markdown_base_path,
		code_base_path=code_base_path,
		code_extensions=code_extensions,
	)
	print(f"Loaded {len(docs)} documents")
	if not docs:
		raise ValueError(
			"No documents loaded. Set INGEST_MARKDOWN_BASE_PATH, INGEST_PDF_BASE_PATH and/or INGEST_CODE_BASE_PATH."
		)
	print("Building chunks and parent elements...")
	chunks, parent_elements = build_parent_child_chunks(docs)
	print(f"Chunks: {len(chunks)}")
	print(f"Parent elements: {len(parent_elements)}")

	print("Writing child chunks to Qdrant")
	recreate_collection_with_documents(chunks)

	print("Writing parent elements to MongoDB")
	recreate_mongo_collection_with_parent_elements(parent_elements)
	print("Ingestion complete")


if __name__ == "__main__":
	main()
