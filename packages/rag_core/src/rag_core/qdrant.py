from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from rag_core.config import settings


def _create_dense_embeddings() -> OpenAIEmbeddings:
	return OpenAIEmbeddings(
		model=settings.embedding_model_name,
		base_url=settings.embedding_model_base_url,
		api_key=settings.embedding_model_api_key,
		check_embedding_ctx_length=False,
	)


def _create_sparse_embeddings() -> FastEmbedSparse:
	# Qdrant/bm25 is a FastEmbed BM25 sparse model.
	return FastEmbedSparse(model_name="Qdrant/bm25")


def recreate_collection_with_documents(chunks: list[Document]) -> QdrantVectorStore:
	"""Delete an existing collection (if present) and index documents in hybrid mode."""
	client = QdrantClient(url=settings.qdrant_url)

	try:
		if client.collection_exists(settings.collection_name):
			client.delete_collection(settings.collection_name)
	except Exception:
		# Fallback for client/version differences: best-effort delete.
		try:
			client.delete_collection(settings.collection_name)
		except Exception:
			pass

	dense_embeddings = _create_dense_embeddings()
	sparse_embeddings = _create_sparse_embeddings()

	return QdrantVectorStore.from_documents(
		documents=chunks,
		embedding=dense_embeddings,
		sparse_embedding=sparse_embeddings,
		url=settings.qdrant_url,
		collection_name=settings.collection_name,
		retrieval_mode=RetrievalMode.HYBRID,
	)


def get_hybrid_vectorstore() -> QdrantVectorStore:
	"""Load an existing collection in hybrid retrieval mode (dense + sparse BM25)."""
	dense_embeddings = _create_dense_embeddings()
	sparse_embeddings = _create_sparse_embeddings()

	return QdrantVectorStore.from_existing_collection(
		embedding=dense_embeddings,
		sparse_embedding=sparse_embeddings,
		url=settings.qdrant_url,
		collection_name=settings.collection_name,
		retrieval_mode=RetrievalMode.HYBRID,
	)
