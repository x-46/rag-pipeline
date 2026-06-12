from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.documents import Document

def get_reranker(top_n: int = 10) -> CrossEncoderReranker:
    """Return a cross-encoder reranker that keeps the top-n most relevant documents."""
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return compressor


def reciprocal_rank_fusion(
    doc_lists: list[list[Document]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[tuple[Document, float]]:
    """
    RRF: promotes documents that appear at high ranks across multiple result lists.
    Each document's score = Σ  w_i / (rank_i + k).
    weights: per-list multipliers (default 1.0 for every list).
    k:       smoothing constant (classic value: 60).
    Returns list of (Document, rrf_score) sorted descending by score.
    """
    if weights is None:
        weights = [1.0] * len(doc_lists)

    scores: dict[tuple, float] = {}
    doc_map: dict[tuple, Document] = {}

    for w, doc_list in zip(weights, doc_lists):
        for rank, doc in enumerate(doc_list):
            key = (doc.metadata.get("source", ""), doc.page_content[:300])
            if key not in doc_map:
                doc_map[key] = doc
                scores[key] = 0.0
            scores[key] += w / (rank + k)

    return [(doc_map[key], scores[key]) for key in sorted(scores, key=scores.__getitem__, reverse=True)]
