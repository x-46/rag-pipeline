import time
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_core.llm import get_llm
from rag_core.metrics import MetricsCallbackHandler, RequestMetrics
from rag_core.mongodb import get_mongo_collection
from rag_core.qdrant import get_vectorstore_retriever
from rag_core.queryRewrite import query_rewriting
from rag_core.reranker import get_reranker, reciprocal_rank_fusion


_ANSWER_PROMPT = ChatPromptTemplate.from_template("""You are an expert assistant for the ASSUME energy simulation framework.
Answer the question using ONLY the information provided in the context below.
If the context does not contain enough information to answer, say "I don't have enough information to answer this question."
Do not make up or infer facts beyond what is explicitly stated in the context.

Context:
{context}

Question: {question}

Answer concisely and precisely:""")


class RagPipeline:
    """
    Stateless-per-request pipeline. Heavy resources (LLM, retriever, reranker)
    are created once at startup. Each invoke() creates its own RequestMetrics
    and MetricsCallbackHandler - no shared mutable state between requests.
    """

    def __init__(self) -> None:
        self._llm = get_llm()
        self._retriever = get_vectorstore_retriever()
        self._reranker = get_reranker(top_n=10)

    def invoke(self, question: str) -> dict:
        metrics = RequestMetrics()
        query_rewrite_handler = MetricsCallbackHandler(metrics.query_rewrite)
        answer_handler = MetricsCallbackHandler(metrics.answer)

        t_start = time.time()
        context_docs = self._retrieve(question, metrics, query_rewrite_handler)
        answer = self._answer(question, context_docs, answer_handler)
        metrics.wall_ms = (time.time() - t_start) * 1000

        return {"answer": answer, "context": context_docs, "metrics": metrics}

    # ── Retrieval ────────────────────────────────────────────────────────

    def _retrieve(
        self,
        question: str,
        metrics: RequestMetrics,
        handler: MetricsCallbackHandler,
    ) -> List[Document]:
        queries = query_rewriting(
            question,
            llm=self._llm,
            handler=handler,
            metrics=metrics,
            retriever=self._retriever,
            reranker=self._reranker,
        )
        print(queries)
        scored_chunks = self._multi_query_retrieve(queries, metrics)
        return self._fetch_parents(scored_chunks, metrics)

    def _multi_query_retrieve(
        self,
        queries: List[str],
        metrics: RequestMetrics,
    ) -> List[tuple[Document, float]]:
        all_doc_lists: list[list] = []

        for q in queries:
            t0 = time.time()
            docs_raw = self._retriever.invoke(q)
            metrics.retrieval.db_ms += (time.time() - t0) * 1000

            t0 = time.time()
            docs = self._reranker.compress_documents(docs_raw, q)
            metrics.retrieval.local_ms += (time.time() - t0) * 1000

            all_doc_lists.append(docs)

        t0 = time.time()
        result = reciprocal_rank_fusion(all_doc_lists, k=40)
        metrics.retrieval.local_ms += (time.time() - t0) * 1000

        return result

    def _fetch_parents(
        self,
        scored_docs: List[tuple[Document, float]],
        metrics: RequestMetrics,
    ) -> List[Document]:
        parent_id_scores: dict[str, float] = {}
        for doc, rrf_score in scored_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id:
                parent_id_scores[parent_id] = parent_id_scores.get(parent_id, 0.0) + rrf_score

        top_parent_ids = sorted(parent_id_scores, key=parent_id_scores.__getitem__, reverse=True)[:10]
        print(f"Top parent IDs: {top_parent_ids}")

        mongo_collection = get_mongo_collection()
        parent_docs: List[Document] = []

        for pid in top_parent_ids:
            t0 = time.time()
            parent = mongo_collection.find_one(
                {"element_id": pid},
                {"page_content": 1, "element_id": 1, "source": 1, "_id": 0},
            )
            metrics.parent_fetch.db_ms += (time.time() - t0) * 1000

            if parent:
                parent_docs.append(
                    Document(
                        page_content=parent["page_content"],
                        metadata={
                            "element_id": parent["element_id"],
                            "retrieval": "parent",
                            "source": parent.get("source", "unknown"),
                        },
                    )
                )

        return parent_docs

    # ── Answer ───────────────────────────────────────────────────────────

    def _answer(
        self,
        question: str,
        context: List[Document],
        handler: MetricsCallbackHandler,
    ) -> str:
        return (_ANSWER_PROMPT | self._llm | StrOutputParser()).invoke(
            {"question": question, "context": context},
            config={"callbacks": [handler]},
        )


def build_pipeline() -> RagPipeline:
    return RagPipeline()
