import json
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from rag_core.metrics import MetricsCallbackHandler, RequestMetrics

query_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert query planner and query rewriter for a RAG system.

The user may ask questions whose answer is not explicitly stated in one passage. Your task is to generate diverse search queries that retrieve the evidence needed to answer the user's question accurately.

Use the user's original question and the optional context below:

Additional context:
{pre_context}

Generate several clearly different search queries that cover the following retrieval strategies:

1. Direct query
   Closely rephrase the original question while preserving its exact intent.

2. Keyword query
   Create a short keyword-based query using the most important terms.

3. Concept queries
   Search for key concepts, related ideas, synonyms, or implied themes from the question.

4. Mechanism queries
   Search for causes, reasons, effects, consequences, mechanisms, or relationships relevant to the question.

5. Technical or domain-specific queries
   Use terminology from the relevant domain, especially when applicable: electricity market simulation, agent-based modeling, reinforcement learning, market design, policy support, validation, interpretability, transparency, or explainability.

6. Alternative-perspective queries
   Reformulate the question from a different but still relevant perspective, such as evaluation, policy relevance, model behavior, stakeholder use, or decision support.

7. Negative or risk queries
   If useful, generate queries that retrieve risks, limitations, or failure modes related to the question, such as market power, price manipulation, black-box behavior, invalid assumptions, lack of validation, biased outcomes, or poor interpretability.

8. Abbreviation/entity queries
   If useful, include abbreviations, key entities, technical terms, or domain-specific shorthand that may appear in documents.

Rules:
- Preserve the original intent exactly.
- Do not invent an answer.
- Do not invent facts, names, numbers, entities, or assumptions.
- Do not create irrelevant or overly broad queries.
- Use the additional context only to improve retrieval, not to change the question.
- Queries may use related concepts, synonyms, and domain terminology.
- Make sure the queries are clearly different from each other.
- Return only a valid JSON list of strings.
- Do not include explanations, Markdown, comments, or text outside the JSON list.
"""
    ),
    (
        "human",
        """Question:
{question}

Number of variants:
{num_queries}"""
    )
])


def query_rewriting(
    question: str,
    llm: ChatOpenAI,
    handler: MetricsCallbackHandler,
    metrics: RequestMetrics,
    retriever,
    reranker,
    max_queries: int = 5,
) -> list[str]:
    """Generate diverse search queries from the original question using the LLM and initial retrieval context."""
    invoke_config = {"callbacks": [handler]}

    t0 = time.time()
    basic_raw = retriever.invoke(question)
    metrics.query_rewrite.db_ms += (time.time() - t0) * 1000

    t0 = time.time()
    basic_info = reranker.compress_documents(basic_raw, question)
    metrics.query_rewrite.local_ms += (time.time() - t0) * 1000

    query_rewriter = query_rewrite_prompt | llm | StrOutputParser()
    raw_queries = query_rewriter.invoke(
        {"question": question, "num_queries": max_queries, "pre_context": basic_info},
        config=invoke_config,
    )
    return parse_queries(raw_queries, question, max_queries)


def parse_queries(raw: str, original_question: str, max_queries: int) -> list[str]:
    """Parse the raw JSON query list, prepend the original question, deduplicate, and cap at max_queries."""
    try:
        queries = json.loads(raw)
        if not isinstance(queries, list):
            queries = []
    except Exception:
        queries = []

    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    queries = [original_question] + queries

    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)

    return unique[:max_queries]
