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

--- STEP 1: Detect query type ---
First decide: is this question about CODE (e.g. functions, classes, algorithms, libraries, APIs, implementations, configurations, error messages, syntax, data structures, pipelines, scripts)?

If YES -> apply the CODE retrieval strategies below (sections A-H).
If NO  -> apply the DOCUMENT retrieval strategies (sections 1-8).

--- CODE retrieval strategies (use only when query is code-related) ---

A. Signature query
   Formulate a query that matches a typical function or class signature, e.g. "def train_agent(env, policy, ...)" or "class MarketSimulator".

B. Docstring / comment query
   Phrase the query as a natural-language description that would appear in a docstring or inline comment explaining the code.

C. Import / dependency query
   Search for the relevant module, package, or import statement (e.g. "import stable_baselines3", "from rl_env import").

D. Error / exception query
   If the question involves a bug or error, search for the exact error message, exception type, or traceback fragment.

E. Variable / parameter query
   Search for key variable names, parameter names, or configuration keys that are central to the functionality.

F. Pattern / idiom query
   Formulate a query around a common code pattern or idiom that implements the requested behavior (e.g. "callback handler", "gym.Env step reset", "reward shaping").

G. Test / example query
   Search for unit tests, usage examples, or notebooks that demonstrate the functionality in question.

H. Broad keyword query
   Short keyword combination of the programming language, framework, and task (e.g. "Python reinforcement learning environment step function").

--- DOCUMENT retrieval strategies (use only when query is NOT code-related) ---

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
   If useful, generate queries that retrieve risks, limitations, or failure modes related to the question.

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
- Return only a valid JSON list of strings. The first element must be "code" or "document" to indicate which strategy set was applied.
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
    queries, query_type = parse_queries(raw_queries, question, max_queries)
    metrics.query_type = query_type
    return queries


_QUERY_TYPE_TAGS = {"code", "document"}


def parse_queries(raw: str, original_question: str, max_queries: int) -> tuple[list[str], str]:
    """Parse the raw JSON query list, strip the leading type tag, prepend the original question, deduplicate, and cap at max_queries.

    Returns a tuple of (queries, query_type) where query_type is "code", "document", or "unknown".
    """
    try:
        queries = json.loads(raw)
        if not isinstance(queries, list):
            queries = []
    except Exception:
        queries = []

    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

    query_type = "unknown"
    if queries and queries[0].lower() in _QUERY_TYPE_TAGS:
        query_type = queries[0].lower()
        queries = queries[1:]

    queries = [original_question] + queries

    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)

    return unique[:max_queries], query_type
