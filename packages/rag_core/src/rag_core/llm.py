from langchain_openai import ChatOpenAI
from rag_core.config import settings


def get_llm() -> ChatOpenAI:
    """Create an LLM instance with no callbacks - pass per-request callbacks via config={"callbacks": [...]}."""
    return ChatOpenAI(
        model=settings.llm_model_name,
        base_url=settings.llm_model_base_url,
        api_key=settings.llm_model_api_key,
    )
