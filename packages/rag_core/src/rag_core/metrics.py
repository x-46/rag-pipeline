import time
from dataclasses import dataclass, field

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


@dataclass
class PhaseMetrics:
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_ms: float = 0.0
    db_ms: float = 0.0
    local_ms: float = 0.0


@dataclass
class RequestMetrics:
    query_rewrite: PhaseMetrics = field(default_factory=PhaseMetrics)
    retrieval: PhaseMetrics = field(default_factory=PhaseMetrics)
    parent_fetch: PhaseMetrics = field(default_factory=PhaseMetrics)
    answer: PhaseMetrics = field(default_factory=PhaseMetrics)
    wall_ms: float = 0.0

    def _total(self) -> PhaseMetrics:
        return PhaseMetrics(
            llm_input_tokens=self.query_rewrite.llm_input_tokens + self.answer.llm_input_tokens,
            llm_output_tokens=self.query_rewrite.llm_output_tokens + self.answer.llm_output_tokens,
            llm_ms=self.query_rewrite.llm_ms + self.answer.llm_ms,
            db_ms=self.query_rewrite.db_ms + self.retrieval.db_ms + self.parent_fetch.db_ms,
            local_ms=self.query_rewrite.local_ms + self.retrieval.local_ms,
        )

    def report(self) -> str:
        def fmt_i(v: int)   -> str: return f"{v:>9,}"    if v else f"{'–':>9}"
        def fmt_f(v: float) -> str: return f"{v:>9,.0f}" if v else f"{'–':>9}"

        def row(label: str, p: PhaseMetrics) -> str:
            return (
                f"  {label:<24}"
                f"{fmt_i(p.llm_input_tokens)}"
                f"{fmt_i(p.llm_output_tokens)}"
                f"{fmt_f(p.llm_ms)}"
                f"{fmt_f(p.db_ms)}"
                f"{fmt_f(p.local_ms)}"
            )

        header = (
            f"  {'Phase':<24}"
            f"{'In Tok':>9}"
            f"{'Out Tok':>9}"
            f"{'LLM ms':>9}"
            f"{'DB ms':>9}"
            f"{'Local ms':>9}"
        )

        lines = [
            "═" * 72,
            "  REQUEST METRICS",
            "═" * 72,
            header,
            "─" * 72,
            row("Query Rewrite",        self.query_rewrite),
            row("Retrieval (multi-q)",   self.retrieval),
            row("Parent Fetch (Mongo)",  self.parent_fetch),
            row("Answer Generation",     self.answer),
            "─" * 72,
            row("TOTAL",                 self._total()),
            "═" * 72,
            f"  Wall time: {self.wall_ms:,.0f} ms",
            "═" * 72,
        ]
        return "\n".join(lines)


class MetricsCallbackHandler(BaseCallbackHandler):
    """
    Per-phase LangChain callback - bound directly to one PhaseMetrics at construction.
    Create one instance per phase and pass via config={"callbacks": [handler]}.
    """

    def __init__(self, phase: PhaseMetrics) -> None:
        self._phase = phase
        self._t0: float = 0.0

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self._t0 = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self._phase.llm_ms += (time.time() - self._t0) * 1000
        usage = (response.llm_output or {}).get("token_usage", {})
        self._phase.llm_input_tokens += usage.get("prompt_tokens", 0)
        self._phase.llm_output_tokens += usage.get("completion_tokens", 0)
