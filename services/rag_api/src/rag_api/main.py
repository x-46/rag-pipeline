import time

from rag_core.pipeline import build_pipeline
from rag_core.mongodb import save_request_metrics


def main() -> None:
    pipeline = build_pipeline()

    question = "Which bidding strategies does ASSUME support?"

    result = pipeline.invoke(question)
    metrics = result["metrics"]

    save_request_metrics(metrics, question)

    print("\n── Answer ──────────────────────────────────────────")
    print(result["answer"])

    print("\n── Sources ─────────────────────────────────────────", len(result["context"]))
    unique_sources: set[str] = set()
    for doc in result["context"]:
        source = doc.metadata.get("source", "unknown")
        if source not in unique_sources:
            unique_sources.add(source)
            print(f"[{len(unique_sources)}] {source}")

    print()
    print(metrics.report())


if __name__ == "__main__":
    main()
