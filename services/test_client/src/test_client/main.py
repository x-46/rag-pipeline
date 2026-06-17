import os

from openai import OpenAI

RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/v1")
RAG_API_KEY = os.getenv("RAG_API_KEY", "none")

QUESTIONS = [
    #"Which bidding strategies does ASSUME support?",
    #"What are the key features of ASSUME?",
    #"How do I get started with ASSUME?",
    "How does the InfrastructureInterface work and how to use it?",
    #"How does the scenario loader work in ASSUME?"
]


def main() -> None:
    client = OpenAI(base_url=RAG_API_URL, api_key=RAG_API_KEY)

    for question in QUESTIONS:
        print(f"\n── Question ─────────────────────────────────────────")
        print(question)

        response = client.chat.completions.create(
            model="rag-model",
            messages=[{"role": "user", "content": question}],
        )

        answer = response.choices[0].message.content
        print(f"\n── Answer ───────────────────────────────────────────")
        print(answer)

        sources = getattr(response, "sources", None)
        if sources:
            print(f"\n── Sources ──────────────────────────────────────────")
            for i, src in enumerate(sources, 1):
                print(f"[{i}] {src['source']}")

        print()


if __name__ == "__main__":
    main()
