from rag_core.config import settings


def main() -> None:
	print("QDRANT_URL:", settings.qdrant_url)


if __name__ == "__main__":
	main()