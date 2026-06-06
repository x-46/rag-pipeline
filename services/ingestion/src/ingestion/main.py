from rag_core.config import settings


def main() -> None:
	print("Ingestion using collection:", settings.collection_name)


if __name__ == "__main__":
	main()
