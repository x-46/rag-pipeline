import argparse
import textwrap
from rag_core.ingestion import chunk_sections, get_loader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run ingestion: load + chunk without writing to any store."
    )
    parser.add_argument("--markdown", default="", metavar="PATH")
    parser.add_argument("--pdf", default="", metavar="PATH")
    parser.add_argument("--code", default="", metavar="PATH")
    parser.add_argument("--ext", default="", metavar="py,ts", help="Comma-separated code file extensions")
    parser.add_argument("--loader", default="docling", choices=["docling", "unstructured"])
    args = parser.parse_args()

    if not any([args.markdown, args.pdf, args.code]):
        parser.error("Provide at least one of --markdown, --pdf, or --code.")

    code_extensions = [e.strip().lstrip(".") for e in args.ext.split(",") if e.strip()]

    print(f"Loader:   {args.loader}")
    print(f"Markdown: {args.markdown or '(none)'}")
    print(f"PDF:      {args.pdf or '(none)'}")
    print(f"Code:     {args.code or '(none)'}" + (f"  extensions: {code_extensions}" if code_extensions else ""))
    print()

    loader = get_loader(
        backend=args.loader,
        pdf_base_path=args.pdf,
        markdown_base_path=args.markdown,
        code_base_path=args.code,
        code_extensions=code_extensions,
    )

    sections = loader.load()
    print(f"Sections loaded: {len(sections)}")
    if not sections:
        print("No documents found - check your paths.")
        return

    for i, s in enumerate(sections, 1):
        preview = textwrap.shorten(s.text, width=120, placeholder=" ...")
        print(f"  [{i:3}] {s.doc_type:<8}  {s.source}  |  {preview}")
    print()

    print("Chunking ...")
    chunks = chunk_sections(sections)
    print(f"Chunks produced: {len(chunks)}")

    for i, c in enumerate(chunks, 1):
        preview = textwrap.shorten(c.text, width=100, placeholder=" ...")
        print(f"  [{i:3}] parent={c.parent_id}  |  {preview}")

    print()
    print("Dry run complete")
