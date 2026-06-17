from __future__ import annotations

import hashlib
from pathlib import Path

from rag_core.ingestion.document import ParentSection

_DOCLING_LABEL_MAP: dict[str, str] = {
    "title": "title",
    "section_header": "title",
    "text": "text",
    "paragraph": "text",
    "narrative_text": "text",
    "table": "table",
    "list_item": "list_item",
    "figure": "image",
    "picture": "image",
    "figure_caption": "figure_caption",
    "code": "code",
    "formula": "formula",
    "page_header": "header",
    "page_footer": "footer",
    "footnote": "text",
    "caption": "text",
    "checkbox_selected": "text",
    "checkbox_unselected": "text",
    "document_index": "text",
}


class DoclingLoader:
    """Loads PDF and Markdown documents with Docling; code files with plain text reading."""

    def __init__(
        self,
        pdf_base_path: str = "",
        markdown_base_path: str = "",
        code_base_path: str = "",
        code_extensions: list[str] | None = None,
    ) -> None:
        self.pdf_base_path = pdf_base_path
        self.markdown_base_path = markdown_base_path
        self.code_base_path = code_base_path
        self.code_extensions: list[str] = code_extensions or []

    def load(self) -> list[ParentSection]:
        sections: list[ParentSection] = []

        if self.pdf_base_path:
            print(f"Loading PDFs from {self.pdf_base_path} via Docling ...")
            sections.extend(self._load_with_docling(self.pdf_base_path, "**/*.pdf", "pdf"))

        if self.markdown_base_path:
            print(f"Loading Markdown from {self.markdown_base_path} via Docling ...")
            sections.extend(
                self._load_with_docling(self.markdown_base_path, "**/*.md", "markdown")
            )

        if self.code_base_path and self.code_extensions:
            print(
                f"Loading code from {self.code_base_path}"
                f" (extensions: {self.code_extensions}) ..."
            )
            for ext in self.code_extensions:
                sections.extend(self._load_code(ext))

        return sections

    def _load_with_docling(
        self, base_path: str, pattern: str, doc_type: str
    ) -> list[ParentSection]:
        # Lazy import - Docling is only required when this backend is selected.
        from docling.document_converter import DocumentConverter 

        paths = list(Path(base_path).glob(pattern))
        if not paths:
            return []

        converter = DocumentConverter()
        sections: list[ParentSection] = []
        for path in paths:
            result = converter.convert(str(path))
            sections.extend(_docling_doc_to_sections(result.document, str(path), doc_type))

        return sections

    def _load_code(self, extension: str) -> list[ParentSection]:
        sections: list[ParentSection] = []
        for path in Path(self.code_base_path).rglob(f"*.{extension}"):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            sections.append(
                ParentSection(
                    text=text,
                    source=str(path),
                    element_id=hashlib.md5(str(path).encode()).hexdigest(),
                    doc_type="code",
                )
            )
        return sections



def _docling_doc_to_sections(document, source: str, doc_type: str) -> list[ParentSection]:
    """One section per title element; body elements get appended to it.

    Same structure as the Unstructured loader: section.text for MongoDB context,
    section.children for element-level chunk boundaries in Qdrant.
    """
    source_hash = hashlib.md5(source.encode()).hexdigest()
    sections: list[ParentSection] = []
    current: ParentSection | None = None
    idx = 0

    for item, _ in document.iterate_items():
        text = _extract_text(item)
        if not text.strip():
            continue

        label_str = str(getattr(item, "label", ""))
        category = _DOCLING_LABEL_MAP.get(label_str, "text")

        if category == "title":
            current = ParentSection(
                text=text, source=source,
                element_id=f"{source_hash}:{idx}",
                doc_type=doc_type,
                children=[text],
            )
            sections.append(current)
        else:
            if current is None:
                current = ParentSection(
                    text="", source=source,
                    element_id=source_hash,
                    doc_type=doc_type,
                )
                sections.append(current)
            current.text = (current.text + "\n\n" + text).lstrip("\n")
            current.children.append(text)

        idx += 1

    return sections


def _extract_text(item) -> str:
    """Text for most elements; tables get exported as markdown."""
    if hasattr(item, "text") and item.text:
        return str(item.text)
    if hasattr(item, "export_to_markdown"):
        try:
            return item.export_to_markdown()
        except Exception:
            pass
    return ""