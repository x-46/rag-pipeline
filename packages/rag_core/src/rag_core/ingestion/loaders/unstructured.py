from __future__ import annotations

import hashlib

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
)
from langchain_core.documents import Document

from rag_core.ingestion.document import ParentSection

_CATEGORY_MAP: dict[str, str] = {
    "Title": "title",
    "NarrativeText": "text",
    "Table": "table",
    "ListItem": "list_item",
    "FigureCaption": "figure_caption",
    "Image": "image",
    "Header": "header",
    "Footer": "footer",
    "CodeSnippet": "code",
    "Formula": "formula",
    "UncategorizedText": "text",
    "PageBreak": "page_break",
    "Address": "text",
    "EmailAddress": "text",
}


class UnstructuredLoader:
    """Loads PDF and Markdown files via Unstructured; code files via ``TextLoader``."""

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
            print(f"Loading PDFs from {self.pdf_base_path} ...")
            sections.extend(_lc_docs_to_sections(self._load_pdfs(), "pdf"))

        if self.markdown_base_path:
            print(f"Loading Markdown from {self.markdown_base_path} ...")
            sections.extend(_lc_docs_to_sections(self._load_markdown(), "markdown"))

        if self.code_base_path and self.code_extensions:
            print(
                f"Loading code from {self.code_base_path}"
                f" (extensions: {self.code_extensions}) ..."
            )
            for ext in self.code_extensions:
                for doc in self._load_code(ext):
                    source = doc.metadata.get("source", "")
                    sections.append(
                        ParentSection(
                            text=doc.page_content,
                            source=source,
                            element_id=hashlib.md5(source.encode()).hexdigest(),
                            doc_type="code",
                        )
                    )

        return sections

    def _load_pdfs(self) -> list[Document]:
        """Each real Document may return multiple Documents."""
        loader = DirectoryLoader(
            self.pdf_base_path,
            glob="**/*.pdf",
            loader_cls=UnstructuredPDFLoader,
            loader_kwargs={
                "mode": "elements",
                "strategy": "hi_res",
                "languages": ["eng"],
                "coordinates": True,
                "infer_table_structure": True,
            },
            show_progress=True,
        )
        return loader.load()

    def _load_markdown(self) -> list[Document]:
        """Each real Document may return multiple Documents."""
        loader = DirectoryLoader(
            self.markdown_base_path,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            loader_kwargs={"mode": "elements"},
            show_progress=True,
        )
        return loader.load()

    def _load_code(self, extension: str) -> list[Document]:
        """Each real Document may return multiple Documents."""
        loader = DirectoryLoader(
            self.code_base_path,
            glob=f"**/*.{extension}",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            recursive=True,
            silent_errors=True,
            show_progress=True,
        )
        return loader.load()



def _lc_docs_to_sections(lc_docs: list[Document], doc_type: str) -> list[ParentSection]:
    """Group Unstructured elements by source file, then by title into ParentSections."""
    by_source: dict[str, list[Document]] = {}
    for doc in lc_docs:
        by_source.setdefault(doc.metadata.get("source", ""), []).append(doc)

    sections: list[ParentSection] = []
    for source, docs in by_source.items():
        sections.extend(_group_by_title(docs, source, doc_type))
    return sections


def _group_by_title(
    docs: list[Document], source: str, doc_type: str
) -> list[ParentSection]:
    """One section per title element; body elements get appended to it.

    section.text = full section text for MongoDB context retrieval.
    section.children = individual elements, used as chunk boundaries in Qdrant.
    """
    sections: list[ParentSection] = []
    current: ParentSection | None = None

    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue

        raw_cat = doc.metadata.get("category", "")
        category = _CATEGORY_MAP.get(raw_cat) or (raw_cat.lower() if raw_cat else None)

        if category == "title":
            eid = doc.metadata.get("element_id") or _fallback_id(source, text)
            current = ParentSection(
                text=text, source=source, element_id=eid, doc_type=doc_type,
                children=[text],
            )
            sections.append(current)
        else:
            if current is None:
                current = ParentSection(
                    text="",
                    source=source,
                    element_id=hashlib.md5(source.encode()).hexdigest(),
                    doc_type=doc_type,
                )
                sections.append(current)
            current.text = (current.text + "\n\n" + text).lstrip("\n")
            current.children.append(text)

    return sections


def _fallback_id(source: str, text: str) -> str:
    return hashlib.md5(f"{source}:{text}".encode()).hexdigest()
