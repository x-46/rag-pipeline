from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParentSection:
    """One heading-grouped section of a source document, stored in MongoDB.

    text is the full section (heading + everything below it) and is what gets
    returned to the user on retrieval.  children holds the individual elements
    (paragraphs, tables, ...) so the chunker can keep those natural boundaries
    rather than re-splitting the merged text at arbitrary positions.

    For code files children is empty; the chunker then just splits text directly.
    """

    text: str
    source: str
    element_id: str
    doc_type: str = "text"
    children: list[str] = field(default_factory=list)

    def to_mongo_dict(self) -> dict:
        return {
            "page_content": self.text,
            "source": self.source,
            "element_id": self.element_id,
        }


@dataclass
class RagDocument:
    """A chunk that goes into Qdrant. parent_id points back to its ParentSection in MongoDB."""

    text: str
    source: str
    doc_type: str = "text"
    parent_id: str | None = None
    metadata: dict = field(default_factory=dict)
