from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from rag_core.ingestion.document import ParentSection, RagDocument

_TOKENIZER_NAME = "Qwen/Qwen3-Embedding-8B"
_CHUNK_SIZE = 320
_CHUNK_OVERLAP = 0
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_sections(
    sections: list[ParentSection],
    splitter=None,
) -> list[RagDocument]:
    """Split sections into Qdrant chunks.

    Iterates over section.children (individual elements) so chunk boundaries
    follow the loader's natural element boundaries. Falls back to section.text
    when children is empty (code files, simple loaders).
    splitter can be injected in tests to skip the tokenizer download.
    """
    if splitter is None:
        tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
            separators=_SEPARATORS,
        )

    chunks: list[RagDocument] = []
    for section in sections:
        texts = section.children if section.children else [section.text]
        for element_text in texts:
            if not element_text.strip():
                continue
            for chunk_text in splitter.split_text(element_text):
                if not chunk_text.strip():
                    continue
                chunks.append(
                    RagDocument(
                        text=chunk_text,
                        source=section.source,
                        doc_type=section.doc_type,
                        parent_id=section.element_id,
                        metadata={"source": section.source},
                    )
                )

    return chunks
