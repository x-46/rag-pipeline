from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader, UnstructuredPDFLoader, TextLoader
from langchain_core.documents import Document
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

def build_parent_child_chunks(docs: list[Document]) -> tuple[list[Document], list[dict]]:
    """Split documents into token-sized child chunks and derive parent sections for retrieval."""
    hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B")

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        hf_tokenizer,
        chunk_size=320,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    parent_elements = get_parent_text_from_elements(docs)

    # Build reverse map: child element_id -> parent element_id
    child_to_parent: dict[str, str] = {}
    for pe in parent_elements:
        pid = pe["element_id"]
        if not pid:
            continue
        child_to_parent[pid] = pid  # title element maps to itself
        for ceid in pe.get("child_element_ids", []):
            if ceid:
                child_to_parent[ceid] = pid

    # Annotate every chunk with parent_id so retrieval is uniform for all doc types
    for chunk in chunks:
        if chunk.metadata.get("doc_type") == "code":
            source = chunk.metadata.get("source", "")
            chunk.metadata["parent_id"] = hashlib.md5(source.encode()).hexdigest()
        else:
            eid = chunk.metadata.get("element_id")
            chunk.metadata["parent_id"] = child_to_parent.get(eid) if eid else None

    return chunks, parent_elements

def load_files(pdf_base_path: str, markdown_base_path: str, code_base_path: str, code_extensions: list[str] = []) -> list[Document]:
    """Load PDF, Markdown, and code documents from the given base paths."""
    pdf_docs = []
    markdown_docs = []
    code_docs = []

    if pdf_base_path:
        print(f"Loading PDF documents from {pdf_base_path}...")
        pdf_docs = _laod_pdfs(pdf_base_path)

    if markdown_base_path:
        print(f"Loading Markdown documents from {markdown_base_path}...")
        markdown_docs = _load_markdown(markdown_base_path)

    if code_base_path and code_extensions:
        print(f"Loading code documents from {code_base_path} with extensions {code_extensions}...")
        for ext in code_extensions:
            code_docs.extend(_load_text(code_base_path, ext))
    
    all_docs = pdf_docs + markdown_docs + code_docs

    return all_docs

def _laod_pdfs(base_path: str) -> list[Document]:
    """Load all PDFs from base_path using Unstructured in elements mode."""
    pdf_loader = DirectoryLoader(
        base_path,
        glob="**/*.pdf",
        loader_cls=UnstructuredPDFLoader,
        loader_kwargs={
            # "mode": "elements",
            "mode": "elements",
            "strategy": "hi_res",
            "languages": ["eng"],
            "coordinates": True, # Wichtig für mehere Spalten
            "infer_table_structure": True,
        },
        show_progress=True,
    )

    pdf_docs = pdf_loader.load()
    return pdf_docs


def _load_markdown(base_path: str) -> list[Document]:
    """Load all Markdown files from base_path using Unstructured in elements mode."""
    loader = DirectoryLoader(
        base_path, 
        glob="**/*.md", 
        loader_cls=UnstructuredMarkdownLoader, 
        loader_kwargs={ "mode": "elements"}, 
        show_progress=True
    )
    docs = loader.load()
    return docs


def _load_text(base_path: str, file_extensions: str) -> list[Document]:
    """Load all text/code files with the given extension from base_path."""
    code_loader = DirectoryLoader(
        base_path,
        glob=f"**/*.{file_extensions}",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8",
        },
        recursive=True,
        silent_errors=True,
        show_progress=True,
    )

    code_docs = code_loader.load()

    for doc in code_docs:
        doc.metadata["doc_type"] = "code"

    return code_docs



def _split_documents_by_title(docs: list[Document]) -> list[dict]:
    """Group elements into sections, each starting at a Title element."""
    sections = []
    current_section = None

    for doc in docs:
        category = doc.metadata.get("category")
        text = doc.page_content.strip()

        if not text:
            continue

        if category == "Title":
            current_section = {
                "title": text,
                "documents": [doc],
                "text": text,
                "metadata": {
                    "source": doc.metadata.get("source"),
                    "page_start": doc.metadata.get("page_number"),
                    "title_element_id": doc.metadata.get("element_id"),
                },
            }
            sections.append(current_section)

        else:
            if current_section is None:
                current_section = {
                    "title": "Ohne Titel",
                    "documents": [],
                    "text": "",
                    "metadata": {
                        "source": doc.metadata.get("source"),
                        "page_start": doc.metadata.get("page_number"),
                        "title_element_id": None,
                    },
                }
                sections.append(current_section)

            current_section["documents"].append(doc)
            current_section["text"] += "\n\n" + text

    return sections


def get_parent_text_from_elements(elements: list[Document]) -> list[dict]:
    """Build parent section dicts from raw elements, handling code and structured doc types separately."""
    # Python-Code-Dateien separat behandeln: kein category/element_id von Unstructured
    code_elements = [e for e in elements if e.metadata.get("doc_type") == "code"]
    structured_elements = [e for e in elements if e.metadata.get("doc_type") != "code"]

    res = []

    # Python Code: jede Datei = eine Parent-Sektion mit stabilem element_id (MD5 des Pfads)
    for doc in code_elements:
        source = doc.metadata.get("source", "unknown")
        eid = hashlib.md5(source.encode()).hexdigest()
        res.append({
            "page_content": doc.page_content,
            "source": source,
            "element_id": eid,
            "child_element_ids": [],
        })
    if code_elements:
        print(f"Processed {len(code_elements)} Python code files as parent sections.")

    # Strukturierte Dokumente (MD/PDF via Unstructured) nach Titel gruppieren
    source_docs = {}
    for element in structured_elements:
        source = element.metadata.get("source")
        if source not in source_docs:
            source_docs[source] = []
        source_docs[source].append(element)
    
    for source, elems in source_docs.items():
        split = _split_documents_by_title(elems)
        for section in split:
            res.append(
                  {
                    "page_content": section["text"],
                    "source": section["metadata"]["source"],
                    "element_id": section["metadata"]["title_element_id"],
                    "child_element_ids": [e.metadata.get("element_id") for e in section["documents"]],
                  }
            )
    return res