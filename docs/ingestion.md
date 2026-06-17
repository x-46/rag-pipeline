# Ingestion

Ingestion reads documents from configured directories, processes them into chunks and parent sections, and writes everything to Qdrant and MongoDB. It is a one-shot operation. It drops both collections and rebuilds them from scratch on every run (this can be easily changed if needed for updates or large-scale ingestion).

The entry point is `services/ingestion/src/ingestion/main.py`.

## What it does

The pipeline has three steps:

**1. Loading** - a loader scans the configured directories and returns a `list[ParentSection]`. Each section covers one heading plus all the body content beneath it. The loader also keeps the individual elements (paragraph, table, list item, ...) separately in `section.children` - more on why below.

**2. Chunking** - `chunk_sections()` iterates over `section.children` and splits each element into token-sized chunks using a tokenizer-aware text splitter (Qwen3-Embedding-8B tokenizer, 320 tokens). Each chunk gets a `parent_id` linking it back to its parent section.

**3. Storage** - child chunks go to Qdrant, indexed with dense and BM25 sparse embeddings. Parent sections go to MongoDB, indexed by `element_id`.

## The data model

**`ParentSection`** represents a logical section of a document and is what loaders produce:

- `text` - the full section text (heading + all body content concatenated); returned to the user on retrieval
- `children` - the individual element texts in order (paragraph, table, ...); used for chunking, not stored in MongoDB
- `source` - file path
- `doc_type` - `"pdf"`, `"markdown"`, or `"code"`
- `element_id` - stable ID for this section; used by chunks to reference their parent

**`RagDocument`** is a child chunk - what the chunker produces and what goes into Qdrant:

- `text` - the chunk content (one element, or a piece of it if the element was longer than 320 tokens)
- `source` - file path (copied from the parent section)
- `doc_type` - copied from the parent section
- `parent_id` - references `ParentSection.element_id`

The Custom ParentSection Data Model allows a clean separation between reading/grouping documents (the loader’s responsibility) and the chunking logic (the chunker’s responsibility).

## What a parent section looks like

For Markdown and PDF files, a parent section is built from a heading element plus all the body content that follows it until the next heading.

For example, this Markdown content produces two parent sections:

```markdown
# Market Simulation

ASSUME models electricity markets using agent-based simulation.
Each agent represents a market participant and submits bids.

| Col A | Col B |
|-------|-------|
| 1     | 2     |

## Reinforcement Learning

Agents can be trained using RL.
```

Section 1:
- `text`: `"Market Simulation\n\nASSUME models ... submits bids.\n\n| Col A ..."`
- `children`: `["Market Simulation", "ASSUME models ... submits bids.", "| Col A ..."]`

Section 2:
- `text`: `"Reinforcement Learning\n\nAgents can be trained using RL."`
- `children`: `["Reinforcement Learning", "Agents can be trained using RL."]`

PDF files work the same way, based on title elements detected by the loader. Content before the first heading goes into an implicit section so nothing is lost.

Code files are always one section per file with `children=[]`. The chunker then falls back to splitting `section.text` directly.

## Why chunks follow element boundaries

The chunks in Qdrant are built from `section.children` rather than from `section.text`. The reason is that each element from the loader's layout analysis already represents a coherent unit of content like a paragraph, a table, a list. Splitting the merged section text at token positions instead would cut across those boundaries at arbitrary points, which makes the embeddings less meaningful and retrieval less precise.

At the same time, returning just a single matched paragraph to the LLM is often too little context. So when retrieval finds a matching chunk, it looks up `parent_id` in MongoDB and returns the full section text. The chunk granularity is for search quality; the section granularity is for answer quality.

## Loader backends

Two backends are available, selectable via `INGEST_LOADER`. Both return `list[ParentSection]` with `text` and `children` populated.

**UnstructuredLoader** (`INGEST_LOADER=unstructured`, default) uses the [Unstructured](https://unstructured.io/) library for PDFs and Markdown and LangChain's `TextLoader` for code files. PDFs are processed with the `hi_res` strategy with OCR and layout detection enabled, table structure inferred. The `element_id` on each section is the stable ID Unstructured assigns to the title element.

**DoclingLoader** (`INGEST_LOADER=docling`) uses [Docling](https://github.com/DS4SD/docling) for PDFs and Markdown, plain UTF-8 reading for code. This implementation is a POC and does not use the full capabilities of Docling.

## How to add a loader

A loader is any class with a `load(self) -> list[ParentSection]` method. The loader is responsible for both reading files and grouping their content into sections. To add one:

1. Create a new file in `packages/rag_core/src/rag_core/ingestion/loaders/`.
2. Group elements by heading into `ParentSection` objects. Set `text` to the full concatenated section text and `children` to the list of individual element texts in document order. The `element_id` should be stable across repeated runs of the same file.
3. For code files, emit one section per file with `children=[]` and `doc_type="code"`.
4. Add a branch for your backend in `get_loader()` in `loaders/__init__.py`. Import lazily inside the branch (same pattern as the existing loaders).

The existing `UnstructuredLoader` is the best reference.

## Chunking details

`chunk_sections()` in `rag_core/ingestion/chunker.py` takes `list[ParentSection]` and returns `list[RagDocument]`.

For each section it iterates over `section.children` (falling back to `[section.text]` when `children` is empty). Each element text is split by `RecursiveCharacterTextSplitter` configured with the Qwen3-Embedding-8B tokenizer (chunk size 320 tokens, no overlap). The separators try paragraph breaks first, then line breaks, then sentence boundaries - character-level splitting only happens as a last resort. Each resulting chunk becomes a `RagDocument` with `parent_id` set to the section's `element_id`. The tokenizer is downloaded from HuggingFace on first run.
