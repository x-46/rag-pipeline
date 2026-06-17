from typing import Protocol

from rag_core.ingestion.document import ParentSection


class DocumentLoader(Protocol):
    """Minimal interface shared by all loader backends.
    
    A loader is responsible for:
    - reading files from one or more source paths
    - returning one ``ParentSection`` per logical document section
    - assigning a stable ``element_id`` and the correct ``doc_type`` on every section
    """

    def load(self) -> list[ParentSection]:
        """Read all configured sources and return one ``ParentSection`` per section."""
        ...
