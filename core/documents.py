"""Tenant-scoped document access shared by the UI and the API."""
from __future__ import annotations

from typing import List, Optional, Tuple

from config.settings import Settings
from rag.pipeline import open_index


def readable_indexes(settings: Settings, principal, doc_ids: Optional[List[str]] = None,
                     vision=None, before_open=None) -> List[Tuple[object, object]]:
    """[(Document row, DocumentIndex)] for current versions this principal may read (retrieval-time authorisation)."""
    from database.repository import get_repository
    from security.policy_engine import PolicyEngine

    repo = get_repository(settings)
    policy = PolicyEngine(settings, repo)
    out = []
    for doc in repo.list_documents(principal.tenant_id):
        if doc_ids and doc.id not in doc_ids:
            continue
        ver = repo.current_version(doc.id)
        if ver is None or not policy.can_read(principal, ver.content_hash):
            continue
        if before_open is not None:  # e.g. restore this document's index from persistent storage first
            before_open(ver.content_hash)
        ix = open_index(ver.content_hash, ver.filename, settings, vision)
        if ix is not None:
            out.append((doc, ix))
    return out
