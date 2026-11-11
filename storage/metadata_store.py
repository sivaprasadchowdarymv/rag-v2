"""
Metadata store (2.1): automatically extracted profiles + user-entered metadata, per document version.

    data/documents/<content_hash>/metadata_profile.json   auto profile (rebuilt when PROFILE_VERSION changes)
    data/documents/<content_hash>/metadata_user.json      {"<scope>": {field: value, ..., "_updated_at", "_by"}}

* No database migration: existing JSON metadata (`meta.json`, index `doc_meta`) and DB rows are untouched.
* User metadata is kept per *scope* (tenant id with accounts; "ws:<workspace>" without), so two tenants that
  happen to upload an identical file never see each other's labels.
* Files live in the document folder, so deleting a document removes them and Hugging Face sync carries them.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Iterable, List, Optional

from config.settings import Settings, get_logger
from rag.metadata_intel import PROFILE_VERSION, USER_FIELDS, build_profile, effective, matches
from rag.models import DocumentIndex
from storage.cache import DocumentPaths, read_json, write_json_atomic

log = get_logger("metadata")
_LOCK = threading.Lock()
MAX_TAGS, MAX_PARTS, MAX_CUSTOM = 30, 50, 30


class MetadataError(ValueError):
    pass


def _clean_text(v: Any, limit: int) -> str:
    text = " ".join(str(v or "").split())
    return "".join(ch for ch in text if ch.isprintable())[:limit]


def sanitize_user_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a user edit. Unknown fields are rejected; values are trimmed and bounded."""
    unknown = set(data) - set(USER_FIELDS)
    if unknown:
        raise MetadataError(f"Unknown metadata field(s): {', '.join(sorted(unknown))}")
    out: Dict[str, Any] = {}
    for k in ("title", "author", "category", "version_label"):
        if k in data and data[k] is not None:
            out[k] = _clean_text(data[k], {"title": 300, "author": 200, "category": 80, "version_label": 60}[k])
    if data.get("description") is not None:
        out["description"] = str(data["description"])[:2000].strip()
    if data.get("category"):
        out["category"] = out["category"].lower()
    if data.get("tags") is not None:
        tags = data["tags"] if isinstance(data["tags"], list) else str(data["tags"]).split(",")
        out["tags"] = list(dict.fromkeys(t for t in (_clean_text(x, 40).lower() for x in tags) if t))[:MAX_TAGS]
    if data.get("part_numbers") is not None:
        parts = data["part_numbers"] if isinstance(data["part_numbers"], list) else str(data["part_numbers"]).split(",")
        out["part_numbers"] = list(dict.fromkeys(p for p in (_clean_text(x, 60).upper() for x in parts) if p))[:MAX_PARTS]
    if data.get("custom") is not None:
        if not isinstance(data["custom"], dict):
            raise MetadataError("custom must be a mapping of field name to value")
        custom = {}
        for key, val in list(data["custom"].items())[:MAX_CUSTOM]:
            key = _clean_text(key, 40)
            if key:
                custom[key] = _clean_text(val, 300)
        out["custom"] = custom
    return out


class MetadataStore:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _paths(self, doc_id: str) -> DocumentPaths:
        return DocumentPaths(self.settings.data_dir, doc_id)

    # ------------------------------------------------------------------ auto
    def profile(self, index: DocumentIndex, file_info: Optional[Dict[str, Any]] = None,
                refresh: bool = False) -> Dict[str, Any]:
        path = self._paths(index.doc_id).doc_dir / "metadata_profile.json"
        cached = None if refresh else read_json(path)
        if isinstance(cached, dict) and cached.get("profile_version") == PROFILE_VERSION:
            if file_info:  # file facts can be learned later (e.g. size from the DB); merge, don't re-extract
                doc = cached.setdefault("document", {})
                for k, src in (("file_size_bytes", "size_bytes"), ("sensitivity", "sensitivity"),
                               ("uploaded_at", "uploaded_at"), ("owner", "owner")):
                    if file_info.get(src):
                        doc[k] = {"value": file_info[src], "source": "file", "confidence": 1.0, "pages": []}
            return cached
        try:
            prof = build_profile(index, file_info)
        except Exception:  # never break the library because of one odd document
            log.exception("metadata extraction failed for one document")
            prof = {"profile_version": PROFILE_VERSION, "doc_id": index.doc_id, "filename": index.filename,
                    "document": {}, "parameters": [], "part_numbers": [], "suggested_tags": [], "error": True}
        prof["extracted_at"] = round(time.time(), 1)
        try:
            write_json_atomic(path, prof)
        except OSError:
            log.warning("could not save the metadata profile")
        return prof

    # ------------------------------------------------------------------ user
    def user(self, doc_id: str, scope: str) -> Dict[str, Any]:
        data = read_json(self._paths(doc_id).doc_dir / "metadata_user.json", {}) or {}
        return dict(data.get(scope) or {})

    def update_user(self, doc_id: str, scope: str, changes: Dict[str, Any], actor: str = "",
                    replace: bool = False) -> Dict[str, Any]:
        clean = sanitize_user_metadata(changes)
        path = self._paths(doc_id).doc_dir / "metadata_user.json"
        with _LOCK:
            data = read_json(path, {}) or {}
            cur = {} if replace else dict(data.get(scope) or {})
            for k, v in clean.items():
                if v in ("", [], {}):
                    cur.pop(k, None)  # empty value = back to the automatic suggestion
                else:
                    cur[k] = v
            cur["_updated_at"], cur["_by"] = round(time.time(), 1), actor[:64]
            data[scope] = cur
            write_json_atomic(path, data)
        return cur

    def reset_user(self, doc_id: str, scope: str) -> None:
        path = self._paths(doc_id).doc_dir / "metadata_user.json"
        with _LOCK:
            data = read_json(path, {}) or {}
            if data.pop(scope, None) is not None:
                write_json_atomic(path, data)

    # ---------------------------------------------------------------- queries
    def view(self, index: DocumentIndex, scope: str, file_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        prof = self.profile(index, file_info)
        usr = self.user(index.doc_id, scope)
        return {"profile": prof, "user": usr, "effective": effective(prof, usr)}

    def filter_doc_ids(self, indexes: Iterable[DocumentIndex], scope: str, **criteria: Any) -> List[str]:
        out = []
        for ix in indexes:
            if matches(self.profile(ix), self.user(ix.doc_id, scope), **criteria):
                out.append(ix.doc_id)
        return out


def scope_for(principal, workspace_id: str = "", auth_mode: str = "users") -> str:
    """Tenant with accounts; the private workspace link without accounts (all users there are 'anonymous')."""
    if auth_mode != "users" and workspace_id:
        return f"ws:{workspace_id}"
    return f"tenant:{getattr(principal, 'tenant_id', 'default')}"


DOC_FILTER_KEYS = ("query", "tags", "category", "part_number", "file_type")


def resolve_doc_filter(flt: Optional[Dict[str, Any]], indexes: Iterable[DocumentIndex], settings: Settings,
                       scope: str) -> Optional[Dict[str, Any]]:
    """Turn document-level metadata criteria ({"tags": [...], "category": ...}) into `doc_ids` (intersected with
    any doc_ids already present). Node-level keys pass through unchanged. No criteria → filter unchanged."""
    if not flt or not any(flt.get(k) for k in DOC_FILTER_KEYS):
        return flt
    crit = {k: flt.get(k) for k in DOC_FILTER_KEYS if flt.get(k)}
    if isinstance(crit.get("tags"), str):
        crit["tags"] = [t.strip() for t in crit["tags"].split(",") if t.strip()]
    ids = set(MetadataStore(settings).filter_doc_ids(indexes, scope, **crit))
    if flt.get("doc_ids"):
        ids &= set(flt["doc_ids"])
    out = {k: v for k, v in flt.items() if k not in DOC_FILTER_KEYS}
    out["doc_ids"] = sorted(ids) or ["__no_document_matches__"]
    return out
