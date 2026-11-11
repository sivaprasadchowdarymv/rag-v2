"""Request / response models of the REST API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TokenRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    token: str
    user_id: str
    tenant_id: str
    roles: List[str]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    llm_mode: Optional[str] = Field(default=None, pattern="^(local|cloud|hybrid)$")
    model: Optional[str] = Field(default=None, max_length=200)
    session_id: str = Field(default="api", max_length=64)


class SourceOut(BaseModel):
    label: str
    document: str
    page: Any
    section: str
    snippet: str


class QueryResponse(BaseModel):
    request_id: str
    kind: str
    answer: str
    sources: List[SourceOut]
    warnings: List[str]
    verification: Dict[str, Any]
    routing: Dict[str, Any]
    agent_trace: List[Dict[str, Any]]
    security: Dict[str, Any]
    provider: str
    model: str
    tokens: int
    cost_usd: float
    latency_s: float
    cached: bool = False                      # 2.1 (additive): answer served from the verified-answer cache
    cache: Dict[str, Any] = Field(default_factory=dict)


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: str = Field(default="rerank", pattern="^(dense|bm25|fuzzy|hybrid|rerank)$")
    k: int = Field(default=5, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None


class PermissionsRequest(BaseModel):
    sensitivity: Optional[str] = Field(default=None, pattern="^(public|internal|confidential|restricted)$")
    allowed_roles: Optional[List[str]] = None


class UserCreate(BaseModel):
    username: str
    password: str
    roles: List[str]


class MetadataUpdate(BaseModel):
    """User-entered metadata (2.1). Every field optional; unknown keys are rejected (422)."""
    model_config = ConfigDict(extra="forbid")
    title: Optional[str] = Field(default=None, max_length=300)
    author: Optional[str] = Field(default=None, max_length=200)
    category: Optional[str] = Field(default=None, max_length=80)
    tags: Optional[List[str]] = None
    description: Optional[str] = Field(default=None, max_length=2000)
    version_label: Optional[str] = Field(default=None, max_length=60)
    part_numbers: Optional[List[str]] = None
    custom: Optional[Dict[str, str]] = None
