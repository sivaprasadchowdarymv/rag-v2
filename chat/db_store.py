"""
Database-backed conversation store (same interface as chat/conversations.ConversationStore).

Used when AUTH_MODE=users: conversations and messages live in the application database (SQLite or
PostgreSQL), scoped by tenant and user, instead of JSON files in a link workspace.
"""
from __future__ import annotations

import time
import uuid
from typing import List, Optional

from sqlalchemy import delete, select

from chat.conversations import ChatMessage, Conversation, ConversationStore, result_to_dict
from database.models import Conversation as ConvRow
from database.models import Message as MsgRow
from rag.models import QueryResult


class DBConversationStore:
    export_markdown = staticmethod(ConversationStore.export_markdown)

    def __init__(self, repo, tenant_id: str, user_id: str):
        self.repo, self.tenant_id, self.user_id = repo, tenant_id, user_id

    def _own(self, s, cid: str) -> Optional[ConvRow]:
        row = s.get(ConvRow, cid)
        return row if row is not None and row.tenant_id == self.tenant_id and row.user_id == self.user_id else None

    def create(self, doc_ids: Optional[List[str]] = None) -> Conversation:
        now = time.time()
        conv = Conversation(uuid.uuid4().hex[:12], "New chat", now, now, list(doc_ids or []))
        with self.repo.session() as s:
            s.add(ConvRow(id=conv.id, tenant_id=self.tenant_id, user_id=self.user_id, title=conv.title,
                          doc_ids=conv.doc_ids, created_at=now, updated_at=now))
        return conv

    def save(self, conv: Conversation) -> None:
        conv.updated = time.time()
        with self.repo.session() as s:
            row = self._own(s, conv.id)
            if row is None:
                row = ConvRow(id=conv.id, tenant_id=self.tenant_id, user_id=self.user_id, created_at=conv.created)
                s.add(row)
            row.title, row.doc_ids, row.updated_at = conv.title, list(conv.doc_ids), conv.updated
            s.execute(delete(MsgRow).where(MsgRow.conversation_id == conv.id))
            s.flush()
            s.add_all(MsgRow(id=m.id, conversation_id=conv.id, seq=i, role=m.role, content=m.content, result=m.result,
                             feedback=m.feedback, variants=m.variants, created_at=m.created)
                      for i, m in enumerate(conv.messages))

    def load(self, cid: str) -> Optional[Conversation]:
        with self.repo.session() as s:
            row = self._own(s, cid)
            if row is None or row.archived:
                return None
            msgs = s.scalars(select(MsgRow).where(MsgRow.conversation_id == cid).order_by(MsgRow.seq)).all()
            return Conversation(row.id, row.title, row.created_at, row.updated_at, list(row.doc_ids or []),
                                [ChatMessage(m.id, m.role, m.content, m.created_at, m.result, m.feedback,
                                             list(m.variants or [])) for m in msgs])

    def list(self) -> List[Conversation]:
        with self.repo.session() as s:
            ids = s.scalars(select(ConvRow.id).where(ConvRow.tenant_id == self.tenant_id,
                                                     ConvRow.user_id == self.user_id, ConvRow.archived.is_(False))
                            .order_by(ConvRow.updated_at.desc()).limit(200)).all()
        return [c for c in (self.load(i) for i in ids) if c]

    def delete(self, cid: str) -> None:
        with self.repo.session() as s:
            if self._own(s, cid) is not None:
                s.execute(delete(MsgRow).where(MsgRow.conversation_id == cid))
                s.execute(delete(ConvRow).where(ConvRow.id == cid))

    def add_exchange(self, conv: Conversation, question: str, result: QueryResult) -> ChatMessage:
        return ConversationStore.add_exchange(self, conv, question, result)  # same logic, DB save()

    def replace_answer(self, conv: Conversation, message_id: str, result: QueryResult) -> None:
        ConversationStore.replace_answer(self, conv, message_id, result)


__all__ = ["DBConversationStore", "result_to_dict"]
