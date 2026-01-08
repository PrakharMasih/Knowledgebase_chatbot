import json
from sqlalchemy import select, delete
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.chatbot.models import Message


class ChatHistoryManager:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def add_message(
        self,
        role: str,
        content: str,
        confidence: Optional[str] = None,
        sources: Optional[List[Dict]] = None,
    ) -> int:
        msg = Message(
            role=role,
            content=content,
            confidence=confidence,
            sources=json.dumps(sources) if sources else None
        )

        self.db.add(msg)
        await self.db.commit()
        await self.db.refresh(msg)

        return msg.id

    async def delete_all_messages(self) -> None:
        await self.db.execute(delete(Message))
        await self.db.commit()

    async def get_chat_history(
            self,
            limit: int = 50,
    ) -> List[Dict[str, Any]]:
        stmt = (
            select(Message)
            .order_by(Message.timestamp.desc())
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        rows = result.scalars().all()

        # reverse to chronological order first
        rows = list(reversed(rows))
        role_priority = {"user": 0, "assistant": 1}
        rows.sort(key=lambda m: (m.timestamp, role_priority.get(m.role, 2)))

        return [
            {
                "role": m.role,
                "content": m.content,
                "confidence": m.confidence,
                "sources": json.loads(m.sources) if m.sources else None,
                "timestamp": m.timestamp,
            }
            for m in rows
        ]

    async def get_recent_context(
        self,
        n_turns: int = 3,
    ) -> List[Dict[str, str]]:
        stmt = (
            select(Message.role, Message.content)
            .order_by(Message.timestamp.desc())
            .limit(n_turns * 2)
        )

        result = await self.db.execute(stmt)
        rows = result.all()

        return [
            {"role": role, "content": content}
            for role, content in reversed(rows)
        ]
