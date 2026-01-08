# app/models/models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, func
from app.config.database import Base


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    confidence = Column(String, nullable=True)
    sources = Column(Text, nullable=True)
    timestamp = Column(DateTime, server_default=func.now())
