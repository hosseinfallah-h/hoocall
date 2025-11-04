# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Text, Integer, ForeignKey, DateTime

class Base(DeclarativeBase):
    pass

class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(120), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    messages: Mapped[list["Message"]] = relationship("Message", back_populates="conversation", cascade="all, delete")

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"))
    role: Mapped[str] = mapped_column(String(20))  # "user" | "assistant" | "system"
    content: Mapped[str] = mapped_column(Text)
    audio_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # optional saved audio
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")
