"""
Chat History Manager for persistent conversation storage.

Features:
- Per-user session history
- Conversation context for follow-ups
- History persistence to JSON files
- Session management
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import json
import threading
import uuid


class ChatMessage:
    """Represents a single chat message."""
    
    def __init__(
        self,
        role: str,
        content: str,
        mode: str,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        self.role = role  # "user" or "assistant"
        self.content = content
        self.mode = mode  # "docs", "video", "product"
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ChatMessage":
        return cls(
            role=data["role"],
            content=data["content"],
            mode=data.get("mode", "docs"),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {}),
        )


class ChatSession:
    """Represents a chat session with history."""
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        mode: str,
        created_at: Optional[str] = None
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.mode = mode
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.messages: List[ChatMessage] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to the session."""
        msg = ChatMessage(role=role, content=content, mode=self.mode, metadata=metadata)
        self.messages.append(msg)
    
    def get_context(self, max_messages: int = 10) -> str:
        """Get conversation context for follow-up questions."""
        recent = self.messages[-max_messages:]
        context_parts = []
        for msg in recent:
            prefix = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{prefix}: {msg.content[:500]}")
        return "\n\n".join(context_parts)
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "mode": self.mode,
            "created_at": self.created_at,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ChatSession":
        session = cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            mode=data.get("mode", "docs"),
            created_at=data.get("created_at"),
        )
        session.messages = [ChatMessage.from_dict(m) for m in data.get("messages", [])]
        session.metadata = data.get("metadata", {})
        return session


class ChatHistoryManager:
    """
    Manages chat history persistence and retrieval.
    
    Structure:
    history/
      user_1/
        session_abc123.json
        session_def456.json
      user_2/
        ...
    """
    
    def __init__(self, history_dir: Optional[Path] = None):
        if history_dir is None:
            history_dir = Path(__file__).parent.parent.parent / "history"
        self.history_dir = history_dir
        self.history_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._active_sessions: Dict[str, ChatSession] = {}
    
    def _get_user_dir(self, user_id: str) -> Path:
        """Get or create user's history directory."""
        user_dir = self.history_dir / user_id
        user_dir.mkdir(exist_ok=True)
        return user_dir
    
    def create_session(self, user_id: str, mode: str) -> ChatSession:
        """Create a new chat session."""
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        session = ChatSession(session_id=session_id, user_id=user_id, mode=mode)
        
        with self._lock:
            self._active_sessions[session_id] = session
        
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get an active session by ID."""
        return self._active_sessions.get(session_id)
    
    def get_or_create_session(self, user_id: str, mode: str, session_id: Optional[str] = None) -> ChatSession:
        """Get existing session or create new one."""
        if session_id and session_id in self._active_sessions:
            return self._active_sessions[session_id]
        
        # Try to load from disk
        if session_id:
            session = self.load_session(user_id, session_id)
            if session:
                self._active_sessions[session_id] = session
                return session
        
        # Create new session
        return self.create_session(user_id, mode)
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict = None
    ):
        """Add a message to a session."""
        session = self._active_sessions.get(session_id)
        if session:
            session.add_message(role, content, metadata)
            self.save_session(session)
    
    def save_session(self, session: ChatSession):
        """Save session to disk."""
        user_dir = self._get_user_dir(session.user_id)
        file_path = user_dir / f"{session.session_id}.json"
        
        with self._lock:
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
    
    def load_session(self, user_id: str, session_id: str) -> Optional[ChatSession]:
        """Load a session from disk."""
        user_dir = self._get_user_dir(user_id)
        file_path = user_dir / f"{session_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return ChatSession.from_dict(data)
        except Exception:
            return None
    
    def get_user_sessions(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get list of user's recent sessions."""
        user_dir = self._get_user_dir(user_id)
        sessions = []
        
        for file_path in sorted(user_dir.glob("session_*.json"), reverse=True)[:limit]:
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "mode": data.get("mode", "docs"),
                    "created_at": data.get("created_at"),
                    "message_count": len(data.get("messages", [])),
                    "preview": data.get("messages", [{}])[0].get("content", "")[:100] if data.get("messages") else "",
                })
            except Exception:
                continue
        
        return sessions
    
    def get_conversation_context(self, session_id: str, max_messages: int = 10) -> str:
        """Get conversation context for a session."""
        session = self._active_sessions.get(session_id)
        if session:
            return session.get_context(max_messages)
        return ""
    
    def clear_session(self, session_id: str):
        """Clear a session from active sessions."""
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
    
    def delete_session(self, user_id: str, session_id: str):
        """Delete a session from disk and memory."""
        self.clear_session(session_id)
        user_dir = self._get_user_dir(user_id)
        file_path = user_dir / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()


# Global chat history manager
chat_history = ChatHistoryManager()
