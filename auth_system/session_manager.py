"""
Session Manager - Manages user sessions for the auth system.
"""

import uuid
import time
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Represents an active session."""
    session_id: str
    user_id: str
    created_at: float
    expires_at: float
    data: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


class SessionManager:
    """Manages user sessions with creation, validation, and cleanup."""

    def __init__(self, session_timeout: int = 3600):
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = session_timeout
        logger.info(f"SessionManager initialized (timeout={session_timeout}s)")

    def create_session(self, user_id: str, data: Optional[Dict[str, Any]] = None) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        now = time.time()
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            expires_at=now + self.session_timeout,
            data=data or {},
        )
        self.sessions[session_id] = session
        logger.info(f"Session created: {session_id} for user {user_id}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID, returns None if expired or not found."""
        session = self.sessions.get(session_id)
        if session is None:
            return None
        if not session.is_active or time.time() > session.expires_at:
            self.invalidate_session(session_id)
            return None
        return session

    def validate_session(self, session_id: str) -> bool:
        """Check if a session is valid."""
        return self.get_session(session_id) is not None

    def refresh_session(self, session_id: str) -> Optional[Session]:
        """Refresh a session's expiration time."""
        session = self.get_session(session_id)
        if session:
            session.expires_at = time.time() + self.session_timeout
            return session
        return None

    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        session = self.sessions.get(session_id)
        if session:
            session.is_active = False
            del self.sessions[session_id]
            logger.info(f"Session invalidated: {session_id}")
            return True
        return False

    def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user."""
        count = 0
        to_remove = [
            sid for sid, s in self.sessions.items()
            if s.user_id == user_id
        ]
        for sid in to_remove:
            self.invalidate_session(sid)
            count += 1
        return count

    def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        now = time.time()
        expired = [
            sid for sid, s in self.sessions.items()
            if now > s.expires_at
        ]
        for sid in expired:
            del self.sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        return len(expired)

    def get_active_count(self) -> int:
        """Get count of active sessions."""
        self.cleanup_expired()
        return len(self.sessions)
