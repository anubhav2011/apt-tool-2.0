"""
Optional audit user id for ORM inserts/updates (created_by / updated_by).
When unset (e.g. background workers), a fixed system id is used.
"""
from __future__ import annotations

import contextvars
from typing import Optional

SYSTEM_AUDIT_USER = "system"

_audit_user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "audit_user_id", default=None
)


def get_audit_user_id() -> str:
    """Effective user id for audit columns; never None."""
    uid = _audit_user_id.get()
    return uid if uid else SYSTEM_AUDIT_USER


def set_audit_user_id(user_id: Optional[str]) -> contextvars.Token:
    """Set current audit user for this context; returns token for reset_audit_user_id."""
    return _audit_user_id.set(user_id)


def reset_audit_user_id(token: contextvars.Token) -> None:
    """Restore previous audit user (e.g. in a finally block)."""
    _audit_user_id.reset(token)
