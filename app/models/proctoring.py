"""
Database Models for Proctoring System
Creates the proctoring database schema:
- proctoring_events_logs: Individual event/frame-level data
- proctoring_reports: Summary reports with risk scores
- proctoring_event_summary: Aggregated event summaries

Proctoring scheduler state lives in ai_interview_status (see that model).
"""

from sqlalchemy import (
    Column, String, DateTime, Float, Integer, Boolean,
    Date, func, Index, ForeignKey, event, BigInteger,
)
import uuid as uuid_lib

from app.core.database import Base


class ProctoringEventLog(Base):
    """
    Proctoring Event Logs Table
    Stores individual event/frame-level detection data.
    References proctoring_reports via interview_id.
    """
    __tablename__ = 'proctoring_events_logs'

    id           = Column(Integer, primary_key=True, autoincrement=True)
    interview_id = Column(
        BigInteger,
        ForeignKey('proctoring_reports.interview_id', ondelete='CASCADE'),
        nullable=False, index=True,
    )

    event_type      = Column(String(100), nullable=False, index=True)
    # Stores video timeline timestamp (e.g. "2:13"), not processing datetime.
    event_timestamp = Column(String(16),  nullable=False, index=True)

    duration   = Column(Float,     nullable=True)
    direction  = Column(String(50), nullable=True)
    intensity  = Column(Float,     nullable=True)
    confidence = Column(Float,     nullable=True)
    velocity   = Column(String(20), nullable=True)   # "slow" / "moderate" / "rapid"
    event_risk = Column(String(20), nullable=True, index=True)

    created = Column(DateTime, server_default=func.now(), index=True)
    updated = Column(DateTime, nullable=True, server_default=func.now(), onupdate=func.now(), index=True)
    created_by = Column(String(36), nullable=True, index=True)
    updated_by = Column(String(36), nullable=True, index=True)
    active = Column(Boolean, nullable=True, default=True)

    __table_args__ = (
        Index('idx_interview_timestamp',    'interview_id', 'event_timestamp'),
        Index('idx_event_type_timestamp',   'event_type',   'event_timestamp'),
        {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'},
    )

    def to_dict(self):
        return {
            'id':              self.id,
            'interview_id':    self.interview_id,
            'event_type':      self.event_type,
            'event_timestamp': self.event_timestamp,
            'duration':        self.duration,
            'direction':       self.direction,
            'intensity':       self.intensity,
            'confidence':      self.confidence,
            'velocity':        self.velocity,
            'event_risk':      self.event_risk,
            'created':         (self.created.isoformat()
                                if self.created else None),
            'updated':         (self.updated.isoformat()
                                if self.updated else None),
            'created_by':      self.created_by,
            'updated_by':      self.updated_by,
            'active':          self.active,
        }


class ProctoringReport(Base):
    """
    Proctoring Reports Table
    One row per interview; `id` is the primary key.
    """
    __tablename__ = 'proctoring_reports'

    id = Column(Integer, primary_key=True, autoincrement=True)
    interview_id = Column(BigInteger, nullable=False, unique=True, index=True)

    interview_date            = Column(Date,    nullable=False, index=True)
    # Cheating likeliness 0.0–10.0 (rounded to 2 decimals before persist)
    cheating_likelihood_score = Column(Float, nullable=False)
    cheating_likelihood_level = Column(String(50), nullable=False, index=True)

    created = Column(DateTime, server_default=func.now(), index=True)
    updated = Column(DateTime, nullable=True, server_default=func.now(), onupdate=func.now(), index=True)
    created_by = Column(String(36), nullable=True, index=True)
    updated_by = Column(String(36), nullable=True, index=True)
    active = Column(Boolean, nullable=True, default=True)

    __table_args__ = (
        Index('idx_interview_created',  'interview_id', 'created'),
        Index('idx_risk_score',         'cheating_likelihood_score'),
        Index('idx_risk_level',         'cheating_likelihood_level'),
        {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'},
    )

    def to_dict(self):
        return {
            'id':                         self.id,
            'interview_id':               self.interview_id,
            'interview_date':             (self.interview_date.isoformat()
                                           if self.interview_date else None),
            'cheating_likelihood_score':  self.cheating_likelihood_score,
            'cheating_likelihood_level':  self.cheating_likelihood_level,
            'created':                    (self.created.isoformat()
                                           if self.created else None),
            'updated':                    (self.updated.isoformat()
                                           if self.updated else None),
            'created_by':                 self.created_by,
            'updated_by':                 self.updated_by,
            'active':                     self.active,
        }


class ProctoringEventSummary(Base):
    """
    Proctoring Event Summary Table
    Aggregated event data per type for a given interview.
    """
    __tablename__ = 'proctoring_event_summary'

    id           = Column(BigInteger, primary_key=True, autoincrement=True)
    interview_id = Column(
        BigInteger,
        ForeignKey('proctoring_reports.interview_id', ondelete='CASCADE'),
        nullable=False, index=True,
    )

    event_type      = Column(String(100), nullable=False, index=True)
    total_count     = Column(Integer,     nullable=False, default=0)
    normal_count    = Column(Integer,     nullable=False, default=0)
    suspicious_count = Column(Integer,   nullable=False, default=0)
    high_risk_count = Column(Integer,    nullable=False, default=0)
    total_duration  = Column(Float,      nullable=False, default=0.0)

    created = Column(DateTime, server_default=func.now(), index=True)
    updated = Column(DateTime, nullable=True, server_default=func.now(), onupdate=func.now(), index=True)
    created_by = Column(String(36), nullable=True, index=True)
    updated_by = Column(String(36), nullable=True, index=True)
    active = Column(Boolean, nullable=True, default=True)

    __table_args__ = (
        Index('idx_interview_event_type', 'interview_id', 'event_type'),
        Index('idx_event_type_created',   'event_type',   'created'),
        {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'},
    )

    def to_dict(self):
        return {
            'id':               self.id,
            'interview_id':     self.interview_id,
            'event_type':       self.event_type,
            'total_count':      self.total_count,
            'normal_count':     self.normal_count,
            'suspicious_count': self.suspicious_count,
            'high_risk_count':  self.high_risk_count,
            'total_duration':   self.total_duration,
            'created':          (self.created.isoformat()
                                 if self.created else None),
            'updated':          (self.updated.isoformat()
                                 if self.updated else None),
            'created_by':       self.created_by,
            'updated_by':       self.updated_by,
            'active':           self.active,
        }


def _register_proctoring_audit_listeners():
    from app.utils.audit_context import get_audit_user_id

    def _before_insert(mapper, connection, target):
        uid = get_audit_user_id()
        if getattr(target, "created_by", None) is None:
            target.created_by = uid
        if getattr(target, "updated_by", None) is None:
            target.updated_by = uid

    def _before_update(mapper, connection, target):
        target.updated_by = get_audit_user_id()

    for model in (ProctoringEventLog, ProctoringReport, ProctoringEventSummary):
        event.listen(model, "before_insert", _before_insert)
        event.listen(model, "before_update", _before_update)


_register_proctoring_audit_listeners()