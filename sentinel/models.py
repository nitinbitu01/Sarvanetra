# sentinel/models.py
"""
sentinel/models.py — SQLAlchemy ORM models for Day 16.2 enterprise platform.
"""

from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, ForeignKey,
    Index, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# ── Officers & Roles ──────────────────────────────────────────────────────────

class Role(Base):
    __tablename__ = "roles"

    id:          Mapped[int]  = mapped_column(Integer, primary_key=True)
    name:        Mapped[str]  = mapped_column(String(50), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at:  Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Officer(Base):
    __tablename__ = "officers"

    id:               Mapped[int]  = mapped_column(Integer, primary_key=True)
    name:             Mapped[str]  = mapped_column(String(255), nullable=False)
    email:            Mapped[str]  = mapped_column(String(255), unique=True, nullable=False)
    hashed_password:  Mapped[str]  = mapped_column(String(255), nullable=False)
    role_id:          Mapped[int]  = mapped_column(ForeignKey("roles.id"), nullable=False)
    is_active:        Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_login_at:    Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at:       Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    role:             Mapped["Role"] = relationship("Role", lazy="joined")
    refresh_tokens:   Mapped[List["RefreshToken"]] = relationship(
        "RefreshToken", back_populates="officer", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_officers_email", "email"),
        Index("idx_officers_active", "is_active"),
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id:            Mapped[int]  = mapped_column(Integer, primary_key=True)
    officer_id:    Mapped[int]  = mapped_column(ForeignKey("officers.id"), nullable=False)
    token_hash:    Mapped[str]  = mapped_column(String(64), unique=True, nullable=False)
    family_id:     Mapped[str]  = mapped_column(String(36), nullable=False)
    revoked:       Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    expires_at:    Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at:    Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    ip_address:    Mapped[Optional[str]] = mapped_column(String(45))

    officer:       Mapped["Officer"] = relationship("Officer", back_populates="refresh_tokens")

    __table_args__ = (
        Index("idx_refresh_token_hash",   "token_hash"),
        Index("idx_refresh_officer_id",   "officer_id"),
        Index("idx_refresh_family",       "family_id"),
        Index("idx_refresh_expires",      "expires_at"),
    )


# ── Cameras ───────────────────────────────────────────────────────────────────

class Camera(Base):
    __tablename__ = "cameras"

    id:             Mapped[int]  = mapped_column(Integer, primary_key=True)
    name:           Mapped[str]  = mapped_column(String(255), nullable=False)
    lat:            Mapped[Optional[float]] = mapped_column(Float)
    lng:            Mapped[Optional[float]] = mapped_column(Float)
    location_label: Mapped[Optional[str]]  = mapped_column(String(255))
    stream_url:     Mapped[Optional[str]]  = mapped_column(Text)
    is_active:      Mapped[bool] = mapped_column(Boolean, default=True)
    created_at:     Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_cameras_active", "is_active"),
        Index("idx_cameras_location", "location_label"),
    )


# ── Alerts ────────────────────────────────────────────────────────────────────

class Alert(Base):
    __tablename__ = "alerts"

    id:         Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id:  Mapped[int]  = mapped_column(ForeignKey("cameras.id"), nullable=False)
    alert_type: Mapped[str]  = mapped_column(String(100), nullable=False)
    severity:   Mapped[str]  = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("idx_alerts_severity_created", "severity", "created_at"),
        Index("idx_alerts_camera_created", "camera_id", "created_at"),
        Index("idx_alerts_created", "created_at"),
    )


# ── Hash-Chained Audit Log ────────────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_log"

    id:           Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type:   Mapped[str]  = mapped_column(String(100), nullable=False)
    actor_id:     Mapped[Optional[int]] = mapped_column(ForeignKey("officers.id"))
    actor_type:   Mapped[str]  = mapped_column(String(50), default="officer")
    target_type:  Mapped[Optional[str]] = mapped_column(String(50))
    target_id:    Mapped[Optional[int]] = mapped_column(Integer)
    payload:      Mapped[Optional[str]] = mapped_column(Text)
    ip_address:   Mapped[Optional[str]] = mapped_column(String(45))
    prev_hash:    Mapped[str]  = mapped_column(String(64), nullable=False)
    row_hash:     Mapped[str]  = mapped_column(String(64), nullable=False, unique=True)
    created_at:   Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_audit_event_type_created", "event_type", "created_at"),
        Index("idx_audit_actor_created",      "actor_id",   "created_at"),
        Index("idx_audit_target",             "target_type", "target_id"),
        Index("idx_audit_row_hash",           "row_hash"),
    )


# ── Alert Feedback ────────────────────────────────────────────────────────────

class AlertFeedback(Base):
    __tablename__ = "alert_feedback"

    id:         Mapped[int] = mapped_column(Integer, primary_key=True)
    alert_id:   Mapped[int] = mapped_column(ForeignKey("alerts.id"), nullable=False)
    officer_id: Mapped[Optional[int]] = mapped_column(ForeignKey("officers.id"))
    verdict:    Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("alert_id", name="uq_feedback_alert"),
        Index("idx_feedback_verdict_created", "verdict", "created_at"),
    )


# ── Analytics Snapshots ────────────────────────────────────────────────────────

class AnalyticsHourlySnapshot(Base):
    __tablename__ = "analytics_hourly_snapshot"

    id:                   Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_hour:        Mapped[datetime] = mapped_column(
        DateTime(timezone=True), unique=True, nullable=False
    )
    total_alerts:         Mapped[int] = mapped_column(Integer, default=0)
    critical_count:       Mapped[int] = mapped_column(Integer, default=0)
    high_count:           Mapped[int] = mapped_column(Integer, default=0)
    medium_count:         Mapped[int] = mapped_column(Integer, default=0)
    low_count:            Mapped[int] = mapped_column(Integer, default=0)
    acked_count:          Mapped[int] = mapped_column(Integer, default=0)
    avg_ack_time_seconds: Mapped[Optional[float]] = mapped_column(Float)
    false_alarm_count:    Mapped[int] = mapped_column(Integer, default=0)
    genuine_count:        Mapped[int] = mapped_column(Integer, default=0)
    push_sent:            Mapped[int] = mapped_column(Integer, default=0)
    push_delivered:       Mapped[int] = mapped_column(Integer, default=0)
    created_at:           Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_hourly_snapshot_hour", "snapshot_hour"),
    )


class AnalyticsDailySnapshot(Base):
    __tablename__ = "analytics_daily_snapshot"

    id:                   Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_date:        Mapped[datetime] = mapped_column(
        DateTime(timezone=True), unique=True, nullable=False
    )
    total_alerts:         Mapped[int] = mapped_column(Integer, default=0)
    critical_count:       Mapped[int] = mapped_column(Integer, default=0)
    high_count:           Mapped[int] = mapped_column(Integer, default=0)
    medium_count:         Mapped[int] = mapped_column(Integer, default=0)
    low_count:            Mapped[int] = mapped_column(Integer, default=0)
    acked_count:          Mapped[int] = mapped_column(Integer, default=0)
    avg_ack_time_seconds: Mapped[Optional[float]] = mapped_column(Float)
    false_alarm_count:    Mapped[int] = mapped_column(Integer, default=0)
    genuine_count:        Mapped[int] = mapped_column(Integer, default=0)
    push_sent:            Mapped[int] = mapped_column(Integer, default=0)
    push_delivered:       Mapped[int] = mapped_column(Integer, default=0)
    created_at:           Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_daily_snapshot_date", "snapshot_date"),
    )
