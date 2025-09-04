"""User and School models."""

from sqlalchemy import Column, String, DateTime, ForeignKey, Enum as SQLEnum, JSON, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from ..database import Base
from .enums import UserRole


class School(Base):
    """School model."""
    __tablename__ = "schools"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    district = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    users = relationship("app.models.user.User", back_populates="school")

    def __repr__(self):
        return f"<School(id={self.id}, name='{self.name}')>"


class User(Base):
    """Application user model for domain tests (separate from auth user)."""
    __tablename__ = "app_users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=True)
    role = Column(SQLEnum(UserRole), nullable=False)
    school_id = Column(String(36), ForeignKey("schools.id"), nullable=True)
    preferences = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    school = relationship("app.models.user.School", back_populates="users")
    assignments = relationship("app.models.assignment.Assignment", back_populates="created_by_user")
    lesson_plans = relationship("app.models.assignment.LessonPlan", back_populates="created_by_user")
    submissions = relationship("app.models.submission.Submission", back_populates="teacher")

    @property
    def is_teacher(self) -> bool:
        return self.role == UserRole.teacher

    @property
    def is_admin(self) -> bool:
        return self.role == UserRole.admin

    @property
    def is_district(self) -> bool:
        return self.role == UserRole.district
