"""Assignment and LessonPlan models."""

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from ..database import Base


class Assignment(Base):
    """Assignment model."""
    __tablename__ = "assignments"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False)
    description = Column(Text)
    grade_level = Column(String(50))
    subject = Column(String(100))
    objectives = Column(JSON, default=[])
    instructions = Column(Text)
    rubric_id = Column(String(36), ForeignKey("rubrics.id"))
    created_by = Column(Integer, ForeignKey("app_users.id"))
    standards_alignment = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    rubric = relationship("Rubric", back_populates="assignments")
    created_by_user = relationship("app.models.user.User", back_populates="assignments")
    submissions = relationship("Submission", back_populates="assignment")

    def __repr__(self):
        return f"<Assignment(id={self.id}, title='{self.title}')>"

    @property
    def submission_count(self):
        """Get count of submissions for this assignment."""
        return len(self.submissions)


class LessonPlan(Base):
    """Lesson plan model."""
    __tablename__ = "lesson_plans"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False)
    unit = Column(String(255))
    grade_level = Column(String(50))
    subject = Column(String(100))
    warm_up = Column(Text)
    guided_practice = Column(Text)
    independent_work = Column(Text)
    assessment_method = Column(Text)
    reflection_prompts = Column(Text)
    standards_alignment = Column(JSON, default={})
    created_by = Column(Integer, ForeignKey("app_users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    created_by_user = relationship("app.models.user.User", back_populates="lesson_plans")

    def __repr__(self):
        return f"<LessonPlan(id={self.id}, title='{self.title}')>"