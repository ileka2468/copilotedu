"""Rubric model."""


from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Boolean, Enum as SQLEnum, Integer, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum

from ..database import Base


class AssignmentType(enum.Enum):
    """Assignment type enumeration."""
    essay = "essay"
    presentation = "presentation"
    math = "math"
    science = "science"
    other = "other"


class Rubric(Base):
    """Rubric model."""
    __tablename__ = "rubrics"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    assignment_type = Column(SQLEnum(AssignmentType))
    criteria = Column(JSON, nullable=False, default=[])
    created_by = Column(Integer, ForeignKey("app_users.id"))
    is_template = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    created_by_user = relationship("app.models.user.User")
    assignments = relationship("Assignment", back_populates="rubric")
    grading_sessions = relationship("GradingSession", back_populates="rubric")

    def __repr__(self):
        return f"<Rubric(id={self.id}, name='{self.name}')>"

    @property
    def total_points(self):
        """Calculate total points for this rubric."""
        if not self.criteria:
            return 0
        return sum(criterion.get('points', 0) for criterion in self.criteria)

    @property
    def criterion_count(self):
        """Get count of criteria in this rubric."""
        return len(self.criteria) if self.criteria else 0

    def get_criterion_by_name(self, name: str):
        """Get a specific criterion by name."""
        if not self.criteria:
            return None
        for criterion in self.criteria:
            if criterion.get('name') == name:
                return criterion
        return None

    def validate_criteria(self):
        """Validate rubric criteria structure."""
        if not self.criteria:
            return False, "Rubric must have at least one criterion"
        
        for i, criterion in enumerate(self.criteria):
            if not isinstance(criterion, dict):
                return False, f"Criterion {i} must be a dictionary"
            
            required_fields = ['name', 'points', 'description']
            for field in required_fields:
                if field not in criterion:
                    return False, f"Criterion {i} missing required field: {field}"
            
            if not isinstance(criterion['points'], (int, float)) or criterion['points'] <= 0:
                return False, f"Criterion {i} points must be a positive number"
        
        return True, "Valid criteria"