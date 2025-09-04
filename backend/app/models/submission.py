"""Submission model."""

from sqlalchemy import Column, String, DateTime, ForeignKey, Enum as SQLEnum, Integer, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum

from ..database import Base


class SubmissionModality(enum.Enum):
    """Submission modality enumeration."""
    text = "text"
    pdf = "pdf"
    pptx = "pptx"
    image = "image"


class ExtractionStatus(enum.Enum):
    """Extraction status enumeration."""
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Submission(Base):
    """Submission model."""
    __tablename__ = "submissions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    teacher_id = Column(Integer, ForeignKey("app_users.id"))
    student_name = Column(String(255))
    assignment_id = Column(String(36), ForeignKey("assignments.id"))
    file_path = Column(String(500))
    mime_type = Column(String(100))
    modality = Column(SQLEnum(SubmissionModality))
    canonical_text_uri = Column(String(500))
    extraction_status = Column(SQLEnum(ExtractionStatus), default=ExtractionStatus.pending)
    extraction_confidence = Column(Float)
    submission_metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    teacher = relationship("app.models.user.User", back_populates="submissions")
    assignment = relationship("Assignment", back_populates="submissions")
    grading_sessions = relationship("GradingSession", back_populates="submission")

    def __repr__(self):
        return f"<Submission(id={self.id}, student_name='{self.student_name}')>"

    @property
    def is_processed(self):
        """Check if submission has been processed."""
        return self.extraction_status == ExtractionStatus.completed

    @property
    def is_processing(self):
        """Check if submission is currently being processed."""
        return self.extraction_status == ExtractionStatus.processing

    @property
    def has_failed(self):
        """Check if submission processing failed."""
        return self.extraction_status == ExtractionStatus.failed

    @property
    def needs_processing(self):
        """Check if submission needs processing."""
        return self.extraction_status == ExtractionStatus.pending

    @property
    def confidence_level(self):
        """Get confidence level as string."""
        if self.extraction_confidence is None:
            return "unknown"
        elif self.extraction_confidence >= 0.9:
            return "high"
        elif self.extraction_confidence >= 0.7:
            return "medium"
        else:
            return "low"

    def get_metadata_value(self, key: str, default=None):
        """Get a specific metadata value."""
        if not self.submission_metadata:
            return default
        return self.submission_metadata.get(key, default)

    def set_metadata_value(self, key: str, value):
        """Set a specific metadata value."""
        if not self.submission_metadata:
            self.submission_metadata = {}
        self.submission_metadata[key] = value

def _submission_metadata_getter(self):
    return self.submission_metadata or {}

def _submission_metadata_setter(self, value):
    self.submission_metadata = value or {}

# Attach a property AFTER class definition so it doesn't interfere with declarative mapping
Submission.metadata = property(_submission_metadata_getter, _submission_metadata_setter)