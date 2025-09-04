"""SQLAlchemy models for Teacher Copilot MVP."""

from .user import School, User
from .enums import UserRole
from .assignment import Assignment, LessonPlan
from .rubric import Rubric, AssignmentType
from .submission import Submission, SubmissionModality, ExtractionStatus
from .grading import GradingSession, EvidenceAnchor, AIActionLog, GradingMode, GradingStatus, AnchorType

__all__ = [
    "User",
    "School",
    "UserRole",
    "Assignment",
    "LessonPlan",
    "AssignmentType",
    "Rubric",
    "Submission",
    "SubmissionModality",
    "ExtractionStatus",
    "GradingSession",
    "EvidenceAnchor",
    "AIActionLog",
    "GradingMode",
    "GradingStatus",
    "AnchorType",
]
