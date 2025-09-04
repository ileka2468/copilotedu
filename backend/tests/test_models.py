"""Test cases for SQLAlchemy models."""

import pytest
from sqlalchemy.exc import IntegrityError
import uuid

from app.models import (
    User, School, UserRole,
    Assignment, LessonPlan,
    Rubric, AssignmentType,
    Submission, SubmissionModality, ExtractionStatus,
    GradingSession, EvidenceAnchor, AIActionLog,
    GradingMode, GradingStatus, AnchorType
)


class TestSchoolModel:
    """Test cases for School model."""

    def test_create_school(self, db_session):
        """Test creating a school."""
        school = School(
            name="Lincoln Elementary",
            district="Springfield District"
        )
        db_session.add(school)
        db_session.commit()
        
        assert school.id is not None
        assert school.name == "Lincoln Elementary"
        assert school.district == "Springfield District"
        assert school.created_at is not None

    def test_school_repr(self, sample_school):
        """Test school string representation."""
        repr_str = repr(sample_school)
        assert "School" in repr_str
        assert str(sample_school.id) in repr_str
        assert sample_school.name in repr_str


class TestUserModel:
    """Test cases for User model."""

    def test_create_user(self, db_session, sample_school):
        """Test creating a user."""
        user = User(
            email="teacher@test.com",
            name="Jane Smith",
            role=UserRole.teacher,
            school_id=sample_school.id,
            preferences={"theme": "dark"}
        )
        db_session.add(user)
        db_session.commit()
        
        assert user.id is not None
        assert user.email == "teacher@test.com"
        assert user.name == "Jane Smith"
        assert user.role == UserRole.teacher
        assert user.school_id == sample_school.id
        assert user.preferences == {"theme": "dark"}

    def test_user_unique_email(self, db_session, sample_school):
        """Test that user email must be unique."""
        user1 = User(
            email="duplicate@test.com",
            name="User One",
            role=UserRole.teacher,
            school_id=sample_school.id
        )
        user2 = User(
            email="duplicate@test.com",
            name="User Two",
            role=UserRole.teacher,
            school_id=sample_school.id
        )
        
        db_session.add(user1)
        db_session.commit()
        
        db_session.add(user2)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_user_role_properties(self, sample_user):
        """Test user role property methods."""
        assert sample_user.is_teacher is True
        assert sample_user.is_admin is False
        assert sample_user.is_district is False

    def test_user_school_relationship(self, sample_user, sample_school):
        """Test user-school relationship."""
        assert sample_user.school == sample_school
        assert sample_user in sample_school.users


class TestRubricModel:
    """Test cases for Rubric model."""

    def test_create_rubric(self, db_session, sample_user):
        """Test creating a rubric."""
        criteria = [
            {"name": "Content", "points": 50, "description": "Quality of content"},
            {"name": "Style", "points": 50, "description": "Writing style"}
        ]
        
        rubric = Rubric(
            name="Writing Rubric",
            description="Rubric for writing assignments",
            assignment_type=AssignmentType.essay,
            criteria=criteria,
            created_by=sample_user.id,
            is_template=True
        )
        db_session.add(rubric)
        db_session.commit()
        
        assert rubric.id is not None
        assert rubric.name == "Writing Rubric"
        assert rubric.assignment_type == AssignmentType.essay
        assert len(rubric.criteria) == 2
        assert rubric.is_template is True

    def test_rubric_total_points(self, sample_rubric):
        """Test rubric total points calculation."""
        assert sample_rubric.total_points == 100

    def test_rubric_criterion_count(self, sample_rubric):
        """Test rubric criterion count."""
        assert sample_rubric.criterion_count == 4

    def test_get_criterion_by_name(self, sample_rubric):
        """Test getting criterion by name."""
        thesis_criterion = sample_rubric.get_criterion_by_name("Thesis")
        assert thesis_criterion is not None
        assert thesis_criterion["points"] == 25
        
        nonexistent = sample_rubric.get_criterion_by_name("Nonexistent")
        assert nonexistent is None

    def test_validate_criteria_valid(self, sample_rubric):
        """Test criteria validation with valid criteria."""
        is_valid, message = sample_rubric.validate_criteria()
        assert is_valid is True
        assert message == "Valid criteria"

    def test_validate_criteria_invalid(self, db_session, sample_user):
        """Test criteria validation with invalid criteria."""
        # Test empty criteria
        rubric = Rubric(
            name="Invalid Rubric",
            criteria=[],
            created_by=sample_user.id
        )
        is_valid, message = rubric.validate_criteria()
        assert is_valid is False
        assert "at least one criterion" in message

        # Test missing required fields
        rubric.criteria = [{"name": "Test"}]  # Missing points and description
        is_valid, message = rubric.validate_criteria()
        assert is_valid is False
        assert "missing required field" in message


class TestAssignmentModel:
    """Test cases for Assignment model."""

    def test_create_assignment(self, db_session, sample_user, sample_rubric):
        """Test creating an assignment."""
        assignment = Assignment(
            title="Research Paper",
            description="Write a research paper on a topic of your choice",
            grade_level="10th",
            subject="English",
            objectives=["Research skills", "Citation format", "Argument development"],
            instructions="Choose a topic and write a 5-page research paper.",
            rubric_id=sample_rubric.id,
            created_by=sample_user.id,
            standards_alignment={"CCSS": ["CCSS.ELA-LITERACY.W.9-10.1"]}
        )
        db_session.add(assignment)
        db_session.commit()
        
        assert assignment.id is not None
        assert assignment.title == "Research Paper"
        assert assignment.grade_level == "10th"
        assert len(assignment.objectives) == 3

    def test_assignment_relationships(self, sample_assignment, sample_user, sample_rubric):
        """Test assignment relationships."""
        assert sample_assignment.created_by_user == sample_user
        assert sample_assignment.rubric == sample_rubric

    def test_submission_count_property(self, sample_assignment):
        """Test submission count property."""
        assert sample_assignment.submission_count == 0


class TestSubmissionModel:
    """Test cases for Submission model."""

    def test_create_submission(self, db_session, sample_user, sample_assignment):
        """Test creating a submission."""
        submission = Submission(
            teacher_id=sample_user.id,
            student_name="Alice Johnson",
            assignment_id=sample_assignment.id,
            file_path="/uploads/alice_essay.pdf",
            mime_type="application/pdf",
            modality=SubmissionModality.pdf,
            canonical_text_uri="/uploads/processed/alice_essay.txt",
            extraction_status=ExtractionStatus.completed,
            extraction_confidence=0.92,
            submission_metadata={"page_count": 3, "word_count": 750}
        )
        db_session.add(submission)
        db_session.commit()
        
        assert submission.id is not None
        assert submission.student_name == "Alice Johnson"
        assert submission.modality == SubmissionModality.pdf
        assert submission.extraction_confidence == 0.92

    def test_submission_status_properties(self, sample_submission):
        """Test submission status properties."""
        assert sample_submission.is_processed is True
        assert sample_submission.is_processing is False
        assert sample_submission.has_failed is False
        assert sample_submission.needs_processing is False

    def test_confidence_level_property(self, sample_submission):
        """Test confidence level property."""
        assert sample_submission.confidence_level == "high"

    def test_metadata_methods(self, sample_submission):
        """Test metadata getter and setter methods."""
        word_count = sample_submission.get_metadata_value("word_count")
        assert word_count == 500
        
        nonexistent = sample_submission.get_metadata_value("nonexistent", "default")
        assert nonexistent == "default"
        
        sample_submission.set_metadata_value("new_field", "new_value")
        assert sample_submission.metadata["new_field"] == "new_value"


class TestGradingSessionModel:
    """Test cases for GradingSession model."""

    def test_create_grading_session(self, db_session, sample_submission, sample_rubric):
        """Test creating a grading session."""
        session = GradingSession(
            submission_id=sample_submission.id,
            rubric_id=sample_rubric.id,
            mode=GradingMode.copilot,
            status=GradingStatus.processing,
            overall_score=78.5,
            ai_confidence=0.85,
            teacher_approved=False
        )
        db_session.add(session)
        db_session.commit()
        
        assert session.id is not None
        assert session.mode == GradingMode.copilot
        assert session.status == GradingStatus.processing
        assert session.overall_score == 78.5

    def test_grading_session_status_properties(self, sample_grading_session):
        """Test grading session status properties."""
        assert sample_grading_session.is_completed is True
        assert sample_grading_session.is_processing is False
        assert sample_grading_session.needs_review is False
        assert sample_grading_session.has_failed is False

    def test_confidence_level_property(self, sample_grading_session):
        """Test AI confidence level property."""
        assert sample_grading_session.confidence_level == "medium"

    def test_evidence_count_property(self, sample_grading_session):
        """Test evidence count property."""
        assert sample_grading_session.evidence_count == 0


class TestEvidenceAnchorModel:
    """Test cases for EvidenceAnchor model."""

    def test_create_evidence_anchor(self, db_session, sample_grading_session):
        """Test creating an evidence anchor."""
        anchor = EvidenceAnchor(
            grading_session_id=sample_grading_session.id,
            criterion_id="thesis",
            anchor_type=AnchorType.span,
            coordinates={"start": 0, "end": 50},
            evidence_text="The thesis statement is clear and arguable.",
            rationale="Strong thesis that takes a clear position.",
            confidence=0.92,
            created_by="ai"
        )
        db_session.add(anchor)
        db_session.commit()
        
        assert anchor.id is not None
        assert anchor.criterion_id == "thesis"
        assert anchor.anchor_type == AnchorType.span
        assert anchor.confidence == 0.92

    def test_evidence_anchor_properties(self, db_session, sample_grading_session):
        """Test evidence anchor properties."""
        anchor = EvidenceAnchor(
            grading_session_id=sample_grading_session.id,
            criterion_id="test",
            anchor_type=AnchorType.span,
            confidence=0.95,
            created_by="ai"
        )
        
        assert anchor.is_ai_generated is True
        assert anchor.is_teacher_generated is False
        assert anchor.confidence_level == "high"

    def test_coordinate_methods(self, db_session, sample_grading_session):
        """Test coordinate getter method."""
        anchor = EvidenceAnchor(
            grading_session_id=sample_grading_session.id,
            criterion_id="test",
            anchor_type=AnchorType.bbox,
            coordinates={"x": 100, "y": 200, "width": 300, "height": 50}
        )
        
        assert anchor.get_coordinate_value("x") == 100
        assert anchor.get_coordinate_value("nonexistent", "default") == "default"


class TestAIActionLogModel:
    """Test cases for AIActionLog model."""

    def test_create_ai_action_log(self, db_session, sample_grading_session):
        """Test creating an AI action log."""
        log = AIActionLog(
            session_id=sample_grading_session.id,
            action_type="grade_criterion",
            model_version="gpt-4-1106-preview",
            prompt_version="v1.2.0",
            toolchain_version="essay-v1.0.0",
            input_hash="abc123",
            output_hash="def456",
            execution_time_ms=1500
        )
        db_session.add(log)
        db_session.commit()
        
        assert log.id is not None
        assert log.action_type == "grade_criterion"
        assert log.model_version == "gpt-4-1106-preview"
        assert log.execution_time_ms == 1500

    def test_execution_time_seconds_property(self, db_session):
        """Test execution time in seconds property."""
        log = AIActionLog(
            action_type="test",
            execution_time_ms=2500
        )
        
        assert log.execution_time_seconds == 2.5


class TestLessonPlanModel:
    """Test cases for LessonPlan model."""

    def test_create_lesson_plan(self, db_session, sample_user):
        """Test creating a lesson plan."""
        lesson_plan = LessonPlan(
            title="Introduction to Photosynthesis",
            unit="Plant Biology",
            grade_level="7th",
            subject="Science",
            warm_up="Quick review of plant parts",
            guided_practice="Diagram photosynthesis process",
            independent_work="Complete worksheet on photosynthesis",
            assessment_method="Exit ticket with key concepts",
            reflection_prompts="What surprised you about photosynthesis?",
            standards_alignment={"NGSS": ["MS-LS1-5"]},
            created_by=sample_user.id
        )
        db_session.add(lesson_plan)
        db_session.commit()
        
        assert lesson_plan.id is not None
        assert lesson_plan.title == "Introduction to Photosynthesis"
        assert lesson_plan.unit == "Plant Biology"
        assert lesson_plan.grade_level == "7th"

    def test_lesson_plan_relationship(self, db_session, sample_user):
        """Test lesson plan user relationship."""
        lesson_plan = LessonPlan(
            title="Test Lesson",
            created_by=sample_user.id
        )
        db_session.add(lesson_plan)
        db_session.commit()
        
        assert lesson_plan.created_by_user == sample_user
        assert lesson_plan in sample_user.lesson_plans