"""Test configuration and fixtures."""

import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db


# Use in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="session")
def engine():
    """Create test database engine."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture(scope="function")
def db_session(engine):
    """Create test database session."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    
    # Start a transaction
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    # Rollback transaction and close
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def sample_school(db_session):
    """Create a sample school for testing."""
    from app.models import School
    school = School(
        name="Test Elementary School",
        district="Test District"
    )
    db_session.add(school)
    db_session.commit()
    db_session.refresh(school)
    return school


@pytest.fixture
def sample_user(db_session, sample_school):
    """Create a sample user for testing."""
    from app.models import User, UserRole
    user = User(
        email="test@example.com",
        name="Test Teacher",
        role=UserRole.teacher,
        school_id=sample_school.id,
        preferences={"theme": "light", "notifications": True}
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def sample_rubric(db_session, sample_user):
    """Create a sample rubric for testing."""
    from app.models import Rubric, AssignmentType
    rubric = Rubric(
        name="Test Essay Rubric",
        description="A test rubric for essays",
        assignment_type=AssignmentType.essay,
        criteria=[
            {"name": "Thesis", "points": 25, "description": "Clear thesis statement"},
            {"name": "Organization", "points": 25, "description": "Logical structure"},
            {"name": "Evidence", "points": 25, "description": "Supporting evidence"},
            {"name": "Grammar", "points": 25, "description": "Proper grammar"}
        ],
        created_by=sample_user.id,
        is_template=True
    )
    db_session.add(rubric)
    db_session.commit()
    db_session.refresh(rubric)
    return rubric


@pytest.fixture
def sample_assignment(db_session, sample_user, sample_rubric):
    """Create a sample assignment for testing."""
    from app.models import Assignment
    assignment = Assignment(
        title="Test Essay Assignment",
        description="Write a 5-paragraph essay",
        grade_level="9th",
        subject="English",
        objectives=["Develop thesis", "Support arguments", "Use proper grammar"],
        instructions="Write a well-structured essay with clear thesis and supporting evidence.",
        rubric_id=sample_rubric.id,
        created_by=sample_user.id,
        standards_alignment={"CCSS": ["CCSS.ELA-LITERACY.W.9-10.1"]}
    )
    db_session.add(assignment)
    db_session.commit()
    db_session.refresh(assignment)
    return assignment


@pytest.fixture
def sample_submission(db_session, sample_user, sample_assignment):
    """Create a sample submission for testing."""
    from app.models import Submission, SubmissionModality, ExtractionStatus
    submission = Submission(
        teacher_id=sample_user.id,
        student_name="John Doe",
        assignment_id=sample_assignment.id,
        file_path="/uploads/test_essay.docx",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        modality=SubmissionModality.text,
        canonical_text_uri="/uploads/processed/test_essay.txt",
        extraction_status=ExtractionStatus.completed,
        extraction_confidence=0.95,
        submission_metadata={"word_count": 500, "page_count": 2}
    )
    db_session.add(submission)
    db_session.commit()
    db_session.refresh(submission)
    return submission


@pytest.fixture
def sample_grading_session(db_session, sample_submission, sample_rubric):
    """Create a sample grading session for testing."""
    from app.models import GradingSession, GradingMode, GradingStatus
    grading_session = GradingSession(
        submission_id=sample_submission.id,
        rubric_id=sample_rubric.id,
        mode=GradingMode.agent,
        status=GradingStatus.completed,
        overall_score=85.5,
        ai_confidence=0.88,
        teacher_approved=False
    )
    db_session.add(grading_session)
    db_session.commit()
    db_session.refresh(grading_session)
    return grading_session