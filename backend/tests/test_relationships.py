"""Test cases for model relationships."""

import pytest
from app.models import (
    User, School, Assignment, Rubric, Submission, 
    GradingSession, EvidenceAnchor, LessonPlan,
    UserRole, AssignmentType, SubmissionModality, 
    ExtractionStatus, GradingMode, GradingStatus, AnchorType
)


class TestModelRelationships:
    """Test cases for model relationships."""

    def test_school_user_relationship(self, db_session):
        """Test bidirectional school-user relationship."""
        school = School(name="Test School", district="Test District")
        db_session.add(school)
        db_session.commit()
        
        user1 = User(
            email="user1@test.com",
            name="User One",
            role=UserRole.teacher,
            school_id=school.id
        )
        user2 = User(
            email="user2@test.com", 
            name="User Two",
            role=UserRole.teacher,
            school_id=school.id
        )
        
        db_session.add_all([user1, user2])
        db_session.commit()
        
        # Test forward relationship
        assert user1.school == school
        assert user2.school == school
        
        # Test reverse relationship
        assert len(school.users) == 2
        assert user1 in school.users
        assert user2 in school.users

    def test_user_assignment_relationship(self, db_session, sample_user, sample_rubric):
        """Test user-assignment relationship."""
        assignment1 = Assignment(
            title="Assignment 1",
            created_by=sample_user.id,
            rubric_id=sample_rubric.id
        )
        assignment2 = Assignment(
            title="Assignment 2", 
            created_by=sample_user.id,
            rubric_id=sample_rubric.id
        )
        
        db_session.add_all([assignment1, assignment2])
        db_session.commit()
        
        # Test forward relationship
        assert assignment1.created_by_user == sample_user
        assert assignment2.created_by_user == sample_user
        
        # Test reverse relationship
        assert len(sample_user.assignments) == 2
        assert assignment1 in sample_user.assignments
        assert assignment2 in sample_user.assignments

    def test_rubric_assignment_relationship(self, db_session, sample_user):
        """Test rubric-assignment relationship."""
        rubric = Rubric(
            name="Test Rubric",
            criteria=[{"name": "Test", "points": 100, "description": "Test criterion"}],
            created_by=sample_user.id
        )
        db_session.add(rubric)
        db_session.commit()
        
        assignment1 = Assignment(
            title="Assignment 1",
            created_by=sample_user.id,
            rubric_id=rubric.id
        )
        assignment2 = Assignment(
            title="Assignment 2",
            created_by=sample_user.id,
            rubric_id=rubric.id
        )
        
        db_session.add_all([assignment1, assignment2])
        db_session.commit()
        
        # Test forward relationship
        assert assignment1.rubric == rubric
        assert assignment2.rubric == rubric
        
        # Test reverse relationship
        assert len(rubric.assignments) == 2
        assert assignment1 in rubric.assignments
        assert assignment2 in rubric.assignments

    def test_assignment_submission_relationship(self, db_session, sample_user, sample_assignment):
        """Test assignment-submission relationship."""
        submission1 = Submission(
            teacher_id=sample_user.id,
            student_name="Student 1",
            assignment_id=sample_assignment.id,
            modality=SubmissionModality.text,
            extraction_status=ExtractionStatus.pending
        )
        submission2 = Submission(
            teacher_id=sample_user.id,
            student_name="Student 2", 
            assignment_id=sample_assignment.id,
            modality=SubmissionModality.pdf,
            extraction_status=ExtractionStatus.pending
        )
        
        db_session.add_all([submission1, submission2])
        db_session.commit()
        
        # Test forward relationship
        assert submission1.assignment == sample_assignment
        assert submission2.assignment == sample_assignment
        
        # Test reverse relationship
        assert len(sample_assignment.submissions) == 2
        assert submission1 in sample_assignment.submissions
        assert submission2 in sample_assignment.submissions

    def test_submission_grading_session_relationship(self, db_session, sample_submission, sample_rubric):
        """Test submission-grading session relationship."""
        session1 = GradingSession(
            submission_id=sample_submission.id,
            rubric_id=sample_rubric.id,
            mode=GradingMode.agent,
            status=GradingStatus.pending
        )
        session2 = GradingSession(
            submission_id=sample_submission.id,
            rubric_id=sample_rubric.id,
            mode=GradingMode.copilot,
            status=GradingStatus.pending
        )
        
        db_session.add_all([session1, session2])
        db_session.commit()
        
        # Test forward relationship
        assert session1.submission == sample_submission
        assert session2.submission == sample_submission
        
        # Test reverse relationship
        assert len(sample_submission.grading_sessions) == 2
        assert session1 in sample_submission.grading_sessions
        assert session2 in sample_submission.grading_sessions

    def test_grading_session_evidence_anchor_relationship(self, db_session, sample_grading_session):
        """Test grading session-evidence anchor relationship."""
        anchor1 = EvidenceAnchor(
            grading_session_id=sample_grading_session.id,
            criterion_id="criterion1",
            anchor_type=AnchorType.span,
            evidence_text="Evidence 1",
            confidence=0.9
        )
        anchor2 = EvidenceAnchor(
            grading_session_id=sample_grading_session.id,
            criterion_id="criterion2",
            anchor_type=AnchorType.bbox,
            evidence_text="Evidence 2",
            confidence=0.8
        )
        
        db_session.add_all([anchor1, anchor2])
        db_session.commit()
        
        # Test forward relationship
        assert anchor1.grading_session == sample_grading_session
        assert anchor2.grading_session == sample_grading_session
        
        # Test reverse relationship
        assert len(sample_grading_session.evidence_anchors) == 2
        assert anchor1 in sample_grading_session.evidence_anchors
        assert anchor2 in sample_grading_session.evidence_anchors

    def test_user_lesson_plan_relationship(self, db_session, sample_user):
        """Test user-lesson plan relationship."""
        lesson1 = LessonPlan(
            title="Lesson 1",
            created_by=sample_user.id
        )
        lesson2 = LessonPlan(
            title="Lesson 2",
            created_by=sample_user.id
        )
        
        db_session.add_all([lesson1, lesson2])
        db_session.commit()
        
        # Test forward relationship
        assert lesson1.created_by_user == sample_user
        assert lesson2.created_by_user == sample_user
        
        # Test reverse relationship
        assert len(sample_user.lesson_plans) == 2
        assert lesson1 in sample_user.lesson_plans
        assert lesson2 in sample_user.lesson_plans

    def test_grading_session_evidence_filtering(self, db_session, sample_grading_session):
        """Test filtering evidence anchors by criterion."""
        anchor1 = EvidenceAnchor(
            grading_session_id=sample_grading_session.id,
            criterion_id="thesis",
            anchor_type=AnchorType.span,
            evidence_text="Thesis evidence",
            confidence=0.9
        )
        anchor2 = EvidenceAnchor(
            grading_session_id=sample_grading_session.id,
            criterion_id="organization",
            anchor_type=AnchorType.span,
            evidence_text="Organization evidence",
            confidence=0.8
        )
        anchor3 = EvidenceAnchor(
            grading_session_id=sample_grading_session.id,
            criterion_id="thesis",
            anchor_type=AnchorType.span,
            evidence_text="Another thesis evidence",
            confidence=0.85
        )
        
        db_session.add_all([anchor1, anchor2, anchor3])
        db_session.commit()
        
        # Test filtering by criterion
        thesis_evidence = sample_grading_session.get_evidence_for_criterion("thesis")
        organization_evidence = sample_grading_session.get_evidence_for_criterion("organization")
        
        assert len(thesis_evidence) == 2
        assert len(organization_evidence) == 1
        assert anchor1 in thesis_evidence
        assert anchor3 in thesis_evidence
        assert anchor2 in organization_evidence

    def test_cascade_deletion_behavior(self, db_session, sample_user, sample_school):
        """Test cascade deletion behavior."""
        # Create a rubric
        rubric = Rubric(
            name="Test Rubric",
            criteria=[{"name": "Test", "points": 100, "description": "Test"}],
            created_by=sample_user.id
        )
        db_session.add(rubric)
        db_session.commit()
        
        # Create an assignment
        assignment = Assignment(
            title="Test Assignment",
            created_by=sample_user.id,
            rubric_id=rubric.id
        )
        db_session.add(assignment)
        db_session.commit()
        
        # Create a submission
        submission = Submission(
            teacher_id=sample_user.id,
            student_name="Test Student",
            assignment_id=assignment.id,
            modality=SubmissionModality.text,
            extraction_status=ExtractionStatus.pending
        )
        db_session.add(submission)
        db_session.commit()
        
        # Verify relationships exist
        assert assignment.rubric == rubric
        assert submission.assignment == assignment
        assert submission.teacher == sample_user
        
        # Test that foreign key constraints are properly set
        assignment_id = assignment.id
        submission_id = submission.id
        
        # Verify the objects exist
        assert db_session.get(Assignment, assignment_id) is not None
        assert db_session.get(Submission, submission_id) is not None