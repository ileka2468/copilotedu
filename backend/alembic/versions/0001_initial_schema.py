"""Initial schema

Revision ID: 0001
Revises: 
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create custom types
    user_role = postgresql.ENUM('teacher', 'admin', 'district', name='user_role')
    user_role.create(op.get_bind())
    
    submission_modality = postgresql.ENUM('text', 'pdf', 'pptx', 'image', name='submission_modality')
    submission_modality.create(op.get_bind())
    
    extraction_status = postgresql.ENUM('pending', 'processing', 'completed', 'failed', name='extraction_status')
    extraction_status.create(op.get_bind())
    
    assignment_type = postgresql.ENUM('essay', 'presentation', 'math', 'science', 'other', name='assignment_type')
    assignment_type.create(op.get_bind())
    
    grading_mode = postgresql.ENUM('copilot', 'agent', name='grading_mode')
    grading_mode.create(op.get_bind())
    
    grading_status = postgresql.ENUM('pending', 'processing', 'completed', 'failed', 'requires_review', name='grading_status')
    grading_status.create(op.get_bind())
    
    anchor_type = postgresql.ENUM('span', 'bbox', 'slide_region', 'line_range', name='anchor_type')
    anchor_type.create(op.get_bind())

    # Create schools table
    op.create_table('schools',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('district', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('role', user_role, nullable=False),
        sa.Column('school_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('preferences', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['school_id'], ['schools.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )

    # Create rubrics table
    op.create_table('rubrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('assignment_type', assignment_type, nullable=True),
        sa.Column('criteria', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('is_template', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create assignments table
    op.create_table('assignments',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('grade_level', sa.String(length=50), nullable=True),
        sa.Column('subject', sa.String(length=100), nullable=True),
        sa.Column('objectives', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('instructions', sa.Text(), nullable=True),
        sa.Column('rubric_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('standards_alignment', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
        sa.ForeignKeyConstraint(['rubric_id'], ['rubrics.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create submissions table
    op.create_table('submissions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('teacher_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('student_name', sa.String(length=255), nullable=True),
        sa.Column('assignment_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('file_path', sa.String(length=500), nullable=True),
        sa.Column('mime_type', sa.String(length=100), nullable=True),
        sa.Column('modality', submission_modality, nullable=True),
        sa.Column('canonical_text_uri', sa.String(length=500), nullable=True),
        sa.Column('extraction_status', extraction_status, nullable=True),
        sa.Column('extraction_confidence', sa.Numeric(precision=3, scale=2), nullable=True),
        sa.Column('submission_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['assignment_id'], ['assignments.id'], ),
        sa.ForeignKeyConstraint(['teacher_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create grading_sessions table
    op.create_table('grading_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('submission_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('rubric_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('mode', grading_mode, nullable=True),
        sa.Column('status', grading_status, nullable=True),
        sa.Column('overall_score', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('ai_confidence', sa.Numeric(precision=3, scale=2), nullable=True),
        sa.Column('teacher_approved', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['rubric_id'], ['rubrics.id'], ),
        sa.ForeignKeyConstraint(['submission_id'], ['submissions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create lesson_plans table
    op.create_table('lesson_plans',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('unit', sa.String(length=255), nullable=True),
        sa.Column('grade_level', sa.String(length=50), nullable=True),
        sa.Column('subject', sa.String(length=100), nullable=True),
        sa.Column('warm_up', sa.Text(), nullable=True),
        sa.Column('guided_practice', sa.Text(), nullable=True),
        sa.Column('independent_work', sa.Text(), nullable=True),
        sa.Column('assessment_method', sa.Text(), nullable=True),
        sa.Column('reflection_prompts', sa.Text(), nullable=True),
        sa.Column('standards_alignment', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create evidence_anchors table
    op.create_table('evidence_anchors',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('grading_session_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('criterion_id', sa.String(length=100), nullable=True),
        sa.Column('anchor_type', anchor_type, nullable=True),
        sa.Column('coordinates', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('evidence_text', sa.Text(), nullable=True),
        sa.Column('rationale', sa.Text(), nullable=True),
        sa.Column('confidence', sa.Numeric(precision=3, scale=2), nullable=True),
        sa.Column('created_by', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['grading_session_id'], ['grading_sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create ai_action_log table
    op.create_table('ai_action_log',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('action_type', sa.String(length=100), nullable=True),
        sa.Column('model_version', sa.String(length=100), nullable=True),
        sa.Column('prompt_version', sa.String(length=100), nullable=True),
        sa.Column('toolchain_version', sa.String(length=100), nullable=True),
        sa.Column('input_hash', sa.String(length=64), nullable=True),
        sa.Column('output_hash', sa.String(length=64), nullable=True),
        sa.Column('execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('idx_users_email', 'users', ['email'], unique=False)
    op.create_index('idx_submissions_teacher_id', 'submissions', ['teacher_id'], unique=False)
    op.create_index('idx_submissions_assignment_id', 'submissions', ['assignment_id'], unique=False)
    op.create_index('idx_grading_sessions_submission_id', 'grading_sessions', ['submission_id'], unique=False)
    op.create_index('idx_evidence_anchors_session_id', 'evidence_anchors', ['grading_session_id'], unique=False)
    op.create_index('idx_ai_action_log_session_id', 'ai_action_log', ['session_id'], unique=False)


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_ai_action_log_session_id', table_name='ai_action_log')
    op.drop_index('idx_evidence_anchors_session_id', table_name='evidence_anchors')
    op.drop_index('idx_grading_sessions_submission_id', table_name='grading_sessions')
    op.drop_index('idx_submissions_assignment_id', table_name='submissions')
    op.drop_index('idx_submissions_teacher_id', table_name='submissions')
    op.drop_index('idx_users_email', table_name='users')

    # Drop tables
    op.drop_table('ai_action_log')
    op.drop_table('evidence_anchors')
    op.drop_table('lesson_plans')
    op.drop_table('grading_sessions')
    op.drop_table('submissions')
    op.drop_table('assignments')
    op.drop_table('rubrics')
    op.drop_table('users')
    op.drop_table('schools')

    # Drop custom types
    op.execute('DROP TYPE IF EXISTS anchor_type')
    op.execute('DROP TYPE IF EXISTS grading_status')
    op.execute('DROP TYPE IF EXISTS grading_mode')
    op.execute('DROP TYPE IF EXISTS assignment_type')
    op.execute('DROP TYPE IF EXISTS extraction_status')
    op.execute('DROP TYPE IF EXISTS submission_modality')
    op.execute('DROP TYPE IF EXISTS user_role')