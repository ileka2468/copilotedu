-- Teacher Copilot MVP Database Initialization

-- Create custom types
CREATE TYPE user_role AS ENUM ('teacher', 'admin', 'district');
CREATE TYPE submission_modality AS ENUM ('text', 'pdf', 'pptx', 'image');
CREATE TYPE extraction_status AS ENUM ('pending', 'processing', 'completed', 'failed');
CREATE TYPE assignment_type AS ENUM ('essay', 'presentation', 'math', 'science', 'other');
CREATE TYPE grading_mode AS ENUM ('copilot', 'agent');
CREATE TYPE grading_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'requires_review');
CREATE TYPE anchor_type AS ENUM ('span', 'bbox', 'slide_region', 'line_range');

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Schools table
CREATE TABLE schools (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    district VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Users and Authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    role user_role NOT NULL DEFAULT 'teacher',
    school_id UUID REFERENCES schools(id),
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Assignments
CREATE TABLE assignments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    grade_level VARCHAR(50),
    subject VARCHAR(100),
    objectives JSONB DEFAULT '[]',
    instructions TEXT,
    rubric_id UUID,
    created_by UUID REFERENCES users(id),
    standards_alignment JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Rubrics and Grading
CREATE TABLE rubrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    assignment_type assignment_type,
    criteria JSONB NOT NULL DEFAULT '[]',
    created_by UUID REFERENCES users(id),
    is_template BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Add foreign key constraint for assignments
ALTER TABLE assignments ADD CONSTRAINT fk_assignment_rubric 
    FOREIGN KEY (rubric_id) REFERENCES rubrics(id);

-- Submissions and Content
CREATE TABLE submissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    teacher_id UUID REFERENCES users(id),
    student_name VARCHAR(255),
    assignment_id UUID REFERENCES assignments(id),
    file_path VARCHAR(500),
    mime_type VARCHAR(100),
    modality submission_modality,
    canonical_text_uri VARCHAR(500),
    extraction_status extraction_status DEFAULT 'pending',
    extraction_confidence DECIMAL(3,2),
    submission_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Grading Sessions
CREATE TABLE grading_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    submission_id UUID REFERENCES submissions(id),
    rubric_id UUID REFERENCES rubrics(id),
    mode grading_mode,
    status grading_status DEFAULT 'pending',
    overall_score DECIMAL(5,2),
    ai_confidence DECIMAL(3,2),
    teacher_approved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Evidence and Annotations
CREATE TABLE evidence_anchors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    grading_session_id UUID REFERENCES grading_sessions(id),
    criterion_id VARCHAR(100),
    anchor_type anchor_type,
    coordinates JSONB DEFAULT '{}',
    evidence_text TEXT,
    rationale TEXT,
    confidence DECIMAL(3,2),
    created_by VARCHAR(50) DEFAULT 'ai',
    created_at TIMESTAMP DEFAULT NOW()
);

-- AI Action Logging
CREATE TABLE ai_action_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID,
    action_type VARCHAR(100),
    model_version VARCHAR(100),
    prompt_version VARCHAR(100),
    toolchain_version VARCHAR(100),
    input_hash VARCHAR(64),
    output_hash VARCHAR(64),
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Lesson Plans
CREATE TABLE lesson_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    unit VARCHAR(255),
    grade_level VARCHAR(50),
    subject VARCHAR(100),
    warm_up TEXT,
    guided_practice TEXT,
    independent_work TEXT,
    assessment_method TEXT,
    reflection_prompts TEXT,
    standards_alignment JSONB DEFAULT '{}',
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_submissions_teacher_id ON submissions(teacher_id);
CREATE INDEX idx_submissions_assignment_id ON submissions(assignment_id);
CREATE INDEX idx_grading_sessions_submission_id ON grading_sessions(submission_id);
CREATE INDEX idx_evidence_anchors_session_id ON evidence_anchors(grading_session_id);
CREATE INDEX idx_ai_action_log_session_id ON ai_action_log(session_id);

-- Insert sample data for development
INSERT INTO schools (name, district) VALUES 
    ('Lincoln Elementary', 'Springfield District'),
    ('Washington High School', 'Springfield District');

INSERT INTO users (email, name, role, school_id) VALUES 
    ('teacher@example.com', 'Jane Smith', 'teacher', (SELECT id FROM schools WHERE name = 'Lincoln Elementary')),
    ('admin@example.com', 'John Doe', 'admin', (SELECT id FROM schools WHERE name = 'Washington High School'));

-- Create a sample rubric
INSERT INTO rubrics (name, description, assignment_type, criteria, created_by, is_template) VALUES 
    ('Essay Grading Rubric', 'Standard 5-paragraph essay rubric', 'essay', 
     '[{"name": "Thesis Statement", "points": 25, "description": "Clear and arguable thesis"}, 
       {"name": "Organization", "points": 25, "description": "Logical structure and flow"}, 
       {"name": "Evidence", "points": 25, "description": "Supporting details and examples"}, 
       {"name": "Grammar", "points": 25, "description": "Proper grammar and mechanics"}]',
     (SELECT id FROM users WHERE email = 'teacher@example.com'), true);