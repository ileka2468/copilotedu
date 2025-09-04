# Teacher Copilot MVP

An AI-powered productivity suite for K-12 teachers that assists with grading, lesson planning, and assignment creation using large language models.

## Architecture

The system is built as a microservices architecture with the following components:

- **Frontend**: Next.js application with Tailwind CSS and shadcn/ui
- **API Gateway**: FastAPI backend service for routing and authentication
- **Grading Service**: Specialized AI-powered grading engine
- **Content Processor**: Document processing and OCR service
- **Database**: PostgreSQL with Redis for caching
- **Message Queue**: Celery with Redis broker for background tasks

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

### Environment Setup

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Update the `.env` file with your API keys:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `ANTHROPIC_API_KEY`: Your Anthropic API key
   - `CLERK_SECRET_KEY`: Your Clerk authentication key

### Running with Docker Compose

1. Build and start all services:
```bash
docker-compose up --build
```

2. The application will be available at:
   - Frontend: http://localhost:3000
   - API Gateway: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Database: localhost:5432
   - Redis: localhost:6379

### Development Setup

For local development without Docker:

1. **Database Setup**:
```bash
# Start PostgreSQL and Redis
docker-compose up postgres redis -d

# Run database migrations (when implemented)
cd backend && alembic upgrade head
```

2. **Backend Services**:
```bash
# API Gateway
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload

# Grading Service
cd grading && pip install -r requirements.txt && python -m grading.main

# Content Processor
cd processor && pip install -r requirements.txt && python -m processor.main
```

3. **Frontend**:
```bash
cd frontend && npm install && npm run dev
```

## Project Structure

```
teacher-copilot-mvp/
├── frontend/                 # Next.js frontend application
│   ├── app/                 # Next.js 13+ app directory
│   ├── components/          # Reusable UI components
│   ├── lib/                 # Utility functions
│   └── public/              # Static assets
├── backend/                 # FastAPI API Gateway
│   ├── app/                 # Application code
│   ├── alembic/             # Database migrations
│   └── tests/               # Backend tests
├── grading/                 # AI Grading Service
│   ├── toolchains/          # AI analysis toolchains
│   ├── models/              # Data models
│   └── services/            # Business logic
├── processor/               # Content Processing Service
│   ├── adapters/            # File format adapters
│   ├── ocr/                 # OCR processing
│   └── storage/             # File storage utilities
├── database/                # Database schemas and migrations
├── nginx/                   # Nginx configuration
└── docker-compose.yml       # Docker services configuration
```

## Services

### Frontend (Port 3000)
- Next.js 14 with App Router
- Tailwind CSS for styling
- shadcn/ui component library
- Real-time collaboration features
- Responsive design for mobile and desktop

### API Gateway (Port 8000)
- FastAPI with automatic OpenAPI documentation
- Authentication and authorization
- Request routing to microservices
- WebSocket support for real-time features
- File upload handling

### Grading Service (Port 8001)
- AI-powered grading engine
- Multiple LLM provider support (OpenAI, Anthropic)
- Specialized toolchains for different content types
- Evidence anchoring and annotation system
- Reliability safeguards against hallucination

### Content Processor (Port 8002)
- Multi-format document processing
- OCR for scanned documents
- Text extraction and normalization
- File validation and metadata extraction
- Batch processing capabilities

#### Thumbnails & Confidence Score
- PDF: first N pages rendered to PNG via PyMuPDF.
  - Saved under `uploads/thumbnails/<sha1>/page_###.png`
  - Metadata includes `preview_thumbnails` and `confidence_score` (0–1)
- PPTX: first N slides rendered to PNG via LibreOffice headless.
  - Saved under `uploads/thumbnails/<sha1>/slide_###.png`
  - Metadata includes `thumbnails` and `confidence_score` (0–1)

The `confidence_score` is a lightweight heuristic indicating extraction reliability. Use it to prioritize results or show warnings for low-confidence content.

## Environment Variables

Key environment variables (see `.env.example` for complete list):

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `OPENAI_API_KEY`: OpenAI API key for LLM access
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude access
- `JWT_SECRET`: Secret key for JWT token signing
- `CLERK_SECRET_KEY`: Clerk authentication service key

### Processor-specific
- `UPLOADS_DIR` (default: `uploads`): Base directory for persisted uploads and thumbnails.
- `THUMBNAILS_MAX` (default: `3`): Max pages/slides rendered for previews.
- `LIBREOFFICE_PATH` (optional; default: `soffice` on PATH): Full path to LibreOffice for PPTX thumbnails.
  - Windows example: `C:\\Program Files\\LibreOffice\\program\\soffice.exe`

The processor package auto-loads a local dotenv file on import:
- Looks for `processor/.env` first, then project root `.env`.
- Use it to set the variables above during local development and testing.

### Google OAuth Setup

To enable direct Google OAuth in the API Gateway:

1. Create OAuth 2.0 Client ID (Web) in Google Cloud Console.
   - Authorized redirect URIs:
     - `http://localhost:8000/auth/oauth/google/callback`
   - Authorized JavaScript origins (for local dev UI):
     - `http://localhost:3000`

2. Set these variables in `.env` (see `.env.example`):

```env
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/oauth/google/callback
# Optional: where the backend will redirect with tokens in URL fragment
FRONTEND_OAUTH_REDIRECT_URI=http://localhost:3000/oauth/callback
```

3. Endpoints (backend):
   - Start login: `GET /auth/oauth/google/login`
   - Callback: `GET /auth/oauth/google/callback`

Notes:
- The backend sets an HttpOnly `oauth_state` cookie and validates it on callback.
- On first login, users are created with default role id `1` (ensure roles are seeded).

## Development

### Adding New Features

1. Update the appropriate service (frontend, backend, grading, processor)
2. Add database migrations if schema changes are needed
3. Update API documentation
4. Add tests for new functionality
5. Update Docker configurations if new dependencies are added

### Testing

```bash
# Backend tests
cd backend && pytest

# Frontend tests
cd frontend && npm test

# Integration tests
docker-compose -f docker-compose.test.yml up --build
```

### Database Migrations

```bash
cd backend
alembic revision --autogenerate -m "Description of changes"
alembic upgrade head
```

## Production Deployment

For production deployment, see the Kubernetes configurations in the `/k8s` directory (to be created in later tasks).

## Contributing

1. Follow the existing code style and patterns
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all services start successfully with Docker Compose

## License

This project is licensed under the MIT License.

## Dev Plan

# Design Document

## Overview

The Teacher Copilot MVP is architected as a microservices-based platform that processes multi-modal educational content through specialized AI toolchains. The system employs a modular adapter pattern for content ingestion, a rubric-driven scoring engine with evidence anchoring, and dual interaction modes (Copilot/Agent) to accommodate different teacher workflows. The architecture prioritizes reliability, transparency, and extensibility while maintaining strict safeguards against AI hallucination and prompt injection.

## Architecture

### High-Level System Architecture

```mermaid
graph TB
    UI[Frontend - Next.js] --> API[API Gateway - FastAPI]
    API --> Auth[Authentication Service]
    API --> Grading[Grading Service]
    API --> Content[Content Generation Service]
    API --> Processing[Content Processing Service]
    
    Processing --> Adapters[Submission Adapters]
    Adapters --> TextAdapter[Text Adapter]
    Adapters --> PDFAdapter[PDF Adapter]
    Adapters --> PPTXAdapter[PPTX Adapter]
    Adapters --> OCRAdapter[OCR Adapter]
    
    Grading --> RubricEngine[Rubric Engine]
    Grading --> Toolchains[AI Toolchains]
    Grading --> Evidence[Evidence Anchoring]
    
    Toolchains --> EssayTools[Essay Analysis]
    Toolchains --> PresentationTools[Presentation Analysis]
    Toolchains --> MathTools[Math Analysis]
    
    API --> Queue[Celery Task Queue]
    Queue --> Workers[Background Workers]
    
    API --> DB[(PostgreSQL)]
    API --> Storage[File Storage - S3]
    API --> Cache[Redis Cache]
    
    Grading --> LLMRouter[LLM Provider Router]
    LLMRouter --> OpenAI[OpenAI API]
    LLMRouter --> Anthropic[Anthropic API]
```

### Service Decomposition

**Frontend Service (Next.js + Tailwind + shadcn/ui)**
- Responsive web application with real-time collaboration features
- Dual-mode interfaces for Copilot and Agent workflows
- Rich text editor with annotation overlay capabilities
- Export functionality for multiple formats (PDF, LMS integration)

**API Gateway (FastAPI)**
- RESTful API with WebSocket support for real-time features
- Request routing and load balancing
- Rate limiting and authentication middleware
- OpenAPI documentation generation

**Authentication Service**
- Google OAuth integration via Clerk or Supabase
- Role-based access control (teacher, admin, district)
- Session management and token validation

**Content Processing Service**
- Orchestrates submission ingestion through adapter pattern
- Manages file upload, validation, and storage
- Coordinates OCR, text extraction, and normalization
- Handles batch processing workflows

**Grading Service**
- Core rubric engine with criterion mapping
- AI toolchain orchestration and execution
- Evidence anchoring and annotation management
- Score calculation and validation logic

**Content Generation Service**
- Assignment and lesson plan generation
- Template management and customization
- Standards alignment and metadata enrichment
- Export format conversion

### Containerization and Deployment

The system is designed for containerized deployment using Docker and Docker Compose for development, with Kubernetes support for production scaling.

**Container Architecture:**
- `frontend`: Next.js application with nginx reverse proxy
- `api-gateway`: FastAPI application with gunicorn WSGI server
- `grading-service`: Specialized container with AI/ML dependencies
- `content-processor`: Container with OCR tools (Tesseract, PaddleOCR)
- `worker`: Celery workers for background task processing
- `postgres`: PostgreSQL database with persistent volumes
- `redis`: Redis for caching and Celery message broker
- `nginx`: Load balancer and static file serving

**Docker Compose Development Setup:**
```yaml
version: '3.8'
services:
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    environment:
      - NEXT_PUBLIC_API_URL=http://api-gateway:8000
  
  api-gateway:
    build: ./backend
    ports: ["8000:8000"]
    depends_on: [postgres, redis]
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/teachercopilot
      - REDIS_URL=redis://redis:6379
  
  grading-service:
    build: ./grading
    depends_on: [postgres, redis]
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
  
  content-processor:
    build: ./processor
    volumes: ["./uploads:/app/uploads"]
    depends_on: [postgres]
  
  worker:
    build: ./backend
    command: celery -A app.celery worker --loglevel=info
    depends_on: [postgres, redis]
  
  postgres:
    image: postgres:15
    volumes: ["postgres_data:/var/lib/postgresql/data"]
    environment:
      - POSTGRES_DB=teachercopilot
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
  
  redis:
    image: redis:7-alpine
    volumes: ["redis_data:/data"]
```

**Production Kubernetes Deployment:**
- Horizontal Pod Autoscaling for API and worker services
- Persistent Volume Claims for database and file storage
- ConfigMaps and Secrets for environment configuration
- Ingress controller for SSL termination and routing

## Components and Interfaces

### Submission Adapter System

The adapter pattern enables extensible support for multiple content types while maintaining a consistent interface for the grading engine.

```python
class SubmissionAdapter(ABC):
    @abstractmethod
    def process(self, file_path: str) -> NormalizedSubmission:
        pass
    
    @abstractmethod
    def extract_text(self) -> str:
        pass
    
    @abstractmethod
    def create_sections(self) -> List[Section]:
        pass
    
    @abstractmethod
    def generate_annotations(self) -> List[AnnotationAnchor]:
        pass

class NormalizedSubmission:
    canonical_text: str
    sections: List[Section]
    modality: SubmissionModality
    confidence: float
    per_page_images: List[str]
    extras: Dict[str, Any]
    metadata: SubmissionMetadata
```

**Text Adapter**: Processes DOCX, Google Docs, TXT files using python-docx and Google Docs API
**PDF Adapter**: Handles digital PDFs with pdfminer.six for text extraction and layout analysis
**PPTX Adapter**: Extracts slide content, speaker notes, and generates slide images using python-pptx
**OCR Adapter**: Processes scanned documents with Tesseract/PaddleOCR and confidence scoring

### Rubric Engine Architecture

```python
class RubricEngine:
    def __init__(self, toolchain_registry: ToolchainRegistry):
        self.registry = toolchain_registry
        self.evidence_tracker = EvidenceTracker()
    
    def grade_submission(self, submission: NormalizedSubmission, 
                        rubric: Rubric, mode: GradingMode) -> GradingResult:
        results = []
        for criterion in rubric.criteria:
            toolchain = self.registry.get_toolchain(criterion.type)
            result = toolchain.analyze(submission, criterion)
            evidence = self.evidence_tracker.anchor_evidence(result, submission)
            results.append(CriterionResult(criterion, result, evidence))
        
        return GradingResult(results, self.calculate_overall_score(results))
```

### AI Toolchain System

Modular toolchains provide specialized analysis for different assignment types and rubric criteria:

**Essay Toolchain**
- `ThesisAnalyzer`: Evaluates thesis clarity and strength
- `ParagraphStructureAnalyzer`: Assesses topic sentences and organization
- `CoherenceAnalyzer`: Checks logical flow and transitions
- `GrammarAnalyzer`: Identifies grammatical errors and style issues

**Presentation Toolchain**
- `SlideStructureAnalyzer`: Evaluates presentation organization
- `ContentAccuracyAnalyzer`: Fact-checks slide content
- `DesignAnalyzer`: Assesses visual design and readability
- `NarrativeFlowAnalyzer`: Checks story progression across slides

**Math Toolchain**
- `EquationParser`: Converts math notation to symbolic form
- `StepValidator`: Verifies algebraic manipulations
- `SolutionChecker`: Validates final answers and units
- `WorkShowingAnalyzer`: Assesses problem-solving process

### Evidence Anchoring System

```python
class EvidenceAnchor:
    id: str
    submission_id: str
    criterion_id: str
    anchor_type: AnchorType  # span, bbox, slide_region, line_range
    coordinates: AnchorCoordinates
    evidence_text: str
    confidence: float
    rationale: str

class AnnotationRenderer:
    def render_text_highlights(self, anchors: List[EvidenceAnchor]) -> str:
        # Generate HTML with highlighted spans
        
    def render_pdf_annotations(self, anchors: List[EvidenceAnchor]) -> bytes:
        # Create annotated PDF with highlights and comments
        
    def render_slide_overlays(self, anchors: List[EvidenceAnchor]) -> List[str]:
        # Generate slide images with bounding box overlays
```

## Data Models

### Core Entities

```sql
-- Users and Authentication
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    role user_role NOT NULL,
    school_id UUID REFERENCES schools(id),
    preferences JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Submissions and Content
CREATE TABLE submissions (
    id UUID PRIMARY KEY,
    teacher_id UUID REFERENCES users(id),
    student_name VARCHAR(255),
    assignment_id UUID REFERENCES assignments(id),
    file_path VARCHAR(500),
    mime_type VARCHAR(100),
    modality submission_modality,
    canonical_text_uri VARCHAR(500),
    extraction_status extraction_status,
    extraction_confidence DECIMAL(3,2),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Rubrics and Grading
CREATE TABLE rubrics (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    assignment_type assignment_type,
    criteria JSONB NOT NULL,
    created_by UUID REFERENCES users(id),
    is_template BOOLEAN DEFAULT FALSE
);

CREATE TABLE grading_sessions (
    id UUID PRIMARY KEY,
    submission_id UUID REFERENCES submissions(id),
    rubric_id UUID REFERENCES rubrics(id),
    mode grading_mode,
    status grading_status,
    overall_score DECIMAL(5,2),
    ai_confidence DECIMAL(3,2),
    teacher_approved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Evidence and Annotations
CREATE TABLE evidence_anchors (
    id UUID PRIMARY KEY,
    grading_session_id UUID REFERENCES grading_sessions(id),
    criterion_id VARCHAR(100),
    anchor_type anchor_type,
    coordinates JSONB,
    evidence_text TEXT,
    rationale TEXT,
    confidence DECIMAL(3,2),
    created_by VARCHAR(50) -- 'ai' or 'teacher'
);

-- AI Action Logging
CREATE TABLE ai_action_log (
    id UUID PRIMARY KEY,
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
```

### Assignment and Content Generation

```sql
CREATE TABLE assignments (
    id UUID PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    grade_level VARCHAR(50),
    subject VARCHAR(100),
    objectives JSONB,
    instructions TEXT,
    rubric_id UUID REFERENCES rubrics(id),
    created_by UUID REFERENCES users(id),
    standards_alignment JSONB
);

CREATE TABLE lesson_plans (
    id UUID PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    unit VARCHAR(255),
    grade_level VARCHAR(50),
    subject VARCHAR(100),
    warm_up TEXT,
    guided_practice TEXT,
    independent_work TEXT,
    assessment_method TEXT,
    reflection_prompts TEXT,
    standards_alignment JSONB,
    created_by UUID REFERENCES users(id)
);
```

## Error Handling

### Graceful Degradation Strategy

**Content Processing Failures**
- OCR confidence below threshold → Request manual review
- File corruption or unsupported format → Clear error message with format requirements
- Large file processing timeout → Chunked processing with progress indicators

**AI Service Failures**
- LLM API unavailable → Fallback to cached responses or alternative provider
- Toolchain execution error → Log error, continue with available criteria
- Evidence anchoring failure → Provide scores without visual anchoring

**Data Integrity Issues**
- Rubric-submission mismatch → Validation warnings before grading
- Missing required fields → Form validation with specific field guidance
- Concurrent editing conflicts → Optimistic locking with merge conflict resolution

### Reliability Safeguards

```python
class ReliabilityGuard:
    def validate_evidence_requirement(self, result: AnalysisResult) -> bool:
        if not result.evidence_anchors:
            raise InsufficientEvidenceError("Score requires evidence anchoring")
        return True
    
    def detect_prompt_injection(self, content: str) -> bool:
        injection_patterns = [
            r"ignore.*previous.*instructions",
            r"system.*prompt.*override",
            r"act.*as.*different.*assistant"
        ]
        return any(re.search(pattern, content, re.IGNORECASE) 
                  for pattern in injection_patterns)
    
    def verify_content_authenticity(self, text: str) -> AuthenticitySignals:
        return AuthenticitySignals(
            perplexity=calculate_perplexity(text),
            burstiness=calculate_burstiness(text),
            ai_likelihood=detect_ai_generation(text)
        )
```

## Testing Strategy

### Unit Testing
- Individual adapter functionality with sample files
- Toolchain analysis accuracy with known-good examples
- Evidence anchoring precision with coordinate validation
- Rubric engine scoring consistency across multiple runs

### Integration Testing
- End-to-end grading workflows for each submission type
- LMS export format validation
- Multi-modal content processing pipelines
- Real-time collaboration features in Copilot mode

### Performance Testing
- Concurrent grading session handling
- Large file processing (50MB PPTX, 200-page PDFs)
- Batch processing throughput optimization
- Database query performance under load

### Security Testing
- Prompt injection attack vectors
- File upload security validation
- Authentication and authorization flows
- Data encryption and privacy compliance

### Educational Accuracy Testing
- Rubric scoring alignment with human graders
- Subject-matter expert validation of AI analysis
- Cross-cultural and linguistic bias detection
- Standards alignment accuracy verification

The testing strategy emphasizes both technical reliability and educational validity, ensuring the system meets the high standards required for student assessment tools.