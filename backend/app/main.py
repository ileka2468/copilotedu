from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os
import logging
from contextlib import asynccontextmanager

import os
from .database import get_db, check_database_connection, create_tables
from .auth import auth_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler replacing deprecated startup/shutdown events."""
    logger.info("Starting up Teacher Copilot API Gateway...")
    # Strict DB connectivity check in production; only skip during pytest
    if os.getenv("PYTEST_CURRENT_TEST") is not None:
        logger.info("Skipping DB connectivity check during tests")
    else:
        if check_database_connection():
            logger.info("Database connection successful")
        else:
            logger.error("Database connection failed")
            raise Exception("Cannot connect to database")
    # Startup complete
    yield
    # Shutdown
    logger.info("Shutting down Teacher Copilot API Gateway...")

app = FastAPI(
    title="Teacher Copilot API Gateway",
    description="API Gateway for Teacher Copilot MVP",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Teacher Copilot API Gateway", "version": "0.1.0"}

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint with database connectivity test."""
    try:
        from sqlalchemy import text
        # Test database connection
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "version": "0.1.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "version": "0.1.0"
        }

@app.get("/models/info")
async def models_info():
    """Get information about available models."""
    return {
        "models": [
            "User", "School", "Assignment", "LessonPlan", 
            "Rubric", "Submission", "GradingSession", 
            "EvidenceAnchor", "AIActionLog"
        ],
        "enums": [
            "UserRole", "SubmissionModality", "ExtractionStatus",
            "AssignmentType", "GradingMode", "GradingStatus", "AnchorType"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)