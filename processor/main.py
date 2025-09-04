import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List
import io

from .service import ContentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Teacher Copilot Content Processor",
    description="Content processing service for Teacher Copilot MVP",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize content processor
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
processor = ContentProcessor(upload_dir=UPLOAD_DIR)

# Ensure upload directory exists
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Teacher Copilot Content Processor",
        "status": "running",
        "version": "0.1.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}

@app.post("/process")
async def process_file(
    file: UploadFile = File(...),
    save_file: bool = True,
    ocr_language: str = "eng"
):
    """
    Process an uploaded file using the appropriate adapter.
    
    Args:
        file: The file to process
        save_file: Whether to save the uploaded file
        ocr_language: Language for OCR processing (default: 'eng')
        
    Returns:
        Processing results including extracted text and metadata
    """
    try:
        # Read file content
        contents = await file.read()
        
        # Create a file-like object
        file_obj = io.BytesIO(contents)
        
        # Process the file
        result = await processor.process_file(
            file=file_obj,
            filename=file.filename,
            save_file=save_file,
            lang=ocr_language
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process file: {str(e)}"
        )
    finally:
        await file.close()

@app.post("/process/batch")
async def process_files(
    files: List[UploadFile] = File(...),
    save_files: bool = True,
    ocr_language: str = "eng"
):
    """
    Process multiple files in a batch.
    
    Args:
        files: List of files to process
        save_files: Whether to save the uploaded files
        ocr_language: Language for OCR processing (default: 'eng')
        
    Returns:
        List of processing results
    """
    results = []
    
    for file in files:
        try:
            # Read file content
            contents = await file.read()
            
            # Create a file-like object
            file_obj = io.BytesIO(contents)
            
            # Process the file
            result = await processor.process_file(
                file=file_obj,
                filename=file.filename,
                save_file=save_files,
                lang=ocr_language
            )
            
            results.append({
                "filename": file.filename,
                "success": True,
                "result": result
            })
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
            
        finally:
            await file.close()
    
    return {
        "processed": len([r for r in results if r["success"]]),
        "failed": len([r for r in results if not r["success"]]),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Teacher Copilot Content Processor")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )