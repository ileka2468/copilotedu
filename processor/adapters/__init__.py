"""
Content processing adapters for different file types.

This module provides a base adapter class and specific implementations
for different file types (PDF, DOCX, PPTX, images, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, BinaryIO, Type, TypeVar, Any
from pathlib import Path
import mimetypes

class ContentProcessingError(Exception):
    """Base exception for content processing errors."""
    pass

class UnsupportedFileTypeError(ContentProcessingError):
    """Raised when a file type is not supported."""
    def __init__(self, file_type: str, message: str = ""):
        self.file_type = file_type
        self.message = message or f"Unsupported file type: {file_type}"
        super().__init__(self.message)

class InvalidFileError(ContentProcessingError):
    """Raised when a file is invalid or corrupted."""
    def __init__(self, filename: str, message: str = ""):
        self.filename = filename
        self.message = message or f"Invalid or corrupted file: {filename}"
        super().__init__(self.message)

class ContentAdapter(ABC):
    """Abstract base class for content processing adapters."""
    
    @classmethod
    @abstractmethod
    def supported_mime_types(cls) -> List[str]:
        """Return a list of MIME types this adapter can handle."""
        pass
    
    @abstractmethod
    async def extract_text(self, file: BinaryIO, **kwargs) -> str:
        """Extract text content from the file."""
        pass
    
    @abstractmethod
    async def extract_metadata(self, file: BinaryIO) -> Dict:
        """Extract metadata from the file."""
        pass
    
    @abstractmethod
    async def is_valid(self, file: BinaryIO) -> bool:
        """Check if the file is valid for this adapter."""
        pass


def get_adapter(file_path: str) -> Optional[ContentAdapter]:
    """
    Factory function to get the appropriate adapter for a file.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        An instance of the appropriate ContentAdapter subclass, or None if no adapter is found.
    """
    # Lazy import to avoid circular imports
    from .text_adapter import TextAdapter
    from .pdf_adapter import PDFAdapter
    from .docx_adapter import DocxAdapter
    from .pptx_adapter import PptxAdapter
    from .image_adapter import ImageAdapter
    
    # Get the MIME type of the file
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        return None
    
    # Try to find an adapter that supports this MIME type
    for adapter_cls in [TextAdapter, PDFAdapter, DocxAdapter, PptxAdapter, ImageAdapter]:
        if mime_type in adapter_cls.supported_mime_types():
            return adapter_cls()
    
    return None
