"""
Content Processing Service

This module provides a service for processing different types of content files
using specialized adapters for each file type. It handles file uploads, processing,
and metadata extraction in a type-safe and efficient manner.

Example:
    >>> processor = ContentProcessor()
    >>> with open('document.pdf', 'rb') as f:
    ...     result = await processor.process_file(f, 'document.pdf')
"""

import os
import logging
import mimetypes
import hashlib
from typing import Dict, Any, Optional, BinaryIO, Type, TypeVar, cast
from pathlib import Path
from contextlib import asynccontextmanager

from .adapters import (
    get_adapter,
    ContentAdapter,
    ContentProcessingError,
    UnsupportedFileTypeError,
    InvalidFileError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type variable for adapter classes
T = TypeVar('T', bound=ContentAdapter)

class ContentProcessor:
    """
    Main content processing service that uses adapters for different file types.
    
    This class provides methods to process various file types using the appropriate
    adapter, extract text and metadata, and save files with proper sanitization.
    
    Args:
        upload_dir: Directory path where uploaded files will be stored.
        max_file_size: Maximum allowed file size in bytes (default: 50MB).
        
    Attributes:
        upload_dir (Path): Directory for storing uploaded files.
        max_file_size (int): Maximum allowed file size in bytes.
    """
    
    def __init__(self, upload_dir: str = "uploads", max_file_size: int = 50 * 1024 * 1024):
        """Initialize the content processor with configuration."""
        self.upload_dir = Path(upload_dir).resolve()
        self.max_file_size = max_file_size
        
        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MIME type detection
        mimetypes.init()
        
        logger.info(f"ContentProcessor initialized with upload directory: {self.upload_dir}")
    
    @asynccontextmanager
    async def _get_file_handle(self, file: BinaryIO) -> BinaryIO:
        """Context manager to ensure proper file handle management."""
        try:
            file.seek(0)
            yield file
        finally:
            file.seek(0)
    
    async def _validate_file_size(self, file: BinaryIO) -> None:
        """Validate that the file size is within allowed limits."""
        current_pos = file.tell()
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(current_pos)  # Reset position
        
        if size > self.max_file_size:
            raise ContentProcessingError(
                f"File size {size} exceeds maximum allowed size of {self.max_file_size} bytes"
            )
    
    async def process_file(
        self,
        file: BinaryIO,
        filename: str,
        save_file: bool = True,
        **adapter_kwargs: Any
    ) -> Dict[str, Any]:
        """
        Process a file using the appropriate adapter.
        
        Args:
            file: File-like object containing the file data.
            filename: Original filename (used for extension detection).
            save_file: Whether to save the uploaded file.
            **adapter_kwargs: Additional arguments to pass to the adapter.
            
        Returns:
            Dict containing processing results with the following keys:
                - success (bool): Whether processing was successful
                - filename (str): Original filename
                - file_path (Optional[str]): Path to saved file if saved
                - content (str): Extracted text content
                - metadata (Dict[str, Any]): Extracted metadata
                - processing_time (float): Time taken to process in seconds
                - file_size (int): Size of the file in bytes
                
        Raises:
            UnsupportedFileTypeError: If no adapter is available for the file type.
            InvalidFileError: If the file is invalid or corrupted.
            ContentProcessingError: For other processing errors.
        """
        import time
        start_time = time.time()
        
        try:
            # Get file size for logging and validation
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to start
            
            logger.info(f"Processing file: {filename} (size: {file_size} bytes)")
            
            # Validate file size
            await self._validate_file_size(file)
            
            # Get the appropriate adapter for this file
            adapter = get_adapter(filename)
            if not adapter:
                mime_type, _ = mimetypes.guess_type(filename)
                raise UnsupportedFileTypeError(
                    file_type=mime_type or 'unknown',
                    message=f"No adapter available for file type: {mime_type or 'unknown'}"
                )
            
            logger.debug(f"Using adapter: {adapter.__class__.__name__} for {filename}")
            
            # Validate the file
            if not await adapter.is_valid(file):
                raise InvalidFileError(
                    filename=filename,
                    message="File is invalid or corrupted"
                )
            
            # Extract text content and metadata
            async with self._get_file_handle(file) as f:
                text = await adapter.extract_text(f, **adapter_kwargs)
            
            async with self._get_file_handle(file) as f:
                metadata = await adapter.extract_metadata(f)
            
            # Save the file if requested
            file_path = None
            if save_file:
                async with self._get_file_handle(file) as f:
                    file_path = await self._save_uploaded_file(f, filename)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare the result
            result = {
                'success': True,
                'filename': filename,
                'file_path': str(file_path.relative_to(self.upload_dir)) if file_path else None,
                'content': text,
                'metadata': metadata,
                'processing_time': round(processing_time, 4),
                'file_size': file_size,
                'file_type': mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            }
            
            logger.info(f"Successfully processed {filename} in {processing_time:.2f}s")
            return result
            
        except (UnsupportedFileTypeError, InvalidFileError, ContentProcessingError):
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error processing {filename}: {str(e)}", exc_info=True)
            raise ContentProcessingError(
                f"An unexpected error occurred while processing {filename}: {str(e)}"
            ) from e
    
    async def _generate_unique_filename(self, filename: str) -> Path:
        """Generate a unique filename to avoid conflicts."""
        safe_name = self._get_safe_filename(filename)
        file_path = self.upload_dir / safe_name
        
        # If file doesn't exist, return the path
        if not file_path.exists():
            return file_path
        
        # Add a counter suffix to make it unique
        name, ext = os.path.splitext(safe_name)
        counter = 1
        
        while True:
            new_name = f"{name}_{counter}{ext}"
            new_path = self.upload_dir / new_name
            if not new_path.exists():
                return new_path
            counter += 1
    
    async def _save_uploaded_file(self, file: BinaryIO, filename: str) -> Path:
        """
        Save an uploaded file to the upload directory with a unique name.
        
        Args:
            file: File-like object containing the file data.
            filename: Original filename.
            
        Returns:
            Path: Path to the saved file.
            
        Raises:
            ContentProcessingError: If the file cannot be saved.
        """
        try:
            # Generate a unique filename
            file_path = await self._generate_unique_filename(filename)
            
            # Ensure the upload directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the file in chunks to handle large files
            file.seek(0)
            with open(file_path, 'wb') as f:
                while chunk := file.read(8192):  # 8KB chunks
                    f.write(chunk)
            
            logger.debug(f"Saved file to {file_path}")
            return file_path
            
        except OSError as e:
            error_msg = f"Failed to save file {filename}: {str(e)}"
            logger.error(error_msg)
            raise ContentProcessingError(error_msg) from e
    
    @staticmethod
    def _get_safe_filename(filename: str) -> str:
        """
        Return a safe version of the filename.
        
        Args:
            filename: Original filename.
            
        Returns:
            str: Sanitized filename with problematic characters replaced.
            
        Example:
            >>> ContentProcessor._get_safe_filename("My Document (draft).pdf")
            'My_Document__draft_.pdf'
        """
        if not filename or not isinstance(filename, str):
            return 'unnamed_file'
        
        # Replace or remove problematic characters
        keep_chars = ('.', '_', '-')
        safe_chars = []
        
        for c in filename:
            if c.isalnum() or c in keep_chars:
                safe_chars.append(c)
            elif c.isspace() or c in ('*', '/', '\\', ':', '!', '@', '#', '$', '%', '^', '&', '(', ')', '+', '=', '[', ']', '{', '}', ';', "'", ',', '~', '`', '|', '"', '<', '>', '?'):
                safe_chars.append('_')
            # Other characters are removed
        
        safe_name = ''.join(safe_chars).strip('_.- ')
        
        # Ensure the filename is not empty and has an extension if original had one
        if not safe_name:
            return 'unnamed_file'
            
        # Ensure the filename has a reasonable length
        max_length = 255
        if len(safe_name) > max_length:
            name, ext = os.path.splitext(safe_name)
            name = name[:max_length - len(ext) - 1]
            safe_name = f"{name}{ext}"
        
        return safe_name
