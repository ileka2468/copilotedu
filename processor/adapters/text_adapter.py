"""
Text file adapter for the content processing service.
"""

from typing import List, Dict, BinaryIO
from . import ContentAdapter, ContentProcessingError

class TextAdapter(ContentAdapter):
    """Adapter for plain text files."""
    
    @classmethod
    def supported_mime_types(cls) -> List[str]:
        return [
            'text/plain',
            'text/markdown',
            'text/csv',
            'text/tab-separated-values',
            'application/json',
            'application/xml',
            'application/yaml',
            'application/x-yaml'
        ]
    
    async def extract_text(self, file: BinaryIO, **kwargs) -> str:
        """Extract text content from a text file."""
        try:
            # Reset file pointer to start
            file.seek(0)
            # Read and decode the file content
            return file.read().decode('utf-8')
        except UnicodeDecodeError as e:
            raise ContentProcessingError(f"Failed to decode text file: {str(e)}")
        except Exception as e:
            raise ContentProcessingError(f"Error processing text file: {str(e)}")
    
    async def extract_metadata(self, file: BinaryIO) -> Dict:
        """Extract basic metadata from the text file."""
        try:
            file.seek(0, 2)  # Move to end of file
            size = file.tell()
            file.seek(0)  # Reset to start
            
            # Count lines (efficiently for large files)
            line_count = 0
            for _ in file:
                line_count += 1
            
            file.seek(0)  # Reset to start again
            
            # Get first few lines for content preview
            preview_lines = []
            for _ in range(5):  # First 5 lines
                line = file.readline()
                if not line:
                    break
                preview_lines.append(line.decode('utf-8', errors='replace').strip())
            
            file.seek(0)  # Reset to start
            
            return {
                'size_bytes': size,
                'line_count': line_count,
                'preview': '\n'.join(preview_lines),
                'encoding': 'utf-8'
            }
            
        except Exception as e:
            raise ContentProcessingError(f"Error extracting metadata: {str(e)}")
    
    async def is_valid(self, file: BinaryIO) -> bool:
        """Check if the file is a valid text file."""
        try:
            # Try to read a small chunk to check if it's valid text
            file.seek(0)
            chunk = file.read(1024)
            # Try to decode as UTF-8 to check if it's valid text
            chunk.decode('utf-8')
            return True
        except UnicodeDecodeError:
            return False
        except Exception:
            return False
        finally:
            file.seek(0)  # Reset file pointer
