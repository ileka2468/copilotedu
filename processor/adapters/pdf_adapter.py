"""
PDF file adapter for the content processing service.
"""

import io
import os
import hashlib
from pathlib import Path
from typing import List, Dict, BinaryIO, Optional

import fitz  # PyMuPDF

from . import ContentAdapter, ContentProcessingError

class PDFAdapter(ContentAdapter):
    """Adapter for PDF files."""
    
    @classmethod
    def supported_mime_types(cls) -> List[str]:
        return [
            'application/pdf',
            'application/x-pdf',
            'application/acrobat',
            'application/vnd.pdf',
            'text/pdf',
            'text/x-pdf'
        ]
    
    async def extract_text(self, file: BinaryIO, **kwargs) -> str:
        """Extract text content from a PDF file."""
        try:
            # Reset file pointer to start
            file.seek(0)
            
            # Read the PDF content
            pdf_data = file.read()
            
            # Open the PDF document
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            
            # Extract text from each page
            text_parts = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                text_parts.append(text)
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            raise ContentProcessingError(f"Error extracting text from PDF: {str(e)}")
    
    async def extract_metadata(self, file: BinaryIO) -> Dict:
        """Extract metadata from the PDF file."""
        try:
            file.seek(0)
            pdf_data = file.read()
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            
            # Get document metadata
            metadata = doc.metadata
            
            # Get page count and extract text stats for confidence + preview
            page_count = len(doc)
            preview_text = ""
            total_chars = 0
            whitespace_chars = 0
            pages_with_text = 0
            if page_count > 0:
                for i in range(page_count):
                    page = doc.load_page(i)
                    t = page.get_text("text") or ""
                    if i == 0:
                        preview_text = t[:500]
                    total_chars += len(t)
                    whitespace_chars += sum(1 for ch in t if ch.isspace())
                    if t.strip():
                        pages_with_text += 1
            
            # Generate preview thumbnails (first N pages)
            thumbnails: List[str] = []
            try:
                max_thumbs = int(os.getenv('THUMBNAILS_MAX', '3'))
                uploads_base = Path(os.getenv('UPLOADS_DIR', 'uploads')).resolve()
                thumb_dir = uploads_base / 'thumbnails' / hashlib.sha1(pdf_data).hexdigest()
                thumb_dir.mkdir(parents=True, exist_ok=True)

                pages_to_render = min(page_count, max_thumbs)
                for i in range(pages_to_render):
                    page = doc.load_page(i)
                    # 2x zoom for better clarity
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    out_name = f"page_{i+1:03d}.png"
                    out_path = thumb_dir / out_name
                    pix.save(str(out_path))
                    # store relative path from uploads base
                    thumbnails.append(str(out_path.relative_to(uploads_base)))
            except Exception:
                # Fail gracefully if rendering not possible
                thumbnails = []

            # Confidence score heuristic
            if page_count > 0:
                density = total_chars / max(1, page_count)
                density_norm = min(density / 1000.0, 1.0)
                ws_ratio = (whitespace_chars / total_chars) if total_chars > 0 else 1.0
                pages_with_text_ratio = pages_with_text / page_count
                confidence_score = max(0.0, min(1.0, 0.2 * density_norm + 0.3 * (1 - ws_ratio) + 0.5 * pages_with_text_ratio))
            else:
                confidence_score = 0.0

            return {
                'page_count': page_count,
                'author': metadata.get('author', ''),
                'title': metadata.get('title', ''),
                'subject': metadata.get('subject', ''),
                'keywords': metadata.get('keywords', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', ''),
                'preview': preview_text,
                'size_bytes': len(pdf_data),
                'preview_thumbnails': thumbnails,
                'confidence_score': confidence_score
            }
            
        except Exception as e:
            raise ContentProcessingError(f"Error extracting PDF metadata: {str(e)}")
        finally:
            file.seek(0)
    
    async def is_valid(self, file: BinaryIO) -> bool:
        """Check if the file is a valid PDF."""
        try:
            file.seek(0)
            # Check PDF magic number
            magic = file.read(4)
            file.seek(0)
            return magic == b'%PDF'
        except Exception:
            return False
        finally:
            file.seek(0)
