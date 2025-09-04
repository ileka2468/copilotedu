"""
DOCX file adapter for the content processing service.
"""

import io
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, BinaryIO
from . import ContentAdapter, ContentProcessingError

class DocxAdapter(ContentAdapter):
    """Adapter for DOCX files."""
    
    @classmethod
    def supported_mime_types(cls) -> List[str]:
        return [
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.ms-word.document.macroenabled.12',
            'application/msword',
            'application/vnd.ms-word',
            'application/word',
            'application/x-msword'
        ]
    
    async def extract_text(self, file: BinaryIO, **kwargs) -> str:
        """Extract text content from a DOCX file."""
        try:
            file.seek(0)
            
            # Read the file content
            docx_data = file.read()
            
            # Create an in-memory file-like object
            docx_io = io.BytesIO(docx_data)
            
            # Extract text from the DOCX file
            text = self._extract_text_from_docx(docx_io)
            
            return text
            
        except Exception as e:
            raise ContentProcessingError(f"Error extracting text from DOCX: {str(e)}")
    
    def _extract_text_from_docx(self, docx_io):
        """Helper method to extract text from a DOCX file."""
        text_parts = []
        
        # Open the DOCX file as a zip archive
        with zipfile.ZipFile(docx_io) as docx_zip:
            # Find the main document part (usually 'word/document.xml')
            if 'word/document.xml' in docx_zip.namelist():
                with docx_zip.open('word/document.xml') as doc_file:
                    # Parse the XML content
                    tree = ET.parse(doc_file)
                    root = tree.getroot()
                    
                    # Define the XML namespaces
                    namespaces = {
                        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
                        'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
                        'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
                        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
                    }
                    
                    # Find all paragraphs
                    paragraphs = root.findall('.//w:p', namespaces)
                    for para in paragraphs:
                        # Extract text from each run in the paragraph
                        para_text = ''
                        for run in para.findall('.//w:t', namespaces):
                            if run.text:
                                para_text += run.text
                        
                        # Add paragraph text if not empty
                        if para_text.strip():
                            text_parts.append(para_text)
        
        return '\n\n'.join(text_parts)
    
    async def extract_metadata(self, file: BinaryIO) -> Dict:
        """Extract metadata from the DOCX file."""
        try:
            file.seek(0)
            docx_data = file.read()
            docx_io = io.BytesIO(docx_data)
            
            metadata = {}
            
            # Open the DOCX file as a zip archive
            with zipfile.ZipFile(docx_io) as docx_zip:
                # Extract core properties
                if 'docProps/core.xml' in docx_zip.namelist():
                    with docx_zip.open('docProps/core.xml') as core_file:
                        tree = ET.parse(core_file)
                        root = tree.getroot()
                        
                        # Define the XML namespaces
                        namespaces = {
                            'cp': 'http://schemas.openxmlformats.org/package/2006/metadata/core-properties',
                            'dc': 'http://purl.org/dc/elements/1.1/',
                            'dcterms': 'http://purl.org/dc/terms/',
                            'dcmitype': 'http://purl.org/dc/dcmitype/',
                            'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
                        }
                        
                        # Extract standard properties
                        metadata['title'] = self._get_xml_text(root, 'dc:title', namespaces)
                        metadata['subject'] = self._get_xml_text(root, 'dc:subject', namespaces)
                        metadata['creator'] = self._get_xml_text(root, 'dc:creator', namespaces)
                        metadata['keywords'] = self._get_xml_text(root, 'cp:keywords', namespaces)
                        metadata['description'] = self._get_xml_text(root, 'dc:description', namespaces)
                        metadata['last_modified_by'] = self._get_xml_text(root, 'cp:lastModifiedBy', namespaces)
                        metadata['revision'] = self._get_xml_text(root, 'cp:revision', namespaces)
                        
                        # Extract creation and modification dates
                        created = self._get_xml_text(root, 'dcterms:created', namespaces)
                        modified = self._get_xml_text(root, 'dcterms:modified', namespaces)
                        
                        if created:
                            metadata['created'] = created
                        if modified:
                            metadata['modified'] = modified
            
            # Get preview text
            preview_text = ""
            try:
                text = await self.extract_text(file)
                preview_text = text[:500]  # First 500 characters
            except:
                pass
            
            metadata.update({
                'size_bytes': len(docx_data),
                'preview': preview_text,
                'file_type': 'DOCX'
            })
            
            return metadata
            
        except Exception as e:
            raise ContentProcessingError(f"Error extracting DOCX metadata: {str(e)}")
        finally:
            file.seek(0)
    
    def _get_xml_text(self, element, xpath, namespaces):
        """Helper method to safely get text from XML element."""
        result = element.find(xpath, namespaces)
        return result.text if result is not None else ""
    
    async def is_valid(self, file: BinaryIO) -> bool:
        """Check if the file is a valid DOCX file."""
        try:
            file.seek(0)
            # Check for DOCX magic number (PK header for ZIP)
            magic = file.read(4)
            file.seek(0)
            
            if magic != b'PK\x03\x04':
                return False
                
            # Try to open as a ZIP file and check for required DOCX files
            with zipfile.ZipFile(io.BytesIO(file.read())) as zipf:
                required_files = ['[Content_Types].xml', '_rels/.rels', 'word/document.xml']
                return all(f in zipf.namelist() for f in required_files)
                
        except Exception:
            return False
        finally:
            file.seek(0)
