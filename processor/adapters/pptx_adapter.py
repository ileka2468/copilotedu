"""
PPTX file adapter for the content processing service.
"""

import io
import zipfile
import xml.etree.ElementTree as ET
import os
import hashlib
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, BinaryIO
from . import ContentAdapter, ContentProcessingError

class PptxAdapter(ContentAdapter):
    """Adapter for PPTX files."""
    
    @classmethod
    def supported_mime_types(cls) -> List[str]:
        return [
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'application/vnd.ms-powerpoint.presentation.macroenabled.12',
            'application/vnd.ms-powerpoint',
            'application/powerpoint',
            'application/x-mspowerpoint'
        ]
    
    async def extract_text(self, file: BinaryIO, **kwargs) -> str:
        """Extract text content from a PPTX file."""
        try:
            file.seek(0)
            
            # Read the file content
            pptx_data = file.read()
            
            # Create an in-memory file-like object
            pptx_io = io.BytesIO(pptx_data)
            
            # Extract text from the PPTX file
            text = self._extract_text_from_pptx(pptx_io)
            
            return text
            
        except Exception as e:
            raise ContentProcessingError(f"Error extracting text from PPTX: {str(e)}")
    
    def _extract_text_from_pptx(self, pptx_io):
        """Helper method to extract text from a PPTX file."""
        text_parts = []
        
        # Open the PPTX file as a zip archive
        with zipfile.ZipFile(pptx_io) as zipf:
            # Get all slide files (usually in ppt/slides/slide*.xml)
            slide_files = [f for f in zipf.namelist() 
                         if f.startswith('ppt/slides/slide') and f.endswith('.xml')]
            
            # Sort slides by number (robust against 'ppt/slides/slide1.xml' and folder names containing 'slides')
            def _slide_index(path: str) -> int:
                # e.g., 'ppt/slides/slide12.xml' -> basename 'slide12' -> index 12
                name = os.path.splitext(os.path.basename(path))[0]
                if name.startswith('slide'):
                    num_part = name[len('slide'):]
                    try:
                        return int(num_part)
                    except ValueError:
                        return 1_000_000  # push unparseable to end
                return 1_000_000

            slide_files.sort(key=_slide_index)
            
            # Process each slide
            for slide_file in slide_files:
                with zipf.open(slide_file) as slide_data:
                    try:
                        # Parse the slide XML
                        tree = ET.parse(slide_data)
                        root = tree.getroot()
                        
                        # Define the XML namespaces
                        namespaces = {
                            'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
                            'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
                            'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
                        }
                        
                        # Find all text elements in the slide
                        text_elements = root.findall('.//a:t', namespaces)
                        if text_elements:
                            slide_text = ' '.join([elem.text for elem in text_elements 
                                                 if elem.text and elem.text.strip()])
                            if slide_text:
                                text_parts.append(slide_text)
                    
                    except Exception as e:
                        # Skip slides that can't be processed
                        continue
        
        return '\n\n'.join(text_parts)
    
    async def extract_metadata(self, file: BinaryIO) -> Dict:
        """Extract metadata from the PPTX file."""
        try:
            file.seek(0)
            pptx_data = file.read()
            pptx_io = io.BytesIO(pptx_data)
            
            metadata = {}
            
            # Open the PPTX file as a zip archive
            with zipfile.ZipFile(pptx_io) as zipf:
                # Extract core properties
                if 'docProps/core.xml' in zipf.namelist():
                    with zipf.open('docProps/core.xml') as core_file:
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
                
                # Count slides
                slide_count = len([f for f in zipf.namelist() 
                                if f.startswith('ppt/slides/slide') and f.endswith('.xml')])
                metadata['slide_count'] = slide_count
            
            # Get preview text and compute text stats for confidence
            preview_text = ""
            total_chars = 0
            slides_with_text = 0
            try:
                text = await self.extract_text(file)
                preview_text = text[:500]  # First 500 characters
                total_chars = len(text)
                # Approximate slides with text by counting non-empty blocks split on two newlines
                slide_chunks = [ch for ch in text.split("\n\n") if ch.strip()]
                slides_with_text = len(slide_chunks)
            except:
                pass

            # Extract speaker notes per slide
            notes_per_slide: List[str] = []
            try:
                notes_per_slide = self._extract_notes_from_pptx(io.BytesIO(pptx_data))
            except Exception:
                notes_per_slide = []
            
            # Attempt to generate slide thumbnails via LibreOffice headless
            thumbnails: List[str] = []
            try:
                max_thumbs = int(os.getenv('THUMBNAILS_MAX', '3'))
                uploads_base = Path(os.getenv('UPLOADS_DIR', 'uploads')).resolve()
                thumb_dir = uploads_base / 'thumbnails' / hashlib.sha1(pptx_data).hexdigest()
                thumb_dir.mkdir(parents=True, exist_ok=True)

                # Write PPTX to a temp file
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir_path = Path(tmpdir)
                    input_path = tmpdir_path / 'input.pptx'
                    with open(input_path, 'wb') as ftmp:
                        ftmp.write(pptx_data)

                    soffice = os.getenv('LIBREOFFICE_PATH', 'soffice')
                    cmd = [soffice, '--headless', '--convert-to', 'png', '--outdir', str(tmpdir_path), str(input_path)]
                    # Run conversion
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=120)

                    # Collect generated PNGs
                    pngs = sorted([p for p in tmpdir_path.glob('*.png')])
                    pages_to_copy = min(len(pngs), max_thumbs)
                    for i in range(pages_to_copy):
                        src = pngs[i]
                        out_name = f"slide_{i+1:03d}.png"
                        out_path = thumb_dir / out_name
                        # Copy file
                        with open(src, 'rb') as s, open(out_path, 'wb') as d:
                            d.write(s.read())
                        thumbnails.append(str(out_path.relative_to(uploads_base)))
            except Exception:
                thumbnails = []

            # Confidence score heuristic
            if slide_count > 0:
                avg_chars = total_chars / max(1, slide_count)
                avg_norm = min(avg_chars / 800.0, 1.0)
                slides_with_text_ratio = min(slides_with_text / slide_count, 1.0)
                confidence_score = max(0.0, min(1.0, 0.6 * slides_with_text_ratio + 0.4 * avg_norm))
            else:
                confidence_score = 0.0

            metadata.update({
                'size_bytes': len(pptx_data),
                'preview': preview_text,
                'file_type': 'PPTX',
                'thumbnails': thumbnails,
                'confidence_score': confidence_score,
                'notes': notes_per_slide,
                'notes_count': sum(1 for n in notes_per_slide if n and n.strip())
            })
            
            return metadata
            
        except Exception as e:
            raise ContentProcessingError(f"Error extracting PPTX metadata: {str(e)}")
        finally:
            file.seek(0)
    
    def _get_xml_text(self, element, xpath, namespaces):
        """Helper method to safely get text from XML element."""
        result = element.find(xpath, namespaces)
        return result.text if result is not None else ""

    def _extract_notes_from_pptx(self, pptx_io: io.BytesIO) -> List[str]:
        """Parse PPTX to extract speaker notes per slide in slide order.
        Returns a list aligned with slide numbering (1-based in filenames),
        containing empty strings for slides without notes."""
        notes_by_slide_index = {}
        with zipfile.ZipFile(pptx_io) as zipf:
            # Collect slide files and sort by numeric index
            slide_files = [f for f in zipf.namelist() if f.startswith('ppt/slides/slide') and f.endswith('.xml')]
            def _slide_index(path: str) -> int:
                name = os.path.splitext(os.path.basename(path))[0]
                if name.startswith('slide'):
                    try:
                        return int(name[len('slide'):])
                    except ValueError:
                        return 1_000_000
                return 1_000_000
            slide_files.sort(key=_slide_index)

            namespaces = {
                'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
                'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
                'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
            }

            for slide_path in slide_files:
                idx = _slide_index(slide_path)
                # Find rels file for this slide to locate notesSlide target
                # Always use POSIX-style paths inside ZIPs
                import posixpath
                rels_path = posixpath.join(posixpath.dirname(slide_path), '_rels', posixpath.basename(slide_path) + '.rels')
                notes_text = ""
                if rels_path in zipf.namelist():
                    try:
                        with zipf.open(rels_path) as rels_file:
                            rels_tree = ET.parse(rels_file)
                            rels_root = rels_tree.getroot()
                            # Relationships namespace is typically http://schemas.openxmlformats.org/package/2006/relationships
                            # but we can match by local-name to avoid ns issues
                            for rel in rels_root:
                                rel_type = rel.attrib.get('Type', '')
                                if rel_type.endswith('/notesSlide'):
                                    target = rel.attrib.get('Target', '')
                                    # Resolve relative path like '../notesSlides/notesSlideN.xml' using POSIX joins
                                    target_path = posixpath.normpath(posixpath.join(posixpath.dirname(slide_path), target))
                                    if target_path in zipf.namelist():
                                        with zipf.open(target_path) as notes_file:
                                            n_tree = ET.parse(notes_file)
                                            n_root = n_tree.getroot()
                                            # Extract all text runs within notes slide
                                            ts = [el.text for el in n_root.findall('.//a:t', namespaces) if el.text and el.text.strip()]
                                            notes_text = ' '.join(ts)
                                    break
                    except Exception:
                        notes_text = ""
                notes_by_slide_index[idx] = notes_text

            # Build ordered list aligned to slides encountered
            ordered_notes = []
            for slide_path in slide_files:
                idx = _slide_index(slide_path)
                ordered_notes.append(notes_by_slide_index.get(idx, ""))

            return ordered_notes
    
    async def is_valid(self, file: BinaryIO) -> bool:
        """Check if the file is a valid PPTX file."""
        try:
            file.seek(0)
            # Check for PPTX magic number (PK header for ZIP)
            magic = file.read(4)
            file.seek(0)
            
            if magic != b'PK\x03\x04':
                return False
                
            # Try to open as a ZIP file and check for required PPTX files
            with zipfile.ZipFile(io.BytesIO(file.read())) as zipf:
                required_files = ['[Content_Types].xml', '_rels/.rels', 'ppt/presentation.xml']
                return all(f in zipf.namelist() for f in required_files)
                
        except Exception:
            return False
        finally:
            file.seek(0)
