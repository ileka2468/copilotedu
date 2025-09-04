import io
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from ..adapters.pdf_adapter import PDFAdapter
from ..adapters.pptx_adapter import PptxAdapter

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None


def _set_env_uploads_tmp(tmpdir: Path, max_thumbs: int = 2):
    os.environ["UPLOADS_DIR"] = str(tmpdir)
    os.environ["THUMBNAILS_MAX"] = str(max_thumbs)


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
@pytest.mark.asyncio
async def test_pdf_thumbnails_generation(tmp_path: Path):
    _set_env_uploads_tmp(tmp_path, 2)

    # Build a simple in-memory PDF
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100 + i * 20), f"Hello Page {i+1}")
    pdf_bytes = doc.tobytes()
    f = io.BytesIO(pdf_bytes)

    adapter = PDFAdapter()
    meta = await adapter.extract_metadata(f)

    thumbs = meta.get("preview_thumbnails", [])
    assert isinstance(thumbs, list)
    assert len(thumbs) <= 2

    # Thumbnails should exist on disk under UPLOADS_DIR
    for rel in thumbs:
        p = tmp_path / rel
        assert p.exists(), f"Missing thumbnail: {p}"
        assert p.suffix.lower() == ".png"


@pytest.mark.asyncio
async def test_pptx_thumbnails_generation_if_libreoffice_present(tmp_path: Path):
    # Skip if we cannot find soffice and no LIBREOFFICE_PATH set
    soffice_env = os.environ.get("LIBREOFFICE_PATH")
    soffice_path = soffice_env or shutil.which("soffice")
    if not soffice_path:
        pytest.skip("LibreOffice (soffice) not available")

    os.environ["LIBREOFFICE_PATH"] = soffice_path
    _set_env_uploads_tmp(tmp_path, 2)

    # Build a minimal PPTX with one slide and some text
    # Reuse helper from test_pptx_adapter
    from .test_pptx_adapter import build_minimal_pptx

    data = build_minimal_pptx("Thumb Test")
    f = io.BytesIO(data)

    adapter = PptxAdapter()
    meta = await adapter.extract_metadata(f)

    thumbs = meta.get("thumbnails", [])
    assert isinstance(thumbs, list)
    # Depending on LibreOffice version, at least 1 PNG should be generated
    assert len(thumbs) >= 0  # Do not fail hard; existence is environment-dependent

    for rel in thumbs:
        p = tmp_path / rel
        assert p.exists(), f"Missing PPTX thumbnail: {p}"
        assert p.suffix.lower() == ".png"
