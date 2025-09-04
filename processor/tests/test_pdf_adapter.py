import io
import pytest

from ..adapters.pdf_adapter import PDFAdapter

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - allow import errors in environments without PyMuPDF
    fitz = None


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not available")
@pytest.mark.asyncio
async def test_pdf_adapter_extract_text_metadata_and_valid():
    adapter = PDFAdapter()

    # Build an in-memory PDF with a single page and some text
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    text = "Hello PDF Adapter"
    page.insert_text((72, 100), text)  # 1 inch margin
    pdf_bytes = doc.tobytes()

    f = io.BytesIO(pdf_bytes)

    assert await adapter.is_valid(f) is True

    extracted = await adapter.extract_text(f)
    assert "Hello" in extracted

    meta = await adapter.extract_metadata(f)
    assert meta.get("page_count") == 1
    assert meta.get("size_bytes") == len(pdf_bytes)
    assert isinstance(meta.get("preview"), str)
    assert "confidence_score" in meta and 0.0 <= meta["confidence_score"] <= 1.0
