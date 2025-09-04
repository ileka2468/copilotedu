import io
import zipfile
import pytest

from ..adapters.pptx_adapter import PptxAdapter


def build_minimal_pptx(slide_text: str) -> bytes:
    content_types = (
        b"<?xml version='1.0' encoding='UTF-8'?>"
        b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>"
        b"<Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>"
        b"<Default Extension='xml' ContentType='application/xml'/>"
        b"<Override PartName='/ppt/presentation.xml' ContentType='application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml'/>"
        b"<Override PartName='/ppt/slides/slide1.xml' ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml'/>"
        b"<Override PartName='/docProps/core.xml' ContentType='application/vnd.openxmlformats-package.core-properties+xml'/>"
        b"</Types>"
    )
    rels = (
        b"<?xml version='1.0' encoding='UTF-8'?>"
        b"<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
        b"<Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='ppt/presentation.xml'/>"
        b"</Relationships>"
    )
    presentation_xml = (
        b"<?xml version='1.0' encoding='UTF-8'?>"
        b"<p:presentation xmlns:p='http://schemas.openxmlformats.org/presentationml/2006/main' "
        b"xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'>"
        b"<p:sldIdLst><p:sldId id='256' r:id='rId1'/></p:sldIdLst>"
        b"</p:presentation>"
    )
    slide1_xml = (
        f"""
        <?xml version='1.0' encoding='UTF-8'?>
        <p:sld xmlns:p='http://schemas.openxmlformats.org/presentationml/2006/main' xmlns:a='http://schemas.openxmlformats.org/drawingml/2006/main'>
          <p:cSld>
            <p:spTree>
              <p:sp>
                <p:txBody>
                  <a:p>
                    <a:r><a:t>{slide_text}</a:t></a:r>
                  </a:p>
                </p:txBody>
              </p:sp>
            </p:spTree>
          </p:cSld>
        </p:sld>
        """.strip().encode("utf-8")
    )
    core_xml = (
        b"<?xml version='1.0' encoding='UTF-8'?>"
        b"<cp:coreProperties xmlns:cp='http://schemas.openxmlformats.org/package/2006/metadata/core-properties' "
        b"xmlns:dc='http://purl.org/dc/elements/1.1/'>"
        b"<dc:title></dc:title>"
        b"</cp:coreProperties>"
    )

    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("ppt/presentation.xml", presentation_xml)
        z.writestr("ppt/slides/slide1.xml", slide1_xml)
        z.writestr("docProps/core.xml", core_xml)
    return bio.getvalue()


def build_pptx_with_notes(slide_text: str, notes_text: str) -> bytes:
    content_types = (
        b"<?xml version='1.0' encoding='UTF-8'?>"
        b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>"
        b"<Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>"
        b"<Default Extension='xml' ContentType='application/xml'/>"
        b"<Override PartName='/ppt/presentation.xml' ContentType='application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml'/>"
        b"<Override PartName='/ppt/slides/slide1.xml' ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml'/>"
        b"<Override PartName='/docProps/core.xml' ContentType='application/vnd.openxmlformats-package.core-properties+xml'/>"
        b"<Override PartName='/ppt/notesSlides/notesSlide1.xml' ContentType='application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml'/>"
        b"</Types>"
    )
    rels = (
        b"<?xml version='1.0' encoding='UTF-8'?>"
        b"<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
        b"<Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='ppt/presentation.xml'/>"
        b"</Relationships>"
    )
    presentation_xml = (
        b"<?xml version='1.0' encoding='UTF-8'?>"
        b"<p:presentation xmlns:p='http://schemas.openxmlformats.org/presentationml/2006/main' "
        b"xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'>"
        b"<p:sldIdLst><p:sldId id='256' r:id='rId1'/></p:sldIdLst>"
        b"</p:presentation>"
    )
    slide1_xml = (
        f"""
        <?xml version='1.0' encoding='UTF-8'?>
        <p:sld xmlns:p='http://schemas.openxmlformats.org/presentationml/2006/main' xmlns:a='http://schemas.openxmlformats.org/drawingml/2006/main'>
          <p:cSld>
            <p:spTree>
              <p:sp>
                <p:txBody>
                  <a:p>
                    <a:r><a:t>{slide_text}</a:t></a:r>
                  </a:p>
                </p:txBody>
              </p:sp>
            </p:spTree>
          </p:cSld>
        </p:sld>
        """.strip().encode("utf-8")
    )
    # Slide rels linking to notes
    slide1_rels = (
        b"<?xml version='1.0' encoding='UTF-8'?>"
        b"<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
        b"<Relationship Id='rId2' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide' Target='../notesSlides/notesSlide1.xml'/>"
        b"</Relationships>"
    )
    notes1_xml = (
        f"""
        <?xml version='1.0' encoding='UTF-8'?>
        <p:notes xmlns:p='http://schemas.openxmlformats.org/presentationml/2006/main' xmlns:a='http://schemas.openxmlformats.org/drawingml/2006/main'>
          <p:cSld>
            <p:spTree>
              <p:sp>
                <p:txBody>
                  <a:p>
                    <a:r><a:t>{notes_text}</a:t></a:r>
                  </a:p>
                </p:txBody>
              </p:sp>
            </p:spTree>
          </p:cSld>
        </p:notes>
        """.strip().encode("utf-8")
    )
    core_xml = (
        b"<?xml version='1.0' encoding='UTF-8'?>"
        b"<cp:coreProperties xmlns:cp='http://schemas.openxmlformats.org/package/2006/metadata/core-properties' "
        b"xmlns:dc='http://purl.org/dc/elements/1.1/'>"
        b"<dc:title></dc:title>"
        b"</cp:coreProperties>"
    )

    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("ppt/presentation.xml", presentation_xml)
        z.writestr("ppt/slides/slide1.xml", slide1_xml)
        z.writestr("ppt/slides/_rels/slide1.xml.rels", slide1_rels)
        z.writestr("ppt/notesSlides/notesSlide1.xml", notes1_xml)
        z.writestr("docProps/core.xml", core_xml)
    return bio.getvalue()


@pytest.mark.asyncio
async def test_pptx_adapter_extract_text_metadata_and_valid():
    adapter = PptxAdapter()
    expected = "Hello PPTX"
    data = build_minimal_pptx(expected)
    f = io.BytesIO(data)

    assert await adapter.is_valid(f) is True

    text = await adapter.extract_text(f)
    assert expected in text

    meta = await adapter.extract_metadata(f)
    assert meta.get("file_type") == "PPTX"
    assert meta.get("size_bytes") == len(data)
    assert meta.get("slide_count") == 1
    assert "preview" in meta
    assert "confidence_score" in meta and 0.0 <= meta["confidence_score"] <= 1.0


@pytest.mark.asyncio
async def test_pptx_notes_extraction_and_alignment_and_confidence():
    adapter = PptxAdapter()
    slide_text = "Slide Content A"
    notes_text = "Speaker notes A"
    data = build_pptx_with_notes(slide_text, notes_text)
    f = io.BytesIO(data)

    assert await adapter.is_valid(f) is True

    meta = await adapter.extract_metadata(f)
    assert meta.get("slide_count") == 1
    # Notes list aligned to slides and contains our notes
    assert isinstance(meta.get("notes"), list)
    assert len(meta["notes"]) == meta["slide_count"]
    assert meta["notes"][0] == notes_text
    assert meta.get("notes_count") == 1
    # Confidence score present and within [0,1]
    assert "confidence_score" in meta and 0.0 <= meta["confidence_score"] <= 1.0
