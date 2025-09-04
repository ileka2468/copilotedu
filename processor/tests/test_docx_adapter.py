import io
import zipfile
import xml.etree.ElementTree as ET
import pytest

from ..adapters.docx_adapter import DocxAdapter


def build_minimal_docx(text: str) -> bytes:
    content_types = (
        b"<?xml version='1.0' encoding='UTF-8'?>"
        b"<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>"
        b"<Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>"
        b"<Default Extension='xml' ContentType='application/xml'/>"
        b"<Override PartName='/word/document.xml' ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'/>"
        b"<Override PartName='/docProps/core.xml' ContentType='application/vnd.openxmlformats-package.core-properties+xml'/>"
        b"</Types>"
    )
    rels = (
        b"<?xml version='1.0' encoding='UTF-8'?>"
        b"<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
        b"<Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='word/document.xml'/>"
        b"</Relationships>"
    )
    # Very small document.xml with one paragraph and one text run
    document_xml = (
        """
        <?xml version='1.0' encoding='UTF-8'?>
        <w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>
          <w:body>
            <w:p>
              <w:r>
                <w:t>{}</w:t>
              </w:r>
            </w:p>
          </w:body>
        </w:document>
        """.strip().format(text)
    ).encode("utf-8")

    core_xml = (
        b"<?xml version='1.0' encoding='UTF-8'?>"
        b"<cp:coreProperties xmlns:cp='http://schemas.openxmlformats.org/package/2006/metadata/core-properties' "
        b"xmlns:dc='http://purl.org/dc/elements/1.1/' "
        b"xmlns:dcterms='http://purl.org/dc/terms/' "
        b"xmlns:dcmitype='http://purl.org/dc/dcmitype/' "
        b"xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>"
        b"<dc:title></dc:title>"
        b"</cp:coreProperties>"
    )

    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document_xml)
        z.writestr("docProps/core.xml", core_xml)
    return bio.getvalue()


@pytest.mark.asyncio
async def test_docx_adapter_extract_text_metadata_and_valid():
    adapter = DocxAdapter()
    expected = "Hello DOCX"
    data = build_minimal_docx(expected)
    f = io.BytesIO(data)

    assert await adapter.is_valid(f) is True

    text = await adapter.extract_text(f)
    assert expected in text

    meta = await adapter.extract_metadata(f)
    assert meta.get("file_type") == "DOCX"
    assert meta.get("size_bytes") == len(data)
    assert "preview" in meta
