import io
import pytest
import pytest_asyncio

from ..adapters.text_adapter import TextAdapter
from ..adapters import ContentProcessingError


@pytest.mark.asyncio
async def test_text_adapter_extract_text_and_metadata():
    adapter = TextAdapter()
    content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6"
    f = io.BytesIO(content.encode("utf-8"))

    # is_valid
    assert await adapter.is_valid(f) is True

    # extract_text
    text = await adapter.extract_text(f)
    assert text == content

    # extract_metadata
    meta = await adapter.extract_metadata(f)
    assert meta["size_bytes"] == len(content.encode("utf-8"))
    assert meta["line_count"] == 6
    assert meta["encoding"] == "utf-8"
    assert "preview" in meta and len(meta["preview"]) > 0


@pytest.mark.asyncio
async def test_text_adapter_invalid_utf8():
    adapter = TextAdapter()
    # invalid utf-8 bytes
    f = io.BytesIO(b"\xff\xfe\xfa\x00")

    # is_valid should be False
    assert await adapter.is_valid(f) is False

    # extract_text should raise decoding error wrapped
    with pytest.raises(ContentProcessingError):
        await adapter.extract_text(f)
