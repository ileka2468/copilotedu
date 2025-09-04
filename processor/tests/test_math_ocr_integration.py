import sys
import types
from io import BytesIO
from typing import Any

import pytest
from PIL import Image, ImageDraw, ImageFont

from processor.adapters.image_adapter import ImageAdapter
from processor.adapters.base_adapter import ExtractionMode
from processor.adapters.config import VisionModelConfig, VisionModelType


def make_math_image() -> BytesIO:
    # Create a small black image with white fraction-like expression
    img = Image.new("RGB", (400, 200), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    text = "(24 x^9 y^5)/(8 x^3 y^12)"
    try:
        # Use a default font; availability differs across systems
        font = ImageFont.load_default()
    except Exception:
        font = None
    d.text((10, 80), text, fill=(255, 255, 255), font=font)
    bio = BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return bio


class DummyLatexOCR:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def __call__(self, image: Image.Image) -> str:
        # Return a deterministic LaTeX for the test image
        return r"\\frac{24 x^{9} y^{5}}{8 x^{3} y^{12}}"


@pytest.fixture(autouse=True)
def mock_pix2tex(monkeypatch):
    # Create a dummy module structure: pix2tex.cli.LatexOCR
    mod_cli = types.ModuleType("pix2tex.cli")
    mod_cli.LatexOCR = DummyLatexOCR
    mod_root = types.ModuleType("pix2tex")
    mod_root.cli = mod_cli
    sys.modules["pix2tex"] = mod_root
    sys.modules["pix2tex.cli"] = mod_cli
    yield
    # Cleanup so other tests aren't affected
    sys.modules.pop("pix2tex", None)
    sys.modules.pop("pix2tex.cli", None)


def test_image_adapter_math_route_extracts_latex():
    cfg = VisionModelConfig(
        model_type=VisionModelType.TESSERACT,
        extraction_mode=ExtractionMode.TEXT_ONLY,
    )
    adapter = ImageAdapter(model_config=cfg)

    # Build synthetic image
    bio = make_math_image()

    # Force math_mode to bypass heuristic variability
    result = adapter.extract_text(bio, math_mode=True)

    assert isinstance(result, dict)
    # We should get plain text OCR plus equation_latex from pix2tex
    assert "text" in result
    assert "equation_latex" in result
    assert result["equation_latex"].startswith("\\frac{24")

    meta = adapter.extract_metadata(bio)
    assert meta.get("math_detected") is True
    assert meta.get("equation_latex") == result["equation_latex"]
    assert meta.get("equation_engine") == "pix2tex"
    # Confidence heuristic should be present and between 0 and 1
    conf = meta.get("math_confidence")
    assert isinstance(conf, float) and 0.0 <= conf <= 1.0
