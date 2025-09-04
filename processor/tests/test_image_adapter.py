import io
import pytest
import base64
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Optional, Dict, Any

from processor.adapters.image_adapter import ImageAdapter, VisionModelConfig, VisionModelType, ExtractionMode

# Test configuration
LOCAL_API_URL = "http://localhost:5001"  # Default JoyCaption API URL
TEST_IMAGE_SIZE = (256, 256)  # Test image size (larger than 2x2 for better testing)

class TestJoyCaptionIntegration:
    """Integration tests for JoyCaption API with ImageAdapter."""
    
    @pytest.fixture
    def create_test_image(self) -> io.BytesIO:
        """Create a test image with some text for OCR testing."""
        # Create a larger image with some colored regions
        img_array = np.zeros((*TEST_IMAGE_SIZE, 3), dtype=np.uint8)
        
        # Add some colored regions
        img_array[:100, :] = [255, 0, 0]      # Red top
        img_array[100:200, :] = [0, 255, 0]   # Green middle
        img_array[200:, :] = [0, 0, 255]      # Blue bottom
        
        # Add some text (this won't be machine-readable but helps with visual testing)
        img = Image.fromarray(img_array)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr

    @pytest.fixture
    def temp_image_file(self, create_test_image) -> str:
        """Save test image to a temporary file and return the path."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(create_test_image.getvalue())
            return tmp.name

    @pytest.mark.asyncio
    async def test_local_joycaption_integration(self, temp_image_file):
        """Test integration with local JoyCaption API."""
        config = {
            "model_type": VisionModelType.OPENAI,
            "model_name": "joycaption",  # Model name used by your API
            "api_base": LOCAL_API_URL,
            "api_key": "dummy-key",  # Dummy key required by OpenAI client
            "extraction_mode": ExtractionMode.BOTH,
            "timeout": 30,
            "text_prompt": "Extract all text from this image:",
            "meaning_prompt": "Describe this image in detail:",
            # JoyCaption specific parameters
            "temperature": 0.7,
            "max_tokens": 512,
            "top_k": 100,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0
        }
        
        adapter = ImageAdapter(model_config=config)
        
        # Test with file path
        with open(temp_image_file, 'rb') as f:
            result = await adapter.extract_text(f)
            
        # Verify results
        assert isinstance(result, dict)
        
        # Check text extraction (may be empty if no text in image)
        if "text" in result:
            assert isinstance(result["text"], str)
        
        # Check meaning extraction
        assert "meaning" in result
        assert isinstance(result["meaning"], str)
        assert len(result["meaning"].strip()) > 0

    @pytest.mark.asyncio
    async def test_different_image_formats(self, temp_image_file):
        """Test with different image formats."""
        config = {
            "model_type": VisionModelType.OPENAI,
            "api_base": LOCAL_API_URL,
            "api_key": "dummy-key",  # Dummy key required by OpenAI client
            "extraction_mode": ExtractionMode.MEANING_ONLY
        }
        
        adapter = ImageAdapter(model_config=config)
        
        # Test with different formats
        formats = ['PNG', 'JPEG', 'BMP']
        
        for fmt in formats:
            # Convert image to target format
            with Image.open(temp_image_file) as img:
                buffer = io.BytesIO()
                img.save(buffer, format=fmt)
                buffer.seek(0)
                
                # Test extraction
                result = await adapter.extract_text(buffer)
                assert "meaning" in result
                assert isinstance(result["meaning"], str)
                assert len(result["meaning"].strip()) > 0

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling with invalid inputs."""
        config = {
            "model_type": VisionModelType.OPENAI,
            "api_base": LOCAL_API_URL,
            "timeout": 5  # Shorter timeout for testing
        }
        
        adapter = ImageAdapter(model_config=config)
        
        # Test with invalid image data
        with pytest.raises(Exception):
            invalid_image = io.BytesIO(b"not an image")
            await adapter.extract_text(invalid_image)
        
        # Test with non-existent endpoint
        adapter.model_config.api_base = "http://localhost:9999"
        with pytest.raises(Exception):
            with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
                tmp.write(b"dummy image data")
                tmp.seek(0)
                await adapter.extract_text(tmp)

    @pytest.fixture(autouse=True)
    def cleanup(self, request, temp_image_file):
        """Cleanup temporary files after tests."""
        def remove_temp_file():
            if Path(temp_image_file).exists():
                Path(temp_image_file).unlink()
                
        request.addfinalizer(remove_temp_file)
