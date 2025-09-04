"""Tests for the content processing service."""

import os
import io
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest_asyncio
from fastapi import UploadFile

from ..service import ContentProcessor
from ..adapters import ContentProcessingError, UnsupportedFileTypeError, InvalidFileError

# Test data
SAMPLE_TEXT = "This is a test file content."
SAMPLE_FILENAME = "test.txt"


@pytest.fixture
def temp_upload_dir():
    """Create a temporary upload directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_file():
    """Create a sample file for testing."""
    return io.BytesIO(SAMPLE_TEXT.encode('utf-8'))


@pytest.fixture
def mock_adapter():
    """Create a mock adapter for testing."""
    adapter = AsyncMock()
    adapter.is_valid.return_value = True
    adapter.extract_text.return_value = SAMPLE_TEXT
    adapter.extract_metadata.return_value = {"size": len(SAMPLE_TEXT), "type": "text/plain"}
    return adapter


@pytest.fixture
def mock_adapter_factory(mock_adapter):
    """Create a mock adapter factory for testing."""
    def factory(filename):
        if filename.endswith('.txt'):
            return mock_adapter
        return None
    return factory


@pytest.fixture
def processor(temp_upload_dir):
    """Create a ContentProcessor instance for testing."""
    return ContentProcessor(upload_dir=str(temp_upload_dir), max_file_size=1024)


class TestContentProcessor:
    """Test cases for ContentProcessor class."""

    @pytest.mark.asyncio
    async def test_process_file_success(self, processor, sample_file, mock_adapter):
        """Test successful file processing."""
        with patch('processor.service.get_adapter', return_value=mock_adapter):
            result = await processor.process_file(sample_file, SAMPLE_FILENAME)
            
            assert result['success'] is True
            assert result['filename'] == SAMPLE_FILENAME
            assert result['content'] == SAMPLE_TEXT
            assert 'metadata' in result
            assert 'processing_time' in result
            assert 'file_size' in result
            
            # Verify adapter methods were called
            mock_adapter.is_valid.assert_awaited_once()
            mock_adapter.extract_text.assert_awaited_once()
            mock_adapter.extract_metadata.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_file_unsupported_type(self, processor, sample_file):
        """Test processing unsupported file type."""
        with patch('processor.service.get_adapter', return_value=None):
            with pytest.raises(UnsupportedFileTypeError):
                await processor.process_file(sample_file, 'unsupported.xyz')

    @pytest.mark.asyncio
    async def test_process_file_invalid_file(self, processor, sample_file, mock_adapter):
        """Test processing an invalid file."""
        mock_adapter.is_valid.return_value = False
        with patch('processor.service.get_adapter', return_value=mock_adapter):
            with pytest.raises(InvalidFileError):
                await processor.process_file(sample_file, SAMPLE_FILENAME)

    @pytest.mark.asyncio
    async def test_process_file_large_file(self, processor, sample_file, mock_adapter):
        """Test processing a file that exceeds size limit."""
        # Create a processor with small size limit
        small_processor = ContentProcessor(max_file_size=10)  # 10 bytes
        with patch('processor.service.get_adapter', return_value=mock_adapter):
            with pytest.raises(ContentProcessingError) as exc_info:
                await small_processor.process_file(sample_file, SAMPLE_FILENAME)
            assert "exceeds maximum allowed size" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_save_uploaded_file(self, processor, sample_file, temp_upload_dir):
        """Test saving an uploaded file."""
        file_path = await processor._save_uploaded_file(sample_file, SAMPLE_FILENAME)
        
        assert file_path.exists()
        assert file_path.parent == temp_upload_dir
        assert file_path.read_text() == SAMPLE_TEXT

    @pytest.mark.asyncio
    async def test_save_uploaded_file_duplicate(self, processor, sample_file, temp_upload_dir):
        """Test saving a file with a duplicate name."""
        # Save first file
        file_path1 = await processor._save_uploaded_file(sample_file, SAMPLE_FILENAME)
        
        # Reset file pointer and save again
        sample_file.seek(0)
        file_path2 = await processor._save_uploaded_file(sample_file, SAMPLE_FILENAME)
        
        assert file_path1 != file_path2
        assert file_path1.read_text() == file_path2.read_text()

    @pytest.mark.asyncio
    async def test_process_file_save_file_false(self, processor, sample_file, mock_adapter):
        """Test processing a file without saving it."""
        with patch('processor.service.get_adapter', return_value=mock_adapter):
            result = await processor.process_file(
                sample_file, 
                SAMPLE_FILENAME, 
                save_file=False
            )
            
            assert result['success'] is True
            assert result['file_path'] is None

    @pytest.mark.asyncio
    async def test_process_file_error_handling(self, processor, sample_file, mock_adapter):
        """Test error handling during file processing."""
        mock_adapter.extract_text.side_effect = Exception("Test error")
        with patch('processor.service.get_adapter', return_value=mock_adapter):
            with pytest.raises(ContentProcessingError) as exc_info:
                await processor.process_file(sample_file, SAMPLE_FILENAME)
            assert "unexpected error" in str(exc_info.value).lower()

    @pytest.mark.parametrize("filename,expected", [
        ("normal_file.txt", "normal_file.txt"),
        ("file with spaces.txt", "file_with_spaces.txt"),
        ("UPPERCASE.TXT", "UPPERCASE.TXT"),
        ("file*with*stars.txt", "file_with_stars.txt"),
        ("file/with/slashes.txt", "file_with_slashes.txt"),
        ("file\\with\\backslashes.txt", "file_with_backslashes.txt"),
        ("file:with:colons.txt", "file_with_colons.txt"),
        ("file with multiple   spaces.txt", "file_with_multiple___spaces.txt"),
        ("  leading_trailing_spaces.txt  ", "leading_trailing_spaces.txt"),
        ("file.with.multiple.dots.txt", "file.with.multiple.dots.txt"),
        ("file-with-hyphens.txt", "file-with-hyphens.txt"),
        ("file_with_underscores.txt", "file_with_underscores.txt"),
        (r"file!@#$%^&()+=[]{};',.~`|\.txt", "file__________________.____.txt"),
        ("", "unnamed_file"),
        (None, "unnamed_file"),
    ])
    def test_get_safe_filename(self, filename, expected):
        """Test filename sanitization."""
        assert ContentProcessor._get_safe_filename(filename) == expected

    @pytest.mark.asyncio
    async def test_process_file_with_additional_metadata(self, processor, sample_file, mock_adapter):
        """Test processing a file with additional metadata."""
        test_metadata = {"author": "Test User", "pages": 5}
        mock_adapter.extract_metadata.return_value = test_metadata
        
        with patch('processor.service.get_adapter', return_value=mock_adapter):
            result = await processor.process_file(
                sample_file, 
                SAMPLE_FILENAME,
                custom_param="test"
            )
            
            assert result['metadata'] == test_metadata
            mock_adapter.extract_text.assert_called_once_with(
                sample_file, 
                custom_param="test"
            )


class TestContentProcessorIntegration:
    """Integration tests for ContentProcessor with actual file operations."""
    
    @pytest.fixture(autouse=True)
    def setup(self, temp_upload_dir):
        self.processor = ContentProcessor(upload_dir=str(temp_upload_dir))
        self.temp_dir = temp_upload_dir
    
    @pytest.mark.asyncio
    async def test_end_to_end_text_file_processing(self):
        """Test complete flow with a text file."""
        # Create a test file
        test_content = "Test content for integration test"
        test_file = io.BytesIO(test_content.encode('utf-8'))
        
        # Process the file
        result = await self.processor.process_file(
            test_file,
            "test_integration.txt"
        )
        
        # Verify results
        assert result['success'] is True
        assert isinstance(result['content'], str)
        assert len(result['content']) > 0
        assert 'metadata' in result
        assert 'processing_time' in result
        
        # Verify file was saved
        saved_file = self.temp_dir / result['file_path']
        assert saved_file.exists()
        assert saved_file.read_text() == test_content

    @pytest.mark.asyncio
    async def test_concurrent_file_processing(self):
        """Test processing multiple files concurrently."""
        import asyncio
        
        async def process_file(i):
            content = f"Test content {i}"
            file_obj = io.BytesIO(content.encode('utf-8'))
            result = await self.processor.process_file(
                file_obj,
                f"concurrent_{i}.txt"
            )
            return result
        
        # Process multiple files concurrently
        tasks = [process_file(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Verify all files were processed successfully
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result['success'] is True
            assert f"Test content {i}" in result['content']
            assert (self.temp_dir / result['file_path']).exists()
