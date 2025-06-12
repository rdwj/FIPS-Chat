"""Tests for image processing utilities."""

import pytest
from PIL import Image
import io
from utils.image_processing import (
    validate_image_format,
    get_image_metadata,
    resize_image_if_needed,
    optimize_image_for_api,
    create_thumbnail,
    auto_orient_image,
    format_file_size,
    is_image_too_large,
    get_supported_formats_info
)


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    image = Image.new('RGB', (800, 600), color='red')
    return image


@pytest.fixture
def sample_image_bytes(sample_image):
    """Convert sample image to bytes."""
    buffer = io.BytesIO()
    sample_image.save(buffer, format='JPEG')
    return buffer.getvalue()


def test_validate_image_format_valid(sample_image_bytes):
    """Test validation of valid image formats."""
    allowed_formats = ['jpg', 'jpeg', 'png']
    
    assert validate_image_format(sample_image_bytes, allowed_formats) == True


def test_validate_image_format_invalid():
    """Test validation of invalid image data."""
    invalid_data = b"not an image"
    allowed_formats = ['jpg', 'jpeg', 'png']
    
    assert validate_image_format(invalid_data, allowed_formats) == False


def test_get_image_metadata(sample_image):
    """Test extraction of image metadata."""
    metadata = get_image_metadata(sample_image)
    
    assert metadata['format'] is None  # New image has no format until saved
    assert metadata['mode'] == 'RGB'
    assert metadata['size'] == (800, 600)
    assert metadata['width'] == 800
    assert metadata['height'] == 600


def test_resize_image_if_needed_no_resize(sample_image):
    """Test that image is not resized when under limit."""
    resized = resize_image_if_needed(sample_image, max_size=(1000, 1000))
    
    # Should return the same image object
    assert resized.size == (800, 600)


def test_resize_image_if_needed_resize_required():
    """Test that image is resized when over limit."""
    large_image = Image.new('RGB', (2000, 1500), color='blue')
    resized = resize_image_if_needed(large_image, max_size=(1000, 1000))
    
    # Image should be resized while maintaining aspect ratio
    assert resized.size[0] <= 1000
    assert resized.size[1] <= 1000
    # Check aspect ratio is maintained (approximately)
    original_ratio = 2000 / 1500
    new_ratio = resized.size[0] / resized.size[1]
    assert abs(original_ratio - new_ratio) < 0.01


def test_create_thumbnail(sample_image):
    """Test thumbnail creation."""
    thumbnail = create_thumbnail(sample_image, size=(100, 100))
    
    assert thumbnail.size[0] <= 100
    assert thumbnail.size[1] <= 100


def test_auto_orient_image(sample_image):
    """Test auto-orientation of image."""
    # For a normal image without EXIF, should return the same image
    oriented = auto_orient_image(sample_image)
    assert oriented.size == sample_image.size


def test_format_file_size():
    """Test file size formatting."""
    assert format_file_size(500) == "500 B"
    assert format_file_size(1536) == "1.5 KB"
    assert format_file_size(2097152) == "2.0 MB"
    assert format_file_size(1073741824) == "1.0 GB"


def test_is_image_too_large():
    """Test image size validation."""
    small_data = b"x" * 1000  # 1KB
    large_data = b"x" * (5 * 1024 * 1024)  # 5MB
    
    assert is_image_too_large(small_data, 2.0) == False
    assert is_image_too_large(large_data, 2.0) == True


def test_optimize_image_for_api(sample_image_bytes):
    """Test image optimization for API."""
    # This should return optimized bytes (might be same or smaller)
    optimized = optimize_image_for_api(sample_image_bytes, max_size_kb=100)
    
    assert isinstance(optimized, bytes)
    assert len(optimized) > 0
    # Should be under the size limit (100KB = 102400 bytes)
    assert len(optimized) <= 102400


def test_get_supported_formats_info():
    """Test getting supported formats information."""
    formats_info = get_supported_formats_info()
    
    assert isinstance(formats_info, dict)
    assert 'jpg' in formats_info
    assert 'png' in formats_info
    assert 'webp' in formats_info
    
    # Check structure
    jpg_info = formats_info['jpg']
    assert 'name' in jpg_info
    assert 'description' in jpg_info


def test_resize_maintains_quality(sample_image):
    """Test that resizing maintains reasonable quality."""
    # Create a more detailed test image
    detailed_image = Image.new('RGB', (1000, 1000), color='white')
    
    resized = resize_image_if_needed(detailed_image, max_size=(500, 500))
    
    assert resized.size == (500, 500)
    assert resized.mode == detailed_image.mode


def test_optimize_image_very_large():
    """Test optimization of very large image."""
    # Create a large image
    large_image = Image.new('RGB', (3000, 3000), color='green')
    buffer = io.BytesIO()
    large_image.save(buffer, format='JPEG', quality=95)
    large_bytes = buffer.getvalue()
    
    # Optimize to very small size
    optimized = optimize_image_for_api(large_bytes, max_size_kb=50)
    
    assert len(optimized) <= 50 * 1024
    
    # Verify it's still a valid image
    optimized_image = Image.open(io.BytesIO(optimized))
    assert optimized_image.format == 'JPEG'