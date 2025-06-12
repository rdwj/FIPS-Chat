"""Image processing utilities for the Ollama Streamlit application."""

from PIL import Image, ImageOps
import io
from typing import Tuple, Optional
import streamlit as st


def validate_image_format(file_data: bytes, allowed_formats: list) -> bool:
    """Validate if uploaded file is a supported image format."""
    try:
        image = Image.open(io.BytesIO(file_data))
        return image.format.lower() in [fmt.lower() for fmt in allowed_formats]
    except Exception:
        return False


def get_image_metadata(image: Image.Image) -> dict:
    """Extract metadata from PIL Image object."""
    metadata = {
        "format": image.format,
        "mode": image.mode,
        "size": image.size,
        "width": image.size[0],
        "height": image.size[1],
    }
    
    # Add additional info if available
    if hasattr(image, 'info'):
        info = image.info
        if 'dpi' in info:
            metadata['dpi'] = info['dpi']
        if 'description' in info:
            metadata['description'] = info['description']
    
    return metadata


def resize_image_if_needed(image: Image.Image, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
    """Resize image if it exceeds maximum dimensions while maintaining aspect ratio."""
    if image.size[0] <= max_size[0] and image.size[1] <= max_size[1]:
        return image
    
    # Calculate new size maintaining aspect ratio
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def optimize_image_for_api(image_data: bytes, max_size_kb: int = 1024) -> bytes:
    """Optimize image for API transmission by reducing size if needed."""
    try:
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary (for JPEG)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Convert to RGB for JPEG compatibility
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            rgb_image.paste(image, mask=image.split()[-1] if len(image.split()) > 3 else None)
            image = rgb_image
        
        # Start with original quality
        quality = 95
        
        while quality > 10:
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=quality, optimize=True)
            
            if len(output.getvalue()) <= max_size_kb * 1024:
                return output.getvalue()
            
            quality -= 10
        
        # If still too large, resize the image
        max_dimension = 800
        while max_dimension > 200:
            resized_image = resize_image_if_needed(image, (max_dimension, max_dimension))
            output = io.BytesIO()
            resized_image.save(output, format='JPEG', quality=80, optimize=True)
            
            if len(output.getvalue()) <= max_size_kb * 1024:
                return output.getvalue()
            
            max_dimension -= 100
        
        # Return final attempt
        return output.getvalue()
    
    except Exception as e:
        st.error(f"Error optimizing image: {str(e)}")
        return image_data


def create_thumbnail(image: Image.Image, size: Tuple[int, int] = (150, 150)) -> Image.Image:
    """Create a thumbnail of the image."""
    thumbnail = image.copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    return thumbnail


def auto_orient_image(image: Image.Image) -> Image.Image:
    """Auto-orient image based on EXIF data."""
    try:
        return ImageOps.exif_transpose(image)
    except Exception:
        return image


def calculate_file_size_reduction(original_size: int, optimized_size: int) -> dict:
    """Calculate file size reduction statistics."""
    reduction_bytes = original_size - optimized_size
    reduction_percent = (reduction_bytes / original_size) * 100 if original_size > 0 else 0
    
    return {
        "original_size": original_size,
        "optimized_size": optimized_size,
        "reduction_bytes": reduction_bytes,
        "reduction_percent": reduction_percent,
        "compression_ratio": original_size / optimized_size if optimized_size > 0 else 1
    }


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def is_image_too_large(image_data: bytes, max_size_mb: float) -> bool:
    """Check if image file size exceeds the maximum allowed size."""
    size_mb = len(image_data) / (1024 * 1024)
    return size_mb > max_size_mb


def get_supported_formats_info() -> dict:
    """Get information about supported image formats."""
    return {
        "jpg": {"name": "JPEG", "description": "Joint Photographic Experts Group"},
        "jpeg": {"name": "JPEG", "description": "Joint Photographic Experts Group"},
        "png": {"name": "PNG", "description": "Portable Network Graphics"},
        "webp": {"name": "WebP", "description": "Google WebP format"},
        "gif": {"name": "GIF", "description": "Graphics Interchange Format"},
        "bmp": {"name": "BMP", "description": "Bitmap Image File"},
        "tiff": {"name": "TIFF", "description": "Tagged Image File Format"}
    }