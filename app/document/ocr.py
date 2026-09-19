"""Tesseract OCR module for extracted image pages and scanned PDFs."""

import os
import logging
from PIL import Image

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Configure default Windows path for Tesseract if present
TESSERACT_PATH_WINDOWS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def configure_tesseract():
    """Attempt to configure Tesseract binary path on Windows."""
    if not PYTESSERACT_AVAILABLE:
        return False

    env_path = os.getenv("TESSERACT_CMD", TESSERACT_PATH_WINDOWS)
    if os.path.exists(env_path):
        pytesseract.pytesseract.tesseract_cmd = env_path
        return True
    
    # Try finding in system PATH
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        logger.warning("Tesseract OCR binary not found. OCR features will be gracefully disabled.")
        return False


def run_ocr_on_image(image: Image.Image) -> str:
    """Run OCR on a PIL Image object."""
    if not PYTESSERACT_AVAILABLE or not configure_tesseract():
        return ""
    
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        logger.warning(f"OCR failed on image: {e}")
        return ""


def run_ocr_on_pixmap(pixmap) -> str:
    """Run OCR directly on a PyMuPDF Pixmap."""
    if not PYTESSERACT_AVAILABLE or not configure_tesseract():
        return ""

    try:
        # Convert pixmap to PIL Image
        mode = "RGBA" if pixmap.alpha else "RGB"
        img = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)
        return run_ocr_on_image(img)
    except Exception as e:
        logger.warning(f"OCR failed on pixmap: {e}")
        return ""
