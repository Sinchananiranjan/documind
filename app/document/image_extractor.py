"""Image extraction and image metadata module."""

import io
import base64
import logging
from typing import List, Dict, Any
from PIL import Image
from app.document.ocr import run_ocr_on_image

logger = logging.getLogger(__name__)

def extract_images_from_page(doc, page, page_num: int) -> List[Dict[str, Any]]:
    """
    Extract embedded images from a PyMuPDF page.
    Returns list of dicts: {'page_num': int, 'image_id': str, 'ocr_text': str, 'base64': str, 'description': str}
    """
    extracted_images = []
    image_list = page.get_images(full=True)

    for img_idx, img in enumerate(image_list):
        try:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Load with PIL
            pil_img = Image.open(io.BytesIO(image_bytes))
            
            # Filter out tiny icons / small line decorations (< 50x50)
            width, height = pil_img.size
            if width < 50 or height < 50:
                continue

            # Convert to base64
            buffered = io.BytesIO()
            pil_img.convert("RGB").save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Run OCR on image to capture text inside figures/diagrams
            ocr_text = run_ocr_on_image(pil_img)
            
            description = f"Image on page {page_num} (Dimensions: {width}x{height}, Format: {image_ext.upper()})"
            if ocr_text:
                description += f" | Extracted Image Text: {ocr_text}"

            extracted_images.append({
                "page_num": page_num,
                "image_id": f"p{page_num}_img{img_idx+1}",
                "ocr_text": ocr_text,
                "base64": img_b64,
                "description": description,
                "width": width,
                "height": height
            })
        except Exception as e:
            logger.warning(f"Failed to extract image {img_idx} on page {page_num}: {e}")

    return extracted_images
