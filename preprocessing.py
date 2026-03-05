import cv2
import numpy as np
import fitz
from PIL import Image
import os

def pdf_to_images(pdf_path, output_dir="data/temp_images"):
    """Converts PDF pages to images — 1.5x res balances quality vs OCR speed."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    doc = fitz.open(pdf_path)
    image_paths = []

    for i in range(len(doc)):
        page = doc.load_page(i)
        # 1.5x is sufficient for OCR — 2x unnecessarily slows things down
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_path = os.path.join(output_dir, f"page_{i+1}.png")
        pix.save(img_path)
        image_paths.append(img_path)

    return image_paths

def preprocess_image(image_path):
    """Applies deskewing, noise reduction, and contrast enhancement."""
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Remove noise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Thresholding for better OCR
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Optional Deskewing (Mental Note: Implementation can be complex, keeping it simple for now)
    # Simple contrast stretch
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    enhanced_path = image_path.replace(".png", "_preprocessed.png")
    cv2.imwrite(enhanced_path, enhanced)
    
    return enhanced_path
