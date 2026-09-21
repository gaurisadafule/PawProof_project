import cv2
import numpy as np
from PIL import Image

def evaluate_sharpness(pil_image, threshold=80.0):
    """Rejects blurry camera shots using OpenCV Laplacian variance."""
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return score >= threshold, score

def apply_clahe_preprocessing(pil_image):
    """Suppresses specular reflections from wet snouts using CLAHE on LAB luminance."""
    img_np = np.array(pil_image)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb_enhanced)