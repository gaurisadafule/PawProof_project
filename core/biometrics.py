import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# 1. YOLO Detector for automated snout cropping
_yolo_model = YOLO("yolov8n.pt")

# 2. Tier 1: MobileNetV2 Global Feature Extractor
_weights = models.MobileNet_V2_Weights.DEFAULT
_mobilenet = models.mobilenet_v2(weights=_weights)
_mobilenet.classifier = torch.nn.Identity()
_mobilenet.eval()

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Tier 2: SIFT Keypoint Extractor for micro-dermal ridges
_sift = cv2.SIFT_create(nfeatures=600)

def detect_and_crop_snout(pil_image):
    """
    Uses YOLOv8 to locate the dog's head/muzzle and isolate the rhinarium.
    Falls back to center crop if unobstructed detection is not possible.
    """
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    results = _yolo_model(img_cv, verbose=False)
    
    boxes = results[0].boxes
    best_box = None
    
    # Class 16 in COCO is 'dog'
    for box in boxes:
        cls_id = int(box.cls[0].item())
        if cls_id == 16:  # Confirmed dog class
            best_box = box.xyxy[0].cpu().numpy()
            break
            
    if best_box is None and len(boxes) > 0:
        best_box = boxes[0].xyxy[0].cpu().numpy()

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)
        box_h = y2 - y1
        box_w = x2 - x1
        
        # Isolate the lower muzzle region containing the rhinarium
        snout_y1 = max(0, y1 + int(box_h * 0.25))
        snout_y2 = min(img_cv.shape[0], y2 - int(box_h * 0.05))
        snout_x1 = max(0, x1 + int(box_w * 0.15))
        snout_x2 = min(img_cv.shape[1], x2 - int(box_w * 0.15))
        
        cropped_cv = img_cv[snout_y1:snout_y2, snout_x1:snout_x2]
        if cropped_cv.size > 0:
            return Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))

    # Fallback to center-weighted 60% crop
    w, h = pil_image.size
    cw, ch = int(w * 0.60), int(h * 0.60)
    return pil_image.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))

def extract_rhinarium_texture(pil_image):
    """
    Crops the snout via YOLO and applies CLAHE with high-pass filtering 
    to isolate biological grooves and suppress coat color.
    """
    snout_crop = detect_and_crop_snout(pil_image)
    gray = cv2.cvtColor(np.array(snout_crop), cv2.COLOR_RGB2GRAY)

    # Equalize sunlight reflections and flash
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # High-pass unsharp mask
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    ridges = cv2.addWeighted(equalized, 1.6, blurred, -0.6, 0)
    return snout_crop, ridges

def extract_features(pil_image):
    """Generates the 1280-D MobileNetV2 embedding for database screening."""
    snout_crop, _ = extract_rhinarium_texture(pil_image)
    tensor = _transform(snout_crop).unsqueeze(0)
    with torch.no_grad():
        feat = _mobilenet(tensor).squeeze().numpy()
    norm = np.linalg.norm(feat)
    return (feat / norm).tolist() if norm > 0 else feat.tolist()

def compute_cosine_similarity(vec_a, vec_b):
    """Fast vector comparison for candidate screening."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    return float(np.dot(a, b))

def compute_local_ridge_similarity(img1_pil, img2_pil):
    """
    Tier 2: Matches microscopic dermal grooves between two snouts using SIFT.
    Differentiates between two visually identical street dogs.
    """
    _, ridges1 = extract_rhinarium_texture(img1_pil)
    _, ridges2 = extract_rhinarium_texture(img2_pil)

    kp1, des1 = _sift.detectAndCompute(ridges1, None)
    kp2, des2 = _sift.detectAndCompute(ridges2, None)

    if des1 is None or des2 is None or len(des1) < 12 or len(des2) < 12:
        return 0.0

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test to filter false matches
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    min_kp = min(len(kp1), len(kp2))
    
    return min(1.0, float((len(good_matches) / min_kp) * 3.5)) if min_kp > 0 else 0.0

def crop_rhinarium_roi(pil_image):
    """Helper returning the YOLO-cropped snout for UI display."""
    return detect_and_crop_snout(pil_image)