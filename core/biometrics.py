# ============================================================
# PawProof - Dog Rhinarium Biometric Matching
# biometrics.py
# ============================================================

import os
import cv2
import numpy as np
import torch

from PIL import Image
from ultralytics import YOLO
from torchvision import models, transforms


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

YOLO_MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolov8n.pt"
)


# ============================================================
# 2. YOLOv8 DOG DETECTOR
# ============================================================

_yolo_model = YOLO(YOLO_MODEL_PATH)


# ============================================================
# 3. MOBILENETV2 FEATURE EXTRACTOR
# ============================================================

_device = torch.device("cpu")

_weights = models.MobileNet_V2_Weights.DEFAULT

_mobilenet = models.mobilenet_v2(
    weights=_weights
)

_mobilenet.classifier = torch.nn.Identity()

_mobilenet.eval()
_mobilenet.to(_device)


_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. IMAGE CONVERSION
# ============================================================

def _to_pil(image):

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):

        if len(image.shape) == 2:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        elif image.shape[2] == 3:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        return Image.fromarray(image).convert("RGB")

    if isinstance(image, str):
        return Image.open(image).convert("RGB")

    raise ValueError(
        "Unsupported image type"
    )


# ============================================================
# 5. YOLO DOG DETECTION
# ============================================================

def _detect_dog(pil_image):

    pil_image = _to_pil(pil_image)

    img = np.array(pil_image)

    results = _yolo_model.predict(
        source=img,
        conf=0.15,
        iou=0.45,
        verbose=False
    )

    if not results:
        return None

    result = results[0]

    if result.boxes is None:
        return None

    best_box = None
    best_conf = 0.0

    for box in result.boxes:

        cls_id = int(
            box.cls[0].item()
        )

        conf = float(
            box.conf[0].item()
        )

        # COCO class 16 = dog
        if cls_id == 16 and conf > best_conf:

            best_conf = conf

            best_box = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

    if best_box is None:
        return None

    return [
        int(best_box[0]),
        int(best_box[1]),
        int(best_box[2]),
        int(best_box[3])
    ]


# ============================================================
# 6. UNIVERSAL SNOUT / RHINARIUM CROP
# ============================================================

def detect_and_crop_snout(pil_image):

    pil_image = _to_pil(pil_image)

    img = np.array(pil_image)

    height, width = img.shape[:2]

    bbox = _detect_dog(
        pil_image
    )

    # --------------------------------------------------------
    # CASE A:
    # YOLO successfully detects the dog
    # --------------------------------------------------------

    if bbox is not None:

        x1, y1, x2, y2 = bbox

        box_w = max(
            1,
            x2 - x1
        )

        box_h = max(
            1,
            y2 - y1
        )

        # The rhinarium is normally located around
        # the lower-middle portion of the face.
        #
        # Use a reasonably large region instead of an
        # extremely tight crop so different camera
        # distances and angles are tolerated.

        rx1 = int(
            x1 + box_w * 0.20
        )

        rx2 = int(
            x1 + box_w * 0.80
        )

        ry1 = int(
            y1 + box_h * 0.40
        )

        ry2 = int(
            y1 + box_h * 0.92
        )

        # Safety margin
        margin_x = int(
            box_w * 0.08
        )

        margin_y = int(
            box_h * 0.04
        )

        rx1 -= margin_x
        rx2 += margin_x
        ry1 -= margin_y
        ry2 += margin_y

        rx1 = max(
            0,
            rx1
        )

        ry1 = max(
            0,
            ry1
        )

        rx2 = min(
            width,
            rx2
        )

        ry2 = min(
            height,
            ry2
        )

        crop = img[
            ry1:ry2,
            rx1:rx2
        ]

        if crop.size > 0:

            return Image.fromarray(
                crop
            ).convert("RGB")

    # --------------------------------------------------------
    # CASE B:
    # IMAGE IS ALREADY A CLOSE-UP SNOUT PHOTO
    # --------------------------------------------------------

    # Central/lower-middle fallback.
    # This allows PawProof to work even when YOLO
    # cannot detect a complete dog.

    rx1 = int(
        width * 0.15
    )

    rx2 = int(
        width * 0.85
    )

    ry1 = int(
        height * 0.30
    )

    ry2 = int(
        height * 0.95
    )

    crop = img[
        ry1:ry2,
        rx1:rx2
    ]

    if crop.size > 0:

        return Image.fromarray(
            crop
        ).convert("RGB")

    return pil_image


# ============================================================
# 7. COMPATIBILITY ALIAS
# ============================================================

def crop_rhinarium_roi(pil_image):

    return detect_and_crop_snout(
        pil_image
    )


# ============================================================
# 8. RHINARIUM TEXTURE EXTRACTION
# ============================================================

def extract_rhinarium_texture(pil_image):

    snout_crop = detect_and_crop_snout(
        pil_image
    )

    # Standard biometric resolution
    snout_crop = snout_crop.resize(
        (512, 384),
        Image.Resampling.LANCZOS
    )

    img = np.array(
        snout_crop
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    equalized = clahe.apply(
        gray
    )

    # --------------------------------------------------------
    # Mild smoothing
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        equalized,
        (3, 3),
        0
    )

    # --------------------------------------------------------
    # Detail enhancement
    # --------------------------------------------------------

    ridges = cv2.addWeighted(
        equalized,
        1.30,
        blurred,
        -0.30,
        0
    )

    return (
        snout_crop,
        ridges
    )


# ============================================================
# 9. MOBILE NET FEATURE EXTRACTION
# ============================================================

def extract_features(pil_image):

    snout_crop, _ = (
        extract_rhinarium_texture(
            pil_image
        )
    )

    tensor = _transform(
        snout_crop
    ).unsqueeze(0).to(
        _device
    )

    with torch.no_grad():

        feat = _mobilenet(
            tensor
        ).squeeze().cpu().numpy()

    # Normalize embedding
    norm = np.linalg.norm(
        feat
    )

    if norm > 0:

        feat = (
            feat / norm
        )

    return feat.tolist()


# ============================================================
# 10. COSINE SIMILARITY
# ============================================================

def compute_cosine_similarity(
    vec_a,
    vec_b
):

    a = np.asarray(
        vec_a,
        dtype=np.float32
    )

    b = np.asarray(
        vec_b,
        dtype=np.float32
    )

    norm_a = np.linalg.norm(
        a
    )

    norm_b = np.linalg.norm(
        b
    )

    if norm_a == 0 or norm_b == 0:

        return 0.0

    similarity = (
        np.dot(a, b)
        /
        (
            norm_a
            *
            norm_b
        )
    )

    return float(
        np.clip(
            similarity,
            -1.0,
            1.0
        )
    )


# ============================================================
# 11. SIFT LOCAL RHINARIUM MATCHING
# ============================================================

def compute_local_ridge_similarity(
    img1_pil,
    img2_pil
):

    _, ridges1 = (
        extract_rhinarium_texture(
            img1_pil
        )
    )

    _, ridges2 = (
        extract_rhinarium_texture(
            img2_pil
        )
    )

    # Create SIFT independently for every comparison
    sift = cv2.SIFT_create(
        nfeatures=1500,
        contrastThreshold=0.015,
        edgeThreshold=12,
        sigma=1.2
    )

    kp1, des1 = (
        sift.detectAndCompute(
            ridges1,
            None
        )
    )

    kp2, des2 = (
        sift.detectAndCompute(
            ridges2,
            None
        )
    )

    if (
        des1 is None
        or des2 is None
        or len(des1) < 5
        or len(des2) < 5
    ):

        return 0.0

    # --------------------------------------------------------
    # FLANN
    # --------------------------------------------------------

    index_params = dict(
        algorithm=1,
        trees=5
    )

    search_params = dict(
        checks=50
    )

    flann = cv2.FlannBasedMatcher(
        index_params,
        search_params
    )

    matches = flann.knnMatch(
        des1,
        des2,
        k=2
    )

    # --------------------------------------------------------
    # Lowe ratio test
    # --------------------------------------------------------

    good_matches = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if (
            m.distance
            <
            0.80 * n.distance
        ):

            good_matches.append(m)

    min_kp = min(
        len(kp1),
        len(kp2)
    )

    if min_kp == 0:

        return 0.0

    match_ratio = (
        len(good_matches)
        /
        min_kp
    )

    # Scale local similarity
    score = min(
        1.0,
        match_ratio * 3.5
    )

    # Additional confidence from absolute
    # number of geometrically useful matches
    if len(good_matches) >= 20:

        score = max(
            score,
            0.75
        )

    elif len(good_matches) >= 12:

        score = max(
            score,
            0.60
        )

    elif len(good_matches) >= 8:

        score = max(
            score,
            0.45
        )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# 12. ORB LOCAL MATCHING
# ============================================================

def compute_orb_similarity(
    img1_pil,
    img2_pil
):

    _, ridges1 = (
        extract_rhinarium_texture(
            img1_pil
        )
    )

    _, ridges2 = (
        extract_rhinarium_texture(
            img2_pil
        )
    )

    orb = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=8
    )

    kp1, des1 = (
        orb.detectAndCompute(
            ridges1,
            None
        )
    )

    kp2, des2 = (
        orb.detectAndCompute(
            ridges2,
            None
        )
    )

    if (
        des1 is None
        or des2 is None
        or len(des1) < 5
        or len(des2) < 5
    ):

        return 0.0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False
    )

    matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if (
            m.distance
            <
            0.78 * n.distance
        ):

            good_matches.append(m)

    min_kp = min(
        len(kp1),
        len(kp2)
    )

    if min_kp == 0:

        return 0.0

    match_ratio = (
        len(good_matches)
        /
        min_kp
    )

    score = min(
        1.0,
        match_ratio * 10.0
    )

    if len(good_matches) >= 30:

        score = max(
            score,
            0.75
        )

    elif len(good_matches) >= 20:

        score = max(
            score,
            0.60
        )

    elif len(good_matches) >= 12:

        score = max(
            score,
            0.45
        )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# 13. COMPLETE BIOMETRIC MATCH
# ============================================================

def biometric_match(
    img1_pil,
    img2_pil
):

    # --------------------------------------------------------
    # Tier 1 - MobileNetV2
    # --------------------------------------------------------

    feat1 = extract_features(
        img1_pil
    )

    feat2 = extract_features(
        img2_pil
    )

    global_score = (
        compute_cosine_similarity(
            feat1,
            feat2
        )
    )

    # --------------------------------------------------------
    # Convert -1..1 cosine score to 0..1
    # --------------------------------------------------------

    global_normalized = (
        global_score + 1.0
    ) / 2.0

    # --------------------------------------------------------
    # Tier 2 - SIFT
    # --------------------------------------------------------

    sift_score = (
        compute_local_ridge_similarity(
            img1_pil,
            img2_pil
        )
    )

    # --------------------------------------------------------
    # Tier 3 - ORB
    # --------------------------------------------------------

    orb_score = (
        compute_orb_similarity(
            img1_pil,
            img2_pil
        )
    )

    # --------------------------------------------------------
    # Weighted fusion
    # --------------------------------------------------------

    final_score = (
        0.55 * global_normalized
        +
        0.30 * sift_score
        +
        0.15 * orb_score
    )

    # --------------------------------------------------------
    # Prevent local features from dominating when the
    # deep visual representation is strongly different.
    # --------------------------------------------------------

    if global_normalized < 0.45:

        final_score *= 0.65

    elif global_normalized < 0.55:

        final_score *= 0.80

    final_score = float(
        np.clip(
            final_score,
            0.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # MATCH THRESHOLD
    # --------------------------------------------------------

    MATCH_THRESHOLD = 0.62

    is_match = (
        final_score
        >=
        MATCH_THRESHOLD
    )

    return {

        "global_similarity":
            float(global_score),

        "global_normalized":
            float(global_normalized),

        "sift_similarity":
            float(sift_score),

        "orb_similarity":
            float(orb_score),

        "final_similarity":
            float(final_score),

        "threshold":
            float(MATCH_THRESHOLD),

        "match":
            bool(is_match)
    }


# ============================================================
# 14. COMPATIBILITY FUNCTION FOR APP.PY
# ============================================================

def compare_images(
    img1_pil,
    img2_pil
):

    result = biometric_match(
        img1_pil,
        img2_pil
    )

    return result["match"]


# ============================================================
# 15. DEBUG / VISUALIZATION FUNCTION
# ============================================================

def get_rhinarium_debug_images(
    pil_image
):

    pil_image = _to_pil(
        pil_image
    )

    crop, ridges = (
        extract_rhinarium_texture(
            pil_image
        )
    )

    ridge_rgb = cv2.cvtColor(
        ridges,
        cv2.COLOR_GRAY2RGB
    )

    ridge_pil = Image.fromarray(
        ridge_rgb
    )

    return (
        pil_image,
        crop,
        ridge_pil
    )


# ============================================================
# 16. MODULE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "PawProof Rhinarium Biometric System"
    )

    print("=" * 60)

    print(
        "\nBiometric module loaded successfully."
    )

    print(
        "\nYOLO model:"
    )

    print(
        YOLO_MODEL_PATH
    )

    print(
        "\nComponents:"
    )

    print(
        "1. YOLOv8 Dog Detector"
    )

    print(
        "2. MobileNetV2 Global Embedding"
    )

    print(
        "3. SIFT Local Rhinarium Matching"
    )

    print(
        "4. ORB Supplementary Matching"
    )

    print(
        "5. Weighted Score Fusion"
    )

    print(
        "\nAll app.py compatibility functions are available."
    )

    print(
        "\nReady for biometric comparison."
    )