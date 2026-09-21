import os
from PIL import Image
from core.biometrics import (
    extract_features, 
    compute_cosine_similarity, 
    compute_local_ridge_similarity,
    crop_rhinarium_roi
)

img_a1_path = "dog_a1.jpg"
img_a2_path = "dog_a2.jpg"
img_b_path  = "dog_b.jpg"

for p in [img_a1_path, img_a2_path, img_b_path]:
    if not os.path.exists(p):
        print(f"[ERROR] Missing test image: {p}. Run 'python generate_test_suite.py' first.")
        exit(1)

print("Loading test images...")
img_a1 = Image.open(img_a1_path).convert("RGB")
img_a2 = Image.open(img_a2_path).convert("RGB")
img_b  = Image.open(img_b_path).convert("RGB")

print("Running Two-Tier Biometric Pipeline (YOLO + MobileNetV2 + SIFT)...")
print("-" * 65)

# --- TIER 1: Deep Feature Extraction & Cosine Screening ---
vec_a1 = extract_features(img_a1)
vec_a2 = extract_features(img_a2)
vec_b  = extract_features(img_b)

cos_same = compute_cosine_similarity(vec_a1, vec_a2)
cos_diff = compute_cosine_similarity(vec_a1, vec_b)

print("TIER 1: GLOBAL EMBEDDING (Cosine Screening)")
print(f"  • Same Dog   (Dog A1 vs A2): {cos_same * 100:.1f}%")
print(f"  • Different Dog (Dog A1 vs B) : {cos_diff * 100:.1f}%")
print("-" * 65)

# --- TIER 2: Micro-Dermal SIFT Ridge Verification ---
sift_same = compute_local_ridge_similarity(img_a1, img_a2)
sift_diff = compute_local_ridge_similarity(img_a1, img_b)

print("TIER 2: MICRO-DERMATOGLYPHIC VERIFICATION (SIFT Ridge Keypoints)")
print(f"  • Same Dog   (Dog A1 vs A2): {sift_same * 100:.1f}% Match")
print(f"  • Different Dog (Dog A1 vs B) : {sift_diff * 100:.1f}% Match")
print("-" * 65)

# --- FINAL SYSTEM VERDICT ---
print("FINAL PIPELINE VERDICT:")

# Same dog evaluation
if cos_same >= 0.70 and sift_same >= 0.60:
    print("  [PASS] Dog A1 vs A2: Confirmed SAME BIOLOGICAL INDIVIDUAL.")
else:
    print(f"  [FAIL] Dog A1 vs A2: False rejection (Cosine: {cos_same:.2f}, SIFT: {sift_same:.2f}).")

# Different dog evaluation
if sift_diff < 0.50:
    print("  [PASS] Dog A1 vs Dog B: Successfully REJECTED as a distinct animal.")
else:
    print(f"  [FAIL] Dog A1 vs Dog B: False positive match detected (SIFT: {sift_diff:.2f}).")

print("-" * 65)