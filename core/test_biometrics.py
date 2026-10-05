from PIL import Image

from biometrics import biometric_match


# ============================================================
# LOAD DOG 5 AND DOG 6
# ============================================================

img1 = Image.open(
    "dog5.jpg"
).convert("RGB")

img2 = Image.open(
    "dog6.jpg"
).convert("RGB")


print("\nProcessing Dog 5 and Dog 6...")


# ============================================================
# RUN PAWPROOF BIOMETRIC MATCH
# ============================================================

result = biometric_match(
    img1,
    img2
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\n========== PAWPROOF RESULT =========="
)

print(
    "MobileNet similarity :",
    round(
        result["global_similarity"],
        4
    )
)

print(
    "SIFT similarity      :",
    round(
        result["sift_similarity"],
        4
    )
)

print(
    "ORB similarity       :",
    round(
        result["orb_similarity"],
        4
    )
)

print(
    "Final similarity     :",
    round(
        result["final_similarity"],
        4
    )
)

print(
    "Threshold            :",
    round(
        result["threshold"],
        4
    )
)

print(
    "-------------------------------------"
)


if result["match"]:

    print(
        "🐶 MATCH"
    )

else:

    print(
        "❌ NO MATCH"
    )


print(
    "====================================="
)