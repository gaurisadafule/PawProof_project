import os
import cv2
import numpy as np
from PIL import Image
import core.biometrics as bio
import core.database as db

def update_db(token, emb_str, img_path):
    query = "UPDATE canine_registry SET biometric_features = %s, image_path = %s WHERE permanent_token = %s"
    params = (emb_str, img_path, token)
    
    # Try common helper functions in your core.database module
    if hasattr(db, "execute_query"):
        db.execute_query(query, params)
    elif hasattr(db, "run_query"):
        db.run_query(query, params)
    elif hasattr(db, "get_db_connection") or hasattr(db, "get_connection"):
        conn_func = getattr(db, "get_db_connection", getattr(db, "get_connection", None))
        conn = conn_func()
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        cur.close()
        conn.close()
    else:
        # Fallback to direct mysql.connector if database.py uses a custom class
        import mysql.connector
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",  # Update password if your MySQL root has one
            database="pawproof_db"
        )
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        cur.close()
        conn.close()

def run():
    print("--- 1. Verification Files ---")
    rec_dir = os.path.join("static", "uploads", "canine_records")
    demo_dir = "demo_assets"
    token = "PMC-SHIV-26-0002"
    ref_path = os.path.join(rec_dir, f"{token}.jpeg")
    verify_path = os.path.join(demo_dir, "demo_verify_scan.jpg")

    print(f"Using reference: {ref_path}")
    print(f"Using verify:    {verify_path}")

    print("\n--- 2. Updating Database Biometric Embedding ---")
    pil_ref = Image.open(ref_path).convert("RGB")
    emb = bio.extract_features(pil_ref)
    emb_str = str(emb)

    try:
        update_db(token, emb_str, ref_path)
        print(f"✓ Database record [{token}] updated successfully.")
    except Exception as e:
        print(f"Database update error: {e}")
        print("Tip: If using a password for MySQL, check user/password in update_db().")

    print("\n--- 3. Testing Biometric Scores ---")
    pil_verify = Image.open(verify_path).convert("RGB")
    vec_b = bio.extract_features(pil_verify)
    cosine = bio.compute_cosine_similarity(emb, vec_b)
    ridge = bio.compute_local_ridge_similarity(pil_ref, pil_verify)

    print(f"Cosine Similarity : {cosine:.4f} (Threshold: 0.55)")
    print(f"Ridge Score       : {ridge:.4f} (Threshold: 0.28)")

    if cosine >= 0.55 and ridge >= 0.28:
        print("\n>>> ALL CHECKS PASSED. READY FOR DEMO! <<<")
    else:
        print("\n>>> Check similarity thresholds if scores are close. <<<")

if __name__ == "__main__":
    run()