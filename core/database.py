import os
import hashlib
import mysql.connector
import numpy as np
from datetime import datetime

STORAGE_DIR = os.path.join("static", "uploads", "canine_records")
LEADS_DIR = os.path.join("static", "uploads", "citizen_leads")
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(LEADS_DIR, exist_ok=True)

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "GauriSQL@14",  # <-- REPLACE WITH YOUR MYSQL PASSWORD
    "database": "pawproof_db"
}

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def init_db():
    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
    cursor.close()
    conn.close()

    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. System Users
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_users (
            username VARCHAR(64) PRIMARY KEY,
            password_hash VARCHAR(64) NOT NULL,
            full_name VARCHAR(128) NOT NULL,
            role ENUM('VET_OFFICE', 'FEEDER', 'TRANSIT', 'MUNICIPAL_CORP') NOT NULL,
            ward VARCHAR(128) NOT NULL,
            badge_or_vci VARCHAR(64) NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)

    # 2. Master Canine Registry
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS canine_registry (
            token_id VARCHAR(64) PRIMARY KEY,
            ward VARCHAR(128) NOT NULL,
            sex ENUM('MALE', 'FEMALE') NOT NULL,
            status ENUM('CAPTURED_IN_TRANSIT', 'STERILIZED_VACCINATED', 'RELEASED') NOT NULL DEFAULT 'CAPTURED_IN_TRANSIT',
            vax_date DATE NULL,
            booster_due DATE NULL,
            arv_batch VARCHAR(64) NULL,
            capture_lat DECIMAL(10, 8) NULL,
            capture_lng DECIMAL(11, 8) NULL,
            capture_landmark VARCHAR(255) NULL,
            enrolled_by VARCHAR(64) NOT NULL,
            role VARCHAR(128) NOT NULL,
            image_path VARCHAR(255) NOT NULL,
            embedding BLOB NOT NULL,
            release_verified TINYINT(1) DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_canine_enrolled_by 
                FOREIGN KEY (enrolled_by) REFERENCES system_users(username)
                ON UPDATE CASCADE ON DELETE RESTRICT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)

    # 3. Optional Citizen / Feeder Leads Queue
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS intake_requests (
            request_id INT AUTO_INCREMENT PRIMARY KEY,
            ward VARCHAR(128) NOT NULL,
            landmark VARCHAR(255) NOT NULL,
            lat DECIMAL(10, 8) NOT NULL,
            lng DECIMAL(11, 8) NOT NULL,
            image_path VARCHAR(255) NOT NULL,
            embedding BLOB NOT NULL,
            reported_by VARCHAR(64) NOT NULL,
            status ENUM('PENDING_PICKUP', 'CAPTURED', 'DISMISSED') DEFAULT 'PENDING_PICKUP',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (reported_by) REFERENCES system_users(username)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    conn.commit()
    cursor.close()
    conn.close()

def create_user(username, password, full_name, role, ward, badge_or_vci=""):
    conn = get_connection()
    cursor = conn.cursor()
    pwd_hash = hash_password(password)
    try:
        sql = """
            INSERT INTO system_users (username, password_hash, full_name, role, ward, badge_or_vci)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (username.strip().lower(), pwd_hash, full_name.strip(), role, ward, badge_or_vci.strip()))
        conn.commit()
        return True, "Account registered successfully!"
    except mysql.connector.IntegrityError:
        return False, "Username already exists. Please choose a different one."
    except Exception as e:
        return False, str(e)
    finally:
        cursor.close()
        conn.close()

def authenticate_user(username, password):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    pwd_hash = hash_password(password)
    cursor.execute("SELECT * FROM system_users WHERE username = %s AND password_hash = %s", (username.strip().lower(), pwd_hash))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

def generate_location_token(ward_name, ward_meta_dict):
    meta = ward_meta_dict.get(ward_name, {"city_code": "GEN", "zone_code": "ZONE"})
    city, zone = meta["city_code"], meta["zone_code"]
    year = datetime.now().strftime("%y")
    prefix = f"{city}-{zone}-{year}-"

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT token_id FROM canine_registry WHERE token_id LIKE %s ORDER BY token_id DESC LIMIT 1", (f"{prefix}%",))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row and row[0]:
        try:
            last_seq = int(row[0].split("-")[-1])
            new_seq = last_seq + 1
        except (ValueError, IndexError):
            new_seq = 1
    else:
        new_seq = 1

    return f"{prefix}{new_seq:04d}"

def save_image_to_disk(pil_image, name_prefix, folder="canine_records"):
    target_dir = os.path.join("static", "uploads", folder)
    os.makedirs(target_dir, exist_ok=True)
    filename = f"{name_prefix}.jpg"
    filepath = os.path.join(target_dir, filename)
    pil_image.save(filepath, format="JPEG", quality=85)
    return filepath

# --- Citizen / Feeder Leads ---
def insert_citizen_request(ward, landmark, lat, lng, image_path, embedding_list, reported_by):
    conn = get_connection()
    cursor = conn.cursor()
    emb_bytes = np.array(embedding_list, dtype=np.float32).tobytes()
    sql = """
        INSERT INTO intake_requests (ward, landmark, lat, lng, image_path, embedding, reported_by)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(sql, (ward, landmark, lat, lng, image_path, emb_bytes, reported_by))
    conn.commit()
    cursor.close()
    conn.close()

def get_pending_requests(ward_filter=None):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    if ward_filter:
        cursor.execute("SELECT * FROM intake_requests WHERE status = 'PENDING_PICKUP' AND ward = %s ORDER BY created_at DESC", (ward_filter,))
    else:
        cursor.execute("SELECT * FROM intake_requests WHERE status = 'PENDING_PICKUP' ORDER BY created_at DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def resolve_citizen_request(request_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE intake_requests SET status = 'CAPTURED' WHERE request_id = %s", (request_id,))
    conn.commit()
    cursor.close()
    conn.close()

# --- Official Registry Intakes ---
def insert_intake_record(token_id, ward, sex, lat, lng, landmark, enrolled_by, role, image_path, embedding_list):
    conn = get_connection()
    cursor = conn.cursor()
    emb_bytes = np.array(embedding_list, dtype=np.float32).tobytes()
    sql = """
        INSERT INTO canine_registry 
        (token_id, ward, sex, status, capture_lat, capture_lng, capture_landmark, enrolled_by, role, image_path, embedding)
        VALUES (%s, %s, %s, 'CAPTURED_IN_TRANSIT', %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(sql, (token_id, ward, sex, lat, lng, landmark, enrolled_by, role, image_path, emb_bytes))
    conn.commit()
    cursor.close()
    conn.close()

def update_clinical_record(token_id, vax_date, booster_due, arv_batch):
    conn = get_connection()
    cursor = conn.cursor()
    sql = """
        UPDATE canine_registry 
        SET status = 'STERILIZED_VACCINATED', vax_date = %s, booster_due = %s, arv_batch = %s
        WHERE token_id = %s
    """
    cursor.execute(sql, (vax_date, booster_due, arv_batch, token_id))
    conn.commit()
    cursor.close()
    conn.close()

def update_release_status(token_id):
    conn = get_connection()
    cursor = conn.cursor()
    sql = "UPDATE canine_registry SET status = 'RELEASED', release_verified = 1 WHERE token_id = %s"
    cursor.execute(sql, (token_id,))
    conn.commit()
    cursor.close()
    conn.close()

def get_all_records(ward_filter=None):
    conn = get_connection()
    cursor = conn.cursor()
    if ward_filter:
        query = """
            SELECT token_id, ward, sex, status, vax_date, booster_due, arv_batch, 
                   capture_lat, capture_lng, capture_landmark, enrolled_by, role, image_path, embedding, release_verified 
            FROM canine_registry WHERE ward = %s
        """
        cursor.execute(query, (ward_filter,))
    else:
        query = """
            SELECT token_id, ward, sex, status, vax_date, booster_due, arv_batch, 
                   capture_lat, capture_lng, capture_landmark, enrolled_by, role, image_path, embedding, release_verified 
            FROM canine_registry
        """
        cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    records = []
    for r in rows:
        records.append({
            "token_id": r[0],
            "ward": r[1],
            "sex": r[2],
            "status": r[3],
            "vax_date": str(r[4]) if r[4] else "N/A",
            "booster_due": str(r[5]) if r[5] else "N/A",
            "arv_batch": r[6] if r[6] else "N/A",
            "lat": float(r[7]) if r[7] else 0.0,
            "lng": float(r[8]) if r[8] else 0.0,
            "landmark": r[9] if r[9] else "N/A",
            "enrolled_by": r[10],
            "role": r[11],
            "image_path": r[12],
            "embedding": np.frombuffer(r[13], dtype=np.float32).tolist(),
            "release_verified": bool(r[14])
        })
    return records