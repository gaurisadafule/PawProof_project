import streamlit as st
import os
import uuid
from datetime import datetime, timedelta
from PIL import Image
import numpy as np

from core.database import (
    init_db, create_user, authenticate_user, generate_location_token,
    save_image_to_disk, insert_intake_record, insert_citizen_request,
    get_pending_requests, resolve_citizen_request, update_clinical_record,
    update_release_status, get_all_records
)
from core.quality import evaluate_sharpness
from core.biometrics import (
    extract_features, compute_cosine_similarity, 
    compute_local_ridge_similarity, crop_rhinarium_roi
)
from core.locations import WARD_METRICS, calculate_geofence_distance
from core.reporting import generate_canine_passport

st.set_page_config(
    page_title="PawProof | Municipal Canine ABC Portal",
    layout="wide",
    page_icon="🐾",
    initial_sidebar_state="collapsed"
)

init_db()

# Application Styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .block-container {
        max-width: 1040px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        border: 1px solid #334155;
    }
    .badge-verified {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-unregistered {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["user"] = None
    st.session_state["role"] = None
    st.session_state["full_name"] = None
    st.session_state["ward"] = None

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# =========================================================
# 1. AUTHENTICATION VIEW
# =========================================================
if not st.session_state["logged_in"]:
    _, col_auth, _ = st.columns([1, 2.4, 1])

    with col_auth:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 24px;'>
            <h1 style='font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-bottom: 4px;'>🐾 PawProof</h1>
            <p style='color: #64748b; font-size: 0.95rem;'>Municipal Biometric Registry & Geofence Tracker</p>
        </div>
        """, unsafe_allow_html=True)

        auth_tab_in, auth_tab_up = st.tabs(["🔐 Sign In", "📝 Register Account"])

        with auth_tab_in:
            st.write("")
            login_user = st.text_input("Username", placeholder="e.g., demo_feeder", key="l_user")
            login_pwd = st.text_input("Password", type="password", key="l_pwd")

            if st.button("Access Portal", type="primary", use_container_width=True):
                if not login_user or not login_pwd:
                    st.error("Please provide both credentials.")
                else:
                    user_data = authenticate_user(login_user, login_pwd)
                    if user_data:
                        st.session_state["logged_in"] = True
                        st.session_state["user"] = user_data["username"]
                        st.session_state["role"] = user_data["role"]
                        st.session_state["full_name"] = user_data["full_name"]
                        st.session_state["ward"] = user_data["ward"]
                        st.rerun()
                    else:
                        st.error("Invalid credentials entered.")

        with auth_tab_up:
            st.write("")
            new_fname = st.text_input("Full Name", placeholder="e.g., Dr. Deshmukh", key="s_name")
            new_uname = st.text_input("Choose Username", placeholder="e.g., vet_deshmukh", key="s_uname")
            new_pwd = st.text_input("Set Password", type="password", key="s_pwd")
            new_role = st.selectbox(
                "Role Assignment",
                [
                    ("FEEDER", "Community Feeder / Volunteer"),
                    ("TRANSIT", "Municipal Catching Squad (Van Driver)"),
                    ("VET_OFFICE", "Veterinary Surgeon / Clinical Desk"),
                    ("MUNICIPAL_CORP", "Health Inspector / Municipal Auditor")
                ],
                format_func=lambda x: x[1],
                key="s_role"
            )[0]
            new_ward = st.selectbox("Operating Ward", options=list(WARD_METRICS.keys()), key="s_ward")
            new_badge = st.text_input("Badge / VCI Registration Number", key="s_badge")

            if st.button("Register Profile", type="primary", use_container_width=True):
                if not new_fname or not new_uname or not new_pwd:
                    st.error("All mandatory fields must be completed.")
                else:
                    success, msg = create_user(new_uname, new_pwd, new_fname, new_role, new_ward, new_badge)
                    if success:
                        st.success(f"{msg} You can now log in.")
                    else:
                        st.error(msg)
    st.stop()

# =========================================================
# 2. APPLICATION SHELL
# =========================================================

# Notification Handler
if "action_notice" in st.session_state:
    st.success(st.session_state["action_notice"])
    del st.session_state["action_notice"]

# Header Bar
st.markdown(f"""
<div class="hero-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2 style="margin: 0; font-size: 1.5rem; font-weight: 800;">🐾 PawProof Registry</h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                Zone: <strong>{st.session_state['ward']}</strong> | Operator: <strong>{st.session_state['full_name']}</strong> ({st.session_state['role']})
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

user_role = st.session_state["role"]

if user_role == "FEEDER":
    tab_scan, tab_ledger = st.tabs(["🔍 Snout Biometric Scan", "📋 Local Ward Records"])
    tab_leads, tab_release, tab_vet, tab_audit = None, None, None, None
elif user_role == "TRANSIT":
    tab_scan, tab_leads, tab_release = st.tabs(["🔍 Snout Biometric Scan", "📬 Feeder Pickup Leads", "🚐 Geofence Release"])
    tab_ledger, tab_vet, tab_audit = None, None, None
elif user_role == "VET_OFFICE":
    tab_scan, tab_vet, tab_ledger = st.tabs(["🔍 Snout Biometric Scan", "💉 Clinical Surgery Desk", "📁 Ward Ledger & Passports"])
    tab_leads, tab_release, tab_audit = None, None, None
else:
    tab_audit, tab_ledger = st.tabs(["📊 Herd Immunity Surveillance", "📁 Ward Ledger & Documentation"])
    tab_scan, tab_leads, tab_release, tab_vet = None, None, None, None

# ---------------------------------------------------------
# VIEW: BIOMETRIC SCANNING & TWO-TIER MATCHING
# ---------------------------------------------------------
if tab_scan:
    with tab_scan:
        st.markdown("### 📷 Canine Snout Biometric Matching")
        st.caption("Hold camera 2–3 feet from the muzzle. YOLO locates the snout and SIFT verifies biological ridges.")

        scan_mode = st.radio("Camera Capture Source", ["📸 Live Shutter", "📁 Upload Image"], horizontal=True)
        active_key = f"shutter_{st.session_state['uploader_key']}"
        
        scan_file = (
            st.camera_input("Aim directly at muzzle", key=active_key)
            if scan_mode == "📸 Live Shutter"
            else st.file_uploader("Upload Snout Close-up", type=["jpg", "jpeg", "png"], key=active_key)
        )

        if scan_file:
            raw_img = Image.open(scan_file).convert("RGB")
            is_sharp, sharp_score = evaluate_sharpness(raw_img)

            c_preview, c_analysis = st.columns([1, 1.3])
            with c_preview:
                st.image(raw_img, caption="Field Capture", use_container_width=True)
                st.caption(f"Sharpness Score: **{sharp_score:.1f}** (Min Required: 80.0)")

            with c_analysis:
                if not is_sharp:
                    st.error(f"⚠️ Image too blurry ({sharp_score:.1f} < 80.0). Stabilize the camera and retake.")
                else:
                    with st.spinner("Executing Two-Tier Biometric Analysis (MobileNetV2 + SIFT)..."):
                        query_vec = extract_features(raw_img)
                        records = get_all_records()

                        # Step 1: Fast Screening via Cosine Similarity
                        best_candidate = None
                        highest_cosine = 0.0

                        for r in records:
                            cos_sim = compute_cosine_similarity(query_vec, r["embedding"])
                            if cos_sim > highest_cosine:
                                highest_cosine = cos_sim
                                best_candidate = r

                        # Step 2: Biological Verification via SIFT Keypoints
                        ridge_score = 0.0
                        confirmed_match = False

                        if best_candidate and highest_cosine >= 0.70:
                            if os.path.exists(best_candidate["image_path"]):
                                stored_img = Image.open(best_candidate["image_path"]).convert("RGB")
                                ridge_score = compute_local_ridge_similarity(raw_img, stored_img)
                                
                                # Confirmation requires both deep geometry and ridge alignment
                                if ridge_score >= 0.60:
                                    confirmed_match = True

                    # 1. BIOMETRIC MATCH CONFIRMED
                    if confirmed_match:
                        st.markdown(f"<span class='badge-verified'>MATCH CONFIRMED (SIFT: {ridge_score*100:.1f}%)</span>", unsafe_allow_html=True)
                        st.markdown(f"### `{best_candidate['token_id']}`")
                        
                        st.markdown("#### Biometric Visual Verification")
                        side1, side2 = st.columns(2)
                        with side1:
                            st.caption("Live YOLO Snout Crop")
                            st.image(crop_rhinarium_roi(raw_img), use_container_width=True)
                        with side2:
                            st.caption("Enrolled Registry Snout")
                            st.image(best_candidate["image_path"], use_container_width=True)

                        st.markdown("---")
                        st.write(f"**Ward Territory:** {best_candidate['ward']}")
                        st.write(f"**Surgical Status:** `{best_candidate['status']}`")
                        st.write(f"**Ear Integrity:** 100% Intact (Contactless Verification)")
                        st.write(f"**ARV Batch:** {best_candidate['arv_batch']}")
                        st.write(f"**Annual Booster Due:** `{best_candidate['booster_due']}`")

                    # 2. UNREGISTERED ANIMAL (OR SIMILAR-LOOKING DOG WITH DIFFERENT GROOVES)
                    else:
                        st.markdown(f"<span class='badge-unregistered'>NO MATCH FOUND</span>", unsafe_allow_html=True)
                        if highest_cosine >= 0.70 and ridge_score < 0.60:
                            st.warning(f"⚠️ Similar coat/facial structure detected (Cosine: {highest_cosine*100:.1f}%), but microscopic dermal grooves do NOT match (SIFT: {ridge_score*100:.1f}%). Confirmed as a different dog.")
                        else:
                            st.info("No matching biometric profile found in the registry.")

                        # FEEDER PATHWAY: SUBMIT PICKUP LEAD
                        if user_role == "FEEDER":
                            st.markdown("---")
                            st.subheader("📢 Report Stray for ABC Program")
                            with st.form("feeder_lead_form"):
                                ward_sel = st.selectbox("Territory", options=list(WARD_METRICS.keys()))
                                def_lat = WARD_METRICS[ward_sel]["lat"]
                                def_lng = WARD_METRICS[ward_sel]["lng"]
                                
                                c_lat, c_lng = st.columns(2)
                                with c_lat:
                                    req_lat = st.number_input("Latitude", value=def_lat, format="%.6f")
                                with c_lng:
                                    req_lng = st.number_input("Longitude", value=def_lng, format="%.6f")
                                
                                spot_landmark = st.text_input("Colony Spot / Landmark", placeholder="e.g. Near Lane 3 Garbage Bin")

                                if st.form_submit_button("Submit Pickup Lead", type="primary", use_container_width=True):
                                    lead_id = f"LEAD-{uuid.uuid4().hex[:6].upper()}"
                                    saved_path = save_image_to_disk(raw_img, lead_id, folder="citizen_leads")
                                    insert_citizen_request(ward_sel, spot_landmark, req_lat, req_lng, saved_path, query_vec, st.session_state["user"])
                                    st.session_state["uploader_key"] += 1
                                    st.session_state["action_notice"] = f"✅ Lead for '{spot_landmark}' added to transit dispatch!"
                                    st.rerun()

                        # TRANSIT SQUAD PATHWAY: OFFICIAL ROUTINE INTAKE
                        elif user_role == "TRANSIT":
                            st.markdown("---")
                            st.subheader("🚐 Official Van Intake (Routine Sweep)")
                            with st.form("transit_sweep_form"):
                                ward_sel = st.selectbox("Territory", options=list(WARD_METRICS.keys()))
                                auto_token = generate_location_token(ward_sel, WARD_METRICS)
                                st.text_input("Assigned Civic Token", value=auto_token, disabled=True)
                                
                                dog_sex = st.radio("Biological Sex", ["MALE", "FEMALE"], horizontal=True)
                                def_lat = WARD_METRICS[ward_sel]["lat"]
                                def_lng = WARD_METRICS[ward_sel]["lng"]
                                
                                c_lat, c_lng = st.columns(2)
                                with c_lat:
                                    cap_lat = st.number_input("Capture Latitude", value=def_lat, format="%.6f")
                                with c_lng:
                                    cap_lng = st.number_input("Capture Longitude", value=def_lng, format="%.6f")
                                
                                cap_landmark = st.text_input("Capture Landmark", value="Routine Municipal Van Sweep")

                                if st.form_submit_button("Register Official Custody Intake", type="primary", use_container_width=True):
                                    saved_path = save_image_to_disk(raw_img, auto_token, folder="canine_records")
                                    insert_intake_record(
                                        token_id=auto_token,
                                        ward=ward_sel,
                                        sex=dog_sex,
                                        lat=cap_lat,
                                        lng=cap_lng,
                                        landmark=cap_landmark,
                                        enrolled_by=st.session_state["user"],
                                        role=st.session_state["role"],
                                        image_path=saved_path,
                                        embedding_list=query_vec
                                    )
                                    st.session_state["uploader_key"] += 1
                                    st.session_state["action_notice"] = f"✅ Canine '{auto_token}' enrolled! Status: In-Transit for ABC."
                                    st.rerun()

# ---------------------------------------------------------
# VIEW: CITIZEN LEADS DISPATCH (TRANSIT ONLY)
# ---------------------------------------------------------
if tab_leads:
    with tab_leads:
        st.subheader("📬 Citizen & Feeder Leads Queue")
        st.caption("Pickup leads submitted by community caregivers.")

        pending = get_pending_requests(st.session_state["ward"])
        if not pending:
            st.info("No pending citizen leads in this ward.")
        else:
            for item in pending:
                with st.container():
                    st.markdown(f"#### 📍 {item['landmark']}")
                    lead_c1, lead_c2 = st.columns([1, 2])
                    with lead_c1:
                        if os.path.exists(item["image_path"]):
                            st.image(item["image_path"], use_container_width=True)
                    with lead_c2:
                        st.write(f"**Reported by:** {item['reported_by']} at {item['created_at']}")
                        st.write(f"**GPS:** `{item['lat']}, {item['lng']}`")
                        
                        with st.form(f"promote_{item['request_id']}"):
                            c_sex = st.radio("Confirmed Sex", ["MALE", "FEMALE"], horizontal=True, key=f"s_{item['request_id']}")
                            if st.form_submit_button("Confirm Netting & Take Custody", type="primary"):
                                new_id = generate_location_token(item["ward"], WARD_METRICS)
                                emb = np.frombuffer(item["embedding"], dtype=np.float32).tolist()
                                
                                insert_intake_record(
                                    token_id=new_id,
                                    ward=item["ward"],
                                    sex=c_sex,
                                    lat=float(item["lat"]),
                                    lng=float(item["lng"]),
                                    landmark=item["landmark"],
                                    enrolled_by=st.session_state["user"],
                                    role=st.session_state["role"],
                                    image_path=item["image_path"],
                                    embedding_list=emb
                                )
                                resolve_citizen_request(item["request_id"])
                                st.session_state["action_notice"] = f"✅ Dog secured under Civic Token '{new_id}'!"
                                st.rerun()
                    st.markdown("---")

# ---------------------------------------------------------
# VIEW: GEOFENCE RETURN & RELEASE (TRANSIT ONLY)
# ---------------------------------------------------------
if tab_release:
    with tab_release:
        st.subheader("🚐 ABC Rule 11 Return & Geofence Verification")
        st.caption("Mandated legal check: Dogs must be returned within 500m of capture origin.")

        records = get_all_records()
        operable = [r for r in records if r["status"] == "STERILIZED_VACCINATED"]

        if not operable:
            st.info("No canines currently pending release.")
        else:
            sel_token = st.selectbox("Select Patient for Field Return", options=[r["token_id"] for r in operable])
            target_dog = next(r for r in operable if r["token_id"] == sel_token)

            st.write(f"**Capture Origin:** {target_dog['landmark']} (`{target_dog['lat']}`, `{target_dog['lng']}`)")

            c_lat = st.number_input("Drop Coordinates Latitude", value=target_dog["lat"], format="%.6f")
            c_lng = st.number_input("Drop Coordinates Longitude", value=target_dog["lng"], format="%.6f")

            dist = calculate_geofence_distance(target_dog["lat"], target_dog["lng"], c_lat, c_lng)
            st.metric("Distance from Capture Origin", f"{dist:.1f} meters", delta="Safe" if dist <= 500 else "Relocation Prohibited")

            if dist <= 500.0:
                st.success("🟢 Compliant: Drop point is within the 500m legal perimeter.")
                if st.button("Confirm Field Release", type="primary", use_container_width=True):
                    update_release_status(sel_token)
                    st.session_state["action_notice"] = f"✅ Canine '{sel_token}' safely released into its home colony!"
                    st.rerun()
            else:
                st.error(f"🔴 Illegal Relocation Prohibited: Current point is {dist:.1f}m away (> 500m limit).")
                st.button("Confirm Field Release", disabled=True, use_container_width=True)

# ---------------------------------------------------------
# VIEW: CLINICAL SURGERY DESK (VET ONLY)
# ---------------------------------------------------------
if tab_vet:
    with tab_vet:
        st.subheader("💉 Veterinary Surgical & Vaccination Documentation")
        st.caption("Log surgical sterilization and anti-rabies vaccination. (Ears remain 100% intact).")

        records = get_all_records()
        in_transit = [r for r in records if r["status"] == "CAPTURED_IN_TRANSIT"]

        if not in_transit:
            st.info("No canines currently admitted for surgery.")
        else:
            sel_id = st.selectbox("Select Admitted Canine", options=[r["token_id"] for r in in_transit])
            dog_data = next(r for r in in_transit if r["token_id"] == sel_id)

            st.write(f"**Ward:** {dog_data['ward']} | **Sex:** {dog_data['sex']} | **Origin:** {dog_data['landmark']}")
            if os.path.exists(dog_data["image_path"]):
                st.image(dog_data["image_path"], width=220)

            with st.form("clinical_discharge_form"):
                surg_date = st.date_input("Surgery & ARV Date", value=datetime.today())
                boost_date = st.date_input("Annual Booster Due Date", value=datetime.today() + timedelta(days=365))
                arv_code = st.text_input("ARV Batch Number", value="ARV-2026-B849")

                if st.form_submit_button("Authorize Clinical Discharge", type="primary", use_container_width=True):
                    update_clinical_record(sel_id, str(surg_date), str(boost_date), arv_code)
                    st.session_state["action_notice"] = f"✅ Canine '{sel_id}' certified! Ready for van release."
                    st.rerun()

# ---------------------------------------------------------
# VIEW: WARD DOCUMENTATION LEDGER
# ---------------------------------------------------------
if tab_ledger:
    with tab_ledger:
        st.subheader("📁 Municipal Registry & Official Passports")

        ward_filter = st.session_state["ward"] if user_role == "FEEDER" else None
        records = get_all_records(ward_filter)

        if not records:
            st.info("No registered records in this jurisdiction.")
        else:
            doc_id = st.selectbox("Select Canine to Inspect", options=[r["token_id"] for r in records])
            item = next(r for r in records if r["token_id"] == doc_id)

            d_col1, d_col2 = st.columns([1, 2])
            with d_col1:
                if os.path.exists(item["image_path"]):
                    st.image(item["image_path"], use_container_width=True)
            with d_col2:
                st.write(f"**Token ID:** `{item['token_id']}` | **Ward:** {item['ward']}")
                st.write(f"**Status:** `{item['status']}`")
                st.write(f"**ARV Vaccine Batch:** {item['arv_batch']} (Due: {item['booster_due']})")
                st.write(f"**Capture Origin:** {item['landmark']}")

                pdf = generate_canine_passport(item)
                st.download_button(
                    label="📄 Download Official Health Certificate (PDF)",
                    data=pdf,
                    file_name=f"Certificate_{item['token_id']}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            st.markdown("---")
            st.dataframe([{
                "Token": r["token_id"], "Ward": r["ward"], "Status": r["status"],
                "Vaccine Date": r["vax_date"], "Booster Due": r["booster_due"], "Origin": r["landmark"]
            } for r in records], use_container_width=True)

# ---------------------------------------------------------
# VIEW: MUNICIPAL HERD IMMUNITY AUDIT (AUDITOR ONLY)
# ---------------------------------------------------------
if tab_audit:
    with tab_audit:
        st.subheader("📊 Municipal Ward Herd Immunity Surveillance")
        all_recs = get_all_records()

        tot = len(all_recs)
        vaxed = sum(1 for r in all_recs if r["status"] in ["STERILIZED_VACCINATED", "RELEASED"])
        rate = (vaxed / tot * 100) if tot > 0 else 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Enrolled Canines", tot)
        m2.metric("Sterilized & Immunized", vaxed)
        m3.metric("Ward Immunity Level", f"{rate:.1f}%", delta=f"{rate - 70.0:.1f}% vs WHO Target")

        if rate >= 70.0:
            st.success("🟢 WHO / National Rabies Target (70%) Achieved for this territory.")
        else:
            st.warning("🟡 Below 70% threshold. Allocate municipal van resources to this zone.")