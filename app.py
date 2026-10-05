import streamlit as st
import os
import re
import base64
import uuid
from datetime import datetime, timedelta
from PIL import Image
import numpy as np

# Safe Module Import
import core.database as db

from core.quality import evaluate_sharpness
from core.biometrics import (
    extract_features, compute_cosine_similarity, 
    compute_local_ridge_similarity, crop_rhinarium_roi,
    biometric_match
)
from core.locations import WARD_METRICS, calculate_geofence_distance
from core.reporting import generate_canine_passport

st.set_page_config(
    page_title="PawProof | Municipal ABC Portal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize MySQL Database
db.init_db()

# =========================================================
# ASSET & LOGO BASE64 ENCODER
# =========================================================
def get_base64_image(image_path: str) -> str:
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(image_path)[1].lstrip(".").lower()
        mime_ext = "jpeg" if ext in ["jpg", "jpeg"] else ext
        return f"data:image/{mime_ext};base64,{base64.b64encode(data).decode()}"
    return ""

# =========================================================
# VALIDATION HELPERS
# =========================================================
def validate_password_strength(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters long / पासवर्ड किमान ८ अक्षरांचा असावा."
    if not re.search(r"[A-Z]", password):
        return False, "Password must include at least one capital letter (A-Z) / पासवर्डमध्ये किमान एक मोठे अक्षर (A-Z) असावे."
    if not re.search(r"[a-z]", password):
        return False, "Password must include at least one small letter (a-z) / पासवर्डमध्ये किमान एक लहान अक्षर (a-z) असावे."
    if not re.search(r"\d", password):
        return False, "Password must include at least one number (0-9) / पासवर्डमध्ये किमान एक अंक (0-9) असावा."
    if not re.search(r"[@$!%*?&#^_\-.]", password):
        return False, "Password must include at least one symbol (@, #, $, etc.) / पासवर्डमध्ये किमान एक विशेष चिन्ह (@, #, $, इ.) वापरा."
    return True, ""

def validate_phone_number(phone: str) -> tuple[bool, str]:
    clean_digits = re.sub(r"\D", "", phone)
    if len(clean_digits) < 10:
        return False, "Please enter a valid 10-digit mobile number / कृपया वैध १० अंकी मोबाईल नंबर टाका."
    return True, ""

# =========================================================
# BILINGUAL DICTIONARY
# =========================================================
TRANSLATIONS = {
    "en": {
        "nav_home": "Home",
        "nav_scan": "Scan Dog",
        "nav_tasks": "My Pickups",
        "nav_surgery": "Doctor's Table",
        "nav_municipal_clearance": "Municipal Authorization",
        "nav_return": "Colony Drop",
        "nav_ledger": "Area Records",
        "nav_audit": "Vaccine Status",
        "signin_heading": "Welcome Back",
        "signin_desc": "Sign in to scan dogs, update health records, or manage ward operations.",
        "username": "Username *",
        "password": "Password *",
        "signin_btn": "Sign In",
        "new_user_prompt": "First time here?",
        "register_link": "Create an account",
        "already_user_prompt": "Already have an account?",
        "signin_link": "Sign in here",
        "register_heading": "Create Your Account",
        "register_desc": "All fields below are mandatory. Please provide accurate details.",
        "fullname": "Your Full Name *",
        "choose_username": "Choose a Username *",
        "choose_password": "Set a Secure Password *",
        "select_role": "What is your role? *",
        "select_ward": "Your Assigned Ward / Area *",
        "contact_phone": "Mobile Number (WhatsApp) *",
        "register_btn": "Create Account & Sign In",
        "feeder_role": "Dog Feeder / Animal Lover (श्वान मित्र)",
        "transit_role": "Dog Catching Van Driver (गाडी चालक)",
        "vet_role": "Veterinary Doctor (डॉक्टर)",
        "audit_role": "Municipal Health Desk / Officer (मनपा आरोग्य कक्ष)",
        "live_camera": "Take Photo",
        "upload_file": "Upload Photo",
        "camera_guide": "Hold camera 1 to 2 feet in front of the dog's nose.",
        "match_found": "DOG ALREADY REGISTERED",
        "no_match": "NEW / UNREGISTERED DOG",
        "signout": "Sign Out",
        "my_profile": "My Account",
        "back_to_ops": "Back to Work",
        "ward_label": "Area",
        "operator": "Signed In As",
        "landmark": "Near Which Spot / Landmark?",
        "submit_lead": "Send Request to Catching Van",
        "admit_intake": "Brought Dog to Hospital",
        "surgery_done": "Complete Surgery & Vaccine Entry",
        "release_done": "Confirm Dog Dropped in Same Area",
        "safe_drop": "Safe: Drop location is within the legal 500m area.",
        "unsafe_drop": "Warning: Location is too far from original spot! Relocation Illegal.",
        "footer_compliance": "Operated in compliance with Animal Birth Control (ABC) Rules, 2023.",
        "footer_support": "PawProof by SG!"
    },
    "mr": {
        "nav_home": "मुख्य पान",
        "nav_scan": "ठसे स्कॅन",
        "nav_tasks": "गाडीची कामे",
        "nav_surgery": "डॉक्टर टेबल",
        "nav_municipal_clearance": "मनपा सोडण्याची मंजुरी",
        "nav_return": "कुत्रा सोडणे",
        "nav_ledger": "भागातील नोंदी",
        "nav_audit": "लसीकरण स्थिती",
        "signin_heading": "स्वागत आहे",
        "signin_desc": "कुत्र्यांची नोंद, शस्त्रक्रिया आणि मनपा मंजुरीसाठी लॉगिन करा.",
        "username": "युझरनेम (Username) *",
        "password": "पासवर्ड (Password) *",
        "signin_btn": "लॉगिन करा",
        "new_user_prompt": "नवीन खाते तयार करायचे आहे?",
        "register_link": "नवीन खाते उघडा",
        "already_user_prompt": "आधीच खाते आहे का?",
        "signin_link": "येथे लॉगिन करा",
        "register_heading": "नवीन खाते तयार करा",
        "register_desc": "खालील सर्व माहिती भरणे बंधनकारक आहे.",
        "fullname": "तुमचे पूर्ण नाव *",
        "choose_username": "युझरनेम निवडा *",
        "choose_password": "सुरक्षित पासवर्ड तयार करा *",
        "select_role": "तुमचे काम काय आहे? *",
        "select_ward": "तुमचा भाग / वॉर्ड *",
        "contact_phone": "मोबाईल नंबर (व्हॉट्सॲप) *",
        "register_btn": "नोंदणी पूर्ण करून लॉगिन करा",
        "feeder_role": "श्वान मित्र / नागरिक फीडर (Feeder)",
        "transit_role": "कुत्रे पकडणारी गाडी / चालक (Catching Van)",
        "vet_role": "पशुवैद्यकीय डॉक्टर (Doctor)",
        "audit_role": "मनपा आरोग्य विभाग / अधिकारी (Municipal Officer)",
        "live_camera": "कॅमेऱ्याने फोटो काढा",
        "upload_file": "फोनमधून फोटो निवडा",
        "camera_guide": "कॅमेरा कुत्र्याच्या नाकासमोर १ ते २ फूट सरळ अंतरावर धरा.",
        "match_found": "या कुत्र्याची नोंद आधीच आहे",
        "no_match": "नवीन कुत्रा (नोंद नाही)",
        "signout": "लॉग आउट",
        "my_profile": "माझे खाते",
        "back_to_ops": "कामावर परत जा",
        "ward_label": "भाग",
        "operator": "कार्यरत व्यक्ती",
        "landmark": "कुठे सापडला? (जवळची खूण)",
        "submit_lead": "गाडीला उचलण्यासाठी पाठवा",
        "admit_intake": "कुत्रा हॉस्पिटलमध्ये आणला",
        "surgery_done": "शस्त्रक्रिया व लस नोंद जतन करा",
        "release_done": "कुत्रा सुरक्षित मूळ जागी सोडल्याची खात्री करा",
        "safe_drop": "जागा योग्य आहे (कुत्रा मूळ गल्लीच्या जवळ आहे)",
        "unsafe_drop": "नियमबाह्य: कुत्रा मूळ जागेपासून खूप लांब सोडता येणार नाही!",
        "footer_compliance": "श्वान नियंत्रण व नसबंदी नियम, २०२३ अंतर्गत कार्यरत.",
        "footer_support": "मदत क्रमांक: १८००-१०३०-२२२ | support@pawproof.org"
    }
}

default_keys = {
    "lang": "en",
    "logged_in": False,
    "user": None,
    "role": None,
    "full_name": None,
    "ward": None,
    "badge": None,
    "phone": None,
    "nav_page": "home",
    "auth_page": "login",
    "prefill_user": "",
    "uploader_key": 0,
    "active_table_dog": None
}

for k, val in default_keys.items():
    if k not in st.session_state:
        st.session_state[k] = val

def perform_logout():
    st.session_state["logged_in"] = False
    st.session_state["user"] = None
    st.session_state["role"] = None
    st.session_state["full_name"] = None
    st.session_state["ward"] = None
    st.session_state["badge"] = None
    st.session_state["phone"] = None
    st.session_state["nav_page"] = "home"
    st.session_state["auth_page"] = "login"
    st.session_state["active_table_dog"] = None

T = TRANSLATIONS[st.session_state["lang"]]

# =========================================================
# MATTE-BROWN THEME CSS
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Noto Sans Devanagari', -apple-system, sans-serif;
    }

    .stApp {
        background-color: #FBF9F5 !important;
        color: #292524 !important;
    }
    
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    .block-container {
        max-width: 1240px;
        padding-top: 1.4rem !important;
        padding-bottom: 4rem !important;
    }

    div[data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E7E1D8 !important;
        border-radius: 16px !important;
        padding: 26px !important;
        box-shadow: 0 4px 16px -4px rgba(69, 26, 3, 0.05) !important;
    }

    button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button,
    div.stDownloadButton > button {
        background: linear-gradient(180deg, #78350F 0%, #5A270B 100%) !important;
        background-color: #78350F !important;
        color: #FDFBF7 !important;
        border: 1px solid #451A03 !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 6px rgba(69, 26, 3, 0.25) !important;
        transition: all 0.15s ease-in-out !important;
    }

    button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    div.stDownloadButton > button:hover {
        background: linear-gradient(180deg, #5A270B 0%, #451A03 100%) !important;
        background-color: #5A270B !important;
        border-color: #291002 !important;
        box-shadow: 0 4px 10px rgba(69, 26, 3, 0.35) !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
    }

    button[kind="secondary"],
    div[data-testid="stPopover"] > button {
        background-color: #F5EFE6 !important;
        color: #451A03 !important;
        border: 1.5px solid #8B3A0F !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        transition: all 0.15s ease-in-out !important;
    }

    button[kind="secondary"]:hover,
    div[data-testid="stPopover"] > button:hover {
        background-color: #E7DBCB !important;
        border-color: #5A270B !important;
        color: #291002 !important;
        transform: translateY(-1px) !important;
    }

    input, textarea, select {
        border-radius: 8px !important;
        border: 1px solid #D6C7B2 !important;
        background-color: #FFFFFF !important;
        color: #292524 !important;
    }
    input:focus, textarea:focus, select:focus {
        border-color: #78350F !important;
        box-shadow: 0 0 0 2px rgba(120, 53, 15, 0.15) !important;
    }

    div[data-testid="stSegmentedControl"] {
        background-color: #EFE8DC !important;
        border: 1px solid #D6C7B2 !important;
        border-radius: 12px !important;
        padding: 4px !important;
    }

    div[data-testid="stSegmentedControl"] button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #573A27 !important;
    }

    div[data-testid="stSegmentedControl"] button[aria-selected="true"] {
        background-color: #78350F !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 4px rgba(120, 53, 15, 0.25) !important;
    }

    div[data-testid="stSegmentedControl"] button[aria-selected="true"] * {
        color: #FFFFFF !important;
    }

    .badge-verified {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-unregistered {
        background-color: #FDF2F0;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }

    .portal-footer {
        margin-top: 50px;
        padding-top: 22px;
        border-top: 1px solid #E7E1D8;
        color: #78716C;
        font-size: 0.82rem;
        text-align: center;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

user_role = st.session_state.get("role")
user_ward = st.session_state.get("ward")
current_username = st.session_state.get("user")

# Role Navigation Routing
if not st.session_state["logged_in"] or not user_role:
    nav_options = [T["nav_home"]]
else:
    if user_role == "FEEDER":
        nav_options = [T["nav_scan"], T["nav_ledger"]]
    elif user_role == "TRANSIT":
        nav_options = [T["nav_scan"], T["nav_tasks"], T["nav_return"]]
    elif user_role == "VET_OFFICE":
        nav_options = [T["nav_surgery"], T["nav_scan"], T["nav_ledger"]]
    else:  # MUNICIPAL_CORP
        nav_options = [T["nav_municipal_clearance"], T["nav_audit"], T["nav_ledger"]]

if st.session_state.get("nav_page") not in nav_options and st.session_state.get("nav_page") != "profile":
    st.session_state["nav_page"] = nav_options[0]

# =========================================================
# TOP NAVBAR (84PX BADGE HEADER)
# =========================================================
nav_left, nav_center, nav_right = st.columns([4.4, 3.4, 2.2], vertical_alignment="center")

with nav_left:
    possible_paths = [
        os.path.join("static", "logo.jpeg"),
        os.path.join("static", "logo.jpg"),
        os.path.join("static", "logo.png"),
        "logo.jpeg",
        "logo.jpg",
        "logo.png"
    ]
    
    logo_path = next((p for p in possible_paths if os.path.exists(p)), None)
    b64_logo = get_base64_image(logo_path) if logo_path else ""

    if b64_logo:
        img_tag = f'<img src="{b64_logo}" style="width: 100%; height: 100%; object-fit: cover; display: block;" />'
    else:
        img_tag = '<span style="font-size: 2.1rem; line-height: 1;">🐾</span>'

    header_html = (
        '<div style="display: flex; align-items: center; gap: 16px;">'
        f'<div style="width: 84px; height: 84px; min-width: 84px; min-height: 84px; border-radius: 50%; overflow: hidden; border: 3px solid #78350F; box-shadow: 0 4px 12px rgba(120, 53, 15, 0.22); display: flex; align-items: center; justify-content: center; background: #FFFFFF; flex-shrink: 0;">'
        f'{img_tag}'
        '</div>'
        '<div>'
        '<div style="font-size: 1.95rem; font-weight: 800; color: #451A03; line-height: 1.05; margin: 0; letter-spacing: -0.03em;">PawProof</div>'
        '<span style="font-size: 0.78rem; font-weight: 700; color: #78350F; background: #F5EFE6; padding: 3px 10px; border-radius: 6px; border: 1px solid #E0D4C3; letter-spacing: 0.05em; text-transform: uppercase; margin-top: 5px; display: inline-block;">Municipal Canine Biometrics</span>'
        '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

with nav_center:
    if st.session_state["logged_in"] and st.session_state.get("nav_page") != "profile":
        selected_nav = st.segmented_control(
            "Navigation",
            options=nav_options,
            default=st.session_state["nav_page"],
            label_visibility="collapsed"
        )
        if selected_nav and selected_nav != st.session_state["nav_page"]:
            st.session_state["nav_page"] = selected_nav
            st.rerun()

with nav_right:
    r_col1, r_col2 = st.columns([1.0, 1.2], vertical_alignment="center")
    with r_col1:
        cur_lang_idx = 0 if st.session_state["lang"] == "en" else 1
        lang_choice = st.selectbox("Lang", options=["English", "मराठी"], index=cur_lang_idx, label_visibility="collapsed")
        new_lang = "en" if lang_choice == "English" else "mr"
        if new_lang != st.session_state["lang"]:
            st.session_state["lang"] = new_lang
            st.rerun()

    with r_col2:
        if st.session_state["logged_in"]:
            initial = st.session_state["full_name"][:1].upper() if st.session_state["full_name"] else "U"
            with st.popover(f"Profile: {initial}", use_container_width=True):
                st.markdown(f"**{st.session_state['full_name']}**")
                st.caption(f"{st.session_state['role']} • {st.session_state['ward']}")
                st.write(f"Username: `@{st.session_state['user']}`")
                if st.session_state.get("badge"):
                    st.write(f"Badge / Reg: `{st.session_state['badge']}`")
                st.markdown("---")
                if st.button(T["my_profile"], use_container_width=True, type="secondary"):
                    st.session_state["nav_page"] = "profile"
                    st.rerun()
                if st.button(T["signout"], use_container_width=True, type="primary"):
                    perform_logout()
                    st.rerun()
        else:
            st.empty()

st.markdown("<hr style='border: 0; border-bottom: 1px solid #E7E1D8; margin: 12px 0 28px 0;'>", unsafe_allow_html=True)

if "action_notice" in st.session_state:
    st.success(st.session_state["action_notice"])
    del st.session_state["action_notice"]

# =========================================================
# 1. AUTHENTICATION
# =========================================================
if not st.session_state["logged_in"]:
    _, col_box, _ = st.columns([1, 2.1, 1])
    with col_box:
        if st.session_state["auth_page"] == "login":
            st.markdown(f"### {T['signin_heading']}")
            st.caption(T["signin_desc"])

            with st.form("clean_login_form"):
                login_user = st.text_input(T["username"], value=st.session_state["prefill_user"], placeholder="e.g. rajesh_transit", key="l_user")
                login_pwd = st.text_input(T["password"], type="password", placeholder="••••••••", key="l_pwd")
                st.write("")
                submit_login = st.form_submit_button(T['signin_btn'], type="primary", use_container_width=True)

                if submit_login:
                    if not login_user.strip() or not login_pwd.strip():
                        st.error("Please enter both username and password / कृपया युझरनेम आणि पासवर्ड दोन्ही टाका.")
                    else:
                        user_data = db.authenticate_user(login_user, login_pwd)
                        if user_data:
                            st.session_state["logged_in"] = True
                            st.session_state["user"] = user_data["username"]
                            st.session_state["role"] = user_data["role"]
                            st.session_state["full_name"] = user_data["full_name"]
                            st.session_state["ward"] = user_data["ward"]
                            st.session_state["badge"] = user_data.get("badge_or_vci", "")
                            st.session_state["phone"] = user_data.get("contact_phone", "")
                            st.session_state["nav_page"] = (
                                T["nav_surgery"] if user_data["role"] == "VET_OFFICE"
                                else T["nav_municipal_clearance"] if user_data["role"] == "MUNICIPAL_CORP"
                                else T["nav_scan"]
                            )
                            st.rerun()
                        else:
                            st.error("Invalid credentials entered / चुकीचे युझरनेम किंवा पासवर्ड.")

            st.markdown(f"<div style='text-align: center; margin-top: 14px; font-size: 0.9rem; color: #78716C;'>{T['new_user_prompt']}</div>", unsafe_allow_html=True)
            if st.button(T["register_link"], use_container_width=True, type="secondary"):
                st.session_state["auth_page"] = "register"
                st.rerun()

        else:
            st.markdown(f"### {T['register_heading']}")
            st.caption(T["register_desc"])

            selected_role = st.selectbox(
                T["select_role"],
                [
                    ("FEEDER", T["feeder_role"]),
                    ("TRANSIT", T["transit_role"]),
                    ("VET_OFFICE", T["vet_role"]),
                    ("MUNICIPAL_CORP", T["audit_role"])
                ],
                format_func=lambda x: x[1],
                key="reg_role_choice"
            )[0]

            with st.form("clean_register_form"):
                new_fname = st.text_input(T["fullname"], placeholder="e.g. Dr. Rajesh Deshmukh", key="s_name")
                new_uname = st.text_input(T["choose_username"], placeholder="e.g. deshmukh_vet", key="s_uname")
                new_pwd = st.text_input(
                    T["choose_password"], 
                    type="password", 
                    help="Min 8 chars, 1 uppercase (A-Z), 1 lowercase (a-z), 1 digit (0-9), and 1 symbol (@, #, $, etc.).",
                    key="s_pwd"
                )
                new_ward = st.selectbox(T["select_ward"], options=list(WARD_METRICS.keys()), key="s_ward")
                new_phone = st.text_input(T["contact_phone"], placeholder="e.g. 9876543210", key="s_phone")

                if selected_role == "TRANSIT":
                    badge_label = "Municipal Van Plate Number / गाडी क्रमांक *"
                elif selected_role == "VET_OFFICE":
                    badge_label = "VCI / Maharashtra Vet Council Reg No. *"
                elif selected_role == "MUNICIPAL_CORP":
                    badge_label = "Municipal Employee ID / कर्मचारी आयडी *"
                else:
                    badge_label = "Volunteer / Feeder Group ID / श्वान मित्र आयडी *"

                new_badge = st.text_input(badge_label, key="s_badge")
                st.write("")
                submit_reg = st.form_submit_button(T['register_btn'], type="primary", use_container_width=True)

                if submit_reg:
                    if not new_fname.strip() or not new_uname.strip() or not new_pwd.strip() or not new_phone.strip() or not new_badge.strip():
                        st.error("All fields are mandatory. Please fill in all fields / सर्व माहिती भरणे बंधनकारक आहे.")
                    else:
                        is_valid_phone, phone_msg = validate_phone_number(new_phone)
                        if not is_valid_phone:
                            st.error(f"⚠️ {phone_msg}")
                        else:
                            is_valid_pwd, pwd_msg = validate_password_strength(new_pwd)
                            if not is_valid_pwd:
                                st.error(f"⚠️ {pwd_msg}")
                            else:
                                success, msg = db.create_user(
                                    new_uname, new_pwd, new_fname, selected_role, new_ward,
                                    badge_or_vci=new_badge, contact_phone=new_phone
                                )
                                if success:
                                    st.session_state["prefill_user"] = new_uname.strip().lower()
                                    st.session_state["auth_page"] = "login"
                                    st.session_state["action_notice"] = "Account created successfully! Please sign in."
                                    st.rerun()
                                else:
                                    st.error(msg)

            st.markdown(f"<div style='text-align: center; margin-top: 14px; font-size: 0.9rem; color: #78716C;'>{T['already_user_prompt']}</div>", unsafe_allow_html=True)
            if st.button(T["signin_link"], use_container_width=True, type="secondary"):
                st.session_state["auth_page"] = "login"
                st.rerun()

    st.markdown(f"""
    <div class="portal-footer">
        <div>{T['footer_compliance']}</div>
        <div style="margin-top: 4px; color: #A8A29E;">{T['footer_support']}</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# =========================================================
# 2. PROFILE / MY ACCOUNT VIEW (FOR ALL ROLES)
# =========================================================
if st.session_state.get("nav_page") == "profile":
    st.markdown(f"### {T['my_profile']}")
    st.caption("Your account details, municipal credentials, and operational area.")

    with st.container():
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown("##### 👤 Personnel Details")
            st.write(f"**Full Name:** {st.session_state['full_name']}")
            st.write(f"**Username:** `@{st.session_state['user']}`")
            st.write(f"**Contact Phone:** {st.session_state.get('phone') or 'Not Registered'}")
            st.write(f"**Statutory Reg / Badge No:** `{st.session_state.get('badge') or 'Not Specified'}`")

        with p_col2:
            st.markdown("##### 🏢 Municipal Work Jurisdiction")
            st.write(f"**Role:** `{user_role}`")
            st.write(f"**Assigned Ward:** {user_ward}")
            ward_info = WARD_METRICS.get(user_ward, {})
            st.write(f"**City Code:** `{ward_info.get('city_code', 'GEN')}`")
            st.write(f"**Zone / PIN:** `{ward_info.get('zone_code', 'N/A')} - {ward_info.get('pin', 'N/A')}`")

    st.markdown("---")
    st.markdown("##### 📊 Operational Activity Log")
    records = db.get_all_records()

    if user_role == "TRANSIT":
        my_tasks = db.get_pending_requests_for_driver(current_username)
        captured = [r for r in records if r["enrolled_by"] == current_username]
        released = [r for r in records if r["ward"] == user_ward and r["release_verified"]]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Pending Pickups", len(my_tasks))
        m2.metric("Canines Secured by My Van", len(captured))
        m3.metric("Colony Releases Completed", len(released))

    elif user_role == "VET_OFFICE":
        admitted = db.get_vet_admitted_patients(ward=user_ward)
        operated = [r for r in records if r.get("assigned_vet") == st.session_state["full_name"] or r["status"] in ["STERILIZED_VACCINATED", "CLEARED_FOR_RELEASE", "RELEASED"]]
        m1, m2 = st.columns(2)
        m1.metric("Pending Surgeries in Queue", len(admitted))
        m2.metric("Operations Certified", len(operated))

    elif user_role == "MUNICIPAL_CORP":
        ward_recs = [r for r in records if r["ward"] == user_ward]
        cleared = [r for r in records if r["status"] in ["CLEARED_FOR_RELEASE", "RELEASED"]]
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Registered Canines", len(ward_recs))
        m2.metric("Official Releases Cleared", len(cleared))
        m3.metric("Ward Health Coverage", f"{(len(cleared)/len(ward_recs)*100) if ward_recs else 0:.1f}%")

    elif user_role == "FEEDER":
        my_leads = [r for r in records if r["enrolled_by"] == current_username]
        m1, m2 = st.columns(2)
        m1.metric("Community Dogs Logged", len(my_leads))
        m2.metric("Active Ward Requests", len([r for r in records if r["ward"] == user_ward]))

    st.write("")
    if st.button(f"← {T['back_to_ops']}", type="primary"):
        st.session_state["nav_page"] = nav_options[0]
        st.rerun()

# =========================================================
# 3. VETERINARY DESK: LIVE INFLOW QUEUE & TABLE-SIDE SCANNER
# =========================================================
elif st.session_state["nav_page"] == T["nav_surgery"] and user_role == "VET_OFFICE":
    st.markdown(f"### 🏥 Doctor's Desk — Hospital Inflow & Surgical Table")
    st.caption(f"Operating Surgeon: **Dr. {st.session_state['full_name']}** | Clinic Ward: **{user_ward}**")

    admitted_patients = db.get_vet_admitted_patients(ward=user_ward)
    all_ward_recs = db.get_all_records(ward_filter=user_ward)
    completed_today = [r for r in all_ward_recs if r["status"] in ["STERILIZED_VACCINATED", "CLEARED_FOR_RELEASE", "RELEASED"]]

    m1, m2, m3 = st.columns(3)
    m1.metric("Incoming Waiting in Pre-Op", len(admitted_patients))
    m2.metric("Active on Table", "1 Patient" if st.session_state.get("active_table_dog") else "None (Table Ready)")
    m3.metric("Operated & Sent to Municipal", len(completed_today))

    st.markdown("---")

    col_board, col_table = st.columns([1.1, 1.4])

    with col_board:
        st.markdown("#### 📋 Hospital Inflow Schedule")
        st.caption("Patients brought by van squads waiting for surgery.")

        if not admitted_patients:
            st.info("No incoming dogs in the surgical intake queue / सध्या शस्त्रक्रियेसाठी दाखल कुत्रे नाहीत.")
        else:
            for pat in admitted_patients:
                with st.container():
                    st.markdown(f"**Token:** `{pat['token_id']}` • **Sex:** `{pat['sex']}`")
                    st.caption(f"Colony Origin: {pat['capture_landmark']} | Van: @{pat['enrolled_by']}")
                    if st.button(f"Call to Table", key=f"btn_load_{pat['token_id']}", use_container_width=True, type="secondary"):
                        st.session_state["active_table_dog"] = pat
                        st.rerun()
                    st.markdown("<hr style='margin: 8px 0; border: 0; border-top: 1px dashed #E0D4C3;'>", unsafe_allow_html=True)

    with col_table:
        st.markdown("#### 🔬 Operating Table: Scan & Treat")
        st.caption("Scan the dog's snout directly on the table to verify identity.")

        table_cam_file = st.camera_input("Table Scanner (Aim at snout)", key="table_scanner_feed")
        
        if table_cam_file:
            t_raw = Image.open(table_cam_file).convert("RGB")
            t_vec = extract_features(t_raw)

            matched_patient = None
            highest_sim = 0.0

            for p in admitted_patients:
                emb = np.frombuffer(p["embedding"], dtype=np.float32).tolist() if isinstance(p["embedding"], (bytes, bytearray)) else p["embedding"]
                score = compute_cosine_similarity(t_vec, emb)
                if score > highest_sim:
                    highest_sim = score
                    matched_patient = p

            if matched_patient and highest_sim >= 0.70:
                if os.path.exists(matched_patient["image_path"]):
                    ref_img = Image.open(matched_patient["image_path"]).convert("RGB")
                    ridge = compute_local_ridge_similarity(t_raw, ref_img)
                    if ridge >= 0.60:
                        st.session_state["active_table_dog"] = matched_patient
                        st.success(f"Verified on Table: Token `{matched_patient['token_id']}` ({ridge*100:.1f}% pattern match)")
                    else:
                        st.warning("General coat matches, but nose ridges differ. Please double check patient.")
            else:
                st.error("Patient snout does not match any admitted pre-op dog.")

        active_patient = st.session_state.get("active_table_dog")

        if active_patient:
            st.markdown("---")
            st.markdown(f"### Procedure: `{active_patient['token_id']}`")
            
            p_img, p_details = st.columns([1, 1.6])
            with p_img:
                if os.path.exists(active_patient["image_path"]):
                    st.image(active_patient["image_path"], caption=f"Admitted Photo - {active_patient['sex']}", use_container_width=True)
            with p_details:
                st.write(f"**Ward:** {active_patient['ward']}")
                st.write(f"**Sex Recorded:** `{active_patient['sex']}`")
                st.write(f"**Colony Spot:** {active_patient['capture_landmark']}")

            with st.form("ot_table_procedure_form"):
                st.markdown("##### Immediate Post-Procedure Entry")
                
                default_proc_idx = 0 if active_patient["sex"] == "FEMALE" else 1
                proc_type = st.selectbox(
                    "Surgical Procedure *",
                    ["Ovariohysterectomy (Female - गर्भाशय शस्त्रक्रिया)", "Castration (Male - नसबंदी शस्त्रक्रिया)"],
                    index=default_proc_idx
                )

                st.info("🐾 Identification Method: Canine Rhinarium Biometric Lock (Zero Mutilation)")

                arv_batch_input = st.text_input("Anti-Rabies Vaccine (ARV) Batch *", value="ARV-PUNE-2026-B1")
                vet_clinical_notes = st.text_input("Surgeon Observations", value="Sutures intact. Healthy vital signs. Shifted to recovery pen.")

                submit_surgery = st.form_submit_button("Complete & Send to Municipal Desk", type="primary", use_container_width=True)

                if submit_surgery:
                    db.complete_surgery_on_table(
                        token_id=active_patient["token_id"],
                        surgery_type=proc_type,
                        arv_batch=arv_batch_input.strip(),
                        doctor_name=st.session_state["full_name"],
                        clinical_notes=vet_clinical_notes.strip()
                    )
                    st.session_state["active_table_dog"] = None
                    st.session_state["action_notice"] = f"Procedure saved for {active_patient['token_id']}. Forwarded to Municipal Office for release clearance."
                    st.rerun()
        else:
            st.info("Operating table is clear. Scan a dog's snout or click 'Call to Table' from the inflow schedule.")

# =========================================================
# 4. MUNICIPAL DESK: COMPLETE CLINICAL DOSSIER & CLEARANCE
# =========================================================
elif st.session_state["nav_page"] == T["nav_municipal_clearance"] and user_role == "MUNICIPAL_CORP":
    st.markdown("### 🏛️ Municipal Animal Welfare Desk — Release Clearance")
    st.caption("Inspect the dog's complete clinical history, surgery records, and post-op recovery timeline to decide a legal release date.")

    records = db.get_all_records(ward_filter=user_ward)
    pending_clearance = [r for r in records if r["status"] == "STERILIZED_VACCINATED"]

    if not pending_clearance:
        st.info("No vaccinated dogs currently awaiting municipal release clearance.")
    else:
        st.markdown("#### Dogs Pending Release Authorization")
        st.dataframe([{
            "Token ID": r["token_id"],
            "Ward": r["ward"],
            "Sex": r["sex"],
            "Surgery / ARV Date": r.get("vax_date", "Recorded"),
            "Vaccine Batch": r.get("arv_batch", "N/A"),
            "Colony Landmark": r["landmark"],
            "Attending Doctor": r.get("assigned_vet", "Dr. Assigned")
        } for r in pending_clearance], use_container_width=True)

        st.markdown("---")
        st.markdown("#### Patient Clinical Dossier & Release Decision")

        sel_clear_id = st.selectbox("Select Canine to Inspect & Authorize", options=[r["token_id"] for r in pending_clearance])
        canine_item = next(r for r in pending_clearance if r["token_id"] == sel_clear_id)

        surg_str = canine_item.get("vax_date")
        try:
            surg_dt = datetime.strptime(surg_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            surg_dt = datetime.today().date()

        days_in_recovery = (datetime.today().date() - surg_dt).days
        min_healing_days = 4 if canine_item["sex"] == "FEMALE" else 3
        earliest_safe_release = surg_dt + timedelta(days=min_healing_days)

        c_photo, c_history = st.columns([1, 2])
        
        with c_photo:
            if os.path.exists(canine_item["image_path"]):
                st.image(canine_item["image_path"], caption=f"Biometric Record: {canine_item['token_id']}", use_container_width=True)
            st.metric("Post-Op Recovery Elapsed", f"{days_in_recovery} Days", 
                      delta="Recovery Complete" if days_in_recovery >= min_healing_days else f"Needs {min_healing_days - days_in_recovery} more day(s)")

        with c_history:
            st.markdown("##### 📜 Clinical & Field History")
            h1, h2 = st.columns(2)
            with h1:
                st.write(f"**Token ID:** `{canine_item['token_id']}`")
                st.write(f"**Ward / Jurisdiction:** {canine_item['ward']}")
                st.write(f"**Sex:** `{canine_item['sex']}`")
                st.write(f"**Pickup Colony Spot:** {canine_item['landmark']}")
                st.write(f"**Admitted by Van:** @{canine_item['enrolled_by']}")
            with h2:
                st.write(f"**Surgery Performed Date:** `{surg_dt}`")
                st.write(f"**Anti-Rabies Batch (ARV):** `{canine_item.get('arv_batch', 'N/A')}`")
                st.write(f"**Next Rabies Booster Due:** `{canine_item.get('booster_due', 'N/A')}`")
                st.write(f"**Operating Surgeon:** Dr. `{canine_item.get('assigned_vet', 'Assigned Doctor')}`")
            
            if canine_item.get("notes"):
                st.info(f"**Doctor's Clinical Notes:** {canine_item['notes']}")

            if days_in_recovery < min_healing_days:
                st.warning(f"⚠️ Mandatory ABC Observation: Under Rule 11, {canine_item['sex'].lower()} dogs require at least {min_healing_days} full days of post-op care. Earliest recommended release date is **{earliest_safe_release}**.")
            else:
                st.success(f"✅ Mandatory {min_healing_days}-day post-op observation period is completed. Dog is clinically fit for release.")

        st.markdown("---")
        
        with st.form("municipal_authorization_form"):
            st.markdown("##### 📝 Municipal Clearance & Release Scheduling")
            
            auth_release_date = st.date_input(
                "Authorized Colony Release Date *",
                value=max(datetime.today().date(), earliest_safe_release),
                min_value=surg_dt,
                help=f"Rule 11 recommends setting on or after {earliest_safe_release} to guarantee recovery."
            )
            
            muni_memo = st.text_input("Municipal Registry / File Memo Number *", placeholder="e.g. PMC/HEALTH/ABC/2026/842")
            officer_notes = st.text_input("Sanitary & Inspector Observation", value="Wound healed, rabies vaccinated, authorized for van squad pickup.")

            clear_btn = st.form_submit_button("Sign Off & Clear for Transit Van Squad", type="primary", use_container_width=True)

            if clear_btn:
                if not muni_memo.strip():
                    st.error("Please enter the official Municipal File Memo Number / नोंदवही क्रमांक टाका.")
                else:
                    db.update_municipal_clearance(
                        token_id=sel_clear_id,
                        release_date=str(auth_release_date),
                        municipal_notes=f"{muni_memo.strip()} - {officer_notes.strip()}",
                        officer_name=st.session_state["full_name"]
                    )
                    st.session_state["action_notice"] = f"Canine '{sel_clear_id}' cleared for release on {auth_release_date}. Added to transit van's drop list."
                    st.rerun()

# =========================================================
# 5. TRANSIT DESK: COLONY DROP (STRICT 500M GEOFENCE)
# =========================================================
elif st.session_state["nav_page"] == T["nav_return"] and user_role == "TRANSIT":
    st.markdown(f"### {T['nav_return']} — Catching Squad Van")
    st.caption("Rule 11 ABC Compliance: Only dogs officially cleared by the Municipal Office can be released, strictly within 500m.")

    records = db.get_all_records(ward_filter=user_ward)
    cleared_for_squad = [r for r in records if r["status"] == "CLEARED_FOR_RELEASE"]

    if not cleared_for_squad:
        st.info("No dogs cleared for release by the Municipal Office right now / सध्या सोडण्यासाठी मनपा मंजुरी बाकी आहे.")
    else:
        sel_token = st.selectbox("Select Cleared Dog to Return / कुत्रा निवडा", options=[r["token_id"] for r in cleared_for_squad])
        target_dog = next(r for r in cleared_for_squad if r["token_id"] == sel_token)

        st.write(f"**Original Pickup Landmark:** {target_dog['landmark']} (`{target_dog['lat']}`, `{target_dog['lng']}`)")
        st.write(f"**Authorized Release Date:** `{target_dog.get('release_date', 'Today')}`")
        if target_dog.get("notes"):
            st.caption(f"Documentation: {target_dog['notes']}")

        c_lat = st.number_input("Current Van Drop Latitude", value=target_dog["lat"], format="%.6f")
        c_lng = st.number_input("Current Van Drop Longitude", value=target_dog["lng"], format="%.6f")

        dist = calculate_geofence_distance(target_dog["lat"], target_dog["lng"], c_lat, c_lng)
        st.metric("Distance from Pickup Spot", f"{dist:.1f} m", delta="Permitted (≤ 500m)" if dist <= 500 else "Relocation Illegal (> 500m)")

        if dist <= 500.0:
            st.success(T['safe_drop'])
            if st.button(T['release_done'], type="primary", use_container_width=True):
                db.update_release_status(sel_token)
                st.session_state["action_notice"] = f"Dog '{sel_token}' safely released at original colony."
                st.rerun()
        else:
            st.error(f"{T['unsafe_drop']} (Current: {dist:.1f}m. Max allowed: 500m)")
            st.button(T['release_done'], disabled=True, use_container_width=True)
            db.log_geofence_breach(sel_token, current_username, c_lat, c_lng, dist)

# =========================================================
# 6. GENERAL SCAN & INTAKE (FEEDER & TRANSIT)
# =========================================================
elif st.session_state["nav_page"] == T["nav_scan"]:
    st.markdown(f"### {T['nav_scan']}")
    st.caption(T["camera_guide"])

    scan_mode = st.radio("Source", [T["live_camera"], T["upload_file"]], horizontal=True, label_visibility="collapsed")
    active_key = f"shutter_{st.session_state['uploader_key']}"

    scan_file = (
        st.camera_input("Aim directly at dog muzzle", key=active_key)
        if scan_mode == T["live_camera"]
        else st.file_uploader("Upload close-up dog snout photo", type=["jpeg", "jpg", "png"], key=active_key)
    )

    if scan_file:
        raw_img = Image.open(scan_file).convert("RGB")
        is_sharp, sharp_score = evaluate_sharpness(raw_img)

        c_prev, c_res = st.columns([1, 1.3])
        with c_prev:
            st.image(raw_img, caption="Capture", use_container_width=True)
            st.caption(f"Clarity: **{sharp_score:.1f}**")

        with c_res:
            if not is_sharp:
                st.error("Photo is blurry. Please hold camera steady and retake.")
            else:
                with st.spinner("Checking nose patterns in database..."):
                    # Keep the existing embedding extraction because it is still
                    # used when a new dog is registered in the database.
                    query_vec = extract_features(raw_img)
                    records = db.get_all_records()

                    best_candidate = None
                    best_result = None
                    highest_score = 0.0

                    # Compare the uploaded snout with the stored image of
                    # every registered dog using the new biometric pipeline.
                    for r in records:
                        image_path = r.get("image_path")

                        if not image_path or not os.path.exists(image_path):
                            continue

                        try:
                            stored_img = Image.open(image_path).convert("RGB")

                            result = biometric_match(
                                raw_img,
                                stored_img
                            )

                            score = result["final_similarity"]

                            if score > highest_score:
                                highest_score = score
                                best_candidate = r
                                best_result = result

                        except Exception:
                            continue

                    confirmed_match = (
                        best_candidate is not None
                        and best_result is not None
                        and best_result["match"]
                    )

                if confirmed_match:
                    st.markdown(
                        f"<span class='badge-verified'>{T['match_found']} "
                        f"({best_result['final_similarity']*100:.1f}%)</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"### `{best_candidate['token_id']}`")
                    st.write(f"**Ward:** {best_candidate['ward']}")
                    st.write(f"**Status:** `{best_candidate['status']}`")
                    st.write(f"**Attending Doctor:** Dr. `{best_candidate.get('assigned_vet', 'Unassigned')}`")
                    st.write(f"**Rabies Vaccine Due:** `{best_candidate['booster_due']}`")

                    # Show the same biometric scores used by the matcher.
                    b1, b2, b3, b4 = st.columns(4)

                    b1.metric(
                        "MobileNet",
                        f"{best_result['global_similarity']*100:.1f}%"
                    )
                    b2.metric(
                        "SIFT",
                        f"{best_result['sift_similarity']*100:.1f}%"
                    )
                    b3.metric(
                        "ORB",
                        f"{best_result['orb_similarity']*100:.1f}%"
                    )
                    b4.metric(
                        "Final Match",
                        f"{best_result['final_similarity']*100:.1f}%"
                    )

                else:
                    st.markdown(
                        f"<span class='badge-unregistered'>{T['no_match']}</span>",
                        unsafe_allow_html=True
                    )

                    if best_result is not None:
                        st.caption(
                            f"Best biometric similarity: "
                            f"{best_result['final_similarity']*100:.1f}%"
                        )

                    if user_role == "FEEDER":
                        st.markdown("---")
                        st.subheader(T['submit_lead'])
                        with st.form("feeder_dispatch_form"):
                            def_lat = WARD_METRICS.get(user_ward, {}).get("lat", 18.5204)
                            def_lng = WARD_METRICS.get(user_ward, {}).get("lng", 73.8567)
                            req_lat = st.number_input("Latitude", value=def_lat, format="%.6f")
                            req_lng = st.number_input("Longitude", value=def_lng, format="%.6f")
                            spot_landmark = st.text_input(T["landmark"], placeholder="e.g. Near Shaniwar Wada North Gate")

                            if st.form_submit_button(T['submit_lead'], type="primary", use_container_width=True):
                                if not spot_landmark.strip():
                                    st.error("Please specify landmark / जवळची खूण नमूद करा.")
                                else:
                                    lead_id = f"LEAD-{uuid.uuid4().hex[:6].upper()}"
                                    saved_path = db.save_image_to_disk(raw_img, lead_id, folder="citizen_leads")
                                    db.insert_citizen_request(user_ward, spot_landmark, req_lat, req_lng, saved_path, query_vec, current_username)
                                    st.session_state["uploader_key"] += 1
                                    st.session_state["action_notice"] = f"Lead dispatched to Van Squad in {user_ward}."
                                    st.rerun()

                    elif user_role == "TRANSIT":
                        st.markdown("---")
                        st.subheader(T['admit_intake'])
                        with st.form("transit_custody_form"):
                            auto_token = db.generate_location_token(user_ward, WARD_METRICS)
                            st.text_input("Assigned Token Number", value=auto_token, disabled=True)
                            dog_sex = st.radio("Sex / लिंग", ["MALE (नर)", "FEMALE (मादी)"], horizontal=True)
                            def_lat = WARD_METRICS.get(user_ward, {}).get("lat", 18.5204)
                            def_lng = WARD_METRICS.get(user_ward, {}).get("lng", 73.8567)
                            cap_lat = st.number_input("Latitude", value=def_lat, format="%.6f")
                            cap_lng = st.number_input("Longitude", value=def_lng, format="%.6f")
                            cap_landmark = st.text_input(T["landmark"], value=f"Van Patrol - {user_ward}")

                            if st.form_submit_button(T['admit_intake'], type="primary", use_container_width=True):
                                saved_path = db.save_image_to_disk(raw_img, auto_token, folder="canine_records")
                                clean_sex = "MALE" if "MALE" in dog_sex else "FEMALE"
                                assigned_vet = db.insert_intake_record(
                                    token_id=auto_token,
                                    ward=user_ward,
                                    sex=clean_sex,
                                    lat=cap_lat,
                                    lng=cap_lng,
                                    landmark=cap_landmark,
                                    enrolled_by=current_username,
                                    role=user_role,
                                    image_path=saved_path,
                                    embedding_list=query_vec
                                )
                                st.session_state["uploader_key"] += 1
                                st.session_state["action_notice"] = f"Dog '{auto_token}' admitted. Routed to Dr. @{assigned_vet}."
                                st.rerun()

# --- B. TRANSIT ASSIGNED TASKS ---
elif st.session_state["nav_page"] == T["nav_tasks"]:
    st.markdown(f"### {T['nav_tasks']}")
    my_tasks = db.get_pending_requests_for_driver(current_username)
    if not my_tasks:
        st.info("No pending pickups for your van right now / सध्या कोणतीही नवीन कामे नाहीत.")
    else:
        for item in my_tasks:
            with st.container():
                st.markdown(f"#### {item['landmark']}")
                lc1, lc2 = st.columns([1, 2])
                with lc1:
                    if os.path.exists(item["image_path"]):
                        st.image(item["image_path"], use_container_width=True)
                with lc2:
                    st.write(f"**Feeder Submitter:** `{item['reported_by']}` at {item['created_at']}")
                    st.write(f"**GPS Coordinates:** `{item['lat']}, {item['lng']}`")

                    with st.form(f"promote_{item['request_id']}"):
                        c_sex = st.radio("Sex on Catching / लिंग", ["MALE (नर)", "FEMALE (मादी)"], horizontal=True, key=f"s_{item['request_id']}")
                        if st.form_submit_button(T['admit_intake'], type="primary"):
                            new_id = db.generate_location_token(item["ward"], WARD_METRICS)
                            emb = np.frombuffer(item["embedding"], dtype=np.float32).tolist() if isinstance(item["embedding"], (bytes, bytearray)) else item["embedding"]
                            clean_sex = "MALE" if "MALE" in c_sex else "FEMALE"

                            assigned_vet = db.insert_intake_record(
                                token_id=new_id,
                                ward=item["ward"],
                                sex=clean_sex,
                                lat=float(item["lat"]),
                                lng=float(item["lng"]),
                                landmark=item["landmark"],
                                enrolled_by=current_username,
                                role=user_role,
                                image_path=item["image_path"],
                                embedding_list=emb
                            )
                            db.resolve_citizen_request(item["request_id"])
                            st.session_state["action_notice"] = f"Dog '{new_id}' secured and admitted. Routed to Dr. @{assigned_vet}."
                            st.rerun()
                st.markdown("---")

# --- C. WARD LEDGER & PASSPORT DOWNLOAD ---
elif st.session_state["nav_page"] == T["nav_ledger"]:
    st.markdown(f"### {T['nav_ledger']}")
    records = db.get_all_records(user_ward if user_role == "FEEDER" else None)

    if not records:
        st.info("No records in this ward jurisdiction yet.")
    else:
        st.markdown("#### Official Health Certificate Download")
        c_pick, c_cert = st.columns([1.2, 1.8])

        with c_pick:
            doc_id = st.selectbox("Select Dog for Health Certificate", options=[r["token_id"] for r in records])
            selected_item = next(r for r in records if r["token_id"] == doc_id)

            if os.path.exists(selected_item["image_path"]):
                st.image(selected_item["image_path"], caption=f"Rhinarium Record: {selected_item['token_id']}", use_container_width=True)

        with c_cert:
            st.write(f"**Token ID:** `{selected_item['token_id']}` • **Sex:** `{selected_item['sex']}`")
            st.write(f"**Status:** `{selected_item['status']}`")
            st.write(f"**Operating Doctor:** Dr. `{selected_item.get('assigned_vet', 'N/A')}`")
            st.write(f"**Rabies Vaccine Date:** `{selected_item.get('vax_date', 'N/A')}` (Batch: `{selected_item.get('arv_batch', 'N/A')}`)")
            st.write(f"**Annual Booster Due:** `{selected_item.get('booster_due', 'N/A')}`")
            st.write(f"**Colony Habitat:** {selected_item['landmark']}")

            # Generate ReportLab Certificate Buffer with photo and QR
            pdf_data = generate_canine_passport(selected_item)
            
            st.download_button(
                label=f"📄 Download Biometric Health Passport ({selected_item['token_id']})",
                data=pdf_data,
                file_name=f"PawProof_Passport_{selected_item['token_id']}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )

        st.markdown("---")
        st.markdown("#### Ward Canine Census Registry")
        st.dataframe([{
            "Token ID": r["token_id"],
            "Ward": r["ward"],
            "Sex": r["sex"],
            "Status": r["status"],
            "Doctor": r.get("assigned_vet", "N/A"),
            "Surgery / ARV Date": r.get("vax_date", "Pending"),
            "Authorized Release Date": r.get("release_date", "Pending"),
            "Colony Landmark": r["landmark"]
        } for r in records], use_container_width=True)

# --- D. SURVEILLANCE AUDIT ---
elif st.session_state["nav_page"] == T["nav_audit"]:
    st.markdown(f"### {T['nav_audit']}")
    all_recs = db.get_all_records()
    tot = len(all_recs)
    vaxed = sum(1 for r in all_recs if r["status"] in ["STERILIZED_VACCINATED", "CLEARED_FOR_RELEASE", "RELEASED"])
    rate = (vaxed / tot * 100) if tot > 0 else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Enrolled Dogs", tot)
    m1.metric("Sterilized & Vaccinated", vaxed)
    m3.metric("Ward Immunity Level", f"{rate:.1f}%", delta=f"{rate - 70.0:.1f}% vs WHO Target")

# =========================================================
# GLOBAL FOOTER
# =========================================================
st.markdown(f"""
<div class="portal-footer">
    <div>{T['footer_compliance']}</div>
    <div style="margin-top: 4px; color: #A8A29E;">{T['footer_support']}</div>
</div>
""", unsafe_allow_html=True)