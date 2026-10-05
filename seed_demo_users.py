import core.database as db

def setup_demo_users():
    ward = "Ward 12 - Shivajinagar"
    default_pwd = "PawProof@2026"

    users = [
        {
            "username": "demo_feeder",
            "full_name": "Demo Citizen Feeder",
            "role": "FEEDER",
            "ward": ward,
            "badge": "VOL-DEMO-001",
            "phone": "9800000001"
        },
        {
            "username": "demo_transit",
            "full_name": "Demo Transit Squad",
            "role": "TRANSIT",
            "ward": ward,
            "badge": "MH-12-DEMO-101",
            "phone": "9800000002"
        },
        {
            "username": "demo_vet",
            "full_name": "Dr. Demo Veterinary Surgeon",
            "role": "VET_OFFICE",
            "ward": ward,
            "badge": "VCI-DEMO-2026",
            "phone": "9800000003"
        },
        {
            "username": "demo_muni",
            "full_name": "Demo Municipal Health Officer",
            "role": "MUNICIPAL_CORP",
            "ward": ward,
            "badge": "PMC-DEMO-MOH",
            "phone": "9800000004"
        }
    ]

    print(f"--- Registering Demo Accounts for: {ward} ---")
    for u in users:
        success, msg = db.create_user(
            username=u["username"],
            password=default_pwd,
            full_name=u["full_name"],
            role=u["role"],
            ward=u["ward"],
            badge_or_vci=u["badge"],
            contact_phone=u["phone"]
        )
        if success:
            print(f"✓ Registered [{u['role']}]: @{u['username']}")
        else:
            print(f"! @{u['username']}: {msg}")

    print("\nRegistration complete. Password for all: PawProof@2026")

if __name__ == "__main__":
    setup_demo_users()