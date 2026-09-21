import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def generate_canine_passport(record):
    """Generates official printable Municipal Canine Certificate."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Header styling
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#1E3A8A"))
    c.drawString(50, 750, "MUNICIPAL CORPORATION - ANIMAL WELFARE DEPARTMENT")
    
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(colors.HexColor("#047857"))
    c.drawString(50, 730, "OFFICIAL CONTACTLESS CANINE HEALTH CERTIFICATE (NO EAR-NOTCH)")
    
    c.setStrokeColor(colors.gray)
    c.line(50, 720, 550, 720)
    
    fields = [
        ("Permanent Token ID:", record["token_id"]),
        ("Administrative Ward:", record["ward"]),
        ("Canine Sex:", record["sex"]),
        ("Clinical Status:", record["status"]),
        ("Anti-Rabies Dose Date:", record["vax_date"]),
        ("Next Annual Booster Due:", record["booster_due"]),
        ("ARV Batch Number:", record["arv_batch"]),
        ("Capture Coordinates:", f"{record['lat']}, {record['lng']}"),
        ("Original Landmark:", record["landmark"]),
        ("Attending Officer:", f"{record['enrolled_by']} ({record['role']})"),
        ("Surgical Specification:", "Sterilized - 100% Intact Ears (Non-Mutilated)"),
        ("Biometric Standard:", "Epidermal Rhinarium Ridge Mapping")
    ]
    
    y = 680
    for label, val in fields:
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(colors.black)
        c.drawString(50, y, label)
        c.setFont("Helvetica", 9)
        c.drawString(220, y, str(val))
        y -= 22
        
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.gray)
    c.drawString(50, 100, "Certified in compliance with ABC Rules 2023. File under physical ward documentation.")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer