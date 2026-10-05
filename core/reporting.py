import os
import io
import qrcode
from PIL import Image
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm

def generate_canine_passport(record: dict) -> bytes:
    """
    Generates a formal municipal Animal Birth Control & Anti-Rabies
    Vaccination Certificate (PDF) containing the dog's snout photo,
    biometric identification hash, clinical surgery details, municipal release memo,
    and statutory Rule 11 ABC compliance declaration.
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm
    )

    styles = getSampleStyleSheet()

    # Color Palette: Matte Earthy Brown, Warm Amber, Clean Slate
    c_primary = colors.HexColor("#78350F")       # Saddle Brown
    c_dark = colors.HexColor("#292524")          # Slate Black
    c_accent = colors.HexColor("#92400E")        # Warm Amber
    c_cream = colors.HexColor("#FDFBF7")         # Card background
    c_border = colors.HexColor("#D6C7B2")        # Neutral border
    c_row_alt = colors.HexColor("#F5EFE6")       # Soft Tan table header

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=c_primary,
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=c_accent,
        alignment=1
    )
    section_head = ParagraphStyle(
        'SecHead',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=c_primary
    )
    label_style = ParagraphStyle(
        'LabelStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#78716C")
    )
    val_style = ParagraphStyle(
        'ValStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=c_dark
    )
    footer_style = ParagraphStyle(
        'FooterStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#A8A29E"),
        alignment=1
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("MUNICIPAL CORPORATION ANIMAL HEALTH REGISTRY", title_style))
    story.append(Paragraph("CANINE BIOMETRIC PASSPORT & ANTI-RABIES IMMUNIZATION RECORD", subtitle_style))
    story.append(Paragraph("Statutory Animal Birth Control Certification • Non-Invasive Rhinarium Identification", ParagraphStyle('SubSub', parent=subtitle_style, fontName='Helvetica', fontSize=7.5, textColor=c_dark)))
    story.append(Spacer(1, 3 * mm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceAfter=8))

    # 2. Canine Identity & Rhinarium Image Card
    token_id = record.get("token_id", "N/A")
    img_path = record.get("image_path", "")

    dog_photo_flowable = None
    if img_path and os.path.exists(img_path):
        try:
            with Image.open(img_path) as im:
                rgb_img = im.convert("RGB")
                img_io = io.BytesIO()
                rgb_img.save(img_io, format="JPEG", quality=90)
                img_io.seek(0)
                dog_photo_flowable = RLImage(img_io, width=54 * mm, height=54 * mm)
        except Exception:
            dog_photo_flowable = None

    if not dog_photo_flowable:
        p_table = Table([[Paragraph("<i>Snout Photo<br/>Not Available</i>", label_style)]], width=54 * mm, height=54 * mm)
        p_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5EFE6")),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        dog_photo_flowable = p_table

    # Verification QR Code
    qr = qrcode.QRCode(box_size=3, border=1)
    qr.add_data(f"PAWPROOF-VERIFIED:{token_id}:{record.get('ward')}")
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="#451A03", back_color="white")
    qr_buf = io.BytesIO()
    qr_img.save(qr_buf, format="JPEG", quality=90)
    qr_buf.seek(0)
    qr_flowable = RLImage(qr_buf, width=26 * mm, height=26 * mm)

    id_data = [
        [Paragraph("CANINE IDENTIFIER", label_style), Paragraph(f"<b>{token_id}</b>", ParagraphStyle('Tk', parent=val_style, fontName='Helvetica-Bold', fontSize=11, textColor=c_primary))],
        [Paragraph("JURISDICTION WARD", label_style), Paragraph(str(record.get("ward", "N/A")), val_style)],
        [Paragraph("SEX", label_style), Paragraph(f"<b>{record.get('sex', 'N/A')}</b>", val_style)],
        [Paragraph("CURRENT STATUS", label_style), Paragraph(f"<b>{record.get('status', 'REGISTERED')}</b>", val_style)],
        [Paragraph("ORIGINAL COLONY SPOT", label_style), Paragraph(str(record.get("landmark", "N/A")), val_style)],
        [Paragraph("GPS COORDINATES", label_style), Paragraph(f"{record.get('lat', 0.0):.6f}, {record.get('lng', 0.0):.6f}", val_style)],
        [Paragraph("IDENTIFICATION METHOD", label_style), Paragraph("Digital Canine Rhinarium Dermal Ridge Print", val_style)]
    ]

    id_details_table = Table(id_data, colWidths=[40 * mm, 74 * mm])
    id_details_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#EFE8DC")),
    ]))

    top_grid = Table([[dog_photo_flowable, id_details_table]], colWidths=[60 * mm, 126 * mm])
    top_grid.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 0), (-1, -1), c_cream),
        ('BOX', (0, 0), (-1, -1), 1, c_border),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(top_grid)
    story.append(Spacer(1, 3.5 * mm))

    # 3. Clinical & Surgery Audit (Clean: Procedure, ARV, Post-Op Recovery)
    story.append(Paragraph("VETERINARY SURGERY & IMMUNIZATION AUDIT", section_head))
    story.append(Spacer(1, 1.5 * mm))

    vax_date_str = str(record.get("vax_date", "Pending"))
    booster_str = str(record.get("booster_due", "Pending"))
    arv_code = str(record.get("arv_batch", "N/A"))
    doctor = str(record.get("assigned_vet", "Registered ABC Surgeon"))
    rel_date = str(record.get("release_date", "Pending Official Sign-off"))

    surg_data = [
        [
            Paragraph("Surgical Procedure", label_style),
            Paragraph("Anti-Rabies Vaccine (ARV)", label_style),
            Paragraph("Annual Booster Due", label_style)
        ],
        [
            Paragraph(f"<b>Sterilized ({record.get('sex')})</b>", val_style),
            Paragraph(f"Batch: <b>{arv_code}</b><br/>Administered: {vax_date_str}", val_style),
            Paragraph(f"<b>{booster_str}</b>", val_style)
        ],
        [
            Paragraph("Operating Surgeon", label_style),
            Paragraph("Scheduled Release Date", label_style),
            Paragraph("Post-Op Recovery Status", label_style)
        ],
        [
            Paragraph(f"Dr. {doctor}", val_style),
            Paragraph(f"<b>{rel_date}</b>", val_style),
            Paragraph("<b>Observation Complete (Fit for Release)</b>", val_style)
        ]
    ]

    surg_table = Table(surg_data, colWidths=[62 * mm, 62 * mm, 62 * mm])
    surg_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_row_alt),
        ('BACKGROUND', (0, 2), (-1, 2), c_row_alt),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
    ]))
    story.append(surg_table)
    story.append(Spacer(1, 3.5 * mm))

    # 4. Statutory Rule 11 Compliance Block
    story.append(Paragraph("RULE 11 ABC COMPLIANCE & MUNICIPAL AUTHENTICATION", section_head))
    story.append(Spacer(1, 1.5 * mm))

    notes_text = record.get("notes") or "Standard ABC Protocol. Biometric token verified. Relocation strictly prohibited."
    legal_text = (
        f"<b>Documentation Memo:</b> {notes_text}<br/>"
        "<b>Statutory Declaration:</b> This canine was captured, surgically sterilized, vaccinated against rabies, "
        "and cleared for colony release strictly within its native 500-meter capture perimeter pursuant to "
        "Rule 11 of the Animal Birth Control Rules, 2023. Unauthorized relocation constitutes a punishable violation under Section 11 of the PCA Act."
    )

    cert_box = Table(
        [[Paragraph(legal_text, ParagraphStyle('Leg', parent=val_style, fontSize=7.5, leading=10.5)), qr_flowable]],
        colWidths=[156 * mm, 30 * mm]
    )
    cert_box.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1, c_border),
        ('BACKGROUND', (0, 0), (-1, -1), c_cream),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(cert_box)
    story.append(Spacer(1, 6 * mm))

    # 5. Dual Sign-off Block
    sign_table = Table([
        [
            Paragraph("____________________________<br/><b>Supervising Veterinary Officer</b><br/>ABC Operating Center", ParagraphStyle('Sg1', parent=val_style, fontSize=7.5, alignment=1)),
            Paragraph("____________________________<br/><b>Medical Officer of Health (MOH)</b><br/>Municipal Corporation Desk", ParagraphStyle('Sg2', parent=val_style, fontSize=7.5, alignment=1))
        ]
    ], colWidths=[93 * mm, 93 * mm])
    story.append(sign_table)
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph(f"Digitally authenticated via PawProof Biometric Registry on {datetime.today().strftime('%B %d, %Y')} • Document Ref: PMC/ARV/{token_id}", footer_style))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()