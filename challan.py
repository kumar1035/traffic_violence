"""
challan.py
Generates a digital e-challan PDF for each violation: violation details,
annotated evidence image, QR code (containing challan ID + plate + a mock
verification URL), and a unique challan ID. This is the artifact that
closes the loop from detection to an actual enforcement document.
"""

import os
import uuid
from datetime import datetime

import qrcode
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


CHALLAN_DIR = "challans"
SEVERITY_LABELS = {
    range(1, 4): ("LOW", HexColor("#2ecc71")),
    range(4, 7): ("MEDIUM", HexColor("#f39c12")),
    range(7, 11): ("HIGH", HexColor("#e74c3c")),
}


def _severity_label(severity):
    for r, (label, color) in SEVERITY_LABELS.items():
        if severity in r:
            return label, color
    return "UNKNOWN", black


def generate_challan_id():
    """Generates a short unique challan ID, e.g. CHL-A1B2C3D4"""
    return f"CHL-{uuid.uuid4().hex[:8].upper()}"


def generate_qr_code(challan_id, plate_number, save_path):
    """
    Generates a QR code containing the challan ID, plate number, and a
    mock verification URL. In a real deployment this URL would point to
    an actual lookup/payment portal.
    """
    verify_url = f"https://trafficvision-demo.local/verify/{challan_id}"
    qr_data = f"Challan: {challan_id}\nPlate: {plate_number}\nVerify: {verify_url}"

    qr = qrcode.QRCode(version=1, box_size=8, border=2)
    qr.add_data(qr_data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(save_path)
    return save_path


def generate_challan_pdf(violation_type, plate_number, confidence, severity,
                          zone, timestamp, reason, evidence_image_path=None,
                          is_repeat_offender=False, repeat_count=0):
    """
    Generates a full e-challan PDF document. Returns (challan_id, pdf_path).
    """
    os.makedirs(CHALLAN_DIR, exist_ok=True)

    challan_id = generate_challan_id()
    pdf_path = os.path.join(CHALLAN_DIR, f"{challan_id}.pdf")

    qr_path = os.path.join(CHALLAN_DIR, f"{challan_id}_qr.png")
    generate_qr_code(challan_id, plate_number or "UNKNOWN", qr_path)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # --- Header ---
    c.setFillColor(HexColor("#1a3c6e"))
    c.rect(0, height - 30 * mm, width, 30 * mm, fill=True, stroke=False)

    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(15 * mm, height - 15 * mm, "TrafficVision AI")
    c.setFont("Helvetica", 10)
    c.drawString(15 * mm, height - 22 * mm, "Automated Traffic Violation Enforcement System")

    c.setFont("Helvetica-Bold", 11)
    c.drawRightString(width - 15 * mm, height - 15 * mm, f"Challan ID: {challan_id}")
    c.setFont("Helvetica", 9)
    c.drawRightString(width - 15 * mm, height - 22 * mm, f"Issued: {timestamp}")

    y = height - 40 * mm

    # --- Severity badge ---
    label, color = _severity_label(severity)
    c.setFillColor(color)
    c.roundRect(15 * mm, y - 8 * mm, 45 * mm, 10 * mm, 2 * mm, fill=True, stroke=False)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(15 * mm + 22.5 * mm, y - 5.5 * mm, f"{label} RISK ({severity}/10)")

    if is_repeat_offender:
        c.setFillColor(HexColor("#c0392b"))
        c.roundRect(65 * mm, y - 8 * mm, 60 * mm, 10 * mm, 2 * mm, fill=True, stroke=False)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(65 * mm + 30 * mm, y - 5.5 * mm, f"REPEAT OFFENDER ({repeat_count}x)")

    y -= 18 * mm

    # --- Violation details ---
    c.setFillColor(black)
    details = [
        ("Violation Type", violation_type.replace("_", " ").title()),
        ("Vehicle Plate", plate_number or "Not identified"),
        ("Detection Confidence", f"{confidence:.1%}"),
        ("Zone / Location", zone or "Not specified"),
        ("Timestamp", timestamp),
    ]

    c.setFont("Helvetica-Bold", 11)
    for label_text, value_text in details:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(15 * mm, y, f"{label_text}:")
        c.setFont("Helvetica", 10)
        c.drawString(70 * mm, y, str(value_text))
        y -= 7 * mm

    y -= 3 * mm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(15 * mm, y, "AI Detection Reasoning:")
    y -= 6 * mm
    c.setFont("Helvetica-Oblique", 9)

    # wrap the reason text manually across multiple lines if long
    max_chars = 95
    reason_text = reason or "No reasoning provided."
    words = reason_text.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 <= max_chars:
            line += (word + " ")
        else:
            c.drawString(15 * mm, y, line.strip())
            y -= 5 * mm
            line = word + " "
    if line:
        c.drawString(15 * mm, y, line.strip())
        y -= 8 * mm

    # --- Evidence image ---
    if evidence_image_path and os.path.exists(evidence_image_path):
        try:
            img = ImageReader(evidence_image_path)
            img_w, img_h = img.getSize()
            max_w = 110 * mm
            max_h = 70 * mm
            scale = min(max_w / img_w, max_h / img_h)
            draw_w, draw_h = img_w * scale, img_h * scale

            c.setFont("Helvetica-Bold", 10)
            c.drawString(15 * mm, y, "Evidence:")
            y -= 5 * mm
            c.drawImage(img, 15 * mm, y - draw_h, width=draw_w, height=draw_h)
            y -= (draw_h + 8 * mm)
        except Exception as e:
            c.setFont("Helvetica", 8)
            c.drawString(15 * mm, y, f"(Evidence image could not be embedded: {e})")
            y -= 8 * mm

    # --- QR code ---
    qr_img = ImageReader(qr_path)
    qr_size = 30 * mm
    c.drawImage(qr_img, width - 45 * mm, 15 * mm, width=qr_size, height=qr_size)
    c.setFont("Helvetica", 7)
    c.drawCentredString(width - 30 * mm, 12 * mm, "Scan to verify challan")

    # --- Footer ---
    c.setFont("Helvetica-Oblique", 7)
    c.setFillColor(HexColor("#666666"))
    c.drawString(15 * mm, 12 * mm,
                 "This is a system-generated prototype document for demonstration purposes.")
    c.drawString(15 * mm, 8 * mm,
                 "Generated by TrafficVision AI — AI-driven traffic enforcement intelligence.")

    c.save()

    return challan_id, pdf_path


if __name__ == "__main__":
    # Standalone test — generates a sample challan PDF
    print("Generating a test challan...")

    challan_id, pdf_path = generate_challan_pdf(
        violation_type="helmet_violation",
        plate_number="KA01AB1234",
        confidence=0.91,
        severity=8,
        zone="MG Road",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        reason="Fine-tuned helmet model classified this head region as 'Without Helmet' "
               "with 0.91 confidence, positioned above a detected motorcycle.",
        evidence_image_path=None,  # set to a real image path to test embedding
        is_repeat_offender=True,
        repeat_count=3,
    )

    print(f"Challan ID: {challan_id}")
    print(f"PDF saved to: {pdf_path}")
    print("\nOpen the PDF in VSCode (or any PDF viewer) to verify it looks correct.")