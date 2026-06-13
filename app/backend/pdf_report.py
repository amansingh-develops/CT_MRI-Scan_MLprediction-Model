"""
ScanSight Medical AI — Clinical Report Generator
=================================================
Generates a professional, multi-page PDF clinical report from scan analysis data.
Uses ReportLab for layout and Pillow for image handling.
"""

import os
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, KeepTogether
)
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate, Frame
from reportlab.graphics.shapes import Drawing, Rect, String, Line


# ── Color Palette ─────────────────────────────────────────────────────────────
NAVY       = colors.HexColor('#0f172a')
DARK_BLUE  = colors.HexColor('#1e3a8a')
MID_BLUE   = colors.HexColor('#1e40af')
LIGHT_BLUE = colors.HexColor('#3b82f6')
SLATE      = colors.HexColor('#475569')
LIGHT_GRAY = colors.HexColor('#f1f5f9')
BORDER     = colors.HexColor('#cbd5e1')
RED_DARK   = colors.HexColor('#991b1b')
RED        = colors.HexColor('#dc2626')
GREEN_DARK = colors.HexColor('#166534')
GREEN      = colors.HexColor('#16a34a')
ORANGE     = colors.HexColor('#ea580c')
WHITE      = colors.white


def _build_header_footer(canvas, doc):
    """Draw header bar and footer on every page."""
    canvas.saveState()
    w, h = letter

    # ── Header bar ──
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 50, w, 50, fill=True, stroke=False)
    canvas.setFillColor(WHITE)
    canvas.setFont('Helvetica-Bold', 14)
    canvas.drawString(72, h - 34, 'ScanSight Medical AI')
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(colors.HexColor('#94a3b8'))
    canvas.drawRightString(w - 72, h - 34, 'Clinical Liver Analysis Report')

    # ── Accent line under header ──
    canvas.setStrokeColor(LIGHT_BLUE)
    canvas.setLineWidth(2)
    canvas.line(0, h - 50, w, h - 50)

    # ── Footer ──
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(SLATE)
    canvas.drawString(72, 30, f'Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}')
    canvas.drawCentredString(w / 2, 30, 'CONFIDENTIAL — For Clinical Use Only')
    canvas.drawRightString(w - 72, 30, f'Page {doc.page}')

    # Footer line
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(72, 42, w - 72, 42)

    canvas.restoreState()


def _section_hr():
    """Return a horizontal rule for visual section breaks."""
    return HRFlowable(width='100%', thickness=0.5, color=BORDER, spaceAfter=12, spaceBefore=4)


def generate_medical_report(scan_data: dict) -> bytes:
    """Generate a professional clinical PDF report.
    
    Args:
        scan_data: Full scan document from Firestore including the `result` object.
    
    Returns:
        PDF file contents as bytes.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=80,   # space for header bar
        bottomMargin=60,
    )

    styles = getSampleStyleSheet()

    # ── Custom Styles ─────────────────────────────────────────────────────────
    styles.add(ParagraphStyle(
        'ReportTitle', parent=styles['Heading1'],
        fontSize=22, spaceAfter=4, textColor=DARK_BLUE, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        'ReportSubtitle', parent=styles['Normal'],
        fontSize=11, spaceAfter=18, textColor=SLATE, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        'SectionHead', parent=styles['Heading2'],
        fontSize=14, spaceBefore=18, spaceAfter=8, textColor=DARK_BLUE,
        fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=10, spaceAfter=8, leading=15, textColor=colors.HexColor('#334155'),
    ))
    styles.add(ParagraphStyle(
        'BodyBold', parent=styles['Normal'],
        fontSize=10, spaceAfter=8, leading=15, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#1e293b'),
    ))
    styles.add(ParagraphStyle(
        'SmallGray', parent=styles['Normal'],
        fontSize=8, textColor=SLATE, leading=11,
    ))
    styles.add(ParagraphStyle(
        'FindingAnomaly', parent=styles['Normal'],
        fontSize=13, fontName='Helvetica-Bold', textColor=RED_DARK,
        spaceBefore=4, spaceAfter=4, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        'FindingClear', parent=styles['Normal'],
        fontSize=13, fontName='Helvetica-Bold', textColor=GREEN_DARK,
        spaceBefore=4, spaceAfter=4, alignment=TA_CENTER,
    ))

    elements = []

    # ── Extract data safely ───────────────────────────────────────────────────
    result = scan_data.get('result') or {}
    # If top-level fields exist (flat Firestore doc), use them as fallback
    if not result:
        result = scan_data

    has_anomaly   = result.get('hasAnomaly', scan_data.get('hasAnomaly', False))
    anomaly_slices = result.get('anomalySlices', scan_data.get('anomalySlices', 0))
    total_slices  = result.get('totalSlices', scan_data.get('totalSlices', 0))
    confidence    = result.get('confidence', scan_data.get('confidence', 0))
    stage         = result.get('estimatedStage', scan_data.get('estimatedStage'))
    liver_vol     = result.get('liverVolumePercent', scan_data.get('liverVolumePercent', 0))
    tumor_ratio   = result.get('tumorToLiverRatio', scan_data.get('tumorToLiverRatio', 0))
    total_liver   = result.get('totalLiverPx', 0)
    total_tumor   = result.get('totalTumorPx', 0)
    proc_time     = result.get('processingTimeMs', scan_data.get('processingTimeMs', 0))
    model_ver     = result.get('modelVersion', scan_data.get('modelVersion', 'U-Net v1.0'))
    slices        = result.get('slices', [])
    max_tumor     = result.get('maxTumorSlice')
    affected_range = result.get('affectedSliceRange')

    scan_ref  = scan_data.get('scanRef', 'N/A')
    file_name = scan_data.get('fileName', scan_data.get('label', 'N/A'))
    scan_type = scan_data.get('scanType', 'N/A')
    modality  = scan_data.get('modality', 'CT')

    # Parse completedAt
    completed_at = scan_data.get('completedAt')
    date_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
    if completed_at:
        try:
            if hasattr(completed_at, 'timestamp'):
                date_str = datetime.fromtimestamp(completed_at.timestamp()).strftime("%B %d, %Y, %I:%M %p")
            elif isinstance(completed_at, str) and len(completed_at) > 5:
                date_str = completed_at
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — Title & Summary
    # ══════════════════════════════════════════════════════════════════════════

    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph("Liver CT Scan Analysis Report", styles['ReportTitle']))
    elements.append(Paragraph(
        f"Automated AI-assisted diagnostic imaging report  ·  {date_str}",
        styles['ReportSubtitle'],
    ))
    elements.append(_section_hr())

    # ── Finding Banner ────────────────────────────────────────────────────────
    if has_anomaly:
        banner_bg = colors.HexColor('#fef2f2')
        banner_border = RED
        banner_text = f"ANOMALY DETECTED — {anomaly_slices} of {total_slices} slices affected"
        banner_style = styles['FindingAnomaly']
    else:
        banner_bg = colors.HexColor('#f0fdf4')
        banner_border = GREEN
        banner_text = f"CLEAR — All {total_slices} slices normal, no anomalies detected"
        banner_style = styles['FindingClear']

    banner_data = [[Paragraph(banner_text, banner_style)]]
    banner_table = Table(banner_data, colWidths=[6.5 * inch])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), banner_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, banner_border),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 16),
        ('RIGHTPADDING', (0, 0), (-1, -1), 16),
    ]))
    elements.append(banner_table)
    elements.append(Spacer(1, 0.25 * inch))

    # ── Patient & Scan Info ───────────────────────────────────────────────────
    elements.append(Paragraph("1. Scan Information", styles['SectionHead']))
    info_rows = [
        ["Scan Reference", scan_ref, "Date of Analysis", date_str],
        ["File Name", file_name, "Imaging Modality", modality],
        ["Scan Type", scan_type, "AI Model", model_ver],
        ["User ID", scan_data.get('userId', 'N/A')[:20] + '...', "Processing Time", f"{proc_time / 1000:.1f}s" if proc_time else "N/A"],
    ]
    info_table = Table(info_rows, colWidths=[1.4*inch, 1.85*inch, 1.4*inch, 1.85*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), LIGHT_GRAY),
        ('BACKGROUND', (2, 0), (2, -1), LIGHT_GRAY),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#334155')),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.2 * inch))

    # ── Summary Metrics ───────────────────────────────────────────────────────
    elements.append(Paragraph("2. Analysis Summary", styles['SectionHead']))

    summary_rows = [
        ["Total CT Slices Analyzed", str(total_slices)],
        ["Slices with Abnormal Tissue", str(anomaly_slices)],
        ["AI Confidence Score", f"{confidence}%"],
        ["Overall Finding", "ANOMALY DETECTED" if has_anomaly else "CLEAR"],
    ]
    if has_anomaly and stage:
        summary_rows.append(["Estimated Clinical Stage", str(stage)])

    finding_color = RED_DARK if has_anomaly else GREEN_DARK
    summary_table = Table(summary_rows, colWidths=[3 * inch, 3.5 * inch])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), SLATE),
        ('TEXTCOLOR', (1, -1), (1, -1), finding_color),
        ('FONTNAME', (1, -1), (1, -1), 'Helvetica-Bold'),
        ('LINEBELOW', (0, 0), (-1, -2), 0.3, BORDER),
        ('LINEBELOW', (0, -1), (-1, -1), 1, finding_color),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.2 * inch))

    # ── Volumetric Assessment ─────────────────────────────────────────────────
    elements.append(Paragraph("3. Volumetric Assessment", styles['SectionHead']))
    vol_rows = [
        ["Liver Parenchyma Coverage", f"{liver_vol}% of total scan area"],
        ["Tumor-to-Liver Volume Ratio", f"{tumor_ratio}%"],
        ["Total Liver Tissue (pixels)", f"{total_liver:,}"],
        ["Total Tumor Tissue (pixels)", f"{total_tumor:,}"],
    ]
    if affected_range:
        vol_rows.append(["Affected Slice Range", f"Slice {affected_range.get('start', 0) + 1} to {affected_range.get('end', 0) + 1}"])

    vol_table = Table(vol_rows, colWidths=[3 * inch, 3.5 * inch])
    vol_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), SLATE),
        ('LINEBELOW', (0, 0), (-1, -1), 0.3, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(vol_table)
    elements.append(Spacer(1, 0.15 * inch))

    # ── Primary Overlay Image ─────────────────────────────────────────────────
    primary_overlay = result.get('overlayImage', '')
    if primary_overlay:
        try:
            file_parts = primary_overlay.split('/static/')
            if len(file_parts) == 2:
                local_img_path = os.path.join('static', file_parts[1])
                if os.path.exists(local_img_path):
                    elements.append(Paragraph("4. Key Diagnostic Image", styles['SectionHead']))
                    img = Image(local_img_path, width=3.8 * inch, height=3.8 * inch)
                    img.hAlign = 'CENTER'
                    elements.append(img)
                    caption = "AI-generated segmentation overlay. "
                    if max_tumor:
                        caption += f'Max tumor concentration: slice "{max_tumor.get("sliceName", "N/A")}" ({max_tumor.get("tumorPercent", 0)}% coverage).'
                    elements.append(Paragraph(caption, ParagraphStyle(
                        'ImgCaption', parent=styles['Normal'],
                        fontSize=8, textColor=SLATE, alignment=TA_CENTER, spaceBefore=4, spaceAfter=12,
                    )))
        except Exception as e:
            print(f"[PDF] Image embedding skipped: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — Per-Slice Findings
    # ══════════════════════════════════════════════════════════════════════════
    valid_slices = [s for s in slices if not s.get('error')]
    if valid_slices:
        elements.append(PageBreak())
        section_num = 5 if primary_overlay else 4
        elements.append(Paragraph(f"{section_num}. Per-Slice Findings", styles['SectionHead']))
        elements.append(Paragraph(
            f"Detailed breakdown of each CT slice analyzed by the AI model. "
            f"Slices marked <b><font color='#dc2626'>ANOMALY</font></b> contain regions of abnormal tissue density.",
            styles['Body'],
        ))

        # Build table
        header = ['Slice #', 'File Name', 'Liver %', 'Tumor %', 'Status']
        table_data = [header]
        for s in valid_slices:
            status = 'ANOMALY' if s.get('hasTumor') else 'CLEAR'
            table_data.append([
                str(s.get('sliceIndex', 0) + 1),
                str(s.get('sliceName', 'N/A'))[:25],
                f"{s.get('liverPercent', 0)}%",
                f"{s.get('tumorPercent', 0)}%",
                status,
            ])

        col_widths = [0.65*inch, 2.3*inch, 0.9*inch, 0.9*inch, 1.3*inch]
        slice_table = Table(table_data, colWidths=col_widths, repeatRows=1)

        # Style — alternate row colors, red/green status text
        table_style = [
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), NAVY),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            # Body
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ALIGN', (0, 1), (0, -1), 'CENTER'),
            ('ALIGN', (2, 1), (3, -1), 'CENTER'),
            ('ALIGN', (4, 1), (4, -1), 'CENTER'),
            ('FONTNAME', (4, 1), (4, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.3, BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]
        # Alternate row backgrounds and status colors
        for i, s in enumerate(valid_slices, start=1):
            if i % 2 == 0:
                table_style.append(('BACKGROUND', (0, i), (-1, i), LIGHT_GRAY))
            if s.get('hasTumor'):
                table_style.append(('TEXTCOLOR', (4, i), (4, i), RED))
            else:
                table_style.append(('TEXTCOLOR', (4, i), (4, i), GREEN))

        slice_table.setStyle(TableStyle(table_style))
        elements.append(slice_table)
        elements.append(Spacer(1, 0.3 * inch))

    # ══════════════════════════════════════════════════════════════════════════
    # Clinical Interpretation
    # ══════════════════════════════════════════════════════════════════════════
    next_section = (6 if primary_overlay else 5) if valid_slices else (5 if primary_overlay else 4)
    elements.append(Paragraph(f"{next_section}. Clinical Interpretation", styles['SectionHead']))

    if has_anomaly:
        interp = (
            f"The AI model analyzed <b>{total_slices} CT scan slices</b> and detected regions of "
            f"abnormal tissue density in <b>{anomaly_slices} slice{'s' if anomaly_slices != 1 else ''}</b>. "
            f"The overall anomaly detection confidence is <b>{confidence}%</b>. "
        )
        if max_tumor:
            interp += (
                f"The highest concentration of suspected tumor tissue was found in slice "
                f"<b>\"{max_tumor.get('sliceName', 'N/A')}\"</b> with tumor coverage of "
                f"<b>{max_tumor.get('tumorPercent', 0)}%</b> of that slice. "
            )
        if stage:
            interp += f"Based on volumetric analysis, the estimated clinical stage is <b>{stage}</b>. "
        interp += (
            f"Liver parenchyma was detected covering <b>{liver_vol}%</b> of the total scan area, "
            f"with a tumor-to-liver volume ratio of <b>{tumor_ratio}%</b>. "
        )
        if affected_range:
            interp += (
                f"Affected tissue spans from slice {affected_range.get('start', 0) + 1} "
                f"to slice {affected_range.get('end', 0) + 1}. "
            )
        interp += "Clinical verification by a qualified hepatologist or radiologist is strongly recommended."
    else:
        interp = (
            f"The AI model analyzed <b>{total_slices} CT scan slices</b> and found "
            f"<b>no signs of abnormal tissue growth</b>. The model confidence for a clear scan is "
            f"<b>{confidence}%</b>. Liver parenchyma was detected covering <b>{liver_vol}%</b> of "
            f"the total scan area. No further action is indicated based on this analysis, however "
            f"routine follow-up imaging should be scheduled per clinical guidelines."
        )
    elements.append(Paragraph(interp, styles['Body']))
    elements.append(Spacer(1, 0.3 * inch))

    # ══════════════════════════════════════════════════════════════════════════
    # Processing Metadata
    # ══════════════════════════════════════════════════════════════════════════
    elements.append(Paragraph(f"{next_section + 1}. Processing Metadata", styles['SectionHead']))
    meta_rows = [
        ["AI Model", model_ver],
        ["Input Resolution", "256 x 256 px (normalized)"],
        ["Processing Time", f"{proc_time / 1000:.1f} seconds" if proc_time else "N/A"],
        ["Slices Processed", str(total_slices)],
        ["Report Generated", datetime.now().strftime("%B %d, %Y at %I:%M:%S %p")],
    ]
    meta_table = Table(meta_rows, colWidths=[2.5 * inch, 4 * inch])
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), SLATE),
        ('LINEBELOW', (0, 0), (-1, -1), 0.3, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 0.4 * inch))

    # ══════════════════════════════════════════════════════════════════════════
    # Disclaimer
    # ══════════════════════════════════════════════════════════════════════════
    disclaimer_text = (
        "<b>DISCLAIMER:</b> This report is generated by ScanSight AI and is intended for investigational "
        "and supportive use only. It does not constitute a medical diagnosis and should not replace "
        "the professional judgment of a qualified radiologist, hepatologist, or physician. AI-based "
        "analysis may contain false positives or false negatives. All findings should be independently "
        "verified through clinical examination and additional diagnostic procedures as appropriate."
    )
    disclaimer_style = ParagraphStyle(
        'DisclaimerBox', parent=styles['Normal'],
        fontSize=8, textColor=SLATE, leading=11, spaceBefore=0, spaceAfter=0,
    )
    disc_data = [[Paragraph(disclaimer_text, disclaimer_style)]]
    disc_table = Table(disc_data, colWidths=[6.5 * inch])
    disc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GRAY),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    elements.append(disc_table)

    # ── Build PDF ─────────────────────────────────────────────────────────────
    doc.build(elements, onFirstPage=_build_header_footer, onLaterPages=_build_header_footer)

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
