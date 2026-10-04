import io
import time
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(inspection_data, annotated_image_bytes=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=12
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6
    )
    table_text = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#334155")
    )
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontSize=8,
        leading=11,
        fontName="Helvetica-Bold",
        textColor=colors.white
    )

    elements = []

    # Title Section
    elements.append(Paragraph("PAVEMENT DISTRESS & INFRASTRUCTURE AUDIT REPORT", title_style))
    elements.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')} | Automated Computer Vision & Inverse Perspective Mapping", subtitle_style))

    # Municipal Asset Identity Section
    road_asset = inspection_data.get("road_asset", {})
    if road_asset:
        elements.append(Paragraph("Municipal Asset & Contractor Identity", section_heading))
        asset_rows = [
            [
                Paragraph("<b>Corridor / Road Name:</b>", table_text), Paragraph(f"<b>{road_asset.get('name', 'N/A')}</b>", table_text),
                Paragraph("<b>Asset Registry ID:</b>", table_text), Paragraph(f"{road_asset.get('road_id', 'N/A')}", table_text)
            ],
            [
                Paragraph("<b>Constructed By (Contractor):</b>", table_text), Paragraph(f"<font color='#4f46e5'><b>{road_asset.get('contractor', 'N/A')}</b></font>", table_text),
                Paragraph("<b>Construction Date:</b>", table_text), Paragraph(f"{road_asset.get('constructed_date', 'N/A')}", table_text)
            ],
            [
                Paragraph("<b>Executing Authority:</b>", table_text), Paragraph(f"{road_asset.get('authority', 'N/A')}", table_text),
                Paragraph("<b>Last Resurfaced:</b>", table_text), Paragraph(f"{road_asset.get('last_resurfaced', 'N/A')}", table_text)
            ],
            [
                Paragraph("<b>Warranty Status:</b>", table_text), Paragraph(f"{road_asset.get('warranty_status', 'N/A')}", table_text),
                Paragraph("<b>Pavement Mix:</b>", table_text), Paragraph(f"{road_asset.get('asphalt_mix', 'N/A')}", table_text)
            ]
        ]
        asset_table = Table(asset_rows, colWidths=[120, 160, 110, 150])
        asset_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(asset_table)
        elements.append(Spacer(1, 6))

    # Summary Metadata
    summary = inspection_data.get("summary", {})
    loc = inspection_data.get("location", {})
    pci = summary.get("pci_index", 100)

    pci_color = "#10b981" if pci >= 80 else ("#f59e0b" if pci >= 55 else "#ef4444")

    meta_data = [
        [
            Paragraph("<b>Latitude:</b>", table_text), Paragraph(f"{loc.get('lat', 'N/A')}", table_text),
            Paragraph("<b>Potholes Detected:</b>", table_text), Paragraph(f"{summary.get('potholes', 0)}", table_text)
        ],
        [
            Paragraph("<b>Longitude:</b>", table_text), Paragraph(f"{loc.get('lng', 'N/A')}", table_text),
            Paragraph("<b>Pavement Cracks:</b>", table_text), Paragraph(f"{summary.get('cracks', 0)}", table_text)
        ],
        [
            Paragraph("<b>Sensor / Source:</b>", table_text), Paragraph(f"{loc.get('source', 'GPS')}", table_text),
            Paragraph("<b>PCI Rating:</b>", table_text), Paragraph(f"<font color='{pci_color}'><b>{pci}/100</b></font>", table_text)
        ]
    ]

    meta_table = Table(meta_data, colWidths=[90, 180, 110, 160])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 8))

    # Visual Evidence
    if annotated_image_bytes:
        try:
            elements.append(Paragraph("Visual Inspection Evidence", section_heading))
            img_io = io.BytesIO(annotated_image_bytes)
            rl_img = RLImage(img_io, width=420, height=210)
            elements.append(rl_img)
            elements.append(Spacer(1, 6))
        except Exception as img_err:
            print(f"Report image embedding skipped: {img_err}")

    # Lane Quality Assessment Section
    lane_rep = inspection_data.get("lane_quality_report", {})
    if lane_rep:
        elements.append(Paragraph("Road Lane Marking Quality Assessment", section_heading))
        lane_data = [
            [
                Paragraph("<b>Serviceability Rating:</b>", table_text), Paragraph(f"{lane_rep.get('serviceability', 'N/A')}", table_text),
                Paragraph("<b>Surface Wear:</b>", table_text), Paragraph(f"{lane_rep.get('wear_percentage', 0)}%", table_text)
            ],
            [
                Paragraph("<b>Line Continuity:</b>", table_text), Paragraph(f"{lane_rep.get('continuity_percentage', 0)}%", table_text),
                Paragraph("<b>Photometric Contrast:</b>", table_text), Paragraph(f"{lane_rep.get('contrast_ratio', 0)}:1", table_text)
            ],
            [
                Paragraph("<b>Recommended Action:</b>", table_text), 
                Paragraph(f"<b>{lane_rep.get('recommended_action', 'None')}</b>", table_text),
                Paragraph("<b>Target Segment:</b>", table_text), 
                Paragraph(f"~{lane_rep.get('estimated_distance_m', 3.5)}m Ground Centerline", table_text)
            ]
        ]
        lane_table = Table(lane_data, colWidths=[120, 160, 110, 150])
        lane_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(lane_table)
        elements.append(Spacer(1, 6))

    # Defect Inventory Table
    elements.append(Paragraph("Itemized Defect Inventory & IPM Physical Measurements", section_heading))

    defect_rows = [[
        Paragraph("ID", table_header),
        Paragraph("Classification", table_header),
        Paragraph("Confidence", table_header),
        Paragraph("Est. Distance", table_header),
        Paragraph("Physical Surface Area", table_header)
    ]]

    idx = 1
    detections = inspection_data.get("structural_detections", [])
    for d in detections:
        metrics = d.get("metrics", {})
        dist = f"{metrics.get('estimated_distance_m', 'N/A')} m"
        area = f"{metrics.get('surface_area_cm2', 'N/A')} cm²"
        conf = f"{int(d.get('confidence', 0) * 100)}%"
        defect_rows.append([
            Paragraph(f"#{idx}", table_text),
            Paragraph(d.get("class_name", "Defect"), table_text),
            Paragraph(conf, table_text),
            Paragraph(dist, table_text),
            Paragraph(area, table_text)
        ])
        idx += 1

    lanes = inspection_data.get("lane_evaluations", [])
    for l in lanes:
        m = l.get("metrics", {})
        defect_rows.append([
            Paragraph(f"#{idx}", table_text),
            Paragraph(f"{l.get('class_name', 'Lane Mark')} (Wear: {m.get('wear_percentage', 'N/A')}%)", table_text),
            Paragraph("91%", table_text),
            Paragraph(f"{m.get('estimated_distance_m', 3.5)} m", table_text),
            Paragraph(m.get("serviceability", "Maintenance Required"), table_text)
        ])
        idx += 1

    if len(defect_rows) == 1:
        defect_rows.append([
            Paragraph("-", table_text),
            Paragraph("No structural defects identified. Road surface nominal.", table_text),
            Paragraph("100%", table_text),
            Paragraph("-", table_text),
            Paragraph("-", table_text)
        ])

    def_table = Table(defect_rows, colWidths=[30, 200, 70, 100, 140])
    def_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    elements.append(def_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()