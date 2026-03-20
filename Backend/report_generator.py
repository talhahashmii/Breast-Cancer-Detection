from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import io
from pathlib import Path


def generate_pdf_report(prediction_data: dict, image_path: str = None) -> bytes:
    """Generate PDF report for prediction results"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=30,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#374151'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("BREAST CANCER DETECTION REPORT", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Report info
    report_date = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.red,
        borderPadding=10
    )
    story.append(Paragraph(
        "<b>⚠️ IMPORTANT DISCLAIMER:</b> This report is for research purposes only. "
        "This tool is NOT FDA approved and should NOT be used for clinical decision-making. "
        "Please consult a qualified radiologist for final diagnosis.",
        disclaimer_style
    ))
    story.append(Spacer(1, 0.3*inch))
    
    # Predictions
    story.append(Paragraph("PREDICTION RESULTS", heading_style))
    
    if "cc_view" in prediction_data:
        # Dual view predictions
        data = [
            ["View", "Prediction", "Confidence"],
            ["CC View", prediction_data["cc_view"]["prediction"].upper(), 
             f"{prediction_data['cc_view']['confidence']}%"],
            ["MLO View", prediction_data["mlo_view"]["prediction"].upper(), 
             f"{prediction_data['mlo_view']['confidence']}%"],
            ["FINAL PREDICTION", prediction_data["final_prediction"].upper(), 
             f"{prediction_data['final_confidence']}%"]
        ]
        
        risk_level = prediction_data["risk_level"]
        risk_color = colors.red if risk_level == "HIGH" else colors.green
    else:
        # Single view prediction
        data = [
            ["Metric", "Value"],
            ["Prediction", prediction_data["prediction"].upper()],
            ["Confidence", f"{prediction_data['confidence']}%"]
        ]
        risk_level = "HIGH" if prediction_data["prediction"] == "malignant" else "LOW"
        risk_color = colors.red if risk_level == "HIGH" else colors.green
    
    table = Table(data, colWidths=[2*inch, 2*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f2937')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), risk_color),
        ('TEXTCOLOR', (0, -1), (-1, -1), colors.whitesmoke),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.3*inch))
    
    # Risk Assessment
    story.append(Paragraph("RISK ASSESSMENT", heading_style))
    risk_text = f"<b>Risk Level:</b> {risk_level}"
    risk_description = (
        "HIGH: Suspicious findings detected. Immediate consultation with a radiologist is recommended."
        if risk_level == "HIGH"
        else "LOW: No suspicious findings detected. Normal screening recommended."
    )
    story.append(Paragraph(risk_text, styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(risk_description, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Model Information
    story.append(Paragraph("MODEL INFORMATION", heading_style))
    model_info = [
        ["Model Architecture", "Swin Transformer"],
        ["Training Dataset", "CBIS-DDSM Mammography Dataset"],
        ["Task", "Binary Classification (Benign vs Malignant)"],
        ["Input Size", "224x224 pixels"],
    ]
    
    model_table = Table(model_info, colWidths=[3*inch, 3*inch])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    
    story.append(model_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Footer
    footer_text = (
        "<font size=8><i>This report is generated automatically and is for informational purposes only. "
        "All medical decisions should be made in consultation with qualified healthcare professionals. "
        "For questions or concerns, please contact your healthcare provider.</i></font>"
    )
    story.append(Paragraph(footer_text, styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
