import io
import datetime
from src.database.database import SessionLocal, Alert, Reading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

def generate_incident_pdf(alert_id: int) -> io.BytesIO:
    """
    Generates a PDF incident report for the given alert ID.
    Includes alert details, context readings, and a plot.
    """
    db = SessionLocal()
    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise ValueError(f"Alert {alert_id} not found")

        reading_id = alert.reading_id
        start_id = max(1, reading_id - 30)
        end_id = reading_id + 30
        readings = db.query(Reading).filter(Reading.id >= start_id, Reading.id <= end_id).order_by(Reading.id).all()
        
        # 1. Create a plot of the context readings
        timestamps = [r.ts for r in readings]
        values = [r.value for r in readings]
        anomaly_indices = [i for i, r in enumerate(readings) if r.is_anomaly]
        
        plt.figure(figsize=(10, 4))
        plt.plot(timestamps, values, color='blue', label='Sensor Value')
        for idx in anomaly_indices:
            plt.scatter(timestamps[idx], values[idx], color='red', marker='x')
            
        plt.title(f"Sensor Value Context - Alert {alert_id}")
        plt.xlabel("Timestamp")
        plt.ylabel("Value")
        plt.xticks([])  # Hide x-axis labels as they get cluttered
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()
        
        # 2. Generate PDF
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Title
        title = Paragraph(f"Incident Report: Alert #{alert_id}", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        # Details
        elements.append(Paragraph(f"<b>Timestamp:</b> {alert.ts}", styles['Normal']))
        elements.append(Paragraph(f"<b>Sensor ID:</b> {alert.sensor_id}", styles['Normal']))
        elements.append(Paragraph(f"<b>Severity:</b> {alert.severity}", styles['Normal']))
        elements.append(Paragraph(f"<b>Score:</b> {alert.score:.4f}", styles['Normal']))
        elements.append(Paragraph(f"<b>Reason:</b> {alert.reason}", styles['Normal']))
        elements.append(Paragraph(f"<b>Status:</b> {alert.status}", styles['Normal']))
        elements.append(Paragraph(f"<b>Feedback:</b> {alert.feedback or 'N/A'}", styles['Normal']))
        elements.append(Paragraph(f"<b>Operator Note:</b> {alert.operator_note or 'N/A'}", styles['Normal']))
        
        elements.append(Spacer(1, 20))
        
        # Plot
        elements.append(Paragraph("<b>Contextual Data Plot</b>", styles['Heading3']))
        elements.append(Image(img_buffer, width=450, height=180))
        elements.append(Spacer(1, 20))
        
        # Table of recent readings
        elements.append(Paragraph("<b>Surrounding Readings (Subset)</b>", styles['Heading3']))
        
        table_data = [["ID", "Timestamp", "Value", "Anomaly Score", "Is Anomaly"]]
        for r in readings[max(0, len(readings)//2 - 5) : min(len(readings), len(readings)//2 + 6)]:
            table_data.append([str(r.id), r.ts, f"{r.value:.2f}", f"{r.anomaly_score:.4f}", "Yes" if r.is_anomaly else "No"])
            
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(t)
        
        doc.build(elements)
        pdf_buffer.seek(0)
        return pdf_buffer
        
    finally:
        db.close()
