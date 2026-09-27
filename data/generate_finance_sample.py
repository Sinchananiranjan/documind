"""Generate a second test PDF on a completely different topic (Finance & Business) for domain testing."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_finance_pdf(output_path: str = "./data/finance_report.pdf"):
    """Generate a sample Finance & Business PDF report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor("#065F46"),
        spaceAfter=12
    )
    story.append(Paragraph("Q3 Corporate Financial & Revenue Analysis", title_style))
    story.append(Paragraph("<b>Department:</b> Global Finance | <b>Fiscal Year:</b> 2026", styles['Normal']))
    story.append(Spacer(1, 15))

    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], textColor=colors.HexColor("#047857"))
    story.append(Paragraph("1. Revenue Overview", h2_style))
    story.append(Paragraph(
        "In Q3 2026, total consolidated net revenue grew by 14.2% year-over-year to $4.85 million. "
        "The primary revenue growth drivers were cloud software subscriptions and international enterprise support services.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 15))

    story.append(Paragraph("2. Financial Performance Table", h2_style))
    table_data = [
        ["Quarter", "Gross Revenue", "Operating Expenses", "Net Margin"],
        ["Q1 2026", "$3.90M", "$2.10M", "46.1%"],
        ["Q2 2026", "$4.25M", "$2.30M", "45.8%"],
        ["Q3 2026", "$4.85M", "$2.45M", "49.4%"],
        ["Q4 (Proj)", "$5.10M", "$2.55M", "50.0%"]
    ]
    t = Table(table_data, colWidths=[110, 130, 140, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#065F46")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#ECFDF5")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#A7F3D0")),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    story.append(Paragraph("3. Profit Margin Calculation", h2_style))
    story.append(Paragraph(
        "Net profit margin is calculated using the formula: Net Margin (%) = ((Gross Revenue - Operating Expenses) / Gross Revenue) * 100. "
        "For Q3 2026, with Gross Revenue of $4.85M and Operating Expenses of $2.45M, the net profit margin equals 49.48%.",
        styles['BodyText']
    ))

    doc.build(story)
    print(f"Finance PDF successfully generated at: {os.path.abspath(output_path)}")
    return output_path

if __name__ == "__main__":
    generate_finance_pdf()
