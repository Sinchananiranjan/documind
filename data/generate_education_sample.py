"""Generate a 3rd test PDF on a General/Educational topic (Biology & Ecosystems) for 3-document isolation testing."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_education_pdf(output_path: str = "./data/educational_doc.pdf"):
    """Generate a sample General/Educational Biology PDF document."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor("#7C2D12"),
        spaceAfter=12
    )
    story.append(Paragraph("Chapter 4: Principles of Ecology & Food Chains", title_style))
    story.append(Paragraph("<b>Subject:</b> Biological Sciences | <b>Grade:</b> Introductory Biology", styles['Normal']))
    story.append(Spacer(1, 15))

    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], textColor=colors.HexColor("#C2410C"))
    story.append(Paragraph("1. Primary Producers and Consumers", h2_style))
    story.append(Paragraph(
        "Primary producers, such as green plants and algae, capture solar energy via photosynthesis to synthesize organic compounds. "
        "Primary consumers (herbivores) feed directly on primary producers, transferring energy up the trophic levels.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 15))

    story.append(Paragraph("2. Trophic Level Energy Table", h2_style))
    table_data = [
        ["Trophic Level", "Organism Type", "Energy Transferred", "Efficiency"],
        ["Level 1", "Primary Producers (Plants)", "10,000 kcal", "100%"],
        ["Level 2", "Primary Consumers (Herbivores)", "1,000 kcal", "10%"],
        ["Level 3", "Secondary Consumers (Carnivores)", "100 kcal", "10%"],
        ["Level 4", "Tertiary Consumers (Apex Predators)", "10 kcal", "10%"]
    ]
    t = Table(table_data, colWidths=[110, 180, 100, 90])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#7C2D12")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#FFEDD5")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#FDBA74")),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    story.append(Paragraph("3. Ecological Calculations", h2_style))
    story.append(Paragraph(
        "Energy transfer efficiency between trophic levels is governed by Lindeman's 10% Rule. "
        "If a primary producer captures 10,000 kcal of energy, the percentage remaining at tertiary level is (10 / 10000) * 100 = 0.1%.",
        styles['BodyText']
    ))

    doc.build(story)
    print(f"Educational PDF successfully generated at: {os.path.abspath(output_path)}")
    return output_path

if __name__ == "__main__":
    generate_education_pdf()
