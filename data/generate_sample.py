"""Generate a multi-page sample PDF containing Text, Tables, Formulas, and Visual Elements for testing."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing, Rect, String, Line

def generate_sample_pdf(output_path: str = "./data/sample_document.pdf"):
    """Generate a sample multimodal test PDF document with text, tables, formulas, and visual elements."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=12
    )
    story.append(Paragraph("DocuMind Architecture & Benchmark Report", title_style))
    story.append(Paragraph("<b>Author:</b> Antigravity Systems | <b>Date:</b> September 2026", styles['Normal']))
    story.append(Spacer(1, 15))

    # Section 1: Text content (Page 1)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], textColor=colors.HexColor("#2563EB"))
    story.append(Paragraph("1. Executive Summary", h2_style))
    story.append(Paragraph(
        "DocuMind is an open-source multimodal document analysis application designed to run entirely locally on lightweight hardware. "
        "It combines PyMuPDF for document parsing, PyTesseract for optical character recognition (OCR), ChromaDB for vector retrieval, "
        "and LangGraph for stateful agent workflow orchestration.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. System Specifications", h2_style))
    story.append(Paragraph(
        "The target hardware deployment consists of a Windows OS with an Intel i3 processor, 8GB RAM, and no dedicated GPU. "
        "All models utilize CPU execution and localized embedding quantization via sentence-transformers (all-MiniLM-L6-v2).",
        styles['BodyText']
    ))
    story.append(Spacer(1, 15))

    # Section 3: Numerical & Formula measurements (Page 1)
    story.append(Paragraph("3. Signal Processing & SNR Calculations", h2_style))
    story.append(Paragraph(
        "In test run 4B, the peak signal amplitude was measured at 3.5 mV, while the background noise floor was recorded at 0.75 mV. "
        "The Signal-to-Noise Ratio (SNR) in decibels is calculated using the formula SNR(dB) = 20 * log10(V_signal / V_noise).",
        styles['BodyText']
    ))
    story.append(Spacer(1, 15))

    # Table 1: Performance benchmark (Page 1)
    story.append(Paragraph("4. Hardware Performance Benchmark Table", h2_style))
    table_data = [
        ["Component", "Resource Allocation", "Average Latency", "Status"],
        ["PyMuPDF Parser", "15 MB RAM", "45 ms / page", "Optimal"],
        ["Sentence Transformer", "90 MB RAM", "120 ms / query", "Active"],
        ["ChromaDB Vector Store", "35 MB RAM", "12 ms / search", "Ready"],
        ["Qwen 2.5 1.5B (LLM)", "1.1 GB RAM", "0.9 s / completion", "Local CPU"]
    ]
    t = Table(table_data, colWidths=[150, 130, 120, 90])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E40AF")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F3F4F6")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    # Page Break -> Page 2
    story.append(PageBreak())

    # Page 2: Visual diagram element
    story.append(Paragraph("5. LangGraph Workflow Routing Diagram", h2_style))
    story.append(Paragraph(
        "The system employs a conditional StateGraph with state nodes for retrieval, fast routing, domain-specific text/table/image analysis, "
        "and hallucination verification with automated retry logic.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 15))

    # Vector drawing acting as embedded diagram image
    d = Drawing(400, 120)
    d.add(Rect(10, 30, 90, 50, fillColor=colors.HexColor("#DBEAFE"), strokeColor=colors.HexColor("#2563EB")))
    d.add(String(25, 52, "Retrieval Node", fontSize=9, fillColor=colors.HexColor("#1E3A8A")))

    d.add(Line(100, 55, 140, 55, strokeColor=colors.HexColor("#94A3B8"), strokeWidth=2))

    d.add(Rect(140, 30, 90, 50, fillColor=colors.HexColor("#FEF08A"), strokeColor=colors.HexColor("#CA8A04")))
    d.add(String(155, 52, "Fast Router", fontSize=9, fillColor=colors.HexColor("#854D0E")))

    d.add(Line(230, 55, 270, 55, strokeColor=colors.HexColor("#94A3B8"), strokeWidth=2))

    d.add(Rect(270, 30, 110, 50, fillColor=colors.HexColor("#DCFCE7"), strokeColor=colors.HexColor("#16A34A")))
    d.add(String(280, 52, "Verify & Output", fontSize=9, fillColor=colors.HexColor("#14532D")))

    story.append(d)
    story.append(Spacer(1, 20))

    story.append(Paragraph("6. Operational Guidelines & Verification", h2_style))
    story.append(Paragraph(
        "When an answer cannot be grounded strictly within the context of the document, the verify_answer node fails validation. "
        "DocuMind returns: 'I couldn't find this in the document.' with verified=False.",
        styles['BodyText']
    ))

    doc.build(story)
    print(f"Sample PDF successfully generated at: {os.path.abspath(output_path)}")
    return output_path

if __name__ == "__main__":
    generate_sample_pdf()
