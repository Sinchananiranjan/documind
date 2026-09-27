"""Generate a comprehensive sample Error Detection & Correction PDF for domain testing."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_error_detection_pdf(output_path: str = "./data/error_detection_report.pdf") -> str:
    """Generate a comprehensive sample PDF on Error Detection and Correction concepts."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=12
    )
    story.append(Paragraph("Data Communication: Error Detection and Correction", title_style))
    story.append(Paragraph("<b>Module:</b> Information Theory & Coding | <b>Author:</b> Telecom Engineering", styles['Normal']))
    story.append(Spacer(1, 15))

    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], textColor=colors.HexColor("#2563EB"))
    
    # Section 1: Error Types
    story.append(Paragraph("1. Types of Errors: Single-Bit vs Burst Errors", h2_style))
    story.append(Paragraph(
        "A single-bit error occurs when only one bit of a given data unit (such as a byte or packet) is altered from 1 to 0 or 0 to 1 during transmission. "
        "In contrast, a burst error occurs when two or more bits in the data unit are corrupted.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "A burst error is significantly more likely than a single-bit error in real-world transmission channels because noise impulse duration is normally much longer than the duration of a single bit. "
        "Therefore, noise typically affects multiple consecutive bits.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 15))

    # Section 2: Hamming Distance & Error Control
    story.append(Paragraph("2. Hamming Distance and XOR Calculation", h2_style))
    story.append(Paragraph(
        "The Hamming distance d(x, y) between two codewords x and y of equal length is the number of positions at which the corresponding bits differ. "
        "It is calculated by applying the bitwise XOR operation (x ⊕ y) between the two codewords and then counting the number of 1s in the result. "
        "For example, the Hamming distance between 000 and 011 is 2 because two bit positions differ.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "To guarantee the detection of up to s errors in any transmitted codeword, the minimum Hamming distance dmin of the block code must satisfy dmin >= s + 1. "
        "To guarantee the correction of up to t errors, the minimum distance must satisfy dmin >= 2t + 1.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 15))

    # Section 3: Parity-Check Codes vs Cyclic Codes & Syndromes
    story.append(Paragraph("3. Parity-Check Codes, Cyclic Codes, and Syndrome Decoding", h2_style))
    story.append(Paragraph(
        "A linear parity-check code appends r parity check bits to k data bits to form an n-bit codeword, where n = k + r. "
        "For a block length n, the total number of possible codewords is 2^n. "
        "A parity-check code differs from a cyclic code in that a parity-check code is a general linear block code structure based on linear parity equations, "
        "whereas a cyclic code is a specific linear block code with the additional algebraic property that any cyclic circular shift of a valid codeword produces another valid codeword.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Hamming distance differs from syndrome in fundamental meaning: Hamming distance measures geometric separation between two codewords (bit difference count), "
        "whereas syndrome is a vector computed at the receiver to detect and locate errors. "
        "During decoding, the receiver calculates the syndrome vector S. If syndrome S = 0, no error occurred. "
        "When the syndrome is non-zero (such as S = 1), an error has occurred during transmission and specifies the error position.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 15))

    # Section 4: Cyclic Redundancy Check (CRC)
    story.append(Paragraph("4. Cyclic Redundancy Check (CRC)", h2_style))
    story.append(Paragraph(
        "Cyclic Redundancy Check (CRC) is a polynomial-based non-linear cyclic redundancy error detection algorithm widely used in digital networks and data storage. "
        "The receiver process in CRC determines whether a received codeword contains an error by dividing the received polynomial by a predefined generator polynomial G(x) using binary modulo-2 division. "
        "If the remainder of the modulo-2 division is zero, the receiver determines that the codeword contains no error and accepts it; "
        "if the remainder is non-zero, the receiver determines that an error occurred and rejects the codeword.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 15))

    # Section 5: Error Detection and Correction Overview
    story.append(Paragraph("5. Error Detection vs Error Correction Principles", h2_style))
    story.append(Paragraph(
        "Error detection is the process of discovering that errors occurred during transmission without identifying exact corrupted bit locations (e.g. via CRC or parity). "
        "Error correction is the process of locating corrupted bit positions and restoring original data bits using redundant bits (Forward Error Correction). "
        "Redundancy is the addition of redundant check bits to the data message. Coding transforms k data bits into n codeword bits. "
        "The role of the receiver is to inspect incoming codewords, evaluate syndrome or modulo-2 remainder, accept valid data, or request retransmission / execute bit correction.",
        styles['BodyText']
    ))

    doc.build(story)
    print(f"Error Detection PDF successfully generated at: {os.path.abspath(output_path)}")
    return output_path

if __name__ == "__main__":
    generate_error_detection_pdf()
