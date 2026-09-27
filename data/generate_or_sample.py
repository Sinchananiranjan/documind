"""Generate a comprehensive sample Operations Research & Linear Programming PDF for testing."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_or_pdf(output_path: str = "./data/operations_research_report.pdf") -> str:
    """Generate a sample PDF on Operations Research and Linear Programming."""
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
    story.append(Paragraph("Operations Research: Linear Programming & Optimization", title_style))
    story.append(Paragraph("<b>Course:</b> Industrial Engineering & OR | <b>Author:</b> Optimization Lab", styles['Normal']))
    story.append(Spacer(1, 15))

    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], textColor=colors.HexColor("#2563EB"))
    
    # Section 1: Linear Programming Fundamentals
    story.append(Paragraph("1. Fundamentals of Linear Programming", h2_style))
    story.append(Paragraph(
        "Linear Programming (LP) is a mathematical optimization method used to achieve the best outcome in a given mathematical model. "
        "The general form of a Linear Programming problem consists of an objective function to maximize or minimize Z = c1*x1 + c2*x2 + ... + cn*xn, "
        "subject to a set of linear structural constraints a_i1*x1 + a_i2*x2 <= b_i and non-negativity constraints x_j >= 0.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Key components of Linear Programming include:\n"
        "• Decision Variables: Represent quantities to be determined (e.g. x1, x2).\n"
        "• Objective Function: Mathematical expression specifying the goal (e.g. Max Z = 80x1 + 55x2).\n"
        "• Constraints: Linear inequalities limiting available resources.\n"
        "• Parameters: Numerical coefficients (c_j, a_ij, b_i) assumed to be known constants.\n"
        "• Assumptions: Linearity, additivity, divisibility, certainty, and non-negativity.\n"
        "• Applications: Production planning, supply chain management, diet problems, and capital budgeting.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 15))

    # Section 2: Graphical Method & Optimal Solutions
    story.append(Paragraph("2. Graphical Method and Corner Point Analysis", h2_style))
    story.append(Paragraph(
        "The graphical method solves two-variable LP problems by plotting constraint lines and identifying the feasible region. "
        "According to the fundamental theorem of LP, an optimal solution must occur at one of the extreme corner points of the convex feasible region.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Consider the objective function Max Z = 80x1 + 55x2. Evaluating corner points gives:\n"
        "• Point (0, 0): Z = 80(0) + 55(0) = 0\n"
        "• Point (0, 10): Z = 80(0) + 55(10) = 550\n"
        "• Point (10, 0): Z = 80(10) + 55(0) = 800\n"
        "• Point (8, 4): Z = 80(8) + 55(4) = 640 + 220 = 860.\n"
        "The point (8, 4) is optimal because it yields the highest objective value Z = 860 among all feasible corner points.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "For objective function Max Z = 4x1 + 3x2, evaluating Z at point (4.5, 2) gives Z = 4(4.5) + 3(2) = 18 + 6 = 24.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 15))

    # Section 3: Special Cases in Graphical Method
    story.append(Paragraph("3. Special Cases in the Graphical Method", h2_style))
    story.append(Paragraph(
        "There are three distinct special cases in graphical optimization:\n"
        "1. Multiple Optimal Solutions (Alternate Optima): Occurs when the objective function line is parallel to a binding constraint boundary line. "
        "Every point on the line segment connecting adjacent optimal corner points produces the same maximum objective value.\n"
        "2. No Optimal Solution (Infeasibility): Occurs when no single point satisfies all constraints simultaneously, resulting in an empty feasible region.\n"
        "3. Unbounded Solution: Occurs when the feasible region extends infinitely in the direction of optimization, allowing Z to increase without limit.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 15))

    # Section 4: Phases of an Operations Research Study
    story.append(Paragraph("4. Complete Phases of an Operations Research Study", h2_style))
    story.append(Paragraph(
        "A formal Operations Research study follows five sequential phases:\n"
        "Phase 1 - Problem Formulation & Definition: Identifying system objectives, decision variables, resource limits, and system boundaries.\n"
        "Phase 2 - Mathematical Model Construction: Translating real-world problem relationships into mathematical equations (objective function & constraints).\n"
        "Phase 3 - Model Solution & Algorithm Execution: Applying mathematical optimization techniques (e.g., Simplex, Graphical Method, Dynamic Programming) to determine optimal values.\n"
        "Phase 4 - Model Validation & Sensitivity Analysis: Testing the model under varying parameter conditions to verify accuracy and stability.\n"
        "Phase 5 - Implementation & Solution Monitoring: Translating mathematical results into operational decisions and monitoring performance.",
        styles['BodyText']
    ))

    doc.build(story)
    print(f"Operations Research PDF generated at: {os.path.abspath(output_path)}")
    return output_path

if __name__ == "__main__":
    generate_or_pdf()
