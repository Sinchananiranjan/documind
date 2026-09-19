"""Table extraction module using PyMuPDF and markdown formatting."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def extract_tables_from_page(page, page_num: int) -> List[Dict[str, Any]]:
    """
    Extract tables from a PyMuPDF page using find_tables() with fallback.
    Returns list of dicts: {'page_num': int, 'table_id': int, 'markdown': str, 'rows_count': int}
    """
    extracted_tables = []
    
    try:
        # PyMuPDF built-in table finder (available in PyMuPDF >= 1.23.0)
        tabs = page.find_tables()
        if tabs and len(tabs.tables) > 0:
            for i, tab in enumerate(tabs.tables):
                try:
                    df = tab.to_pandas()
                    md = df.to_markdown(index=False)
                    extracted_tables.append({
                        "page_num": page_num,
                        "table_id": i + 1,
                        "markdown": md,
                        "rows_count": len(df)
                    })
                except Exception as e:
                    # Fallback to direct text list extraction from table
                    logger.debug(f"to_pandas failed, falling back to raw list: {e}")
                    raw_extract = tab.extract()
                    if raw_extract:
                        headers = raw_extract[0]
                        rows = raw_extract[1:]
                        md_lines = [
                            "| " + " | ".join([str(c) if c else "" for c in headers]) + " |",
                            "| " + " | ".join(["---"] * len(headers)) + " |"
                        ]
                        for row in rows:
                            md_lines.append("| " + " | ".join([str(c) if c else "" for c in row]) + " |")
                        extracted_tables.append({
                            "page_num": page_num,
                            "table_id": i + 1,
                            "markdown": "\n".join(md_lines),
                            "rows_count": len(rows)
                        })
    except Exception as e:
        logger.warning(f"PyMuPDF find_tables failed on page {page_num}: {e}")
    
    # Fallback heuristic for structured text tables if no tables found via find_tables
    if not extracted_tables:
        heuristic_table = _extract_table_heuristic(page.get_text("text"), page_num)
        if heuristic_table:
            extracted_tables.append(heuristic_table)

    return extracted_tables


def _extract_table_heuristic(text: str, page_num: int) -> Dict[str, Any] | None:
    """Fallback heuristic detection for tab or multi-space aligned lines."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    table_lines = []
    
    for line in lines:
        # Look for lines with multiple pipe separators or multiple tab characters
        if "|" in line and line.count("|") >= 2:
            table_lines.append(line)
        elif "\t" in line and line.count("\t") >= 2:
            cols = [c.strip() for c in line.split("\t") if c.strip()]
            table_lines.append("| " + " | ".join(cols) + " |")

    if len(table_lines) >= 2:
        return {
            "page_num": page_num,
            "table_id": 1,
            "markdown": "\n".join(table_lines),
            "rows_count": len(table_lines)
        }
    return None
