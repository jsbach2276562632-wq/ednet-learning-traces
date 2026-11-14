"""Build the English report from report.md and the checked-in figures."""

import json
import re
from html import escape
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, PolyLine, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
BASE = "https://github.com/jsbach2276562632-wq/ednet-learning-traces"
INK = colors.HexColor("#17263c")
BLUE = colors.HexColor("#235e83")
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=9, textColor=INK)
HEADING = ParagraphStyle("heading", parent=BODY, fontName="Helvetica-Bold", fontSize=15, leading=19,
                         spaceBefore=10, spaceAfter=7, keepWithNext=True)
TITLE = ParagraphStyle("title", parent=HEADING, fontSize=24, leading=30, spaceAfter=14)
CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=8.5, leading=11)


def prose(text):
    text = escape(text.replace("−", "-"))
    def link(match):
        label, path = match.groups()
        relative = (REPORT / path).resolve().relative_to(ROOT).as_posix()
        return f'<link href="{BASE}/blob/main/{relative}" color="#235e83">{label}</link>'
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)


def calibration_plot():
    data = json.loads((ROOT / "results/kt_diagnostics/diagnostics.json").read_text(encoding="utf-8"))
    plot = Drawing(480, 160)
    palette = [BLUE, colors.HexColor("#c56827"), colors.HexColor("#40836b")]
    for split, left, title in (("test_id", 56, "Seen bundles"), ("test_ood", 274, "Unseen bundles")):
        bottom, size = 35, 103
        plot.add(String(left, 148, title, fontName="Helvetica-Bold", fontSize=9, fillColor=INK))
        for step in (0, 0.5, 1):
            delta = step * size
            plot.add(Line(left, bottom + delta, left + size, bottom + delta, strokeColor=colors.lightgrey))
            plot.add(String(left - 7, bottom + delta - 3, f"{step:g}", textAnchor="end", fontSize=7))
            plot.add(String(left + delta, bottom - 12, f"{step:g}", textAnchor="middle", fontSize=7))
        plot.add(Line(left, bottom, left + size, bottom + size, strokeColor=colors.grey, strokeDashArray=[3, 3]))
        for (name, model), color in zip(data["models"].items(), palette):
            points = [(left + size * row["predicted"], bottom + size * row["observed"])
                      for row in model[split]["bins"]]
            plot.add(PolyLine(points, strokeColor=color, strokeWidth=1.5))
        plot.add(String(left + size / 2, 13, "Predicted probability", textAnchor="middle", fontSize=8))
    plot.add(String(3, 148, "Observed", fontSize=8))
    for index, (name, color) in enumerate(zip(data["models"], palette)):
        y = 121 - 15 * index
        plot.add(Line(385, y + 2, 397, y + 2, strokeColor=color, strokeWidth=1.5))
        plot.add(String(401, y, name, fontSize=7.5))
    return plot


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor("#d3dce5"))
    canvas.line(52, 42, A4[0] - 52, 42)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(BLUE)
    canvas.drawString(52, 28, "EDNET LEARNING TRACES")
    canvas.linkURL(BASE, (52, 25, 170, 39))
    canvas.drawRightString(A4[0] - 52, 28, str(doc.page))


def build():
    story = []
    for block in (REPORT / "report.md").read_text(encoding="utf-8").strip().split("\n\n"):
        if block.startswith("# "):
            story.append(Paragraph(prose(block[2:]), TITLE))
        elif block.startswith("## "):
            if block == "## Early low-accuracy alerts":
                story.extend([calibration_plot(), Paragraph("Figure 1. Next-answer calibration by bundle group.", CAPTION),
                              PageBreak()])
            story.append(Paragraph(prose(block[3:]), HEADING))
        elif block.startswith("|"):
            rows = [[Paragraph(prose(cell.strip()), BODY) for cell in line.strip("|").split("|")]
                    for i, line in enumerate(block.splitlines()) if i != 1]
            table = Table(rows, colWidths=[145, 95, 96, 143], hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce8f2")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7f9")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(table)
        elif block.startswith("!["):
            path = re.search(r"\(([^)]+)\)", block).group(1)
            figure = Image(str(REPORT / path), width=420, height=246, kind="proportional")
            figure.hAlign = "CENTER"
            story.extend([figure, Paragraph("Figure 2. Cases reached at equal review budgets.", CAPTION)])
        else:
            story.append(Paragraph(prose(block), BODY))
    doc = SimpleDocTemplate(str(REPORT / "report.pdf"), pagesize=A4, leftMargin=52, rightMargin=52,
                            topMargin=43, bottomMargin=54, title="Student Knowledge Tracing on EdNet")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build()
