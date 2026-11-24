"""Convert report/RL_Summative_Report.docx -> .pdf (docx -> HTML via mammoth -> headless Chrome).

    uv run --no-project --with mammoth --with pypdf report/make_pdf.py
"""
import subprocess
import tempfile
from pathlib import Path

import mammoth
import pypdf

HERE = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 1.9cm 2.0cm; }
body { font-family: Calibri, Carlito, 'Helvetica Neue', Arial, sans-serif; font-size: 11pt; line-height: 1.22; color: #111; }
h1 { font-size: 14pt; color: #1F3A5F; margin: 12pt 0 3pt; break-after: avoid; }
h2 { font-size: 12pt; color: #1F3A5F; margin: 8pt 0 2pt; break-after: avoid; }
p { margin: 0 0 4pt; }
table { border-collapse: collapse; width: 100%; margin: 3pt 0 6pt; }
table * { font-size: 7.5pt !important; line-height: 1.15; }
td, th { border: 1px solid #888; padding: 1.5pt 3pt; vertical-align: top; }
td p { margin: 0; }
tr { break-inside: avoid; }
tr:first-child td, th { background: #DCE6F2; font-weight: bold; text-align: left; }
img { display: block; margin: 3pt auto 0; width: 88%; break-inside: avoid; }
.alu { font-size: 8pt; font-weight: bold; color: #555; margin-bottom: 8pt; }
body > p:nth-child(2) strong { font-size: 16pt; }
"""

with open(HERE / "RL_Summative_Report.docx", "rb") as f:
    body = mammoth.convert_to_html(f).value
ALU = ("<div class='alu'>AFRICAN LEADERSHIP UNIVERSITY<br>[BSE]<br>[ML TECHNIQUES II]<br>[SUMMATIVE ASSIGNMENT]</div>")
body = ALU + body
html = HERE / "_report.html"
html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>")
out = HERE / "RL_Summative_Report.pdf"
subprocess.run([CHROME, "--headless=new", f"--user-data-dir={tempfile.mkdtemp()}", "--no-pdf-header-footer",
                f"--print-to-pdf={out}", html.as_uri()], capture_output=True, timeout=120)
html.unlink()
print("pages:", len(pypdf.PdfReader(out).pages))
