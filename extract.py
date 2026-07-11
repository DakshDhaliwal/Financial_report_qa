import fitz  # this is pymupdf
import pdfplumber
import json

# ── STEP A: Extract all text ──────────────────────────
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():  # skip blank pages
            all_text.append({
                "page": page_num + 1,
                "content": text.strip()
            })

    print(f"✅ Extracted text from {len(all_text)} pages")
    return all_text


# ── STEP B: Extract all tables ────────────────────────
def extract_tables(pdf_path):
    all_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                if table:
                    all_tables.append({
                        "page": page_num + 1,
                        "table": table
                    })

    print(f"✅ Extracted {len(all_tables)} tables")
    return all_tables


# ── STEP C: Save everything ───────────────────────────
def save_output(text_data, table_data):
    with open("text_output.json", "w") as f:
        json.dump(text_data, f, indent=2)

    with open("tables_output.json", "w") as f:
        json.dump(table_data, f, indent=2)

    print("✅ Saved to text_output.json and tables_output.json")


# ── RUN IT ────────────────────────────────────────────
pdf_path = "report.pdf"

text_data = extract_text(pdf_path)
table_data = extract_tables(pdf_path)
save_output(text_data, table_data)