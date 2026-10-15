
import pymupdf

pdf_path = "sample_research.pdf"

document = pymupdf.open(pdf_path)

print("Total pages:", len(document))
print()

for page_number, page in enumerate(document, start=1):
    text = page.get_text()

    print(f"--- Page {page_number} ---")
    print(text)

document.close()
