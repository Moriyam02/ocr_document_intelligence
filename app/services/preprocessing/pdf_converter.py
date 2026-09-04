import fitz  # PyMuPDF
import os

def convert_pdf_to_images(pdf_path: str) -> list[str]:
    print(f"--- Starting PDF conversion for: {pdf_path} ---")
    dir_name, file_name = os.path.split(pdf_path)
    base_name = os.path.splitext(file_name)[0]

    doc = fitz.open(pdf_path)
    print(f"--- Total pages found in PDF: {len(doc)} ---")
    extracted_image_paths = []

    for idx, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        page_filename = f"{base_name}_page_{idx + 1}.png"
        page_path = os.path.join(dir_name, page_filename)
        
        pix.save(page_path)
        print(f"--- Saved page image to: {page_path} ---")
        extracted_image_paths.append(page_path)

    doc.close()
    return extracted_image_paths