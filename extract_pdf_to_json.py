import os
import fitz  # PyMuPDF
import pdfplumber
import json
import re

class PdfToJson:
    def extract_pdf_content(pdf_path, index, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)
        image_dir = os.path.join(output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        doc = fitz.open(pdf_path)
        json_data = {}

        with pdfplumber.open(pdf_path) as plumber_pdf:
            for page_num, (page_fitz, page_plumber) in enumerate(zip(doc, plumber_pdf.pages), start=1):
                page_key = f"page_{page_num}"
                json_data[page_key] = {
                    "text": "",
                    "images": [],
                    "tables": []
                }

                # Texto completo e linhas individuais
                text = page_plumber.extract_text()
                text_lines = text.split("\n") if text else []
                json_data[page_key]["text"] = text.strip() if text else ""

                # Tabelas com tentativa de capturar label anterior
                tables = page_plumber.extract_tables()
                for i, table in enumerate(tables):
                    label = None

                    # Tenta encontrar um label na linha anterior à tabela
                    if i < len(text_lines):
                        for j in range(len(text_lines)-1, -1, -1):
                            line = text_lines[j].strip()
                            if not line:
                                continue
                            if re.match(r'(?i)^tabela\s+\w+[:\.\-]?', line):
                                label = line
                                break

                    table_entry = {"data": table}
                    if label:
                        table_entry["label"] = label

                    json_data[page_key]["tables"].append(table_entry)

                # Imagens
                for img_index, img in enumerate(page_fitz.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]
                    filename = f"page{page_num}_img{img_index}.{ext}"
                    filepath = os.path.join(image_dir, filename)

                    with open(filepath, "wb") as f:
                        f.write(image_bytes)

                    json_data[page_key]["images"].append(filepath)

        # Salva JSON
        json_path = os.path.join(output_dir, f"extracted_content({index}).json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

        print(f"Extração concluída. Arquivo JSON salvo em: {json_path}")
        return json_path