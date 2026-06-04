import os
import json
import fitz

from tqdm import tqdm

os.makedirs(
    "data/processed",
    exist_ok=True
)

PDF_FOLDER = "data/papers"

OUTPUT_FILE = (
    "data/processed/parsed_papers.json"
)


def extract_text_from_pdf(pdf_path):

    pages = []

    try:

        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):

            page = doc[page_num]

            text = page.get_text()

            pages.append(
                {
                    "page": page_num + 1,
                    "text": text
                }
            )

        doc.close()

    except Exception as e:

        print(
            f"Error reading {pdf_path}: {e}"
        )

    return pages


def parse_all_pdfs():

    pdf_files = [
        file
        for file in os.listdir(PDF_FOLDER)
        if file.endswith(".pdf")
    ]

    parsed_papers = []

    for pdf_file in tqdm(pdf_files):

        pdf_path = os.path.join(
            PDF_FOLDER,
            pdf_file
        )

        arxiv_id = pdf_file.replace(
            ".pdf",
            ""
        )

        pages = extract_text_from_pdf(
            pdf_path
        )

        if len(pages) == 0:
            continue

        parsed_papers.append(
            {
                "arxiv_id": arxiv_id,
                "pdf_file": pdf_file,
                "total_pages": len(pages),
                "pages": pages
            }
        )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            parsed_papers,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nParsed {len(parsed_papers)} papers."
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    return parsed_papers


if __name__ == "__main__":
    parse_all_pdfs()