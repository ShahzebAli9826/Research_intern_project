import json
import os
import requests
from tqdm import tqdm

os.makedirs("data/papers", exist_ok= True)

def load_metadata():
    with open(
        "data/metadata/papers_metadata.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)

def download_pdf(pdf_url, save_path):
    try:
        response = requests.get(
            pdf_url,
            timeout = 30
        )
        if response.status_code == 200:
            with open(save_path, "wb") as pdf_file:
                pdf_file.write(
                    response.content
                )
            return True
        
    except Exception as e:
        print(f"Error downloading {save_path}:{e}")

    return False

def download_all_papers():
    papers = load_metadata()
    downloaded = 0
    for paper in tqdm(papers):
        arxiv_id = paper["arxiv_id"]
        save_path = os.path.join(
            "data/papers",
            f"{arxiv_id}.pdf"
        )
        if os.path.exists(save_path):
            continue

        pdf_url = paper["pdf_url"]
        success = download_pdf(
            pdf_url,
            save_path
        )
        if success:
            downloaded += 1
    print(f"\nDownloaded {downloaded} new papers.")


if __name__ == "__main__":
    download_all_papers()