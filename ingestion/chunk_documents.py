import json

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

PARSED_FILE = (
    "data/processed/parsed_papers.json"
)

METADATA_FILE = (
    "data/metadata/papers_metadata.json"
)

OUTPUT_FILE = (
    "data/processed/chunks.json"
)


def load_parsed_papers():

    with open(
        PARSED_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def load_metadata():

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    return {
        paper["arxiv_id"]: paper
        for paper in metadata
    }


# Better chunking strategy
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


def create_chunks(
    parsed_papers,
    metadata
):

    chunks = []

    for paper in tqdm(
        parsed_papers,
        desc="Creating Chunks"
    ):

        arxiv_id = paper["arxiv_id"]

        paper_meta = metadata.get(
            arxiv_id,
            {}
        )

        title = paper_meta.get(
            "title",
            ""
        )

        authors = paper_meta.get(
            "authors",
            []
        )

        full_text = ""

        for page in paper["pages"]:

            page_number = page["page"]

            page_text = page["text"]

            if not page_text.strip():
                continue

            full_text += (
                f"\n[PAGE_{page_number}]\n"
                + page_text
                + "\n"
            )

        if not full_text.strip():
            continue

        split_texts = splitter.split_text(
            full_text
        )

        for idx, text in enumerate(
            split_texts
        ):

            chunk = {

                "chunk_id":
                    f"{arxiv_id}_chunk_{idx}",

                "arxiv_id":
                    arxiv_id,

                "title":
                    title,

                "authors":
                    authors,

                "chunk_index":
                    idx,

                "text":
                    text
            }

            chunks.append(chunk)

    return chunks


def save_chunks(chunks):

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )


if __name__ == "__main__":

    print("Loading parsed papers...")

    parsed_papers = load_parsed_papers()

    print("Loading metadata...")

    metadata = load_metadata()

    print("Creating chunks...")

    chunks = create_chunks(
        parsed_papers,
        metadata
    )

    save_chunks(chunks)

    print(
        f"\nCreated {len(chunks)} chunks."
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )