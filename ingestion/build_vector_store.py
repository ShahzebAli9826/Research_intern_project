import json
import os
import shutil

from tqdm import tqdm

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


CHUNKS_FILE = (
    "data/processed/chunks.json"
)

VECTOR_DB_PATH = (
    "data/processed/chroma_db"
)


def load_chunks():

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def build_documents(chunks):

    documents = []

    for chunk in tqdm(
        chunks,
        desc="Preparing Documents"
    ):

        doc = Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id":
                    chunk["chunk_id"],

                "arxiv_id":
                    chunk["arxiv_id"],

                "title":
                    chunk.get(
                        "title",
                        ""
                    ),

                "authors":
                    ", ".join(
                        chunk.get(
                            "authors",
                            []
                        )
                    ),

                "chunk_index":
                    chunk.get(
                        "chunk_index",
                        0
                    )
            }
        )

        documents.append(doc)

    return documents


def create_vector_store(
    documents
):

    if os.path.exists(
        VECTOR_DB_PATH
    ):
        shutil.rmtree(
            VECTOR_DB_PATH
        )

    print(
        "Loading embedding model..."
    )

    embeddings = (
        HuggingFaceEmbeddings(
            model_name=
            "BAAI/bge-base-en-v1.5",

            encode_kwargs={
                "normalize_embeddings": True
            }
        )
    )

    print(
        "Creating ChromaDB..."
    )

    vector_store = (
        Chroma.from_documents(
            documents=documents,

            embedding=embeddings,

            persist_directory=
            VECTOR_DB_PATH
        )
    )

    print(
        "ChromaDB creation completed."
    )

    return vector_store


if __name__ == "__main__":

    print(
        "Loading chunks..."
    )

    chunks = load_chunks()

    print(
        f"Loaded {len(chunks)} chunks."
    )

    # Development mode
    # Change 5000 -> len(chunks)
    # later if needed

    chunks = chunks[:5000]

    print(
        f"Using {len(chunks)} chunks."
    )

    print(
        "Preparing documents..."
    )

    documents = build_documents(
        chunks
    )

    print(
        f"Prepared {len(documents)} documents."
    )

    print(
        "Building vector database..."
    )

    create_vector_store(
        documents
    )

    print(
        "\nVector DB saved to:"
    )

    print(
        VECTOR_DB_PATH
    )