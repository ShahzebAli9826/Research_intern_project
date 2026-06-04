from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DB_PATH = (
    "data/processed/chroma_db"
)


class VectorRetriever:

    def __init__(self):

        print("Loading embedding model...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

        print("Loading ChromaDB...")

        self.vector_store = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=self.embeddings
        )

    def retrieve(
        self,
        query: str,
        k: int = 10
    ):

        results = self.vector_store.similarity_search(
            query,
            k=k
        )

        return results

    def retrieve_with_scores(
        self,
        query: str,
        k: int = 30
    ):

        results = (
            self.vector_store
            .similarity_search_with_score(
                query,
                k=k
            )
        )

        return results


if __name__ == "__main__":

    retriever = VectorRetriever()

    query = "What is Reflexion?"

    results = retriever.retrieve_with_scores(
        query=query,
        k=30
    )

    print(
        f"\nRetrieved {len(results)} documents."
    )

    for idx, (doc, score) in enumerate(results):

        print("\n" + "=" * 80)

        print(
            f"Result {idx + 1}"
        )

        print(
            f"Score: {score:.4f}"
        )

        print(
            f"Title: {doc.metadata.get('title', 'Unknown')}"
        )

        print(
            f"Arxiv ID: {doc.metadata.get('arxiv_id', 'Unknown')}"
        )

        print(
            f"Chunk Index: {doc.metadata.get('chunk_index', 0)}"
        )

        print("\nPreview:\n")

        print(
            doc.page_content[:500]
        )