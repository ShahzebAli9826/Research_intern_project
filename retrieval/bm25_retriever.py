import json
import re

from rank_bm25 import BM25Okapi

CHUNKS_FILE = (
    "data/processed/chunks.json"
)


class BM25Retriever:

    @staticmethod
    def tokenize(text):

        return re.findall(
            r"\w+",
            text.lower()
        )

    def __init__(self):

        print("Loading chunks...")

        with open(
            CHUNKS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            self.chunks = json.load(file)

        print(
            f"Loaded {len(self.chunks)} chunks."
        )

        self.corpus = [
            chunk["text"]
            for chunk in self.chunks
        ]

        self.tokenized_corpus = [
            self.tokenize(doc)
            for doc in self.corpus
        ]

        print(
            "Building BM25 index..."
        )

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

    def retrieve(
        self,
        query,
        k=10
    ):

        tokenized_query = (
            self.tokenize(query)
        )

        scores = (
            self.bm25.get_scores(
                tokenized_query
            )
        )

        ranked_indices = (
            scores.argsort()[::-1][:100]
        )

        results = []

        for idx in ranked_indices:

            if scores[idx] <= 0:
                continue

            results.append(
                {
                    "score":
                        float(scores[idx]),

                    "chunk":
                        self.chunks[idx]
                }
            )

            if len(results) >= k:
                break

        return results


if __name__ == "__main__":

    retriever = BM25Retriever()

    results = retriever.retrieve(
        "What is Reflexion?",
        k=5
    )
    results2 = retriever.retrieve("Self-RAG", k=5)

    results3 = retriever.retrieve("multi agent planning", k=5)

    results4 = retriever.retrieve("agent memory", k=5)

    results5 = retriever.retrieve("web navigation agent", k=5)

    print(
        f"\nRetrieved {len(results)} chunks."
    )

    for idx, result in enumerate(results):

        print("\n" + "=" * 80)

        print(
            f"Result {idx + 1}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Title: "
            f"{result['chunk']['title']}"
        )

        print(
            result["chunk"]["text"][:500]
        )

    
    for idx, result in enumerate(results2):

        print("\n" + "=" * 80)

        print(
            f"Result {idx + 1}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Title: "
            f"{result['chunk']['title']}"
        )

        print(
            result["chunk"]["text"][:500]
        )
        
    for idx, result in enumerate(results3):

        print("\n" + "=" * 80)

        print(
            f"Result {idx + 1}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Title: "
            f"{result['chunk']['title']}"
        )

        print(
            result["chunk"]["text"][:500]
        )
    
    for idx, result in enumerate(results4):

        print("\n" + "=" * 80)

        print(
            f"Result {idx + 1}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Title: "
            f"{result['chunk']['title']}"
        )

        print(
            result["chunk"]["text"][:500]
        )
    
    for idx, result in enumerate(results5):

        print("\n" + "=" * 80)

        print(
            f"Result {idx + 1}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Title: "
            f"{result['chunk']['title']}"
        )

        print(
            result["chunk"]["text"][:500]
        )