from sentence_transformers import CrossEncoder

from retrieval.hybrid_retriever import (
    HybridRetriever
)


class Reranker:

    def __init__(self):

        print(
            "Loading Cross Encoder..."
        )

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length = 512
        )

    def rerank(
        self,
        query,
        retrieved_results,
        top_k=5
    ):

        pairs = []

        for result in retrieved_results:

            text = result["data"].get(
                "text",
                ""
            )

            pairs.append(
                (
                    query,
                    text
                )
            )

        scores = self.model.predict(
            pairs,
            show_progress_bar = True
        )

        reranked = []

        for result, score in zip(
            retrieved_results,
            scores
        ):

            reranked.append(
                {
                    "rerank_score":
                        float(score),

                    "result":
                        result
                }
            )

        reranked.sort(
            key=lambda x:
            x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]


if __name__ == "__main__":

    hybrid = HybridRetriever()

    reranker = Reranker()

    query = (
        "What is Reflexion?"
    )

    print(
        "\nRetrieving documents..."
    )

    retrieved = hybrid.retrieve(
        query=query,
        final_k=20
    )

    print(
        f"Retrieved {len(retrieved)} chunks."
    )

    print(
        "\nReranking..."
    )

    results = reranker.rerank(
        query=query,
        retrieved_results=retrieved,
        top_k=5
    )

    print(
        f"\nTop {len(results)} Results"
    )

    for idx, item in enumerate(
        results,
        start=1
    ):

        print(
            "\n" + "=" * 80
        )

        print(
            f"Rank {idx}"
        )

        print(
            f"Cross Encoder Score: "
            f"{item['rerank_score']:.4f}"
        )

        data = item["result"]["data"]

        print(
            f"Title: "
            f"{data['metadata'].get('title', 'No Title')}"
        )

        print(
            f"Arxiv ID: "
            f"{data['metadata'].get('arxiv_id', '')}"
        )

        print(
            data["text"][:500]
        )