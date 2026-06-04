from collections import defaultdict
from retrieval.vector_retriever import VectorRetriever
from retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:

    def __init__(self):

        print(
            "Initializing Vector Retriever..."
        )

        self.vector_retriever = (
            VectorRetriever()
        )

        print(
            "Initializing BM25 Retriever..."
        )

        self.bm25_retriever = (
            BM25Retriever()
        )

    def reciprocal_rank_fusion(
        self,
        vector_results,
        bm25_results,
        k=60
    ):

        fused_scores = defaultdict(
            float
        )

        chunk_lookup = {}

        

        for rank, (
            doc,
            score
        ) in enumerate(
            vector_results,
            start=1
        ):

            chunk_id = (
                doc.metadata[
                    "chunk_id"
                ]
            )

            fused_scores[
                chunk_id
            ] += 1.5 / (
                k + rank
            )

            chunk_lookup[
                chunk_id
            ] = {
                "text":
                    doc.page_content,

                "metadata":
                    doc.metadata
            }

       

        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            chunk = result["chunk"]

            chunk_id = (
                chunk["chunk_id"]
            )

            fused_scores[
                chunk_id
            ] += 1 / (
                k + rank
            )

            chunk_lookup[
                chunk_id
            ] = {
                "text":
                    chunk["text"],

                "metadata": {
                    "chunk_id":
                        chunk["chunk_id"],

                    "arxiv_id":
                        chunk["arxiv_id"],

                    "title":
                        chunk["title"],

                    "authors":
                        ", ".join(
                            chunk["authors"]
                        ),

                    "chunk_index":
                        chunk["chunk_index"]
                }
            }

        ranked = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        final_results = []
        seen_titles = set()

        for chunk_id, score in ranked:

            data = chunk_lookup[chunk_id]

            title = data["metadata"].get(
                "title",
                ""
            )

            if title in seen_titles:
                continue

            seen_titles.add(title)

            final_results.append(
                {
                    "chunk_id": chunk_id,
                    "rrf_score": score,
                    "data": data
                }
            )
        return final_results

    def retrieve(
        self,
        query,
        vector_k=100,
        bm25_k=100,
        final_k=30
    ):

        vector_results = (
            self.vector_retriever
            .retrieve_with_scores(
                query=query,
                k=vector_k
            )
        )

        bm25_results = (
            self.bm25_retriever
            .retrieve(
                query=query,
                k=bm25_k
            )
        )

        fused_results = (
            self.reciprocal_rank_fusion(
                vector_results,
                bm25_results
            )
        )

        return fused_results[
            :final_k
        ]


if __name__ == "__main__":

    retriever = (
        HybridRetriever()
    )

    query = (
        "What is Reflexion?"
    )

    results = retriever.retrieve(
        query=query,
        final_k=5
    )

    print(
        f"\nRetrieved "
        f"{len(results)} results."
    )

    for idx, result in enumerate(
        results
    ):

        print(
            "\n" + "=" * 80
        )

        print(
            f"Result {idx + 1}"
        )

        print(
            f"RRF Score: "
            f"{result['rrf_score']:.6f}"
        )

        data = result["data"]

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