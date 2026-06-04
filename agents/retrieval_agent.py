from retrieval.hybrid_retriever import (
    HybridRetriever
)

from retrieval.reranker import (
    Reranker
)

from graph.state import AgentState


class RetrievalAgent:

    def __init__(self):

        print(
            "Initializing Retrieval Agent..."
        )

        self.hybrid = HybridRetriever()

        self.reranker = Reranker()

    def retrieve(
        self,
        query: str
    ):

        hybrid_results = (
            self.hybrid.retrieve(
                query=query,
                vector_k=100,
                bm25_k=100,
                final_k=30
            )
        )

        reranked_results = (
            self.reranker.rerank(
                query=query,
                retrieved_results=hybrid_results,
                top_k=10
            )
        )

        return reranked_results


retrieval_agent = RetrievalAgent()


def retrieval_node(
    state: AgentState
):

    print(
        "\n[Retrieval Agent] Starting retrieval..."
    )

    # Clear previous retrieval results
    # whenever a new retrieval cycle starts

    state["retrieved_docs"] = []

    all_results = []

    rewritten_query = (
        state.get(
            "rewritten_query",
            ""
        )
    )

    # First retrieval pass:
    # use planner-generated sub-questions

    if not rewritten_query:

        questions = (
            state.get(
                "sub_questions",
                []
            )
        )

        if not questions:

            questions = [
                state["query"]
            ]

    # Reflection loop retrieval:
    # use rewritten query directly

    else:

        questions = [
            rewritten_query
        ]

        print(
            "\n[Retrieval Agent] "
            "Using rewritten query."
        )

    for question in questions:

        print(
            f"\nRetrieving for: "
            f"{question}"
        )

        results = (
            retrieval_agent.retrieve(
                question
            )
        )

        all_results.extend(
            results
        )

    # Deduplicate chunks

    unique_results = []

    seen_chunks = set()

    for item in all_results:

        try:

            chunk_id = (
                item["result"]
                ["chunk_id"]
            )

        except KeyError:

            chunk_id = (
                item["result"]
                ["data"]
                ["metadata"]
                .get(
                    "chunk_id",
                    ""
                )
            )

        if chunk_id in seen_chunks:

            continue

        seen_chunks.add(
            chunk_id
        )

        unique_results.append(
            item
        )

    state["retrieved_docs"] = (
        unique_results
    )

    state["retrieval_scores"] = [

        item["result"].get(
            "rrf_score",
            0.0
        )

        for item in unique_results
    ]

    state["rerank_scores"] = [

        item.get(
            "rerank_score",
            0.0
        )

        for item in unique_results
    ]

    print(
        f"\n[Retrieval Agent] "
        f"{len(unique_results)} "
        f"unique chunks retrieved."
    )

    return state


if __name__ == "__main__":

    state = {

        "query":
            "What is Reflexion?",

        "sub_questions": [

            "What is Reflexion?"
        ],

        "retrieved_docs": [],

        "answer": "",

        "citations": [],

        "reflection": {},

        "rewritten_query": "",

        "iteration_count": 0,

        "is_sufficient": False,

        "retrieval_scores": [],

        "rerank_scores": []
    }

    updated_state = (
        retrieval_node(
            state
        )
    )

    print(
        "\nRetrieved Docs:"
    )

    print(
        len(
            updated_state[
                "retrieved_docs"
            ]
        )
    )