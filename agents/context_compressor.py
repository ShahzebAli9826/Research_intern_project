from collections import defaultdict

from graph.state import AgentState


MAX_DOCS = 15

MAX_CHUNKS_PER_PAPER = 4


def get_chunk_id(doc):

    try:

        return (
            doc["result"]["data"]
            ["metadata"]
            ["chunk_id"]
        )

    except Exception:

        return None


def get_paper_id(doc):

    try:

        metadata = (
            doc["result"]["data"]
            ["metadata"]
        )

        return metadata.get(
            "arxiv_id",
            "unknown"
        )

    except Exception:

        return "unknown"


def get_rerank_score(doc):

    return doc.get(
        "rerank_score",
        0.0
    )


def context_compressor_node(
    state: AgentState
):

    print(
        "\n[Context Compressor] "
        "Starting compression..."
    )

    docs = state.get(
        "retrieved_docs",
        []
    )

    if not docs:

        print(
            "[Context Compressor] "
            "No documents found."
        )

        return state

    original_count = len(docs)

    # =====================================================
    # Sort by reranker score
    # =====================================================

    docs = sorted(
        docs,
        key=get_rerank_score,
        reverse=True
    )

    # =====================================================
    # Remove duplicate chunks
    # =====================================================

    unique_docs = []

    seen_chunks = set()

    for doc in docs:

        chunk_id = get_chunk_id(
            doc
        )

        if chunk_id is None:
            unique_docs.append(doc)
            continue

        if chunk_id in seen_chunks:
            continue

        seen_chunks.add(
            chunk_id
        )

        unique_docs.append(
            doc
        )

    # =====================================================
    # Paper Diversity
    # Avoid 12 chunks from same paper
    # =====================================================

    paper_counts = defaultdict(
        int
    )

    compressed_docs = []

    for doc in unique_docs:

        paper_id = get_paper_id(
            doc
        )

        if (
            paper_counts[
                paper_id
            ]
            >= MAX_CHUNKS_PER_PAPER
        ):

            continue

        compressed_docs.append(
            doc
        )

        paper_counts[
            paper_id
        ] += 1

        if (
            len(compressed_docs)
            >= MAX_DOCS
        ):

            break

    # =====================================================
    # Compression Stats
    # =====================================================

    compressed_count = len(
        compressed_docs
    )

    compression_ratio = (

        compressed_count
        / original_count

        if original_count > 0
        else 0
    )

    avg_rerank_score = (

        sum(
            get_rerank_score(doc)

            for doc in compressed_docs
        )

        / compressed_count

        if compressed_count > 0
        else 0
    )

    unique_papers = len(

        {
            get_paper_id(doc)

            for doc in compressed_docs
        }
    )

    state[
        "retrieved_docs"
    ] = compressed_docs

    state[
        "compression_stats"
    ] = {

        "before":
            original_count,

        "after":
            compressed_count,

        "compression_ratio":
            compression_ratio,

        "unique_papers":
            unique_papers,

        "avg_rerank_score":
            avg_rerank_score
    }

    print(
        f"[Context Compressor] "
        f"{original_count} → "
        f"{compressed_count}"
    )

    print(
        f"[Context Compressor] "
        f"Unique Papers: "
        f"{unique_papers}"
    )

    print(
        f"[Context Compressor] "
        f"Compression Ratio: "
        f"{compression_ratio:.2f}"
    )

    return state


if __name__ == "__main__":

    mock_state = {

        "retrieved_docs": []
    }

    updated_state = (
        context_compressor_node(
            mock_state
        )
    )

    print(
        updated_state.get(
            "compression_stats",
            {}
        )
    )