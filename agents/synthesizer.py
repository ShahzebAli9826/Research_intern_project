from langchain_core.prompts import ChatPromptTemplate

from config.llm import llm
from graph.state import AgentState


class Synthesizer:

    def __init__(self):

        self.prompt = ChatPromptTemplate.from_template(
            """
You are an expert research assistant.

Answer the user's question using ONLY the provided sources.

Rules:

1. Do NOT use outside knowledge.
2. Every factual claim must be supported by evidence.
3. Cite sources inline using [arXiv:<id>].
4. Combine information from multiple papers when appropriate.
5. If papers disagree, explain the disagreement.
6. If evidence is insufficient, explicitly say so.
7. Mention paper titles when relevant.
8. Produce a well-structured answer.
9. Prioritize consensus across multiple papers.
10. Do not overemphasize claims from a single paper.
11. Distinguish between describing a method and criticizing a method.
12. Prefer foundational definitions when available.

Question:
{query}

Sources:
{context}

Answer:
"""
        )

        self.chain = self.prompt | llm

    def generate_answer(
        self,
        query: str,
        context: str
    ):

        response = self.chain.invoke(
            {
                "query": query,
                "context": context
            }
        )

        return response.content


synthesizer = Synthesizer()


def build_context(retrieved_docs):

    context_parts = []

    evidence = []

    for idx, item in enumerate(
        retrieved_docs,
        start=1
    ):

        data = item["result"]["data"]

        metadata = data["metadata"]

        title = metadata.get(
            "title",
            "Unknown"
        )

        arxiv_id = metadata.get(
            "arxiv_id",
            "Unknown"
        )

        chunk_id = metadata.get(
            "chunk_id",
            ""
        )

        text = data.get(
            "text",
            ""
        )

        context_parts.append(
            f"""
[Source {idx}]

Title: {title}

Arxiv ID: {arxiv_id}

Chunk ID: {chunk_id}

Content:
{text}
"""
        )

        evidence.append(
            {
                "source_id": idx,
                "source_label": f"Source {idx}",
                "title": title,
                "arxiv_id": arxiv_id,
                "chunk_id": chunk_id,
                "text": text
            }
        )

    return (
        "\n\n".join(context_parts),
        evidence
    )


def synthesizer_node(
    state: AgentState
):

    print(
        "\n[Synthesizer] Generating answer..."
    )

    # Safety check

    if not state.get(
        "retrieved_docs"
    ):

        print(
            "[Synthesizer] No documents retrieved."
        )

        state["answer"] = (
            "No relevant evidence was retrieved."
        )

        state["citations"] = []

        state["evidence"] = []

        return state

    context, evidence = build_context(
        state["retrieved_docs"]
    )

    answer = synthesizer.generate_answer(
        query=state["query"],
        context=context
    )

    state["answer"] = answer

    state["evidence"] = evidence

    state["citations"] = [
        {
            "title": item["title"],
            "arxiv_id": item["arxiv_id"],
            "chunk_id": item["chunk_id"]
        }
        for item in evidence
    ]

    print(
        "[Synthesizer] Answer generated."
    )

    return state


if __name__ == "__main__":

    from agents.retrieval_agent import (
        retrieval_node
    )

    state = {

        "query":
            "What is Reflexion?",

        "sub_questions": [
            "What is Reflexion?"
        ],

        "retrieved_docs": [],

        "answer": "",

        "citations": [],

        "evidence": [],

        "reflection": "",

        "rewritten_query": "",

        "iteration_count": 0,

        "is_sufficient": False,

        "retrieval_scores": [],

        "rerank_scores": []
    }

    state = retrieval_node(
        state
    )

    state = synthesizer_node(
        state
    )

    print("\n")

    print("=" * 100)

    print("\nANSWER:\n")

    print(
        state["answer"]
    )

    print("\n")

    print("=" * 100)

    print(
        f"\nEvidence Chunks: "
        f"{len(state['evidence'])}"
    )