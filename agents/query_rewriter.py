from langchain_core.prompts import (
    ChatPromptTemplate
)

from config.llm import llm
from graph.state import AgentState


class QueryRewriter:

    def __init__(self):

        self.prompt = (
            ChatPromptTemplate.from_template(
                """
You are an expert research query optimizer.

The previous retrieval cycle did not fully answer
the user's question.

Your task is to generate a better retrieval query.

Original Query:
{query}

Sub Questions:
{sub_questions}

Reflection Reason:
{reason}

Missing Topics:
{missing_topics}

Rules:

1. Preserve the original intent.
2. Explicitly include missing topics.
3. Add important technical keywords.
4. Improve both semantic retrieval and BM25 retrieval.
5. Avoid vague wording.
6. Return ONLY the rewritten query.
7. Do not explain your reasoning.

Rewritten Query:
"""
            )
        )

        self.chain = (
            self.prompt
            | llm
        )

    def rewrite(
        self,
        query: str,
        sub_questions: str,
        reason: str,
        missing_topics: str
    ):

        response = self.chain.invoke(
            {
                "query": query,
                "sub_questions": sub_questions,
                "reason": reason,
                "missing_topics": missing_topics
            }
        )

        return (
            response.content
            .strip()
        )


rewriter = QueryRewriter()


def query_rewriter_node(
    state: AgentState
):

    print(
        "\n[Query Rewriter] "
        "Starting..."
    )

    # No rewrite needed

    if state.get(
        "is_sufficient",
        False
    ):

        print(
            "[Query Rewriter] "
            "Answer already sufficient."
        )

        return state

    reflection = state.get(
        "reflection",
        {}
    )

    missing_topics = (
        reflection.get(
            "missing_topics",
            []
        )
    )

    reason = (
        reflection.get(
            "reason",
            ""
        )
    )

    if not missing_topics:

        missing_topics = [
            "Additional supporting evidence"
        ]

    rewritten_query = (
        rewriter.rewrite(
            query=state["query"],

            sub_questions="\n".join(
                state.get(
                    "sub_questions",
                    [state["query"]]
                )
            ),

            reason=reason,

            missing_topics="\n".join(
                missing_topics
            )
        )
    )

    state[
        "rewritten_query"
    ] = rewritten_query

    state.setdefault(
        "query_history",
        []
    )

    if rewritten_query not in state["query_history"]:

        state["query_history"].append(
            rewritten_query
        )

    state[
        "iteration_count"
    ] = (
        state.get(
            "iteration_count",
            0
        )
        + 1
    )

    print(
        "[Query Rewriter] "
        "Rewritten Query:"
    )

    print(
        rewritten_query
    )

    print(
        f"[Query Rewriter] "
        f"Iteration "
        f"{state['iteration_count']}"
    )

    return state


if __name__ == "__main__":

    mock_state = {

        "query":
            (
                "How does Reflexion "
                "differ from ReAct?"
            ),

        "sub_questions": [

            "What is Reflexion?",

            "What is ReAct?",

            (
                "What are the "
                "differences between "
                "Reflexion and ReAct?"
            )
        ],

        "reflection": {

            "is_sufficient":
                False,

            "reason":
                (
                    "Answer did not "
                    "adequately explain "
                    "ReAct and the "
                    "comparison."
                ),

            "missing_topics": [

                "Definition of ReAct",

                (
                    "Differences between "
                    "Reflexion and ReAct"
                ),

                (
                    "Strengths and "
                    "limitations of "
                    "both approaches"
                )
            ]
        },

        "is_sufficient":
            False,

        "rewritten_query":
            "",

        "query_history":
            [],

        "iteration_count":
            0
    }

    updated_state = (
        query_rewriter_node(
            mock_state
        )
    )

    print("\n")

    print("=" * 80)

    print(
        "FINAL REWRITTEN QUERY"
    )

    print("=" * 80)

    print(
        updated_state[
            "rewritten_query"
        ]
    )

    print("\n")

    print(
        "QUERY HISTORY:"
    )

    print(
        updated_state[
            "query_history"
        ]
    )

    print("\n")

    print(
        "ITERATION COUNT:"
    )

    print(
        updated_state[
            "iteration_count"
        ]
    )