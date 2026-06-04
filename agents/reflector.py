from typing import List

from langchain_core.prompts import (
    ChatPromptTemplate
)

from config.llm import llm
from graph.state import AgentState


class Reflector:

    def __init__(self):

        self.prompt = (
            ChatPromptTemplate.from_template(
                """
You are an expert research evaluator.

Your task is to determine whether the answer is
sufficient to answer the user's query.

You must evaluate:

1. Completeness
   - Are all sub-questions answered?

2. Coverage
   - Are important concepts missing?

3. Evidence Quality
   - Does the answer appear supported?

4. Research Quality
   - Would a researcher be satisfied?

User Query:
{query}

Sub Questions:
{sub_questions}

Answer:
{answer}

Citation Verification:
Support Ratio = {support_ratio}

Return your response in EXACTLY this format:

SUFFICIENT: YES or NO

REASON:
<short explanation>

MISSING_TOPICS:
- topic 1
- topic 2

Do not write anything else.
"""
            )
        )

        self.chain = (
            self.prompt
            | llm
        )

    def reflect(
        self,
        query: str,
        answer: str,
        sub_questions: List[str],
        support_ratio: float
    ):

        response = self.chain.invoke(
            {
                "query": query,
                "answer": answer,
                "sub_questions":
                    "\n".join(
                        sub_questions
                    ),
                "support_ratio":
                    support_ratio
            }
        )

        return response.content


reflector = Reflector()


def parse_reflection(
    reflection_text: str
):

    result = {

        "is_sufficient": False,

        "reason": "",

        "missing_topics": []
    }

    lines = [
        line.strip()
        for line in reflection_text.split(
            "\n"
        )
        if line.strip()
    ]

    current_section = None

    for line in lines:

        upper = line.upper()

        if upper.startswith(
            "SUFFICIENT:"
        ):

            value = (
                line.split(
                    ":",
                    1
                )[1]
                .strip()
                .upper()
            )

            result[
                "is_sufficient"
            ] = (
                value == "YES"
            )

        elif upper.startswith(
            "REASON:"
        ):

            current_section = (
                "reason"
            )

            result[
                "reason"
            ] = (
                line.split(
                    ":",
                    1
                )[1]
                .strip()
            )

        elif upper.startswith(
            "MISSING_TOPICS:"
        ):

            current_section = (
                "topics"
            )

        elif (
            current_section
            == "reason"
        ):

            result[
                "reason"
            ] += (
                " " + line
            )

        elif (
            current_section
            == "topics"
        ):

            if line.startswith("-"):

                result[
                    "missing_topics"
                ].append(
                    line.replace(
                        "-",
                        ""
                    ).strip()
                )

    return result


def reflector_node(
    state: AgentState
):

    print(
        "\n[Reflector] "
        "Evaluating answer..."
    )

    verification = (
        state.get(
            "verification_report",
            {}
        )
    )

    support_ratio = (
        verification.get(
            "support_ratio",
            0.0
        )
    )

    # Hard failure condition

    if support_ratio < 0.50:

        state["reflection"] = {

            "is_sufficient":
                False,

            "reason":
                "Citation support ratio is too low.",

            "missing_topics":
                []
        }

        state[
            "is_sufficient"
        ] = False

        print(
            "[Reflector] "
            "Failed citation threshold."
        )

        return state

    reflection_text = (
        reflector.reflect(
            query=state["query"],
            answer=state["answer"],
            sub_questions=
                state.get(
                    "sub_questions",
                    [state["query"]]
                ),
            support_ratio=
                support_ratio
        )
    )

    reflection = (
        parse_reflection(
            reflection_text
        )
    )

    # Require both:
    #
    # 1. LLM says sufficient
    # 2. Citation quality acceptable

    is_sufficient = (

        reflection[
            "is_sufficient"
        ]

        and

        support_ratio >= 0.80
    )

    reflection[
        "support_ratio"
    ] = support_ratio

    state[
        "reflection"
    ] = reflection

    state[
        "is_sufficient"
    ] = is_sufficient

    print(
        f"[Reflector] "
        f"Sufficient: "
        f"{is_sufficient}"
    )

    print(
        f"[Reflector] "
        f"Reason: "
        f"{reflection['reason']}"
    )

    if (
        reflection[
            "missing_topics"
        ]
    ):

        print(
            "[Reflector] "
            "Missing Topics:"
        )

        for topic in (
            reflection[
                "missing_topics"
            ]
        ):

            print(
                f"  - {topic}"
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
                "differences?"
            )
        ],

        "answer":
            """
Reflexion is a framework that
uses verbal reinforcement learning.

It improves agent performance
through self-reflection.
""",

        "verification_report": {

            "verified": True,

            "support_ratio": 0.92
        },

        "reflection": "",

        "is_sufficient": False
    }

    updated_state = (
        reflector_node(
            mock_state
        )
    )

    print("\n")

    print("=" * 80)

    print(
        "REFLECTION:"
    )

    print(
        updated_state[
            "reflection"
        ]
    )

    print("\n")

    print(
        "IS SUFFICIENT:"
    )

    print(
        updated_state[
            "is_sufficient"
        ]
    )