from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config.llm import llm
from graph.state import AgentState



planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a research planning agent.

Your task is to decompose a research question
into smaller sub-questions that can be answered
through document retrieval.

Return ONLY valid JSON.

Format:

{{
    "sub_questions": [
        "...",
        "...",
        "..."
    ]
}}
"""
        ),
        (
            "human",
            "{query}"
        )
    ]
)


parser = JsonOutputParser()


planner_chain = (
    planner_prompt
    | llm
    | parser
)


def planner_node(
    state: AgentState
):

    print(
        "\n[Planner] Creating sub-questions..."
    )

    result = planner_chain.invoke(
        {
            "query":
                state["query"]
        }
    )

    sub_questions = result.get(
        "sub_questions",
        []
    )

    print(
        f"[Planner] Generated "
        f"{len(sub_questions)} sub-questions."
    )

    state["sub_questions"] = (
        sub_questions
    )

    return state


if __name__ == "__main__":

    state = {

        "query":
            "What is Reflexion and how does it differ from ReAct?",

        "sub_questions": [],

        "retrieved_docs": [],

        "answer": "",

        "citations": [],

        "reflection": "",

        "rewritten_query": "",

        "iteration_count": 0,

        "is_sufficient": False,

        "retrieval_scores": [],

        "rerank_scores": []
    }

    updated_state = (
        planner_node(state)
    )

    print(
        "\nGenerated Sub Questions:\n"
    )

    for idx, question in enumerate(
        updated_state["sub_questions"],
        start=1
    ):

        print(
            f"{idx}. {question}"
        )