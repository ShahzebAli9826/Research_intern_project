import json

from graph.workflow import (
    research_graph
)


QUESTION_ID = "q01"


def run():

    with open(
        "questions.jsonl",
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            item = json.loads(line)

            if item["id"] == QUESTION_ID:

                question = item["question"]

                break

        else:

            print(
                f"Question {QUESTION_ID} not found."
            )

            return

    print("\n" + "=" * 100)

    print(
        f"RUNNING {QUESTION_ID}"
    )

    print("=" * 100)

    print(
        "\nQUESTION:\n"
    )

    print(question)

    state = {

        "query": question,

        "sub_questions": [],

        "retrieved_docs": [],

        "answer": "",

        "reflection": "",

        "rewritten_query": "",

        "iteration_count": 0,

        "is_sufficient": False
    }

    result = (
        research_graph.invoke(
            state
        )
    )

    print("\n" + "=" * 100)

    print(
        "GRAPH OUTPUT KEYS"
    )

    print("=" * 100)

    print(result.keys())

    print("\n" + "=" * 100)

    print(
        "FINAL ANSWER"
    )

    print("=" * 100)

    print(
        result.get(
            "answer",
            ""
        )
    )

    print("\n" + "=" * 100)

    print(
        "REFLECTION"
    )

    print("=" * 100)

    print(
        result.get(
            "reflection",
            {}
        )
    )

    print("\n" + "=" * 100)

    print(
        "VERIFICATION REPORT"
    )

    print("=" * 100)

    print(
        result.get(
            "verification_report",
            {}
        )
    )

    print("\n" + "=" * 100)

    print(
        "CITATIONS"
    )

    print("=" * 100)

    print(
        result.get(
            "citations",
            []
        )
    )


if __name__ == "__main__":

    run()