import json
import re
from collections import Counter

QUESTIONS_FILE = "questions.jsonl"

paper_counter = Counter()

patterns = [

    # Quoted paper names
    r'"([^"]+)"',

    # Paper-like names
    r'\b(?:Mem0|SWE-agent|OpenHands|OSWorld|AppWorld|UI-TARS-2|UI-TARS|Tau)\b',

    # arXiv IDs
    r'\b\d{4}\.\d{5}\b'
]

with open(
    QUESTIONS_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        item = json.loads(line)

        question = item.get(
            "question",
            ""
        )

        for pattern in patterns:

            matches = re.findall(
                pattern,
                question,
                flags=re.IGNORECASE
            )

            for match in matches:

                paper_counter[
                    match.strip()
                ] += 1

print("\n" + "=" * 80)
print("POTENTIAL PAPER REFERENCES")
print("=" * 80)

for name, count in (
    paper_counter.most_common()
):

    print(
        f"{count:2d} | {name}"
    )

print("\nTotal Unique References:",
      len(paper_counter))