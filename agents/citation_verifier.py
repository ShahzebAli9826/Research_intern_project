import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from graph.state import AgentState


class CitationVerifier:

    def __init__(self):

        print(
            "\nInitializing Citation Verifier..."
        )

        self.model_name = (
            "cross-encoder/nli-deberta-v3-base"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_name
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.model_name
            )
        )

        self.labels = [
            "contradiction",
            "neutral",
            "entailment"
        ]

    def extract_citations(
        self,
        answer
    ):

        citation_pattern = (
            r"\[arXiv:([^\]]+)\]"
        )

        citations = re.findall(
            citation_pattern,
            answer
        )

        return list(
            set(citations)
        )

    def split_evidence(
        self,
        text,
        window_size=350,
        stride=250
    ):

        tokens = text.split()

        windows = []

        for i in range(
            0,
            len(tokens),
            stride
        ):

            chunk = tokens[
                i:i + window_size
            ]

            if chunk:

                windows.append(
                    " ".join(chunk)
                )

        return windows

    def verify_claim(
        self,
        claim,
        evidence
    ):

        inputs = self.tokenizer(
            claim,
            evidence,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        with torch.no_grad():

            logits = (
                self.model(
                    **inputs
                ).logits
            )

        probs = torch.softmax(
            logits,
            dim=1
        )[0]

        result = {
            label: probs[idx].item()
            for idx, label in enumerate(
                self.labels
            )
        }

        return result


verifier = CitationVerifier()


def verifier_node(
    state: AgentState
):

    print(
        "\n[Verifier] Verifying citations..."
    )

    answer = state["answer"]

    retrieved_docs = (
        state["retrieved_docs"]
    )

    verification_results = []

    claims = re.split(
        r"(?<=[.!?])\s+",
        answer
    )

    for claim in claims:

        claim = claim.strip()

        if (
            not claim
            or
            len(claim.split()) < 5
        ):
            continue

        best_result = None

        best_entailment = -1

        best_chunk = ""

        best_arxiv = ""

        for item in retrieved_docs:

            data = (
                item["result"]["data"]
            )

            metadata = (
                data["metadata"]
            )

            chunk = data["text"]

            arxiv_id = metadata.get(
                "arxiv_id",
                ""
            )

            windows = (
                verifier.split_evidence(
                    chunk
                )
            )

            for window in windows:

                current_result = (
                    verifier.verify_claim(
                        claim=claim,
                        evidence=window
                    )
                )

                if (
                    current_result[
                        "entailment"
                    ]
                    > best_entailment
                ):

                    best_entailment = (
                        current_result[
                            "entailment"
                        ]
                    )

                    best_result = (
                        current_result
                    )

                    best_chunk = (
                        window[:500]
                    )

                    best_arxiv = (
                        arxiv_id
                    )

        if best_result:

            verification_results.append(
                {
                    "claim":
                        claim,

                    "best_match":
                        best_chunk,

                    "arxiv_id":
                        best_arxiv,

                    "scores":
                        best_result
                }
            )

    final_verification = []

    for result in verification_results:

        scores = result["scores"]

        supported = (
            scores["entailment"] >= 0.50
            and
            scores["entailment"]
            >
            scores["contradiction"]
        )

        final_verification.append(
            {
                "claim":
                    result["claim"],

                "supported":
                    supported,

                "arxiv_id":
                    result["arxiv_id"],

                "scores":
                    scores,

                "evidence":
                    result["best_match"]
            }
        )

    state["verification_results"] = (
        final_verification
    )

    supported_count = sum(
        1
        for r in final_verification
        if r["supported"]
    )

    total = len(
        final_verification
    )

    print(
        f"[Verifier] "
        f"Supported Claims: "
        f"{supported_count}/{total}"
    )

    support_ratio = (
        supported_count / total
        if total > 0
        else 0.0
    )

    state["verification_report"] = {
        "verified": support_ratio >= 0.80,
        "support_ratio": support_ratio,
        "supported_claims": supported_count,
        "unsupported_claims": total - supported_count,
        "total_claims": total,
        "details": final_verification
    }

    state["is_sufficient"] = (
        support_ratio >= 0.80
    )


    return state

