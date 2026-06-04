from typing import TypedDict
from typing import List
from typing import Dict
from typing import Any


class AgentState(TypedDict):

    query: str

    sub_questions: List[str]

    rewritten_query: str

    query_history: List[str]

    retrieved_docs: List[Dict[str, Any]]

    answer: str

    citations: List[Dict[str, Any]]

    evidence: list

    reflection: str

    rewritten_query: str

    iteration_count: int

    is_sufficient: bool

    retrieval_scores: List[float]

    rerank_scores: List[float]

    verification_report: dict

    compression_stats: dict