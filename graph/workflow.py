from langgraph.graph import (
    StateGraph,
    END
)

from graph.state import AgentState

from agents.planner import (
    planner_node
)

from agents.context_compressor import (
    context_compressor_node
)

from agents.retrieval_agent import (
    retrieval_node
)

from agents.synthesizer import (
    synthesizer_node
)

from agents.citation_verifier import (
    verifier_node
)

from agents.reflector import (
    reflector_node
)

from agents.query_rewriter import (
    query_rewriter_node
)


# =====================================================
# Configuration
# =====================================================

MAX_ITERATIONS = 3


# =====================================================
# Routing Logic
# =====================================================

def should_continue(
    state: AgentState
):

    # Reflection says answer is good

    if state.get(
        "is_sufficient",
        False
    ):

        print(
            "\n[Graph] Answer accepted."
        )

        return END

    # Prevent infinite loops

    if (
        state.get(
            "iteration_count",
            0
        )
        >= MAX_ITERATIONS
    ):

        print(
            "\n[Graph] "
            "Maximum iterations reached."
        )

        reflection = state.get(
            "reflection",
            {}
        )

        reason = reflection.get(
            "reason",
            ""
        )

        reflection[
            "reason"
        ] = (
            reason
            + " Maximum refinement limit reached."
        )

        state[
            "reflection"
        ] = reflection

        return END

    print(
        "\n[Graph] "
        "Answer insufficient."
    )

    print(
        "[Graph] "
        "Launching Query Rewriter..."
    )

    return "query_rewriter"


# =====================================================
# Build Graph
# =====================================================

builder = StateGraph(
    AgentState
)


# -----------------------------------------------------
# Nodes
# -----------------------------------------------------

builder.add_node(
    "planner",
    planner_node
)

builder.add_node(
    "retrieval",
    retrieval_node
)

builder.add_node(
    "synthesizer",
    synthesizer_node
)

builder.add_node(
    "citation_verifier",
    verifier_node
)

builder.add_node(
    "context_compressor",
    context_compressor_node
)

builder.add_node(
    "reflector",
    reflector_node
)

builder.add_node(
    "query_rewriter",
    query_rewriter_node
)


# -----------------------------------------------------
# Entry Point
# -----------------------------------------------------

builder.set_entry_point(
    "planner"
)


# -----------------------------------------------------
# Main Flow
# -----------------------------------------------------

builder.add_edge(
    "planner",
    "retrieval"
)

builder.add_edge(
    "retrieval",
    "context_compressor"
)

builder.add_edge(
    "context_compressor",
    "synthesizer"
)

builder.add_edge(
    "synthesizer",
    "citation_verifier"
)

builder.add_edge(
    "citation_verifier",
    "reflector"
)


# -----------------------------------------------------
# Reflection Loop
# -----------------------------------------------------

builder.add_conditional_edges(
    "reflector",
    should_continue
)

builder.add_edge(
    "query_rewriter",
    "retrieval"
)


# =====================================================
# Compile Graph
# =====================================================

research_graph = (
    builder.compile()
)


# =====================================================
# Local Test
# =====================================================

if __name__ == "__main__":

    print("\n" + "=" * 80)

    print(
        "Research Agentic RAG Graph "
        "Compiled Successfully"
    )

    print("=" * 80)

    print(
        f"MAX_ITERATIONS = "
        f"{MAX_ITERATIONS}"
    )

    print(
        "\nWorkflow:"
    )

    print(
        """
planner
↓
retrieval
↓
context_compressor
↓
synthesizer
  ↓
Citation Verifier
  ↓
Reflector
  ↓
Sufficient?
  ├── YES → END
  └── NO
         ↓
   Query Rewriter
         ↓
     Retrieval
"""
    )