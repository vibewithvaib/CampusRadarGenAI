from langgraph.graph import END, StateGraph

from app.graph.state import RecommendationState
from app.services.ranking_service import rank_profiles
from app.services.reflection_service import reflect_recommendations
from app.services.retrieval_service import retrieve_profiles


def retrieval_node(
    state: RecommendationState
):
    retrieved_profiles = retrieve_profiles(
        query=state["source_profile"],
        document_type=state["target_document_type"]
    )

    return {
        "retrieved_profiles": retrieved_profiles
    }


def ranking_node(
    state: RecommendationState
):
    ranked_profiles = rank_profiles(
        source_profile=state["source_profile"],
        retrieved_profiles=state["retrieved_profiles"]
    )

    return {
        "ranked_profiles": ranked_profiles,
        "recommendations": ranked_profiles
    }


def reflection_node(
    state: RecommendationState
):
    reflection = reflect_recommendations(
        source_profile=state["source_profile"],
        recommendations=state["ranked_profiles"]
    )

    return {
        "reflection": reflection,
        "retry": reflection.get("retry", False)
    }


def should_retry(
    state: RecommendationState
):
    if state["retry"]:
        return "retrieve"

    return END


builder = StateGraph(RecommendationState)

builder.add_node(
    "retrieve",
    retrieval_node
)

builder.add_node(
    "rank",
    ranking_node
)

builder.add_node(
    "reflect",
    reflection_node
)

builder.set_entry_point("retrieve")

builder.add_edge(
    "retrieve",
    "rank"
)

builder.add_edge(
    "rank",
    "reflect"
)

builder.add_conditional_edges(
    "reflect",
    should_retry,
    {
        "retrieve": "retrieve",
        END: END
    }
)

recommendation_graph = builder.compile()