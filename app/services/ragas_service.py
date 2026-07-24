from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness
)


def evaluate_recommendations(
    source_profile: str,
    retrieved_profiles: list[str],
    recommendations: list[str]
):
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": source_profile,
                "retrieved_contexts": retrieved_profiles,
                "response": "\n\n".join(recommendations),
                "reference": "\n\n".join(retrieved_profiles)
            }
        ]
    )

    evaluation = evaluate(
        dataset=dataset,
        metrics=[
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall
        ]
    )

    return evaluation