import json

from app.ai.llm import llm
from app.ai.prompts import RANKING_PROMPT


def rank_profiles(
    source_profile: str,
    retrieved_profiles: list[str]
) -> list[str]:
    if not retrieved_profiles:
        return []

    prompt = RANKING_PROMPT.format(
        source_summary=source_profile,
        retrieved_profiles="\n\n".join(retrieved_profiles)
    )

    response = llm.invoke(prompt)

    ranked_indices = json.loads(response.content)

    ranked_profiles = [
        retrieved_profiles[index]
        for index in ranked_indices
        if 0 <= index < len(retrieved_profiles)
    ]

    return ranked_profiles