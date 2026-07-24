import json

from app.ai.llm import llm
from app.ai.prompts import REFLECTION_PROMPT


def reflect_recommendations(
    source_profile: str,
    recommendations: list[str]
) -> dict:
    prompt = REFLECTION_PROMPT.format(
        source_profile=source_profile,
        recommendations="\n\n".join(recommendations)
    )

    response = llm.invoke(prompt)

    reflection = json.loads(response.content)

    return reflection