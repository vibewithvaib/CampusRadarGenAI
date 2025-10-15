import json
import re
from typing import List

from app.ai.chat import invoke_llm
from app.ai.prompts import RANKING_PROMPT
from app.core.config import settings


def rank_profiles(
    source_profile: str,
    target_profiles: List[str]
) -> List[int]:
    

    target_context = "\n---\n".join(target_profiles)

    prompt = RANKING_PROMPT.format(
        source_profile=source_profile,
        target_profiles=target_context,
        max_recommendations=settings.MAX_RECOMMENDATIONS
    )

    response = invoke_llm(prompt)

    print(f"LLM Raw Response: '{response}'")

    try:
        match = re.search(r"\[(.*?)\]", response)

        if not match:
            raise ValueError("No JSON array found in LLM response.")

        json_string = f"[{match.group(1)}]"

        recommended_ids = json.loads(json_string)

        if not isinstance(recommended_ids, list):
            raise ValueError("Parsed response is not a list.")

        return [int(profile_id) for profile_id in recommended_ids]

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing LLM response: {e}")
        return []