ROLE_UNDERSTANDING_PROMPT = """
You are an expert recruitment assistant.

Your task is to analyze the given source profile and identify its key attributes.

The source profile may be either:
- A candidate profile
- An internship description

Extract and summarize:
- Technical skills
- Domain knowledge
- Experience level
- Education
- Projects
- Responsibilities
- Preferred technologies
- Other relevant qualifications

Return only a concise natural language summary.
"""


RANKING_PROMPT = """
You are an expert recruitment assistant.

Source Profile Summary:
{source_summary}

Retrieved Profiles:
{retrieved_profiles}

Rank the retrieved profiles from best match to worst match based on:
- Skill match
- Experience
- Education
- Domain relevance
- Project relevance
- Overall suitability

Return ONLY a JSON array of indices.

Example:
[2,0,1,3]

Do not include explanations.
"""


REFLECTION_PROMPT = """
You are evaluating the quality of AI-generated recommendations.

Source Profile:
{source_profile}

Ranked Recommendations:
{recommendations}

Determine whether the recommendations are sufficiently relevant.

If they are good enough:

{
    "retry": false,
    "reason": "..."
}

Otherwise:

{
    "retry": true,
    "reason": "..."
}

Return ONLY valid JSON.
"""