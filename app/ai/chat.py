from langchain_core.messages import HumanMessage

from app.ai.openai_client import llm


def invoke_llm(prompt: str) -> str:

    response = llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )

    return response.content