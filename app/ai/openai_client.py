from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import settings

llm = ChatOpenAI(
    model=settings.GPT_MODEL,
    temperature=0,
    api_key=settings.OPENAI_API_KEY
)

embeddings = OpenAIEmbeddings(
    model=settings.EMBEDDING_MODEL,
    api_key=settings.OPENAI_API_KEY
)