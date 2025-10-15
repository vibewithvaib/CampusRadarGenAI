from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    OPENAI_API_KEY: str

    GPT_MODEL: str = "gpt-4.1-mini"

    EMBEDDING_MODEL: str = "text-embedding-3-small"

    CHROMA_DB_PATH: str = "./chroma_db"

    CHROMA_COLLECTION_NAME: str = "recruitment_db"

    MAX_RECOMMENDATIONS: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()