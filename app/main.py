from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="AI Recruitment Recommendation API",
    description="FastAPI application for internship and candidate recommendation using OpenAI and ChromaDB.",
    version="1.0.0"
)

app.include_router(router)


@app.get("/")
def health_check():
    return {
        "status": "success",
        "message": "AI Recruitment Recommendation API is running."
    }