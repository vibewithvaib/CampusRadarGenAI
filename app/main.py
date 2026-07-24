from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="CampusRadar AI Service",
    version="1.0.0"
)

app.include_router(
    router,
    prefix="/api",
    tags=["CampusRadar"]
)


@app.get("/")
def health_check():
    return {
        "status": "success",
        "message": "CampusRadar AI Service is running."
    }