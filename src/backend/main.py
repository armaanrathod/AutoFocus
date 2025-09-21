from fastapi import FastAPI
from .routes import router

app = FastAPI(title="AutoFocus API")

app.include_router(router)

@app.get("/")
def home():
    return {"message": "AutoFocus backend is running!"}
