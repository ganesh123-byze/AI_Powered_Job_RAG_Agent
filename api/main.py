import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes import router

app = FastAPI(
    title="Job RAG Agent",
    description="Agentic RAG app for graduate job searching",
    version="1.0.0",
)

# Configure basic logging so module loggers emit to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_DIR = os.path.join(BASE_DIR, "ui")

app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(UI_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
