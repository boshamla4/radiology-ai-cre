from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.models.database import init_db
from app.routers import upload, analysis, reports


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    Path("uploads").mkdir(exist_ok=True)
    Path("pdfs").mkdir(exist_ok=True)
    print("✓ Database initialised")
    print("✓ Radiology AI system ready")
    yield
    # Shutdown (nothing to clean up for now)


app = FastAPI(
    title="Radiology AI — Centre de Radiologie Emilie",
    description=(
        "Système d'aide au diagnostic radiologique par segmentation sémantique "
        "et génération automatique de comptes rendus."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(analysis.router)
app.include_router(reports.router)

# Serve overlay images and PDFs statically
if Path("uploads").exists():
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
if Path("pdfs").exists():
    app.mount("/pdfs", StaticFiles(directory="pdfs"), name="pdfs")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Radiology AI Backend"}
