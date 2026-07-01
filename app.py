"""
app.py
نقطة تشغيل السيرفر الرئيسية (FastAPI).
شغليها بـ: uvicorn app:app --host 0.0.0.0 --port 7860
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ALLOWED_ORIGINS
from services.model_loader import model_loader
from routers import ai

app = FastAPI(
    title="Egyptian Museum — Artifact Story Teller API",
    description="ترفع صورة تمثال من المتحف المصري، والـ API يرجّعلك اسمه وقصته وصوته (واختياري نموذج 3D).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    model_loader.load()


@app.get("/")
def root():
    return {
        "message": "Egyptian Museum API شغّال ✅",
        "docs": "/docs",
        "health": "/ai/health",
        "detect": "/ai/detect (POST, multipart/form-data: image, language, with_3d)",
    }


app.include_router(ai.router)
