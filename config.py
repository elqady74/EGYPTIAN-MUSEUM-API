"""
Config.py
كل الإعدادات والمفاتيح بتتقرأ من متغيرات البيئة (Environment Variables / .env)
مفيش أي مفتاح سري مكتوب هنا في الكود عشان الأمان.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # يقرأ ملف .env لو موجود (للتشغيل المحلي فقط)

BASE_DIR = Path(__file__).resolve().parent

# ------------------- المسارات -------------------
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "temp"

CLASSIFIER_PATH = os.environ.get(
    "CLASSIFIER_PATH", str(CHECKPOINTS_DIR / "statue_classifier.pth")
)
CLASSES_PATH = os.environ.get(
    "CLASSES_PATH", str(CHECKPOINTS_DIR / "classes.txt")
)

# ------------------- مفاتيح الـ API -------------------
# لازم تضيفيها من Settings -> Repository secrets في الـ Hugging Face Space
# ومتكتبيهاش هنا نهائي.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# ------------------- إعدادات الموديل -------------------
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.85"))

# ------------------- توليد الـ 3D -------------------
# توليد الـ 3D بـ Shap-E تقيل جدًا على CPU وياخد وقت طويل.
# سيبيها False على الـ free tier وشغّليها True لو معاكي GPU.
ENABLE_3D = os.environ.get("ENABLE_3D", "false").lower() == "true"

# ------------------- CORS -------------------
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
