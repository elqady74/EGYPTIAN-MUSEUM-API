"""
model_loader.py

تحميل الموديلات مرة واحدة فقط وقت تشغيل السيرفر.
- تحميل موديل تصنيف التماثيل من Hugging Face Hub
- تحميل classes.txt من Hugging Face Hub
- إعداد Gemini
"""

import os
import torch
import torch.nn as nn
from torchvision import models
import google.generativeai as genai
from huggingface_hub import hf_hub_download

from config import GEMINI_API_KEY, GEMINI_MODEL_NAME

# اسم الـ Model Repository على Hugging Face
HF_REPO_ID = "ahmedelqady88/egyptian-statue-classifier"


class ModelLoader:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = []
        self.detection_model = None
        self.gemini = None
        self._loaded = False

    def load(self):
        """يتم استدعاؤها مرة واحدة عند Startup"""

        if self._loaded:
            return

        print("⬇️ Downloading model from Hugging Face Hub...")

        hf_token = os.getenv("HF_TOKEN")

        # تحميل ملفات الموديل من Hugging Face
        classifier_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="statue_classifier.pth",
            token=hf_token,
        )

        classes_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="classes.txt",
            token=hf_token,
        )

        # قراءة أسماء الكلاسات
        with open(classes_path, "r", encoding="utf-8") as f:
            self.class_names = [
                line.strip()
                for line in f.readlines()
                if line.strip()
            ]

        print(f"✅ Loaded {len(self.class_names)} classes")

        # إنشاء موديل ResNet50
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.class_names))

        # تحميل الـ Weights
        state_dict = torch.load(
            classifier_path,
            map_location=self.device,
        )

        model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()

        self.detection_model = model

        # إعداد Gemini
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini = genai.GenerativeModel(GEMINI_MODEL_NAME)
            print("✅ Gemini initialized")
        else:
            print("⚠️ GEMINI_API_KEY not found")

        self._loaded = True

        print("=" * 50)
        print("✅ All models loaded successfully")
        print(f"Device : {self.device}")
        print(f"Classes: {len(self.class_names)}")
        print("=" * 50)


# Singleton
model_loader = ModelLoader()