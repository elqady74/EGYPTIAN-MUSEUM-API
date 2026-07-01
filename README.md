---
title: Egyptian Museum API
emoji: 🏛️
colorFrom: yellow
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🏛️ Egyptian Museum — Artifact Story Teller API

FastAPI بتصنّف صورة تمثال من المتحف المصري (من ضمن 30 تمثال)، وترجع:
- اسم التمثال
- قصة عنه بالعربي أو الإنجليزي (Gemini)
- ملف صوت للقصة (Edge TTS)
- (اختياري) نموذج 3D بصيغة GLB (Shap-E)

## 📁 هيكل المشروع

```
EGYPTIAN-MUSEUM-API/
├── models/
│   └── checkpoints/        # statue_classifier.pth + classes.txt (ضيفيهم إنتِ)
├── outputs/                # ملفات مؤقتة ناتجة (صوت/3D)
├── routers/
│   └── ai.py                # Endpoints
├── services/
│   ├── model_loader.py       # تحميل الموديلات مرة واحدة
│   ├── inference.py          # التصنيف + القصة + الصوت
│   └── threed.py              # توليد 3D (اختياري)
├── temp/
├── uploads/
├── utils/
│   └── statues_data.py        # بيانات الـ 30 تمثال
├── app.py
├── config.py
├── Dockerfile
├── requirements.txt
└── .env.example
```

## ⚙️ قبل الرفع على Hugging Face Space

1. **ضيفي ملفات الموديل** (غير موجودة هنا لأنها بتاعتك):
   - `models/checkpoints/statue_classifier.pth`
   - `models/checkpoints/classes.txt`

2. **متكتبيش أي مفتاح سري في الكود.** روحي على صفحة الـ Space:
   `Settings → Variables and secrets → New secret` وضيفي:
   - `GEMINI_API_KEY` = مفتاح Gemini بتاعك
   - (اختياري) `CONFIDENCE_THRESHOLD` = `0.85`
   - (اختياري) `ENABLE_3D` = `false` أو `true` (لو معاكي GPU)

   > ⚠️ كان في مفتاح Gemini وتوكن ngrok مكتوبين صريح جوه النوتبوك الأصلي — لازم تلغيهم/تعملي لهم regenerate فورًا لأنهم بقوا مكشوفين.

3. اختياري: لو عايزة تجربي محليًا، انسخي `.env.example` لملف اسمه `.env` واملي فيه القيم.

## 🚀 التشغيل محليًا

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

بعد التشغيل: افتحي `http://localhost:7860/docs` لتجربة الـ API من واجهة Swagger.

## 📡 Endpoints

| Method | Path | الوصف |
|---|---|---|
| GET | `/` | رسالة ترحيب |
| GET | `/ai/health` | فحص إن السيرفر شغّال |
| POST | `/ai/detect` | ترفع صورة وتاخد القصة + الصوت (+ 3D اختياري) |

### مثال استخدام `/ai/detect`

```bash
curl -X POST "https://<space-url>/ai/detect" \
  -F "image=@statue.jpg" \
  -F "language=ar" \
  -F "with_3d=false"
```

الرد:
```json
{
  "matched": true,
  "class": "2",
  "name": "تمثال حجرى ضخم للملك رمسيس الثاني",
  "gender": "male",
  "confidence": 0.97,
  "language": "ar",
  "story": "...",
  "audio_base64": "..."
}
```

## 📦 رفعه على Hugging Face Spaces

1. اعملي Space جديد من نوع **Docker**.
2. ارفعي كل ملفات المجلد ده (بعد إضافة ملفات الموديل والـ secrets) — إما بالسحب والإفلات على الموقع، أو عبر `git push`.
3. الـ Space هيعمل build تلقائي ويشغّل السيرفر على البورت 7860.

## ⚠️ ملاحظة عن الـ 3D

توليد الـ 3D بـ Shap-E تقيل جدًا ومحتاج GPU. على الـ Hardware المجاني (CPU) هيكون بطيء جدًا أو ممكن يفشل من قلة الميموري.
سايباها متوقفة (`ENABLE_3D=false`) إلا لو رقّيتي الـ Space لـ Hardware فيه GPU.
