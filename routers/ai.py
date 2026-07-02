"""
routers/ai.py
كل الـ Endpoints بتاعة الذكاء الاصطناعي (تصنيف + قصة + صوت + 3D اختياري).
"""

import os
import uuid
import base64

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from config import UPLOADS_DIR, ENABLE_3D
from services import inference

router = APIRouter(prefix="/ai", tags=["AI"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/detect")
async def detect(
    image: UploadFile = File(..., description="صورة القطعة الأثرية"),
    language: str = Form("ar", description="ar أو en"),
    with_3d: bool = Form(False, description="توليد نموذج 3D (لو مفعّل على السيرفر)"),
):
    """
    بيرفع صورة القطعة الأثرية، يصنّفها، يولّد قصة بصوتها،
    واختياريًا نموذج 3D لو الميزة مفعّلة على السيرفر (ENABLE_3D=true).
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="لازم تبعتي ملف صورة صحيح (image/*)")

    request_id = uuid.uuid4().hex[:12]
    temp_path = str(UPLOADS_DIR / f"{request_id}_{image.filename}")

    with open(temp_path, "wb") as f:
        f.write(await image.read())

    try:
        (
    predicted_class,
    name,
    gender,
    confidence,
    prompt_3d,
    matched,
) = inference.detect_artifact(temp_path)

        if not matched:
            msg = (
                "معلش، الصورة دي ملهاش تطابق كافي مع أي تمثال من الـ30 اللي المودل متدرب عليهم "
                f"(أعلى نسبة تطابق كانت {confidence * 100:.1f}%). جربي صورة أوضح."
                if language == "ar"
                else
                f"Sorry, this doesn't match any of the 30 trained statues closely enough "
                f"(best match was only {confidence * 100:.1f}%). Try a clearer photo."
            )
            return JSONResponse(
                status_code=200,
                content={
                    "matched": False,
                    "confidence": round(confidence, 4),
                    "message": msg,
                },
            )

        story = inference.generate_story(name, gender, language)
        audio_path = await inference.text_to_speech(story, gender, language, request_id)

        response = {
            "matched": True,
            "class": predicted_class,
            "name": name,
            "gender": gender,
            "confidence": round(confidence, 4),
            "language": language,
            "story": story,
        }

        if os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                response["audio_base64"] = base64.b64encode(f.read()).decode("utf-8")

        if with_3d:
            if not ENABLE_3D:
                response["glb_base64"] = None
                response["3d_note"] = "توليد الـ 3D متوقف على هذا السيرفر (ENABLE_3D=false)."
            else:
                from services import threed

                glb_path = threed.generate_3d(prompt_3d, request_id)
                response["glb_base64"] = threed.glb_to_base64(glb_path)

        return response

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حصل خطأ: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)