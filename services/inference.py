"""
inference.py
منطق الـ Pipeline:
1) detect_artifact   -> تصنيف الصورة + threshold
2) generate_story     -> توليد القصة بالعربي/الإنجليزي عبر Gemini
3) text_to_speech     -> تحويل القصة لصوت عبر Edge TTS
"""

import asyncio
import torch
from PIL import Image
from torchvision import transforms

from config import CONFIDENCE_THRESHOLD, OUTPUTS_DIR
from services.model_loader import model_loader
from utils.statues_data import STATUES, STATUES_3D_PROMPT

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def detect_artifact(image_path: str, threshold: float = CONFIDENCE_THRESHOLD):
    """يرجع: predicted_class, name, gender, confidence, prompt_3d"""
    img = Image.open(image_path).convert("RGB")
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0).to(model_loader.device)

    with torch.no_grad():
        outputs = model_loader.detection_model(batch_t)
        prob = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(prob, 0)
        confidence = confidence.item()
        pred_idx = pred_idx.item()

    if confidence < threshold:
        return "unknown", "تمثال غير معروف", "male", confidence, "ancient Egyptian artifact statue"

    predicted_class = model_loader.class_names[pred_idx]
    info = STATUES.get(str(predicted_class), {"name": "قطعة أثرية", "gender": "male"})
    prompt_3d = STATUES_3D_PROMPT.get(str(predicted_class), "ancient Egyptian artifact statue")
    return predicted_class, info["name"], info["gender"], confidence, prompt_3d


def _story_prompt(artifact_name: str, gender: str, language: str) -> str:
    if language == "ar":
        if gender == "female":
            return (
                f"انتِ {artifact_name}، قطعة اثرية في المتحف المصري.\n"
                "احكي قصتك في جملتين او ثلاثة بضمير المتكلم المؤنث (انا).\n"
                "اذكري: اسمك، متى صُنعتِ، لماذا صُنعتِ، وجملة مشوقة عن سرك.\n"
                "بالعربي الفصيح، مختصر وجذاب.\n"
                "مهم جدا: اكتبي الرد بالكلمات فقط بدون اي رموز مثل النجمة او الشرطة المائلة او الهاش او اي علامات تنسيق."
            )
        return (
            f"انتَ {artifact_name}، قطعة اثرية في المتحف المصري.\n"
            "احكِ قصتك في جملتين او ثلاثة بضمير المتكلم المذكر (انا).\n"
            "اذكر: اسمك، متى صُنعت، لماذا صُنعت، وجملة مشوقة عن سرك.\n"
            "بالعربي الفصيح، مختصر وجذاب.\n"
            "مهم جدا: اكتب الرد بالكلمات فقط بدون اي رموز مثل النجمة او الشرطة المائلة او الهاش او اي علامات تنسيق."
        )

    if gender == "female":
        return (
            f"You are {artifact_name}, a female artifact in the Egyptian Museum.\n"
            "Tell your story in 2-3 sentences in first person (I am...).\n"
            "Include: your name, when and why you were made, and one captivating secret.\n"
            "Keep it short and elegant.\n"
            "IMPORTANT: Write only plain words. Do not use any symbols like asterisks, "
            "slashes, hashes, bullet points, or any formatting marks."
        )
    return (
        f"You are {artifact_name}, an artifact in the Egyptian Museum.\n"
        "Tell your story in 2-3 sentences in first person (I am...).\n"
        "Include: your name, when and why you were made, and one captivating secret.\n"
        "Keep it short and elegant.\n"
        "IMPORTANT: Write only plain words. Do not use any symbols like asterisks, "
        "slashes, hashes, bullet points, or any formatting marks."
    )


def generate_story(artifact_name: str, gender: str, language: str = "ar") -> str:
    if model_loader.gemini is None:
        raise RuntimeError("Gemini client غير مهيأ — تأكدي من GEMINI_API_KEY في الـ secrets.")
    prompt = _story_prompt(artifact_name, gender, language)
    response = model_loader.gemini.generate_content(prompt)
    return response.text


async def _tts_async(text: str, gender: str, language: str, out_path: str):
    import edge_tts

    if language == "ar":
        voice = "ar-EG-ShakirNeural" if gender == "male" else "ar-EG-SalmaNeural"
    else:
        voice = "en-US-GuyNeural" if gender == "male" else "en-US-JennyNeural"

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def text_to_speech(text: str, gender: str = "male", language: str = "ar", request_id: str = "output") -> str:
    out_path = str(OUTPUTS_DIR / f"{request_id}.mp3")
    asyncio.run(_tts_async(text, gender, language, out_path))
    return out_path
