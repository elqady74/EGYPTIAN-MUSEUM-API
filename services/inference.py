"""
inference.py
AI Pipeline
1. Detect Artifact
2. Generate Story using Gemini
3. Generate Text-To-Speech
4. Generate 3D Model
"""

import asyncio

import torch
from PIL import Image
from torchvision import transforms

from config import (
    CONFIDENCE_THRESHOLD,
    OUTPUTS_DIR,
)

from services.model_loader import model_loader
from utils.statues_data import (
    STATUES,
    STATUES_3D_PROMPT,
)

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


def detect_artifact(
    image_path: str,
    threshold: float = CONFIDENCE_THRESHOLD,
):
    """
    Detect Artifact
    Returns
    -------
    predicted_class
    artifact_name
    gender
    confidence
    prompt_3d
    matched
    """

    image = Image.open(image_path).convert("RGB")

    tensor = preprocess(image)

    batch = tensor.unsqueeze(0).to(
        model_loader.device
    )

    with torch.no_grad():

        outputs = model_loader.detection_model(batch)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )[0]

        confidence, prediction = torch.max(
            probabilities,
            dim=0,
        )

    confidence = float(confidence.item())

    prediction = int(prediction.item())

    predicted_class = model_loader.class_names[
        prediction
    ]

    statue = STATUES.get(
        str(predicted_class),
        {
            "name": predicted_class,
            "gender": "male",
        },
    )

    prompt_3d = STATUES_3D_PROMPT.get(
        str(predicted_class),
        "ancient Egyptian artifact statue",
    )

    matched = confidence >= threshold

    return (
        predicted_class,
        statue["name"],
        statue["gender"],
        confidence,
        prompt_3d,
        matched,
    )


def _story_prompt(
    artifact_name: str,
    gender: str,
    language: str,
):
    """
    Gemini Prompt
    """

    if language == "ar":

        if gender == "female":

            return f"""
أنتِ {artifact_name}.
احكي قصتك بصيغة المتكلم.
اذكري:
- اسمك.
- العصر الذي تنتمين إليه.
- لماذا صُنعتِ.
- سرًا مميزًا عنك.
الرد يكون بالعربية الفصحى فقط.
بدون أي رموز أو تنسيق.
"""

        return f"""
أنتَ {artifact_name}.
احكِ قصتك بصيغة المتكلم.
اذكر:
- اسمك.
- العصر الذي تنتمي إليه.
- لماذا صُنعت.
- سرًا مميزًا عنك.
الرد يكون بالعربية الفصحى فقط.
بدون أي رموز أو تنسيق.
"""

    return f"""
You are {artifact_name}.
Tell your story in first person.
Include:
- Your name.
- Historical period.
- Why you were made.
- One interesting secret.
Use plain English only.
No markdown.
No symbols.
"""


def generate_story(
    artifact_name: str,
    gender: str,
    language: str = "ar",
):
    """
    Generate Story using Gemini
    """

    if model_loader.gemini is None:
        raise RuntimeError(
            "Gemini API Key is missing."
        )

    prompt = _story_prompt(
        artifact_name,
        gender,
        language,
    )

    response = model_loader.gemini.generate_content(
        prompt
    )

    return response.text.strip()


async def _tts_async(
    text: str,
    gender: str,
    language: str,
    output_path: str,
):

    import edge_tts

    if language == "ar":
        voice = (
            "ar-EG-SalmaNeural"
            if gender == "female"
            else "ar-EG-ShakirNeural"
        )
    else:
        voice = (
            "en-US-JennyNeural"
            if gender == "female"
            else "en-US-GuyNeural"
        )

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
    )

    await communicate.save(output_path)


def text_to_speech(
    text: str,
    gender: str = "male",
    language: str = "ar",
    request_id: str = "output",
):
    """
    Generate MP3 file
    """

    output_path = str(
        OUTPUTS_DIR / f"{request_id}.mp3"
    )

    asyncio.run(
        _tts_async(
            text=text,
            gender=gender,
            language=language,
            output_path=output_path,
        )
    )

    return output_path