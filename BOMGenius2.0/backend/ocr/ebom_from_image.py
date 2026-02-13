import json
import pandas as pd
from .vision_ollama import run_vision_inference

VISION_MODEL = "glm-ocr:q8_0"

def ebom_from_image(image_path: str) -> pd.DataFrame:
    prompt = """
You are an OCR + PLM extraction agent.

Extract the Engineering BOM from this image.
Return ONLY valid JSON array of objects.

Each object must contain:
- part_no
- description
- qty

Example:
[
  {"part_no": "P-100", "description": "Steel Bracket", "qty": 2},
  {"part_no": "BOLT-M6", "description": "Hex Bolt M6", "qty": 8}
]
No markdown. No commentary. Only JSON.
"""

    raw_text = run_vision_inference(VISION_MODEL, image_path, prompt)

    try:
        data = raw_text
        return pd.DataFrame(data)
    except Exception as e:
        raise ValueError(f"OCR model did not return valid JSON:\n{raw_text}\nComplete error: {e}")
