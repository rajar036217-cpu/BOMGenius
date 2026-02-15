import pandas as pd
from .vision_ollama import run_vision_inference

VISION_MODEL = "glm-ocr:q8_0"


def ebom_from_image(image_path: str) -> pd.DataFrame:
    
    prompt = """
You are an OCR + PLM structured data extraction agent.

Your task:
Extract the Engineering BOM table EXACTLY as it appears in the image.

Instructions:

1. Detect the table structure.
2. Identify ALL column headers exactly as written in the image.
3. Preserve header spelling, spacing, capitalization, and symbols.
4. Extract every row under those headers.
5. If a cell is empty, return an empty string "".
6. Do NOT invent columns.
7. Do NOT summarize.
8. Do NOT normalize names.
9. Do NOT drop any columns.

Output Requirements:

• Return ONLY a valid JSON array of objects.
• Each object must contain ALL detected column headers as keys.
• Every row must have the same keys.
• No markdown.
• No explanation.
• No commentary.
• No extra text.
• Only raw JSON.

Example format (structure example only):

If the image headers are:
Item | Part No | Description | Qty | UOM

Then return:

[
  {
    "Item": "1",
    "Part No": "P-100",
    "Description": "Steel Bracket",
    "Qty": "2",
    "UOM": "EA"
  }
]

Extract the complete Engineering BOM table now.
"""

    raw_text = run_vision_inference(VISION_MODEL, image_path, prompt)

    try:
        data = raw_text
        return pd.DataFrame(data)
    except Exception as e:
        raise ValueError(
            f"OCR model did not return valid JSON:\n{raw_text}\nComplete error: {e}"
        )