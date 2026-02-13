import ollama
import os
from pydantic import BaseModel
from typing import List
import json


# TODO: Class names Row and Rows should have better names
class Row(BaseModel):
    part_no: str
    description: str
    qty: int


class Rows(BaseModel):
    rows: List[Row]


schema = Rows.model_json_schema()


def run_vision_inference(model_name: str, image_path: str, prompt: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    response = ollama.chat(
        model=model_name,
        format=schema,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_path],
            }
        ],
    )

    json_data = json.loads(response["message"]["content"])
    return json_data["rows"]  # key 'rows' is comming from Rows class
