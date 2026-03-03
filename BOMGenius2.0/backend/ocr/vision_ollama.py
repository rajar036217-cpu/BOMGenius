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


dVISION_MODEL = "llava" 

def run_vision_inference(model_name, image_path, prompt):
    print(f"--- Running OCR Inference using {model_name} ---")
    try:
        response = ollama.chat(
            model=model_name,  # This will now use "llava"
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path]
                }
            ],
            options={"temperature": 0.0} # Keep it strict
        )
        return response['message']['content']
    except Exception as e:
        print(f"Vision API Error: {e}")
        return "{}"

    json_data = json.loads(response["message"]["content"])
    return json_data["rows"]  # key 'rows' is comming from Rows class