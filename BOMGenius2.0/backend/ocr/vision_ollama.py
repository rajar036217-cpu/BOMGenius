import json
import re
import os
import ollama


def run_vision_inference(model_name: str, image_path: str, prompt: str):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_path],
            }
        ],
    )

    content = response["message"]["content"].strip()

    print("RAW MODEL RESPONSE:\n", content)

    # Remove markdown
    content = content.replace("```json", "").replace("```", "").strip()

    # Extract all JSON objects
    json_objects = re.findall(r"\{.*?\}", content, re.DOTALL)

    parsed_rows = []

    for obj in json_objects:
        try:
            parsed = json.loads(obj)
            parsed_rows.append(parsed)
        except Exception as e:
            print("Skipping invalid JSON block:", e)

    return parsed_rows