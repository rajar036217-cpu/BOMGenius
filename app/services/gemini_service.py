"""Google Gemini AI service for BOM analysis and CAD image processing"""
import os
import json
import base64
from typing import Any
import google.generativeai as genai
from app.models import DesignBomEntry, MatchResult


API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)


def find_key(obj: dict, keywords: list[str]) -> str | None:
    """Find a key in object that matches any of the keywords"""
    keys = list(obj.keys())
    for keyword in keywords:
        for k in keys:
            if keyword.lower() in k.lower():
                return k
    return None


async def analyze_cad_image(
    image_base64: str,
    mime_type: str
) -> list[dict[str, Any]]:
    """
    Analyze a CAD image and extract components.
    
    Args:
        image_base64: Base64-encoded image data
        mime_type: MIME type of the image (e.g., "image/png")
    
    Returns:
        List of component dictionaries with Part Name, Quantity, and Description
    """
    prompt = """
    You are an expert mechanical engineer. Analyze the provided image of a CAD assembly model.
    Your task is to identify all distinct components and list them.
    For each component, provide a descriptive name, estimate the quantity shown in the assembly, and write a brief technical description.
    
    Return the result as a JSON array of objects, where each object has the following keys: "Part Name", "Quantity", and "Description".
    Example:
    [
        { "Part Name": "M6 Hex Bolt", "Quantity": 8, "Description": "Standard M6 hexagonal head bolt, likely steel." },
        { "Part Name": "Mounting Plate", "Quantity": 1, "Description": "Main structural plate with four mounting holes." }
    ]
    """

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        image_data = base64.b64decode(image_base64)
        
        response = model.generate_content(
            [
                prompt,
                {
                    "mime_type": mime_type,
                    "data": image_data,
                },
            ],
        )
        
        response_text = response.text
        parsed = json.loads(response_text)
        
        if isinstance(parsed, list):
            return parsed
        return parsed.get("components", [])

    except Exception as e:
        raise Exception(f"CAD analysis failed: {str(e)}")


async def compute_semantic_matches(
    design_bom: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    global_threshold: float = 0.7,
) -> list[MatchResult]:
    """
    Match design BOM entries with inventory using semantic AI matching.
    
    Args:
        design_bom: List of design part dictionaries
        inventory: List of inventory dictionaries
        global_threshold: Confidence threshold for matching
    
    Returns:
        List of MatchResult objects
    """
    prompt = f"""
    You are a BOM matching expert. Your task is to match design parts with inventory items and provide confidence scores.
    
    Design BOM:
    {json.dumps(design_bom, indent=2)}
    
    Available Inventory:
    {json.dumps(inventory, indent=2)}
    
    For each design part, find the best matching inventory item. Consider part names, specifications, and quantities.
    Return a JSON array where each object contains:
    - "Design Part (eBOM)": name of the design part
    - "Required Qty": quantity needed
    - "Matched Inventory (mBOM)": matched inventory part name
    - "Confidence": confidence score (0.0-1.0)
    - "Unit Price ($)": estimated unit price
    - "Total Cost ($)": total cost
    - "Stock Status": "In Stock" or "Out of Stock"
    - "Match Status": "Exact" or "Substitute"
    
    Apply a confidence threshold of {global_threshold}. Only include matches above this threshold.
    """

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        response = model.generate_content(prompt)
        response_text = response.text
        parsed = json.loads(response_text)
        
        if isinstance(parsed, list):
            results = parsed
        else:
            results = parsed.get("matches", [])
        
        match_results = []
        for result in results:
            match_results.append(MatchResult(
                design_part=result.get("Design Part (eBOM)", ""),
                required_qty=result.get("Required Qty", 0),
                matched_inventory=result.get("Matched Inventory (mBOM)", ""),
                confidence=result.get("Confidence", "0.0"),
                unit_price=result.get("Unit Price ($)", 0),
                total_cost=result.get("Total Cost ($)", 0),
                stock_status=result.get("Stock Status", "Unknown"),
                match_status=result.get("Match Status", "Unknown"),
            ))
        
        return match_results

    except Exception as e:
        raise Exception(f"Semantic matching failed: {str(e)}")
