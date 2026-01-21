"""FastAPI application with routes for BOM Genius"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import json
import base64
import os
from datetime import datetime

from app.models import GlobalState, PageEnum, MatchResult
from app.services.gemini_service import analyze_cad_image, compute_semantic_matches
from app.services.csv_utils import parse_csv, convert_to_csv


app = FastAPI(title="BOM Genius Enterprise Pro")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Global application state
global_state = GlobalState()


@app.get("/")
async def index(request: Request):
    """Render main dashboard page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "pages": [page.value for page in PageEnum],
        "current_page": PageEnum.DASHBOARD.value
    })


@app.get("/api/state")
async def get_state():
    """Get global application state"""
    return {
        "global_threshold": global_state.global_threshold,
        "generated_bom": global_state.generated_bom,
    }


@app.post("/api/state/threshold")
async def set_threshold(threshold: float):
    """Update global threshold"""
    global_state.global_threshold = threshold
    return {"global_threshold": global_state.global_threshold}


@app.post("/api/bom/convert")
async def convert_bom(
    design_file: UploadFile = File(...),
    inventory_file: UploadFile = File(...)
):
    """
    Convert design BOM to manufacturing BOM by matching with inventory.
    
    Accepts two CSV files:
    1. Design BOM (eBOM)
    2. Available Inventory (mBOM reference)
    
    Returns: Matched results with confidence scores and pricing
    """
    try:
        design_content = await design_file.read()
        inventory_content = await inventory_file.read()
        
        design_bom = parse_csv(design_content.decode("utf-8"))
        inventory = parse_csv(inventory_content.decode("utf-8"))
        
        if not design_bom or not inventory:
            raise HTTPException(status_code=400, detail="CSV files are empty")
        
        # Compute semantic matches using Gemini
        match_results = await compute_semantic_matches(
            design_bom,
            inventory,
            global_state.global_threshold
        )
        
        # Convert to dict for JSON response
        results = [result.model_dump() for result in match_results]
        
        return {
            "success": True,
            "results": results,
            "design_file": design_file.filename,
            "inventory_file": inventory_file.filename,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bom/export")
async def export_bom(results: list[dict]):
    """Export BOM matching results as CSV"""
    try:
        csv_content = convert_to_csv(results)
        
        return {
            "csv": csv_content,
            "filename": f"bom_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cad/analyze")
async def analyze_cad(
    file: UploadFile = File(...)
):
    """
    Analyze CAD image and extract components.
    
    Accepts: CAD model image (PNG, JPG, etc.)
    Returns: Extracted parts list with quantities and descriptions
    """
    try:
        # Validate file type
        if file.content_type not in ["image/png", "image/jpeg", "image/jpg", "image/webp"]:
            raise HTTPException(
                status_code=400,
                detail="Only PNG, JPG, and WebP images are supported"
            )
        
        content = await file.read()
        image_base64 = base64.b64encode(content).decode("utf-8")
        
        # Analyze image
        components = await analyze_cad_image(
            image_base64,
            file.content_type
        )
        
        # Store in global state for potential use in BOM converter
        global_state.generated_bom = [
            {"data": component} for component in components
        ]
        
        return {
            "success": True,
            "components": components,
            "image_file": file.filename,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/json/import")
async def import_json(
    json_data: str = Form(...)
):
    """
    Import BOM data from JSON format.
    
    Accepts: JSON string with BOM entries
    Returns: Parsed data ready for matching
    """
    try:
        data = json.loads(json_data)
        
        if isinstance(data, list):
            bom_data = data
        elif isinstance(data, dict) and "bom" in data:
            bom_data = data["bom"]
        else:
            raise ValueError("Invalid JSON format")
        
        return {
            "success": True,
            "data": bom_data,
            "count": len(bom_data),
            "timestamp": datetime.now().isoformat()
        }
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "BOM Genius"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
