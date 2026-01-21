"""Data models for BOM Genius"""
from typing import Any, Dict
from enum import Enum
from pydantic import BaseModel


class PageEnum(str, Enum):
    DASHBOARD = "Dashboard"
    BOM_CONVERTER = "BOM Converter"
    FEDERATED_LEARNING = "Federated Learning"
    CAD_ANALYZER = "CAD Analyzer"
    JSON_IMPORTER = "JSON Importer"


class DesignBomEntry(BaseModel):
    """Design BOM entry with flexible key-value structure"""
    data: Dict[str, Any]


class InventoryEntry(BaseModel):
    """Inventory entry with flexible key-value structure"""
    data: Dict[str, Any]


class MatchResult(BaseModel):
    """BOM matching result"""
    design_part: str
    required_qty: int | str
    matched_inventory: str
    confidence: str
    unit_price: int | str
    total_cost: int | str
    stock_status: str
    match_status: str


class ChartData(BaseModel):
    """Chart data point"""
    name: str
    value: int


class ThresholdHistory(BaseModel):
    """Threshold optimization history"""
    step: int
    threshold: float


class GlobalState(BaseModel):
    """Global application state"""
    global_threshold: float = 0.7
    generated_bom: list[DesignBomEntry] | None = None
