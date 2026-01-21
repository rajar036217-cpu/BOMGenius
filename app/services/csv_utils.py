"""CSV parsing and conversion utilities"""
import csv
import io
from typing import Any


def parse_csv(text: str) -> list[dict[str, Any]]:
    """
    Parse CSV text into a list of dictionaries.
    
    Args:
        text: CSV content as string
    
    Returns:
        List of row dictionaries
    """
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        if row:
            rows.append(dict(row))
    return rows


def convert_to_csv(data: list[dict[str, Any]]) -> str:
    """
    Convert list of dictionaries to CSV string.
    
    Args:
        data: List of row dictionaries
    
    Returns:
        CSV content as string
    """
    if not data:
        return ""
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()
