import os
import re
import json
import time
import pandas as pd
import requests
from typing import Optional, Dict, List
from collections import defaultdict
import ollama

import pdfplumber
import httpx
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI

# ==========================================
# 1. SETUP OLLAMA CLIENT WITH INSTRUCTOR
# ==========================================
MODEL_NAME = "llama3.2 3b"
http_client = httpx.Client(timeout=30.0)
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        http_client=http_client
    ),
    mode=instructor.Mode.JSON
)

class MBomComponent(BaseModel):
    """Manufacturing BOM Component Data Structure"""
    part_number: str = Field(description="The unique identifier or part number")
    description: str = Field(description="Name or description of the part")
    quantity: int = Field(default=1, description="Number of units required")
    consumables: List[str] = Field(default_factory=list)
    routing_step: Optional[str] = Field(default="NA")

class PartAnalysis(BaseModel):
    step_by_step_reasoning: str
    predicted_consumable: str

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def load_global_rules() -> dict:
    rule_path = os.path.join(os.path.dirname(__file__), "..", "federated", "global_rules.json")
    if os.path.exists(rule_path):
        try:
            with open(rule_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("name_correction_map", {})
        except Exception as e:
            print(f"Warning: Could not load global rules - {e}")
    return {}

def _clean_str(x):
    if pd.isna(x): return ""
    return str(x).strip()

def _json_between_tags(text: str) -> str:
    match = re.search(r"<JSON>\s*(.*?)\s*</JSON>", text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    match_md = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match_md: return match_md.group(1).strip()
    return text.strip()

def _safe_json_loads(s: str):
    s = s.strip()
    if not (s.startswith('[') and s.endswith(']')):
        start = s.find('[')
        end = s.rfind(']')
        if start != -1 and end != -1:
            s = s[start:end+1]
    return json.loads(s)

# ==========================================
# CONFIDENCE SCORING
# ==========================================
def compute_confidence_score(row: Dict) -> float:
    """
    Heuristic confidence score in [0, 1] for each mBOM row.

    We avoid changing the overall pipeline structure by computing a lightweight
    score from the presence/consistency of key derived fields.
    """
    score = 0.50

    make_buy = str(row.get("Make/Buy", "") or "").strip().title()
    if make_buy in ("Make", "Buy"):
        score += 0.10
    else:
        score -= 0.10

    work_center = str(row.get("Work Center", "") or "").strip()
    if work_center and work_center.upper() not in ("NA", "N/A", "NONE"):
        score += 0.10

    routing = str(row.get("Operations (Routing Embedded)", "") or "").strip()
    if make_buy == "Make":
        if routing and routing.upper() not in ("NA", "N/A", "NONE"):
            score += 0.10
        else:
            score -= 0.05
    else:
        # For Buy parts, routing is usually NA; don't penalize.
        score += 0.02

    cons = str(row.get("Consumables", "") or "").strip()
    if cons and cons.upper() not in ("NA", "N/A", "NONE"):
        score += 0.10

    inv = str(row.get("Inventory Status", "") or "").strip()
    if inv and inv.lower() not in ("unknown", "na", "n/a", "none"):
        score += 0.05

    supplier = str(row.get("Approved_Supplier", "") or "").strip()
    if supplier and supplier.upper() not in ("NA", "N/A", "NONE"):
        score += 0.05

    action = str(row.get("Procurement Action", "") or "").strip()
    if action and action.upper() not in ("NA", "N/A", "NONE"):
        score += 0.05

    # Clamp to [0, 0.99] to keep buckets stable
    score = max(0.0, min(0.99, float(score)))
    return round(score, 2)

# ==========================================
# 3. EXTRACTION PIPELINE (PDF)
# ==========================================
def extract_ebom_from_pdf_hybrid(pdf_file_path):
    print(f"--- Phase 3: Hybrid Extraction for {pdf_file_path} ---")
    messy_raw_text = ""
    try:
        with pdfplumber.open(pdf_file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table_num, table in enumerate(tables):
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0]) 
                        messy_raw_text += f"Page {page_num+1} - Table {table_num+1}:\n{df.to_string()}\n\n"
    except Exception as e:
        print(f"PDF Extraction failed: {e}")
        return None

    prompt = f"""
    You are an expert Data Engineer. Extract Bill of Materials (BOM) data 
    from this PDF text and output a STRICT JSON ARRAY of objects.
    Required Keys: "Part Number", "Description", "Qty"
    Raw Text: {messy_raw_text}
    """
    try:
        response = requests.post("http://localhost:11434/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt, "format": "json", "stream": False, "keep_alive": "0", "options": {"temperature": 0.0}}
        )
        return pd.DataFrame(json.loads(response.json().get('response', '[]')))
    except Exception as e:
        print(f"LLM Refinement failed: {e}")
        return pd.DataFrame()

# ==========================================
# 4. DETERMINISTIC PIPELINE (STRUCTURING & RULES)
# ==========================================
def preprocess_ebom_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    print("--- Phase 1: Deterministic Normalization ---")
    
    # 1. Strict Regex Mapping (Added Part Type & Make/Buy)
    col_patterns = {
        r'.*part.*no.*|.*part.*num.*|.*item.*id.*': 'Part Number',
        r'.*parent.*|.*top.*level.*': 'Parent Part Number',
        r'.*desc.*|.*name.*': 'Description',
        r'.*qty.*|.*quantity.*': 'Qty',
        r'.*type.*|.*category.*': 'Part Type',
        r'.*make.*buy.*|.*source.*': 'Make/Buy',
        r'.*material.*': 'Material'
    }
    
    new_cols = []
    for col in raw_df.columns:
        standardized = str(col).strip()
        for pattern, std_name in col_patterns.items():
            if re.search(pattern, standardized, re.IGNORECASE):
                standardized = std_name
                break
        new_cols.append(standardized)
    raw_df.columns = new_cols

    # Ensure required columns exist
    for req_col in ['Part Number', 'Parent Part Number', 'Description', 'Part Type', 'Make/Buy']:
        if req_col not in raw_df.columns:
            raw_df[req_col] = ''

    print("--- Phase 2: DFS Hierarchy Reconstruction ---")
    
    # 2. Reverse Lookup: Map Parent Names to correct Part IDs
    name_to_id = {}
    for _, row in raw_df.iterrows():
        p_id = str(row.get('Part Number', '')).strip()
        p_name = str(row.get('Description', '')).strip().lower()
        if p_id and p_name:
            name_to_id[p_name] = p_id

    # Normalize Parent Column
    for idx, row in raw_df.iterrows():
        parent_val = str(row.get('Parent Part Number', '')).strip()
        parent_val_lower = parent_val.lower()
        
        # If the Parent column holds a Name, replace it with the actual ID
        if parent_val_lower in name_to_id:
            raw_df.at[idx, 'Parent Part Number'] = name_to_id[parent_val_lower]
        elif parent_val_lower in ['top level', 'root', '', 'none', 'nan', 'na']:
            raw_df.at[idx, 'Parent Part Number'] = 'ROOT'
    
    # 3. Build DFS Tree
    tree = defaultdict(list)
    part_details = {}
    for _, row in raw_df.iterrows():
        child = str(row.get('Part Number', '')).strip()
        parent = str(row.get('Parent Part Number', '')).strip()
        if child:
            tree[parent].append(child)
            part_details[child] = row.to_dict()

    all_children = set(part_details.keys())
    roots = [p for p in tree.keys() if p not in all_children]
    structured_data = []

    def dfs(node_id, current_level, path_history):
        if node_id in part_details:
            node_data = part_details[node_id].copy()
            node_data['Level'] = current_level
            node_data['Hierarchy Path'] = " > ".join(path_history)
            structured_data.append(node_data)
        for child_id in tree.get(node_id, []):
            dfs(child_id, current_level + 1, path_history + [child_id])

    for root in roots:
        for top_level_part in tree[root]:
            dfs(top_level_part, 0, [top_level_part])

    return pd.DataFrame(structured_data) if structured_data else raw_df

def enforce_manufacturing_rules(structured_df: pd.DataFrame) -> pd.DataFrame:
    print("--- Phase 3: Rule-Based Manufacturing Logic ---")

    cols = ["Make/Buy", "Work Center", "Operations (Routing Embedded)"]

    # ✅ 1) Remove duplicate columns (keep first occurrence)
    structured_df = structured_df.loc[:, ~structured_df.columns.duplicated()].copy()

    # ✅ 2) If df is empty, just create cols and return (NO apply)
    if structured_df.empty:
        for c in cols:
            if c not in structured_df.columns:
                structured_df[c] = []
        return structured_df

    def apply_rules(row):
        try:
            part_name = str(row.get("Description", "") or "").lower()
            node_type = str(row.get("Part Type", "Component") or "Component").lower()

            make_buy = str(row.get("Make/Buy", "") or "").strip().title()
            is_process_or_test = (
                any(kw in part_name for kw in ["test", "process", "measurement", "inspection"])
                or node_type in ["process", "test"]
            )

            if make_buy not in ["Make", "Buy"]:
                make_buy = "Buy"
                buy_keywords = ["screw", "nut", "bolt", "cable", "wire", "label", "box", "tape", "pin"]
                if any(kw in part_name for kw in buy_keywords) or node_type in ["fastener", "material"]:
                    make_buy = "Buy"
                elif node_type in ["assembly", "sub-assembly", "sub-assy"] or is_process_or_test:
                    make_buy = "Make"

            work_center, routing = "GENERAL_STORE", "NA"

            if make_buy == "Make":
                if any(kw in part_name for kw in ["pcb", "board", "smd", "circuit", "inverter"]):
                    work_center = "SMT_LINE"
                    routing = "10: SMT Placement | 20: Reflow Soldering | 30: AOI Inspection | 40: Functional Test"
                elif any(kw in part_name for kw in ["frame", "weld", "cradle", "chassis", "structure"]):
                    work_center = "WELDING_STATION"
                    routing = "10: Jig Setup | 20: Spot Welding | 30: Seam Welding | 40: CMM Inspection"
                elif any(kw in part_name for kw in ["housing", "shell", "plastic", "cover"]):
                    work_center = "INJECTION_MOLDING"
                    routing = "10: Injection Molding | 20: Cooling & Trimming | 30: Visual QC"
                elif is_process_or_test:
                    work_center = "QA_STATION"
                    routing = "10: Setup Equipment | 20: Execute Process/Test | 30: Log Results"
                else:
                    work_center = "MECH_LINE"
                    routing = "50: Mechanical Assembly | 60: Torque Tightening | 70: Final Testing | 80: Packing"
            else:
                work_center = "INCOMING_QC"
                routing = "NA"

            # ✅ ALWAYS return exactly 3 values
            return [make_buy, work_center, routing]

        except Exception:
            return ["Buy", "INCOMING_QC", "NA"]

    # ✅ 3) Do NOT use result_type="expand" (it breaks on empty/odd cases sometimes)
    out = structured_df.apply(apply_rules, axis=1)

    # ✅ 4) Build a guaranteed 3-column DataFrame, aligned to index
    assign_df = pd.DataFrame(out.tolist(), index=structured_df.index, columns=cols)

    # ✅ 5) Assign safely
    structured_df[cols] = assign_df

    return structured_df

# ==========================================
# 5. AI ENRICHMENT (Consumables & ERP)
# ==========================================
def predict_consumable_hybrid(description, material):
    desc = str(description).lower()
    mat = str(material).lower()
    
    if any(kw in desc for kw in ['pcb', 'smd', 'board', 'circuit']): return "Solder"
    if any(kw in desc for kw in ['housing', 'shell', 'casing', 'plastic']): return "Adhesive"
    if any(kw in desc for kw in ['cable', 'wire', 'harness']): return "Cable Tie"
    if any(kw in desc for kw in ['screw', 'bolt', 'nut', 'fastener']): return "Threadlocker"
    
    prompt = f"You are a Manufacturing Engineer. Output ONLY ONE consumable name. No raw materials. Part: {description} Material: {material}. Answer:"
    try:
        response = requests.post("http://localhost:11434/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False, "keep_alive": "0", "options": {"temperature": 0.0}}
        )
        ans = response.json().get('response', 'NA').strip()
        if "requires" in ans.lower() or len(ans.split()) > 2: return "NA"
        return ans.title()
    except:
        return "NA"

def ai_enrich(base_rows: List[Dict], inventory_json: str = "[]", chunk_size: int = 15) -> List[Dict]:
    results = []
    for offset in range(0, len(base_rows), chunk_size):
        chunk = base_rows[offset: offset + chunk_size]
        prompt = f"""
SYSTEM: You are an ERP procurement intelligence assistant. You ONLY output valid JSON.
TASK: Enrich each BOM row with ONLY inventory + procurement fields.
RULES: Output MUST be a JSON array of length = {len(chunk)}. Output EXACT indexes 0..{len(chunk)-1}.
OUTPUT KEYS: "Inventory Status", "Stock_Qty", "Store_Location", "Procurement Action", "Approved_Supplier", "Lead_Time_Days", "Procurement Steps" (array), "index".
INVENTORY MATCHING: Use INVENTORY_JSON.
INPUT_ROWS:
{json.dumps([{"index": i, "row": r} for i, r in enumerate(chunk)], ensure_ascii=False)}
INVENTORY_JSON:
{inventory_json}
Return JSON ONLY between tags: <JSON> [ ... ] </JSON>
"""
        try:
            response = requests.post("http://localhost:11434/api/generate",
                json={"model": MODEL_NAME, "prompt": prompt, "stream": False, "keep_alive": "0", "options": {"temperature": 0.0}}
            )
            raw = _json_between_tags(response.json().get('response', '[]'))
            arr = _safe_json_loads(raw)
            by_index = {int(obj["index"]): obj for obj in arr if isinstance(obj, dict) and "index" in obj}
        except:
            by_index = {}

        for i in range(len(chunk)):
            obj = by_index.get(i, {})
            results.append({
                "Inventory Status": obj.get("Inventory Status", "Unknown"),
                "Stock_Qty": obj.get("Stock_Qty", 0),
                "Store_Location": obj.get("Store_Location", "NA"),
                "Procurement Action": obj.get("Procurement Action", "NA"),
                "Approved_Supplier": obj.get("Approved_Supplier", "NA"),
                "Lead_Time_Days": obj.get("Lead_Time_Days", 0),
                "Procurement Steps": obj.get("Procurement Steps", []),
            })
    return results

# ==========================================
# 6. ORCHESTRATOR ENGINE
# ==========================================
def generate_mbom(ebom_df: pd.DataFrame, inv_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    global_rules = load_global_rules()
    inventory_json = "[]" if inv_df is None or inv_df.empty else inv_df.to_json(orient="records")
    
    start_total = time.perf_counter()
    
    # 1. Pipeline Transformations
    ebom_df = preprocess_ebom_pipeline(ebom_df)
    ebom_df = enforce_manufacturing_rules(ebom_df)

    # 2. Map to Base Rows
    pn_to_desc = dict(zip(ebom_df['Part Number'], ebom_df['Description'])) if 'Part Number' in ebom_df.columns else {}
    base_rows = []
    
    for _, row in ebom_df.iterrows():
        child_pn = str(row.get("Part Number", ""))
        child_desc = str(row.get("Description", ""))
        child_desc = global_rules.get(child_desc, child_desc) # Apply Global Rules Dictionary
        parent_pn = str(row.get("Parent Part Number", "ROOT"))
        
        base_rows.append({
            "Level": int(row.get("Level", 0)),
            "Parent Part Number": parent_pn,
            "Parent Description": pn_to_desc.get(parent_pn, "Top Level"),
            "Child Part Number": child_pn,
            "Child Description": child_desc,
            "Qty": int(row.get("Qty", 1)) if str(row.get("Qty", 1)).isdigit() else 1,
            "UOM": "EA",
            "Node Type": str(row.get("Part Type", "Component")),
            "Make/Buy": str(row.get("Make/Buy", "Buy")),
            "Work Center": str(row.get("Work Center", "NA")),
            "Operations (Routing Embedded)": str(row.get("Operations (Routing Embedded)", "NA")),
            "Hierarchy Path": str(row.get("Hierarchy Path", child_pn)),
            "Material": str(row.get("Material", "")),
            "Revision": str(row.get("Revision", "NA")),
            "Effective Date": str(row.get("Valid From", "")),
        })

    # 3. AI Enrichment (Batch)
    start_ai = time.perf_counter()
    ai_rows = ai_enrich(
        base_rows=[{
            "Part Number": r["Child Part Number"], "Part Name": r["Child Description"],
            "Part Type": r["Node Type"], "Make/Buy": r["Make/Buy"], "Qty": r["Qty"]
        } for r in base_rows],
        inventory_json=inventory_json,
        chunk_size=15
    )
    end_ai = time.perf_counter()

    # 4. Post Processing & Merging
    start_post = time.perf_counter()
    for r, ai in zip(base_rows, ai_rows):
        r.update({
            "Inventory Status": ai.get("Inventory Status", "Unknown"),
            "Stock_Qty": ai.get("Stock_Qty", 0),
            "Store_Location": ai.get("Store_Location", "NA"),
            "Procurement Action": ai.get("Procurement Action", "NA"),
            "Approved_Supplier": ai.get("Approved_Supplier", "NA"),
            "Lead_Time_Days": ai.get("Lead_Time_Days", 0),
        })
        
        steps = ai.get("Procurement Steps", [])
        if not steps or steps == "NA" or steps == []:
            steps = ["PR", "PO", "GRN", "Incoming QC", "Putaway", "Issue"] if r["Make/Buy"] == "Buy" else ["Kitting", "Assembly", "In-process QC", "Final Test", "Packing", "FG Receipt"]
        
        r["Procurement Steps"] = " -> ".join([str(x) for x in steps if str(x).strip()]) if isinstance(steps, list) else str(steps)
        r["Consumables"] = predict_consumable_hybrid(r["Child Description"], r["Material"])
        r["Confidence_Score"] = compute_confidence_score(r)
    end_post = time.perf_counter()
    end_total = time.perf_counter()

    # 5. Grouping & Aggregation
    df = pd.DataFrame(base_rows)
    group_cols = [c for c in [
        "Level", "Parent Part Number", "Parent Description", "Child Description", "UOM", 
        "Node Type", "Make/Buy", "Work Center", "Procurement Steps", "Operations (Routing Embedded)", "Consumables"
    ] if c in df.columns]

    def join_unique(x):
        s = pd.Series(x).astype(str).str.strip().replace({"nan": "", "None": ""})
        return ", ".join([u for u in s.unique().tolist() if u])

    agg_dict = {
    "Qty": "sum",
    "Child Part Number": join_unique,
    "Hierarchy Path": "first",
    "Revision": "first",
    "Effective Date": "first",
}

# Add Confidence aggregation only if present
    if "Confidence_Score" in df.columns:
        agg_dict["Confidence_Score"] = "mean"
    for c in ["Inventory Status", "Stock_Qty", "Store_Location", "Procurement Action", "Approved_Supplier", "Lead_Time_Days"]:
        if c in df.columns: agg_dict[c] = "first"

    print("\n--- PERFORMANCE METRICS ---")
    print(f"AI Generation Time   : {end_ai - start_ai:.4f} sec")
    print(f"Post-processing Time : {end_post - start_post:.4f} sec")
    print(f"Total MBOM Time      : {end_total - start_total:.4f} sec\n")

# Ensure confidence column always exists before aggregation
    if "Confidence_Score" not in df.columns:
        df["Confidence_Score"] = 0.0
    else:
        df["Confidence_Score"] = pd.to_numeric(df["Confidence_Score"], errors="coerce").fillna(0.0)
    print("COLUMNS:", df.columns.tolist())

    out_df = df.groupby(group_cols, as_index=False).agg(agg_dict).fillna("")
    if "Confidence_Score" in out_df.columns:
        out_df["Confidence_Score"] = pd.to_numeric(out_df["Confidence_Score"], errors="coerce").fillna(0.0).round(2)
    return out_df