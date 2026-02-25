import os
import re
import json
import pandas as pd
import ollama
from datetime import datetime
from typing import Optional, Dict, List, Tuple

print("=== HYBRID mBOM ENGINE LOADED (AI + Deterministic Validation) ===")

MODEL_NAME = "llama3.2:3b"
GLOBAL_RULES_PATH = "federated/global_rules.json"


# =========================================================
# Utilities
# =========================================================

def load_global_rules():
    if os.path.exists(GLOBAL_RULES_PATH):
        with open(GLOBAL_RULES_PATH, "r") as f:
            return json.load(f)
    return {}

GLOBAL_RULES = load_global_rules()

def _clean_str(x):
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def smart_find_column(df, candidates):
    norm = {}
    for c in df.columns:
        key = re.sub(r"[\s_]+", "", str(c).strip().lower())
        norm[key] = c

    for cand in candidates:
        key = re.sub(r"[\s_]+", "", str(cand).strip().lower())
        if key in norm:
            return norm[key]

    for cand in candidates:
        key = re.sub(r"[\s_]+", "", str(cand).strip().lower())
        for k, real in norm.items():
            if key in k:
                return real
    return None

def format_routing_text(ops) -> str:
    if ops is None:
        return "NA"
    if isinstance(ops, str):
        return ops.strip() or "NA"
    if isinstance(ops, list):
        parts = []
        for o in ops:
            if isinstance(o, dict):
                seq = o.get("Op_Seq", "")
                opn = o.get("Operation", "")
                if str(seq).strip() and str(opn).strip():
                    parts.append(f"{seq}: {opn}")
            elif isinstance(o, str) and o.strip():
                parts.append(o.strip())
        return " | ".join(parts) if parts else "NA"
    return "NA"

# =========================================================
# Normalization
# =========================================================

def normalize_ebom(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    part_no = smart_find_column(df, ["part number", "part_no", "partno"])
    part_name = smart_find_column(df, ["part name", "description"])
    parent_asm = smart_find_column(df, ["parent assembly", "parent"])
    qty = smart_find_column(df, ["quantity", "qty"])
    rev = smart_find_column(df, ["revision"])
    part_type = smart_find_column(df, ["part type"])
    std_custom = smart_find_column(df, ["standard vs custom"])
    valid_from = smart_find_column(df, ["valid from"])
    material = smart_find_column(df, ["material", "raw material"])

    out = pd.DataFrame()
    out["Part Number"] = df[part_no].apply(_clean_str)
    out["Part Name"] = df[part_name].apply(_clean_str)
    out["Parent Assembly"] = df[parent_asm].apply(_clean_str) if parent_asm else ""
    out["Quantity"] = pd.to_numeric(df[qty], errors="coerce").fillna(1).astype(int) if qty else 1
    out["Revision"] = df[rev].apply(_clean_str) if rev else "NA"
    out["Part Type"] = df[part_type].apply(_clean_str) if part_type else ""
    out["Standard vs Custom"] = df[std_custom].apply(_clean_str) if std_custom else ""
    out["Valid From"] = df[valid_from].apply(_clean_str) if valid_from else ""
    out["Material"] = df[material].apply(_clean_str) if material else ""

    return out.fillna("")


# =========================================================
# Hierarchy
# =========================================================

def build_parent_child(ebom):
    name_to_pn = dict(zip(ebom["Part Name"], ebom["Part Number"]))

    def resolve_parent(x):
        return name_to_pn.get(_clean_str(x), "")

    ebom["Parent Part Number"] = ebom["Parent Assembly"].apply(resolve_parent)
    return ebom


def compute_levels(ebom):
    children = {}
    for _, r in ebom.iterrows():
        parent = _clean_str(r["Parent Part Number"])
        child = _clean_str(r["Part Number"])
        if parent:
            children.setdefault(parent, []).append(child)

    roots = ebom.loc[ebom["Parent Part Number"] == "", "Part Number"].tolist()
    if not roots:
        roots = ebom["Part Number"].tolist()[:1]  # fallback to first row as root
    level = {}
    path = {}

    def dfs(node, lvl, stack):
        level[node] = lvl
        path[node] = " > ".join(stack + [node])
        for ch in children.get(node, []):
            dfs(ch, lvl + 1, stack + [node])

    for r in roots:
        dfs(r, 0, [])

    return level, path

# --- ADD this builder (base rows = deterministic "ground truth") ---

def build_base_rows(ebom_df: pd.DataFrame) -> List[Dict]:
    ebom = normalize_ebom(ebom_df)
    ebom = build_parent_child(ebom)
    level_map, path_map = compute_levels(ebom)

    pn_to_name = dict(zip(ebom["Part Number"], ebom["Part Name"]))

    base_rows: List[Dict] = []
    for _, r in ebom.iterrows():
        child_pn = _clean_str(r["Part Number"])
        child_name = pn_to_name.get(child_pn, _clean_str(r["Part Name"]))
        parent_pn = _clean_str(r["Parent Part Number"])
        parent_name = pn_to_name.get(parent_pn, "")

        nt = node_type_rule(_clean_str(r["Part Type"]), child_name)
        mb = makebuy_rule(_clean_str(r["Part Type"]), _clean_str(r["Standard vs Custom"]), child_name)
        wc = work_center_rule(nt, child_name, mb)
        ops = routing_rule(nt, child_name, mb)

        base_rows.append({
            "Level": int(level_map.get(child_pn, 0)),
            "Parent Part Number": parent_pn,
            "Parent Description": parent_name,
            "Child Part Number": child_pn,
            "Child Description": child_name,
            "Qty": int(r["Quantity"]) if str(r["Quantity"]).isdigit() else 1,
            "UOM": "EA",
            "Revision": _clean_str(r["Revision"]) or "NA",
            "Effective Date": _clean_str(r["Valid From"]) or "",
            "Node Type": nt,
            "Make/Buy": mb,
            "Work Center": wc,
            "Operations (Routing Embedded)": format_routing_text(ops),
            "Hierarchy Path": path_map.get(child_pn, child_pn),
            "Material": _clean_str(r.get("Material", "")),
            "Consumables": "NA",
            # AI placeholders (kept internal, not necessarily output)
            "Inventory Status": "Unknown",
            "Stock_Qty": 0,
            "Store_Location": "NA",
            "Procurement Action": "NA",
            "Approved_Supplier": "NA",
            "Lead_Time_Days": 0,
            "Procurement Steps": [],
        })

    return base_rows
# =========================================================
# Deterministic Rules
# =========================================================

# --- ADD these deterministic rule functions (Rules = NodeType/MakeBuy/WorkCenter/Routing) ---

def node_type_rule(part_type: str, part_name: str) -> str:
    pt = (part_type or "").lower()
    nm = (part_name or "").lower()

    if "process" in pt or any(k in nm for k in ["process", "test", "inspection", "burn", "hipot", "reflow", "smt", "aoi"]):
        return "Process"
    if any(k in nm for k in ["label", "box", "packing", "carton", "manual", "sticker"]):
        return "Packaging"
    if "assembly" in pt or any(k in nm for k in ["assembly", "sub-assembly", "adapter assembly", "pcb assembly", "cable assembly"]):
        return "Assembly"
    if pt in ["material"] or "material" in pt:
        return "Material"
    return "Component"

def makebuy_rule(part_type: str, std_custom: str, part_name: str) -> str:
    pt = (part_type or "").lower()
    sc = (std_custom or "").lower()
    nm = (part_name or "").lower()

    if any(k in nm for k in ["cable assembly", "housing", "top shell", "bottom shell", "label", "carton", "box"]):
        return "Buy"
    if "assembly" in pt or "sub-assembly" in pt:
        return "Make"
    if sc == "custom":
        return "Make"
    return "Buy"

def work_center_rule(node_type: str, part_name: str, make_buy: str) -> str:
    nm = (part_name or "").lower()
    if node_type == "Process":
        if any(k in nm for k in ["smt"]): return "SMT_LINE"
        if any(k in nm for k in ["reflow"]): return "REFLOW_OVEN"
        if any(k in nm for k in ["aoi"]): return "AOI_STATION"
        if any(k in nm for k in ["hipot", "hi-pot", "burn", "thermal", "load", "regulation", "efficiency"]): return "QA_STATION"
        return "NA"

    if node_type == "Packaging":
        return "PACK_LINE"

    if make_buy == "Make":
        # PCB assembly vs mechanical assembly split (simple keyword)
        if "pcb" in nm:
            return "SMT_LINE"
        return "MECH_LINE"

    return "NA"

def routing_rule(node_type: str, part_name: str, make_buy: str) -> List[Dict]:
    nm = (part_name or "").lower()

    # Only attach routing to assemblies (avoid per-resistor routing)
    if node_type != "Assembly":
        return []

    if "pcb" in nm:
        return [
            {"Op_Seq": 10, "Operation": "SMT Placement", "Work_Center": "SMT_LINE"},
            {"Op_Seq": 20, "Operation": "Reflow Soldering", "Work_Center": "REFLOW_OVEN"},
            {"Op_Seq": 30, "Operation": "AOI Inspection", "Work_Center": "AOI_STATION"},
            {"Op_Seq": 40, "Operation": "Functional Test", "Work_Center": "PCB_TEST"},
        ]

    # final/mech assembly
    return [
        {"Op_Seq": 50, "Operation": "Mechanical Assembly", "Work_Center": "MECH_LINE"},
        {"Op_Seq": 80, "Operation": "Hi-Pot Test", "Work_Center": "QA_STATION"},
        {"Op_Seq": 90, "Operation": "Load Regulation Test", "Work_Center": "QA_STATION"},
        {"Op_Seq": 100, "Operation": "Burn-In Test", "Work_Center": "BURN_IN_RACK"},
        {"Op_Seq": 110, "Operation": "Label & Packing", "Work_Center": "PACK_LINE"},
    ]


# =========================================================
# AI Draft Generator
# =========================================================

def _json_between_tags(text: str) -> str:
    m = re.search(r"<JSON>\s*(.*?)\s*</JSON>", text, flags=re.S)
    if m:
        return m.group(1).strip()

    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end != -1 and end > start:
        return text[start:end].strip()
    raise ValueError("No JSON array found in model output.")

def _safe_json_loads(s: str):
    s = s.strip()

    # normalize smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # remove trailing commas before ] or }
    s = re.sub(r",\s*([\]}])", r"\1", s)

    # strip code fences if any
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    return json.loads(s)
    
def ai_enrich(base_rows: List[Dict], inventory_json: str = "[]", chunk_size: int = 15) -> List[Dict]:
    """
    Returns list[dict] same length as base_rows.
    Each dict contains ONLY:
      Inventory Status, Stock_Qty, Store_Location,
      Procurement Action, Approved_Supplier, Lead_Time_Days, Procurement Steps
    """
    results: List[Dict] = []

    for offset in range(0, len(base_rows), chunk_size):
        chunk = base_rows[offset: offset + chunk_size]

        prompt = f"""
SYSTEM:
You are an ERP procurement intelligence assistant.
You ONLY output valid JSON. No markdown. No commentary.

TASK:
Enrich each BOM row with ONLY inventory + procurement fields.
Do NOT output any other keys.

HARD RULES:
1) Output MUST be a JSON array of length = {len(chunk)}.
2) Preserve row order exactly.
3) If unknown, use "Unknown"/"NA", 0 for numbers, [] for arrays.
4) Never include extra keys. Never include explanations.
5) Output MUST contain exactly the indexes 0..{len(chunk)-1}, no extras.

OUTPUT KEYS (exact):
- "Inventory Status" : one of ["In Stock","Low Stock","Out of Stock","Unknown"]
- "Stock_Qty" : number
- "Store_Location" : string
- "Procurement Action" : one of ["Issue from Stock","Purchase","Manufacture","Expedite","NA"]
- "Approved_Supplier" : string
- "Lead_Time_Days" : number
- "Procurement Steps" : array of short strings (max 6)
- "index" : integer (0..{len(chunk)-1})

INVENTORY MATCHING:
Use INVENTORY_JSON (may be empty).
Match by exact "Part Number" if possible, else fuzzy by "Part Name"/"Part Type".
If no match, Inventory Status="Unknown", Stock_Qty=0, Store_Location="NA".

INPUT_ROWS (index -> row):
{json.dumps([{ "index": i, "row": r } for i, r in enumerate(chunk)], ensure_ascii=False)}

INVENTORY_JSON:
{inventory_json}

Return JSON ONLY between tags:
<JSON>
[ ... ]
</JSON>
"""

        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={"temperature": 0.0, "num_predict": 900, "top_p": 0.9, "repeat_penalty": 1.15},
        )

        raw = _json_between_tags(response["response"])
        try:
            arr = _safe_json_loads(raw)
        except json.JSONDecodeError:
            repair_prompt = f"""
        SYSTEM: You are a JSON repair tool. Output ONLY valid JSON.

        Fix this into a valid JSON array (only syntax fixes: commas/quotes/brackets).
        BROKEN_JSON:
        {raw}

        Return ONLY the repaired JSON array.
        """
            fixed = ollama.generate(
                model=MODEL_NAME,
                prompt=repair_prompt,
                options={"temperature": 0.0, "num_predict": 900}
            )
            fixed_raw = _json_between_tags(fixed["response"]) if "<JSON>" in fixed["response"] else fixed["response"]
            arr = _safe_json_loads(fixed_raw)

        # Expect list of objects with "index"
        if not isinstance(arr, list):
            raise ValueError("AI output is not a list")

        by_index = {}
        for obj in arr:
            if isinstance(obj, dict) and "index" in obj:
                by_index[int(obj["index"])] = obj

        fixed = []
        for i in range(len(chunk)):
            obj = by_index.get(i, {})
            fixed.append({
                "Inventory Status": obj.get("Inventory Status", "Unknown"),
                "Stock_Qty": obj.get("Stock_Qty", 0),
                "Store_Location": obj.get("Store_Location", "NA"),
                "Procurement Action": obj.get("Procurement Action", "NA"),
                "Approved_Supplier": obj.get("Approved_Supplier", "NA"),
                "Lead_Time_Days": obj.get("Lead_Time_Days", 0),
                "Procurement Steps": obj.get("Procurement Steps", []),
            })

        results.extend(fixed)

    if len(results) != len(base_rows):
        raise ValueError("AI row count mismatch (post-chunk).")

    return results

def get_ai_consumables(description, material):
    prompt = f"""
You are a Manufacturing Engineer.

Part Description: {description}
Material: {material}

If this part requires a consumable during assembly 
(e.g., Glue, Grease, Solder, Cable Tie, Threadlocker),
return ONE word only.

If no consumable needed, return NA.
"""

    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={"temperature": 0.0, "num_predict": 15}
        )
        return response["response"].strip().replace(".", "")
    except:
        return "NA"

def generate_mbom(ebom_df: pd.DataFrame, inv_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if inv_df is None or inv_df.empty:
        inventory_json = "[]"
    else:
        inventory_json = inv_df.to_json(orient="records")

    base_rows = build_base_rows(ebom_df)

    ai_rows = ai_enrich(
        base_rows=[{
            "Part Number": r["Child Part Number"],
            "Part Name": r["Child Description"],
            "Part Type": r["Node Type"],
            "Make/Buy": r["Make/Buy"],
            "Qty": r["Qty"],
        } for r in base_rows],
        inventory_json=inventory_json,
        chunk_size=15
    )

    for r, ai in zip(base_rows, ai_rows):
        r["Inventory Status"] = ai.get("Inventory Status", r["Inventory Status"])
        r["Stock_Qty"] = ai.get("Stock_Qty", r["Stock_Qty"])
        r["Store_Location"] = ai.get("Store_Location", r["Store_Location"])
        r["Procurement Action"] = ai.get("Procurement Action", r["Procurement Action"])
        r["Approved_Supplier"] = ai.get("Approved_Supplier", r["Approved_Supplier"])
        r["Lead_Time_Days"] = ai.get("Lead_Time_Days", r["Lead_Time_Days"])
        r["Procurement Steps"] = ai.get("Procurement Steps", r["Procurement Steps"])

        r["Consumables"] = get_ai_consumables(
            description=r.get("Child Description", ""),
            material=r.get("Material", "")
        ) or "NA"

        if isinstance(r.get("Procurement Steps"), list):
            r["Procurement Steps"] = " -> ".join([str(x) for x in r["Procurement Steps"] if str(x).strip()]) or "NA"
        else:
            r["Procurement Steps"] = str(r.get("Procurement Steps") or "NA")

        r["Operations (Routing Embedded)"] = format_routing_text(r.get("Operations (Routing Embedded)"))

    df = pd.DataFrame(base_rows)

    # UI screenshot columns ONLY
    final_cols = [
        "Level",
        "Parent Part Number",
        "Parent Description",
        "Child Description",
        "UOM",
        "Node Type",
        "Make/Buy",
        "Work Center",
        "Procurement Steps",
        "Operations (Routing Embedded)",
        "Consumables",
        "Qty",
        "Child Part Number",
        "Hierarchy Path",
        "Revision",
        "Effective Date"
    ]

    for c in final_cols:
        if c not in df.columns:
            df[c] = "NA"

    return df[final_cols].fillna("NA")

