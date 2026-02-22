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

    out = pd.DataFrame()
    out["Part Number"] = df[part_no].apply(_clean_str)
    out["Part Name"] = df[part_name].apply(_clean_str)
    out["Parent Assembly"] = df[parent_asm].apply(_clean_str) if parent_asm else ""
    out["Quantity"] = pd.to_numeric(df[qty], errors="coerce").fillna(1).astype(int) if qty else 1
    out["Revision"] = df[rev].apply(_clean_str) if rev else "NA"
    out["Part Type"] = df[part_type].apply(_clean_str) if part_type else ""
    out["Standard vs Custom"] = df[std_custom].apply(_clean_str) if std_custom else ""
    out["Valid From"] = df[valid_from].apply(_clean_str) if valid_from else ""

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


# =========================================================
# Deterministic Rules
# =========================================================

def node_type(part_type):
    pt = part_type.lower()
    if "assembly" in pt:
        return "Assembly"
    return "Component"


def makebuy(part_type, std_custom):
    pt = part_type.lower()
    sc = std_custom.lower()

    if "assembly" in pt:
        return "Make"
    if sc == "custom":
        return "Make"
    return "Buy"


def work_center(make_buy, node_type):
    if make_buy == "Make":
        return "Final Assembly Line"
    return "Incoming Inspection (General)"


def procurement_steps(make_buy):
    if make_buy == "Make":
        return "Issue Components -> Assemble -> In-process QC -> Final Test -> FG Receipt"
    return "Vendor Selection -> PR -> PO -> GRN -> Incoming QC -> Putaway -> Issue to Line"


def routing(make_buy):
    if make_buy == "Make":
        return "10: Kitting | 20: Assembly | 30: Functional Test | 40: Packing"
    return "10: PR/PO | 20: Incoming Inspection | 30: Putaway"


# =========================================================
# AI Draft Generator
# =========================================================

def generate_ai_draft(ebom_df):
    prompt = f"""
You are a Manufacturing Engineer.

Convert this eBOM into structured mBOM table.
Return CSV only.
"""

    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={"temperature": 0.0}
    )

    from io import StringIO
    try:
        return pd.read_csv(StringIO(response["response"]))
    except:
        return pd.DataFrame()


# =========================================================
# Validation Layer
# =========================================================

def validate_and_correct(ai_df, ebom_df):

    ebom = normalize_ebom(ebom_df)
    ebom = build_parent_child(ebom)
    level_map, path_map = compute_levels(ebom)

    pn_to_name = dict(zip(ebom["Part Number"], ebom["Part Name"]))

    rows = []

    for _, r in ebom.iterrows():

        child = r["Part Number"]
        parent = r["Parent Part Number"]
        name = pn_to_name.get(child, "")

        lvl = level_map.get(child, 0)
        nt = node_type(r["Part Type"])
        mb = makebuy(r["Part Type"], r["Standard vs Custom"])
        wc = work_center(mb, nt)

        rows.append({
            "Level": lvl,
            "Parent Part Number": parent,
            "Parent Description": pn_to_name.get(parent, ""),
            "Child Description": name,
            "UOM": "EA",
            "Node Type": nt,
            "Make/Buy": mb,
            "Work Center": wc,
            "Procurement Steps": procurement_steps(mb),
            "Operations (Routing Embedded)": routing(mb),
            "Consumables": "NA",
            "Qty": r["Quantity"],
            "Child Part Number": child,
            "Hierarchy Path": path_map.get(child, child),
            "Revision": r["Revision"],
            "Effective Date": r["Valid From"]
        })

    df = pd.DataFrame(rows)

    return df


# =========================================================
# Final Formatter (Exact UI Match)
# =========================================================

def format_output(df):

    cols = [
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

    for c in cols:
        if c not in df.columns:
            df[c] = ""

    df = df[cols]
    df = df.sort_values(["Hierarchy Path", "Level"]).reset_index(drop=True)

    return df.fillna("")


# =========================================================
# PUBLIC ENTRY
# =========================================================

def generate_mbom(ebom_df, inv_df=None):

    # AI draft (optional influence)
    _ = generate_ai_draft(ebom_df)

    # Deterministic validation output
    validated = validate_and_correct(_, ebom_df)

    # Strict UI formatting
    return format_output(validated)
