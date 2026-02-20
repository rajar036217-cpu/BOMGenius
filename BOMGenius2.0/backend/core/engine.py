import os
import re
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Tuple

print("--- core/engine.py Loaded: SAP-style mBOM (WITH/WITHOUT inventory) + Consumables + Safe Aggregation ---")

# =============================
# Helpers
# =============================
def _col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in cols:
            return cols[key]
    return None

def _parse_date(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return str(datetime.strptime(s, fmt).date())
        except Exception:
            pass
    return s

def _clean_str(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s

# =============================
# Normalize eBOM
# =============================
def normalize_ebom(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    part_no = _col(df, ["part number", "part_number", "part_no", "partno"])
    part_name = _col(df, ["part name", "part_name", "description", "desc", "item name"])
    parent_asm = _col(df, ["parent assembly", "parent_assembly", "parent", "parent name", "parent_description"])
    qty = _col(df, ["quantity", "qty", "qty_per", "qty per"])
    rev = _col(df, ["revision", "rev"])
    part_type = _col(df, ["part type", "part_type", "type"])
    std_custom = _col(df, ["standard vs custom", "standard_vs_custom", "std_custom"])
    material = _col(df, ["material", "raw material"])
    valid_from = _col(df, ["valid from", "valid_from", "effective date", "start date"])

    if not part_no or not part_name:
        raise ValueError(
            f"eBOM missing required columns. Found columns: {list(df.columns)}. "
            f"Need: Part Number + Part Name/Description"
        )

    out = pd.DataFrame()
    out["Part Number"] = df[part_no].astype(str).str.strip()
    out["Part Name"] = df[part_name].astype(str).str.strip()
    out["Parent Assembly"] = df[parent_asm].astype(str).str.strip() if parent_asm else ""
    out["Quantity"] = pd.to_numeric(df[qty], errors="coerce").fillna(1).astype(int) if qty else 1
    out["Revision"] = df[rev].astype(str).str.strip() if rev else "NA"
    out["Part Type"] = df[part_type].astype(str).str.strip() if part_type else ""
    out["Standard vs Custom"] = df[std_custom].astype(str).str.strip() if std_custom else ""
    out["Material"] = df[material].astype(str).str.strip() if material else ""
    out["Valid From"] = df[valid_from].apply(_parse_date) if valid_from else ""

    out = out.replace({"nan": "", "NaN": "", "None": ""})
    return out

# =============================
# Normalize Inventory
# (supports file without Part Number too)
# =============================
def normalize_inventory(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "Part Number", "Part Name", "In_Inventory", "Stock_Qty",
            "Store_Location", "Approved_Supplier", "Lead_Time_Days"
        ])

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    pn = _col(df, ["part number", "part_number", "part_no", "partno", "item code", "item_code", "material", "partnumber"])
    pname = _col(df, ["part name", "part_name", "description", "item name"])
    qty = _col(df, ["stock_qty", "stock qty", "qty_on_hand", "on hand", "stock", "quantity"])
    loc = _col(df, ["store_location", "store location", "bin", "location", "stocking type", "stocking_type"])
    inv_flag = _col(df, ["in_inventory", "in inventory", "available", "inventory"])
    sup = _col(df, ["approved_supplier", "supplier", "vendor"])
    lead = _col(df, ["lead_time_days", "lead time days", "lead time", "leadtime"])

    out = pd.DataFrame()
    out["Part Number"] = df[pn].astype(str).str.strip() if pn else ""
    out["Part Name"] = df[pname].astype(str).str.strip() if pname else ""

    out["Stock_Qty"] = pd.to_numeric(df[qty], errors="coerce").fillna(0).astype(int) if qty else 0
    out["Store_Location"] = df[loc].astype(str).str.strip() if loc else ""
    out["Approved_Supplier"] = df[sup].astype(str).str.strip() if sup else ""
    out["Lead_Time_Days"] = pd.to_numeric(df[lead], errors="coerce").fillna(0).astype(int) if lead else 0

    if inv_flag:
        out["In_Inventory"] = df[inv_flag].astype(str).str.strip()
        out["In_Inventory"] = out["In_Inventory"].replace({"1": "Yes", "0": "No", "TRUE": "Yes", "FALSE": "No"})
    else:
        out["In_Inventory"] = out["Stock_Qty"].apply(lambda x: "Yes" if int(x) > 0 else "No")

    out = out.replace({"nan": "", "NaN": "", "None": ""})
    return out

# =============================
# Manufacturing Rules (Deterministic)
# =============================
ELECTRONICS_KW = [
    "pcb", "board", "ic", "capacitor", "resistor", "diode", "mosfet", "controller",
    "transformer", "inductor", "connector", "plug", "cable", "wire", "fuse", "rectifier", "opto"
]

def node_type_from_part_type(part_type: str) -> str:
    pt = (part_type or "").lower()
    if pt == "assembly" or "assembly" in pt:
        return "Assembly"
    if "sub" in pt:
        return "Sub-Assembly"
    return "Component"

def makebuy_rule(part_type: str, std_custom: str, part_name: str) -> str:
    pt = (part_type or "").lower()
    sc = (std_custom or "").lower()
    nm = (part_name or "").lower()

    if "assembly" in pt:
        return "Make"
    if "mechanical" in pt and sc == "custom":
        return "Make"
    if sc == "standard":
        return "Buy"
    if any(k in nm for k in ELECTRONICS_KW):
        return "Buy"
    return "Make" if "mechanical" in pt else "Buy"

def work_center_rule(make_buy: str, part_type: str, part_name: str) -> str:
    pt = (part_type or "").lower()
    nm = (part_name or "").lower()

    if make_buy == "Make":
        if "assembly" in pt:
            return "Final Assembly Line"
        if "mechanical" in pt:
            return "Injection Molding / Mechanical Assembly"
        if "pcb" in nm or "board" in nm:
            return "SMT Line"
        return "Manufacturing Cell"

    # Buy
    if any(k in nm for k in ["pcb", "ic", "capacitor", "resistor", "diode", "transformer", "connector"]):
        return "Incoming Inspection (Electronics)"
    return "Incoming Inspection (General)"

def procurement_steps_rule(make_buy: str, inventory_status: Optional[str] = None) -> str:
    if make_buy == "Buy":
        if inventory_status == "Available in Stock":
            return "Material Issue from Stores -> Line Supply"
        return "Vendor Selection -> PR -> PO -> GRN -> Incoming QC -> Putaway -> Issue to Line"
    return "Issue Components -> Manufacture/Assemble -> In-process QC -> Final Test -> FG Receipt"

def routing_embedded_rule(make_buy: str, node_type: str, level: int, work_center: str) -> str:
    if make_buy == "Buy":
        return "10: Create PR/PO | 20: Incoming Inspection | 30: Putaway/Issue"
    if node_type in ["Assembly", "Sub-Assembly"] or level <= 1:
        return f"10: Kitting/Issue | 20: Assembly ({work_center}) | 30: Functional Test | 40: Packing"
    return f"10: Material Issue | 20: Primary Process ({work_center}) | 30: Finishing | 40: In-Process Inspection"

# =============================
# Consumables (AI-ish but safe)
# =============================
def heuristic_consumables(part_name: str, material: str) -> str:
    d = (part_name or "").lower()
    m = (material or "").lower()

    # basic, super stable rules (no hallucination)
    if any(x in d for x in ["pcb", "board"]):
        return "Solder"
    if any(x in d for x in ["wire", "cable"]):
        return "Cable Tie"
    if any(x in d for x in ["housing", "shell", "cover", "case"]):
        return "Adhesive"
    if any(x in d for x in ["screw", "bolt", "nut", "washer"]):
        return "Thread Locker"
    if "plastic" in m and any(x in d for x in ["housing", "shell", "cover"]):
        return "Adhesive"
    if "metal" in m and any(x in d for x in ["screw", "bolt"]):
        return "Lubricant"
    return "NA"

def llm_consumables_optional(part_name: str, material: str) -> str:
    """
    Optional LLM (Ollama) call.
    Toggle with env: USE_LLM_CONSUMABLES=1
    Keeps token cost under control; fallback always heuristic.
    """
    if os.getenv("USE_LLM_CONSUMABLES", "0") != "1":
        return heuristic_consumables(part_name, material)

    try:
        import ollama
        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        prompt = (
            "You are a manufacturing engineer.\n"
            f"Item: {part_name}\n"
            f"Material: {material}\n"
            "Return ONE consumable word needed for assembly (Glue/Adhesive, Grease, Solder, Cable Tie, Threadlocker).\n"
            "If none, return NA.\n"
            "Answer with ONLY the word."
        )
        resp = ollama.generate(model=model, prompt=prompt, options={"temperature": 0.0, "num_predict": 6})
        ans = str(resp.get("response", "")).strip()
        ans = re.sub(r"[^A-Za-z ]", "", ans).strip()
        if not ans:
            return heuristic_consumables(part_name, material)
        # normalize common outputs
        ans_low = ans.lower()
        if "glue" in ans_low or "adhes" in ans_low:
            return "Adhesive"
        if "solder" in ans_low:
            return "Solder"
        if "tie" in ans_low:
            return "Cable Tie"
        if "grease" in ans_low or "lub" in ans_low:
            return "Lubricant"
        if "thread" in ans_low or "locker" in ans_low:
            return "Thread Locker"
        if ans.upper() == "NA":
            return "NA"
        return ans.title()
    except Exception:
        return heuristic_consumables(part_name, material)

# =============================
# Hierarchy build
# =============================
def build_parent_child(ebom: pd.DataFrame) -> pd.DataFrame:
    ebom = ebom.copy()

    # map name -> part number (first occurrence)
    name_to_pn: Dict[str, str] = {}
    for _, r in ebom.iterrows():
        pn = _clean_str(r.get("Part Number"))
        nm = _clean_str(r.get("Part Name"))
        if pn and nm and nm not in name_to_pn:
            name_to_pn[nm] = pn

    def resolve_parent_pn(parent_asm_name: str) -> str:
        p = _clean_str(parent_asm_name)
        if not p:
            return ""
        return name_to_pn.get(p, "")

    ebom["Parent Part Number"] = ebom["Parent Assembly"].apply(resolve_parent_pn)
    return ebom

def compute_levels(ebom: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, str], List[str]]:
    children: Dict[str, List[str]] = {}
    for _, r in ebom.iterrows():
        child = _clean_str(r.get("Part Number"))
        parent = _clean_str(r.get("Parent Part Number"))
        if parent:
            children.setdefault(parent, []).append(child)

    roots = (
        ebom.loc[ebom["Parent Part Number"].astype(str).str.strip() == "", "Part Number"]
        .astype(str).str.strip().tolist()
    )

    level: Dict[str, int] = {}
    path: Dict[str, str] = {}

    def dfs(node: str, lvl: int, stack: List[str]):
        if node in stack:
            return
        new_stack = stack + [node]
        if node not in level or lvl < level[node]:
            level[node] = lvl
            path[node] = " > ".join(new_stack)
        for ch in children.get(node, []):
            dfs(ch, lvl + 1, new_stack)

    for rt in roots:
        if rt:
            dfs(rt, 0, [])

    # orphan safety
    for pn in ebom["Part Number"].astype(str).str.strip().tolist():
        if pn and pn not in level:
            dfs(pn, 0, [])

    return level, path, roots

# =============================
# Aggregation (rollup) - SAFE
# =============================
def smart_rollup_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine rows ONLY within same parent + same manufacturing classification.
    - Qty summed
    - Child Part Number joined (unique)
    - Hierarchy Path keep first
    """
    if df is None or df.empty:
        return df

    # ensure consumables exists for groupby
    if "Consumables" not in df.columns:
        df["Consumables"] = "NA"

    group_cols = [
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
    ]
    group_cols = [c for c in group_cols if c in df.columns]  # safety

    # build agg dict safely
    agg_dict = {
        "Qty": "sum",
        "Child Part Number": lambda x: ", ".join(pd.Series(x).astype(str).str.strip().replace("nan", "").unique()),
        "Hierarchy Path": "first",
        "Revision": "first",
        "Effective Date": "first",
    }

    # optional inventory cols
    for col in ["Inventory Status", "Stock_Qty", "Store_Location", "Procurement Action", "Approved_Supplier", "Lead_Time_Days"]:
        if col in df.columns:
            agg_dict[col] = "first"

    # Parent info should be stable (already in group_cols mostly)
    out = df.groupby(group_cols, as_index=False).agg(agg_dict)

    # Keep Parent Part Number empty display clean
    if "Parent Part Number" in out.columns:
        out["Parent Part Number"] = out["Parent Part Number"].replace({"nan": ""}).fillna("")

    return out

# =============================
# Generate mBOM
# =============================
def generate_mbom(ebom_df: pd.DataFrame, inv_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    ebom = normalize_ebom(ebom_df)
    ebom = build_parent_child(ebom)
    level_map, path_map, roots = compute_levels(ebom)

    pn_to_name = dict(zip(ebom["Part Number"], ebom["Part Name"]))

    inv_norm = normalize_inventory(inv_df) if (inv_df is not None and not inv_df.empty) else pd.DataFrame()

    inventory_active = (
        inv_norm is not None
        and not inv_norm.empty
        and (("Part Number" in inv_norm.columns and (inv_norm["Part Number"] != "").any())
             or ("Part Name" in inv_norm.columns and (inv_norm["Part Name"] != "").any()))
    )

    def inv_logic(child_pn: str, child_name: str, mb: str):
        if not inventory_active:
            return None

        if mb == "Make":
            return ("N/A (Manufactured)", 0, "", "Produce In-house", "", 0)

        # Try by PN
        rec = pd.DataFrame()
        if "Part Number" in inv_norm.columns:
            rec = inv_norm[inv_norm["Part Number"].astype(str).str.strip() == str(child_pn).strip()]

        # fallback by Name
        if rec.empty and "Part Name" in inv_norm.columns:
            rec = inv_norm[
                inv_norm["Part Name"].astype(str).str.lower().str.strip()
                == str(child_name).lower().strip()
            ]

        if rec.empty:
            return ("Unknown (Not in Inventory List)", 0, "", "Trigger PR", "", 0)

        row = rec.iloc[0]  # duplicates-safe

        in_inv = str(row.get("In_Inventory", "")).strip().lower()
        stock = int(row.get("Stock_Qty", 0) or 0)
        loc = str(row.get("Store_Location", "")).strip()
        sup = str(row.get("Approved_Supplier", "")).strip()
        lead = int(row.get("Lead_Time_Days", 0) or 0)

        if in_inv in ["yes", "y", "true", "1"] and stock > 0:
            return ("Available in Stock", stock, loc, "Issue from Stores", sup, lead)

        return ("Not Available", stock, loc, "Trigger PR", sup, lead)

    def add_inventory_cols(base: dict, inv_tuple):
        if inv_tuple is None:
            return base
        inv_status, stock, loc, action, sup, lead = inv_tuple
        base.update({
            "Inventory Status": inv_status,
            "Stock_Qty": stock,
            "Store_Location": loc,
            "Procurement Action": action,
            "Approved_Supplier": sup,
            "Lead_Time_Days": lead,
        })
        return base

    rows = []

    # ========== Roots ==========
    for rt in roots:
        r0 = ebom[ebom["Part Number"] == rt]
        part_type = _clean_str(r0["Part Type"].iloc[0]) if not r0.empty else ""
        stdc = _clean_str(r0["Standard vs Custom"].iloc[0]) if not r0.empty else ""
        eff = _clean_str(r0["Valid From"].iloc[0]) if not r0.empty else ""
        rev = _clean_str(r0["Revision"].iloc[0]) if not r0.empty else "NA"
        mat = _clean_str(r0["Material"].iloc[0]) if not r0.empty else ""

        lvl = int(level_map.get(rt, 0))
        name = _clean_str(pn_to_name.get(rt, ""))
        mb = makebuy_rule(part_type, stdc, name)
        nt = node_type_from_part_type(part_type)
        wc = work_center_rule(mb, part_type, name)

        inv_tuple = inv_logic(rt, name, mb)
        cons = llm_consumables_optional(name, mat)

        base = {
            "Level": lvl,
            "Parent Part Number": "",
            "Parent Description": "",
            "Child Part Number": rt,
            "Child Description": name,
            "Qty": 1,
            "UOM": "EA",
            "Revision": rev,
            "Node Type": nt,
            "Make/Buy": mb,
            "Work Center": wc,
            "Effective Date": eff,
            "Procurement Steps": procurement_steps_rule(mb, inv_tuple[0] if inv_tuple else None),
            "Operations (Routing Embedded)": routing_embedded_rule(mb, nt, lvl, wc),
            "Consumables": cons,
            "Hierarchy Path": path_map.get(rt, rt),
        }
        rows.append(add_inventory_cols(base, inv_tuple))

    # ========== Edges ==========
    for _, r in ebom.iterrows():
        child = _clean_str(r.get("Part Number"))
        parent = _clean_str(r.get("Parent Part Number"))
        if not parent:
            continue

        lvl = int(level_map.get(child, 0))
        child_name = _clean_str(r.get("Part Name"))
        part_type = _clean_str(r.get("Part Type"))
        stdc = _clean_str(r.get("Standard vs Custom"))
        mat = _clean_str(r.get("Material"))
        eff = _clean_str(r.get("Valid From"))
        rev = _clean_str(r.get("Revision")) or "NA"
        qty = int(r.get("Quantity", 1) or 1)

        mb = makebuy_rule(part_type, stdc, child_name)
        nt = node_type_from_part_type(part_type)
        wc = work_center_rule(mb, part_type, child_name)

        inv_tuple = inv_logic(child, child_name, mb)
        cons = llm_consumables_optional(child_name, mat)

        base = {
            "Level": lvl,
            "Parent Part Number": parent,
            "Parent Description": _clean_str(pn_to_name.get(parent, "")),
            "Child Part Number": child,
            "Child Description": child_name,
            "Qty": qty,
            "UOM": "EA",
            "Revision": rev,
            "Node Type": nt,
            "Make/Buy": mb,
            "Work Center": wc,
            "Effective Date": eff,
            "Procurement Steps": procurement_steps_rule(mb, inv_tuple[0] if inv_tuple else None),
            "Operations (Routing Embedded)": routing_embedded_rule(mb, nt, lvl, wc),
            "Consumables": cons,
            "Hierarchy Path": path_map.get(child, child),
        }
        rows.append(add_inventory_cols(base, inv_tuple))

    df = pd.DataFrame(rows).fillna("")

    # ✅ Rollup aggregation (this fixes Qty sum + merge by description)
    df = smart_rollup_aggregation(df)

    # ✅ Sort stable for UI
    if "Hierarchy Path" in df.columns:
        df["_root"] = df["Hierarchy Path"].astype(str).str.split(" > ").str[0]
        df = df.sort_values(["_root", "Level", "Parent Part Number", "Child Description"]).drop(columns=["_root"])

    return df