import os
import re
import json
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Tuple

print("--- core/engine.py Loaded: SAP-style mBOM (WITH/WITHOUT inventory) + Consumables + Safe Aggregation ---")


GLOBAL_RULES_PATH = "federated/global_rules.json"

def load_global_rules():
    if os.path.exists(GLOBAL_RULES_PATH):
        with open(GLOBAL_RULES_PATH, "r") as f:
            return json.load(f)
    return {}

GLOBAL_RULES = load_global_rules()


# =========================================================
# Helpers
# =========================================================
def _clean_str(x) -> str:
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def smart_find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find best matching column from candidates (case-insensitive, ignores spaces/_)."""
    norm = {}
    for c in df.columns:
        key = re.sub(r"[\s_]+", "", str(c).strip().lower())
        norm[key] = c

    for cand in candidates:
        key = re.sub(r"[\s_]+", "", str(cand).strip().lower())
        if key in norm:
            return norm[key]

    # fallback: contains match
    for cand in candidates:
        key = re.sub(r"[\s_]+", "", str(cand).strip().lower())
        for k, real in norm.items():
            if key in k:
                return real
    return None


def _parse_date(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return str(datetime.strptime(s, fmt).date())
        except Exception:
            pass
    return s


# =========================================================
# Normalize eBOM
# =========================================================
def normalize_ebom(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    part_no = smart_find_column(df, ["part number", "part_no", "partno", "part_number", "child_part_no"])
    part_name = smart_find_column(df, ["part name", "description", "desc", "item name", "part_name"])
    parent_asm = smart_find_column(df, ["parent assembly", "parent", "parent_description", "parent_assembly"])
    qty = smart_find_column(df, ["quantity", "qty", "qty_per", "qty per", "amount"])
    rev = smart_find_column(df, ["revision", "rev"])
    part_type = smart_find_column(df, ["part type", "type", "part_type"])
    std_custom = smart_find_column(df, ["standard vs custom", "standard_vs_custom", "std_custom"])
    valid_from = smart_find_column(df, ["valid from", "valid_from", "effective date", "start date"])

    if not part_no or not part_name:
        raise ValueError(
            f"eBOM missing required columns. Found columns: {list(df.columns)}. "
            f"Need: Part Number + Part Name/Description"
        )

    out = pd.DataFrame()
    out["Part Number"] = df[part_no].apply(_clean_str)
    out["Part Name"] = df[part_name].apply(_clean_str)
    out["Parent Assembly"] = df[parent_asm].apply(_clean_str) if parent_asm else ""
    out["Quantity"] = pd.to_numeric(df[qty], errors="coerce").fillna(1).astype(int) if qty else 1
    out["Revision"] = df[rev].apply(_clean_str) if rev else "NA"
    out["Part Type"] = df[part_type].apply(_clean_str) if part_type else ""
    out["Standard vs Custom"] = df[std_custom].apply(_clean_str) if std_custom else ""
    out["Valid From"] = df[valid_from].apply(_parse_date) if valid_from else ""

    return out.fillna("")


# =========================================================
# Normalize Inventory (flexible)
# =========================================================
def normalize_inventory(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "Part Number", "Part Name", "In_Inventory", "Stock_Qty",
            "Store_Location", "Approved_Supplier", "Lead_Time_Days"
        ])

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    pn = smart_find_column(df, ["part number", "part_no", "partno", "part_number", "item code", "item_code", "material"])
    pname = smart_find_column(df, ["part name", "description", "item name", "part_name"])
    qty = smart_find_column(df, ["stock_qty", "stock qty", "qty_on_hand", "on hand", "stock", "quantity"])
    loc = smart_find_column(df, ["store_location", "store location", "bin", "location", "stocking type", "stocking_type"])
    inv_flag = smart_find_column(df, ["in_inventory", "in inventory", "available", "inventory"])
    sup = smart_find_column(df, ["approved_supplier", "supplier", "vendor"])
    lead = smart_find_column(df, ["lead_time_days", "lead time days", "lead time", "leadtime"])

    out = pd.DataFrame()
    out["Part Number"] = df[pn].apply(_clean_str) if pn else ""
    out["Part Name"] = df[pname].apply(_clean_str) if pname else ""
    out["Stock_Qty"] = pd.to_numeric(df[qty], errors="coerce").fillna(0).astype(int) if qty else 0
    out["Store_Location"] = df[loc].apply(_clean_str) if loc else ""
    out["Approved_Supplier"] = df[sup].apply(_clean_str) if sup else ""
    out["Lead_Time_Days"] = pd.to_numeric(df[lead], errors="coerce").fillna(0).astype(int) if lead else 0

    if inv_flag:
        raw = df[inv_flag].astype(str).str.strip()
        raw = raw.replace({"1": "Yes", "0": "No", "TRUE": "Yes", "FALSE": "No", "true": "Yes", "false": "No"})
        out["In_Inventory"] = raw
    else:
        out["In_Inventory"] = out["Stock_Qty"].apply(lambda x: "Yes" if int(x) > 0 else "No")

    return out.fillna("")


# =========================================================
# Rules (Manufacturing)
# =========================================================
ELECTRONICS_KW = [
    "pcb", "board", "ic", "capacitor", "resistor", "diode", "mosfet", "controller",
    "transformer", "inductor", "connector", "plug", "cable", "wire", "fuse"
]


def node_type_from_part_type(part_type: str) -> str:
    pt = (part_type or "").lower()
    if "assembly" in pt and "sub" not in pt:
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
    return "Buy"


def work_center_rule(make_buy: str, part_type: str, part_name: str) -> str:
    pt = (part_type or "").lower()
    nm = (part_name or "").lower()

    if make_buy == "Make":
        if "assembly" in pt:
            return "Final Assembly Line"
        if "mechanical" in pt:
            return "Mechanical Assembly Cell"
        if "pcb" in nm:
            return "SMT Line"
        return "Manufacturing Cell"

    if any(k in nm for k in ["pcb", "ic", "capacitor", "resistor", "diode", "transformer", "connector"]):
        return "Incoming Inspection (Electronics)"
    return "Incoming Inspection (General)"


def procurement_steps_rule(make_buy: str, inventory_status: Optional[str]) -> str:
    if make_buy == "Buy":
        if inventory_status == "Available in Stock":
            return "Material Issue from Stores -> Line Supply"
        return "Vendor Selection -> PR -> PO -> GRN -> Incoming QC -> Putaway -> Issue to Line"
    return "Issue Components -> Assemble -> In-process QC -> Final Test -> FG Receipt"


def routing_embedded_rule(make_buy: str, node_type: str, level: int, work_center: str) -> str:
    if make_buy == "Buy":
        return "10: PR/PO | 20: Incoming Inspection | 30: Putaway/Issue"
    if node_type in ["Assembly", "Sub-Assembly"] or level <= 1:
        return f"10: Kitting | 20: Assembly ({work_center}) | 30: Functional Test | 40: Packing"
    return f"10: Material Issue | 20: Primary Process ({work_center}) | 30: Finishing | 40: In-Process Inspection"


# =========================================================
# Consumables (Heuristic + optional LLM via Ollama)
# =========================================================
def heuristic_consumables(part_name: str, material: str) -> str:
    desc = (part_name or "").lower()
    mat = (material or "").lower()

    if "pcb" in desc or "board" in desc:
        return "Solder"
    if "wire" in desc or "cable" in desc:
        return "Cable Tie"
    if "housing" in desc or "shell" in desc or "plastic" in mat:
        return "Adhesive"
    if "screw" in desc or "bolt" in desc or "metal" in mat:
        return "Thread Locker"
    return "NA"


def llm_consumables_optional(part_name: str, material: str) -> str:
    """
    Optional LLM (Ollama) call.
    Enable only if you want: set env USE_LLM_CONSUMABLES=1
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
            "Return ONE consumable needed for assembly: Adhesive, Solder, Cable Tie, Thread Locker, Lubricant.\n"
            "If none, return NA.\n"
            "Answer with ONLY the word."
        )
        resp = ollama.generate(model=model, prompt=prompt, options={"temperature": 0.0, "num_predict": 6})
        ans = str(resp.get("response", "")).strip()
        ans = re.sub(r"[^A-Za-z ]", "", ans).strip()
        if not ans:
            return heuristic_consumables(part_name, material)

        low = ans.lower()
        if "glue" in low or "adhes" in low:
            return "Adhesive"
        if "solder" in low:
            return "Solder"
        if "tie" in low:
            return "Cable Tie"
        if "thread" in low or "locker" in low:
            return "Thread Locker"
        if "lub" in low or "grease" in low:
            return "Lubricant"
        if low == "na":
            return "NA"
        return ans.title()
    except Exception:
        return heuristic_consumables(part_name, material)


# =========================================================
# Hierarchy build (Parent PN from Parent Assembly name)
# =========================================================
def build_parent_child(ebom: pd.DataFrame) -> pd.DataFrame:
    ebom = ebom.copy()

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

    for pn in ebom["Part Number"].astype(str).str.strip().tolist():
        if pn and pn not in level:
            dfs(pn, 0, [])

    return level, path, roots


# =========================================================
# Aggregation (rollup)
# =========================================================
def smart_rollup_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

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
    group_cols = [c for c in group_cols if c in df.columns]

    def join_unique(x):
        s = pd.Series(x).astype(str).str.strip()
        s = s.replace({"nan": "", "None": ""})
        uniq = [u for u in s.unique().tolist() if u]
        return ", ".join(uniq)

    agg_dict = {
        "Qty": "sum",
        "Child Part Number": join_unique,
        "Hierarchy Path": "first",
        "Revision": "first",
        "Effective Date": "first",
    }

    # inventory cols if present
    for c in ["Inventory Status", "Stock_Qty", "Store_Location", "Procurement Action", "Approved_Supplier", "Lead_Time_Days"]:
        if c in df.columns:
            agg_dict[c] = "first"

    out = df.groupby(group_cols, as_index=False).agg(agg_dict)
    return out.fillna("")


# =========================================================
# Generate mBOM (single entry point)
# =========================================================
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

        rec = pd.DataFrame()
        if "Part Number" in inv_norm.columns:
            rec = inv_norm[inv_norm["Part Number"].astype(str).str.strip() == str(child_pn).strip()]

        if rec.empty and "Part Name" in inv_norm.columns:
            rec = inv_norm[
                inv_norm["Part Name"].astype(str).str.lower().str.strip()
                == str(child_name).lower().strip()
            ]

        if rec.empty:
            return ("Unknown (Not in Inventory List)", 0, "", "Trigger PR", "", 0)

        row = rec.iloc[0]
        in_inv = str(row.get("In_Inventory", "")).strip().lower()
        stock = int(row.get("Stock_Qty", 0) or 0)
        loc = _clean_str(row.get("Store_Location", ""))
        sup = _clean_str(row.get("Approved_Supplier", ""))
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

    # Root rows (Level 0)
    for rt in roots:
        r0 = ebom[ebom["Part Number"] == rt]
        part_type = _clean_str(r0["Part Type"].iloc[0]) if not r0.empty else ""
        stdc = _clean_str(r0["Standard vs Custom"].iloc[0]) if not r0.empty else ""
        eff = _clean_str(r0["Valid From"].iloc[0]) if not r0.empty else ""
        rev = _clean_str(r0["Revision"].iloc[0]) if not r0.empty else "NA"

        lvl = level_map.get(rt, 0)
        name = pn_to_name.get(rt, "")
        name = GLOBAL_RULES.get(name, name)
        mb = makebuy_rule(part_type, stdc, name)
        nt = node_type_from_part_type(part_type)
        wc = work_center_rule(mb, part_type, name)

        inv_tuple = inv_logic(rt, name, mb)
        cons = llm_consumables_optional(name, "")  # no material column in ebom by default

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

    # Parent-child edges
    for _, r in ebom.iterrows():
        child = _clean_str(r.get("Part Number"))
        parent = _clean_str(r.get("Parent Part Number"))
        if not parent:
            continue

        lvl = level_map.get(child, 0)
        child_name = _clean_str(r.get("Part Name"))
        child_name = GLOBAL_RULES.get(child_name, child_name)
        part_type = _clean_str(r.get("Part Type"))
        stdc = _clean_str(r.get("Standard vs Custom"))
        rev = _clean_str(r.get("Revision")) or "NA"
        eff = _clean_str(r.get("Valid From"))
        qty = int(r.get("Quantity", 1) or 1)

        mb = makebuy_rule(part_type, stdc, child_name)
        nt = node_type_from_part_type(part_type)
        wc = work_center_rule(mb, part_type, child_name)

        inv_tuple = inv_logic(child, child_name, mb)
        cons = llm_consumables_optional(child_name, "")  # add material if you want later

        base = {
            "Level": lvl,
            "Parent Part Number": parent,
            "Parent Description": pn_to_name.get(parent, ""),
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

    # sort like SAP-ish view
    df["_root"] = df["Hierarchy Path"].astype(str).str.split(" > ").str[0]
    df = df.sort_values(["_root", "Level", "Parent Part Number", "Child Part Number"]).drop(columns=["_root"])

    # ✅ aggregation (rollup)
    df = smart_rollup_aggregation(df)

    return df.fillna("")