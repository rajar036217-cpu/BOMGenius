# Manufacturing BOM (MBOM) Data Structure

This document defines the schema and logic for the Manufacturing Bill of Materials. It details the specific fields required to bridge Engineering Design (EBOM) with Factory Logistics and Production Planning.

---

### 1. Manufacturing Part Number (`Mfg_Part_No`)
*   **Definition:** The unique alphanumeric identifier used to track the physical item on the factory floor and in the warehouse.
*   **Technical Role:** This is the **Primary Key** for the ERP (Enterprise Resource Planning) system. It must match the Inventory Master Data exactly.
*   **Business Logic:** While Engineering might call a part `CAD-101`, Manufacturing might stock it as `STEEL-ROD-5MM`. This field represents the *stocked* item.

### 2. Make/Buy Flag (`Make_Buy_Code`)
*   **Definition:** A classification code indicating whether the item is manufactured internally or purchased from an external vendor.
*   **Technical Role:** The "Switch" for the planning engine.
    *   **'M' (Make):** Triggers a **Work Order** (Production Job).
    *   **'B' (Buy):** Triggers a **Purchase Order** (Procurement).
*   **Business Logic:** If an item is a "Make," the system looks for a child BOM (sub-components). If it is a "Buy," the system looks at the Vendor List and Inventory On-Hand.

### 3. Operation Sequence Number (`Op_Sequence`)
*   **Definition:** A numerical value (e.g., 10, 20, 30) representing the chronological step in the manufacturing process where this part is consumed.
*   **Technical Role:** Defines the **Time** aspect of the BOM. It ensures Just-In-Time (JIT) delivery.
*   **Business Logic:** Parts required for "Final Packaging" (Seq 90) should not be delivered to the "Welding Station" (Seq 10). This prevents clutter on the shop floor.

### 4. Work Center ID (`Work_Center`)
*   **Definition:** The specific location, machine, or station in the factory where the assembly or processing takes place.
*   **Technical Role:** Links the BOM to **Capacity Planning**. It helps calculate labor costs and machine utilization.
*   **Business Logic:** If `Work_Center = PAINT_BOOTH`, the system knows this step requires specific resources (painters, sprayers) and cannot happen at the `ASSEMBLY_BENCH`.

### 5. BOM Level (`BOM_Level`)
*   **Definition:** An integer indicating the depth of the part in the assembly hierarchy.
*   **Technical Role:** Used for **Tree Traversal** algorithms.
    *   `0` = Finished Good.
    *   `1` = Major Sub-Assembly.
    *   `2` = Component.
*   **Business Logic:** The ERP system uses this to schedule production. Level 2 parts must be built/bought *before* Level 1 parts can be started.

### 6. Inventory Location (`Bin_Location`)
*   **Definition:** The precise physical address (Aisle, Rack, Shelf, Bin) where the picker can find the item.
*   **Technical Role:** Populates the **Pick List**. It maps the digital record to physical space.
*   **Business Logic:** Essential for efficiency. Without this, workers waste time searching for parts. If a part is "Floor Stock," the location might be the specific Work Center.

### 7. Backflush Indicator (`Backflush_Ind`)
*   **Definition:** A Boolean flag (True/False) determining *how* inventory is deducted from the system.
*   **Technical Role:** Automates inventory transactions.
*   **Business Logic:**
    *   **True:** Inventory is automatically deducted when the parent assembly is reported complete (used for cheap items like screws/washers).
    *   **False:** Inventory must be manually scanned/issued before the job starts (used for expensive items like Engines or Gold).

### 8. Item Type (`MBOM_Item_Type`)
*   **Definition:** Categorizes the item based on its manufacturing behavior.
*   **Technical Role:** Determines how the system treats the line item.
    *   **Standard:** A physical part tracked in inventory.
    *   **Consumable:** Items like Glue, Tape, or Grease (often added in MBOM but missing in EBOM).
    *   **Phantom:** A logical grouping of parts that is assembled and immediately consumed (never stocked).
*   **Business Logic:** Allows the MBOM to include necessary items that engineers don't draw in CAD (e.g., "5 grams of Loctite").

### 9. Total Required Quantity (`Total_Qty_Req`)
*   **Definition:** The calculated quantity needed, including the base quantity plus anticipated waste.
*   **Technical Role:** `Formula: Quantity * (1 + Scrap_Factor)`. This is the number sent to the Purchasing Department.
*   **Business Logic:** If you need 100 stickers and the machine usually ruins 5%, the MBOM demands 105 stickers to ensure the production run doesn't stop short.

### 10. EBOM Reference ID (`EBOM_Ref_ID`)
*   **Definition:** The original Engineering Part Number that corresponds to this Manufacturing Part.
*   **Technical Role:** Establishes the **Digital Thread** and Traceability.
*   **Business Logic:** If a manufacturing defect is found, this field allows the Quality Team to trace the issue back to the original Engineering Design and CAD drawings.
