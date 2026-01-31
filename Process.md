> Note: The following steps represent the traditional manual EBOM-to-MBOM conversion process.  
> This project uses AI to automate and intelligently assist these steps.
> Scope Note: ERP execution (SAP/MES) is outside the scope of this POC.  
> The focus is on intelligent MBOM generation at the PLM layer.

***

# EBOM Structure: Detailed Field Definitions

The Engineering Bill of Materials (EBOM) is the foundational data set for product design. Below is a detailed explanation of the attributes required for a professional-grade EBOM.

### 1. Parent Part Number (`Parent_Part_No`)
*   **Definition:** The unique identifier of the higher-level assembly that "owns" the components.
*   **Technical Role:** It establishes the **Hierarchy**. In a database, this acts as the "Foreign Key" to link sub-parts to a main product.
*   **Business Logic:** If this field is empty, the part is considered a "Top-Level Assembly" (the final product).

### 2. Child Part Number (`Child_Part_No`)
*   **Definition:** The unique identifier for the specific component or sub-assembly being used.
*   **Technical Role:** This is the **Primary Key** for the part itself. It links directly to the Engineering Drawing and the Inventory Master Data.
*   **Business Logic:** Every physical item in the factory must have a unique `Child_Part_No` to ensure the correct item is picked for assembly.

### 3. Revision (`Revision`)
*   **Definition:** A version control code (e.g., A, B, C or 01, 02) indicating the design iteration.
*   **Technical Role:** Facilitates **Change Management**. It ensures that the factory is not building an obsolete version of a part.
*   **Business Logic:** When a part is improved (e.g., made stronger or cheaper), the Revision is "up-issued." The EBOM must always point to the "Released" revision.

### 4. Sequence Number / Find Number (`Sequence_No`)
*   **Definition:** The numerical order of the part as it appears on the Engineering Drawing.
*   **Technical Role:** It acts as a **Visual Link** between the data and the blueprint.
*   **Business Logic:** When a technician looks at a 2D drawing, they see "bubbles" with numbers. The `Sequence_No` tells them that "Bubble #10" on the drawing is "Part X" in the system.

### 5. Quantity (`Quantity`)
*   **Definition:** The exact amount of the child part required to build **one unit** of the parent assembly.
*   **Technical Role:** Used for **Demand Calculation**. (Total Needed = Production Goal × EBOM Quantity).
*   **Business Logic:** Must be precise. For hardware, it is usually an integer (e.g., 4 screws); for liquids or raw materials, it is a decimal (e.g., 0.5kg of resin).

### 6. Unit of Measure (`UoM`)
*   **Definition:** The standard unit used to count or weigh the part.
*   **Standard Values:**
    *   **EA (Each):** Used for discrete parts (screws, motors, screens).
    *   **MM/M (Millimeters/Meters):** Used for wires, pipes, or cables.
    *   **KG/G (Kilograms/Grams):** Used for raw materials, grease, or chemicals.
*   **Business Logic:** Prevents catastrophic ordering errors (e.g., ordering 10 *Kilograms* of screws instead of 10 *Individual* screws).

### 7. Part Type (`Part_Type`)
*   **Definition:** The classification of the item's physical and logical state.
*   **Classifications:**
    *   **Assembly:** A part that consists of other parts (has its own BOM).
    *   **Component:** A "leaf node" part that cannot be disassembled (e.g., a single bolt).
    *   **Phantom:** A logical grouping used in CAD to organize files. It does not exist as a physical stocked item and is "blown through" during manufacturing.

### 8. Effectivity Date (`Effectivity_Date`)
*   **Definition:** The timestamp (ISO-8601 format) defining when this part becomes active in the design.
*   **Technical Role:** Enables **Time-Based Filtering**.
*   **Business Logic:** Used for "Phase-In/Phase-Out" strategies. It allows engineers to schedule a part change for a future date (e.g., "Start using the new battery on March 1st").

### 9. Scrap Factor (`Scrap_Factor`) — *Advanced Field*
*   **Definition:** The anticipated percentage of wastage during the manufacturing process.
*   **Business Logic:** If a process typically breaks 2% of the parts, the Scrap Factor allows the system to order 102 parts to successfully build 100 units.

### 10. Alternative Part (`Alternative_Part_No`) — *Advanced Field*
*   **Definition:** A pre-approved substitute part number.
*   **Business Logic:** If the primary `Child_Part_No` is out of stock, this field tells the factory which other part is safe to use without stopping the production line.

***

# Factory Inventory: Detailed Data Specification


## Detailed Field Definitions

### 1. Part Number (`Part_Number`)
*   **Definition:** The unique identifier for the material (linked to the EBOM/MBOM).
*   **Professional Role:** Acts as the **Foreign Key**. It ensures that the physical part in the bin matches the engineering specification required for the build.
*   **Constraint:** Must be a mandatory field to prevent "anonymous" stock.

### 2. Location ID (`Location_ID`)
*   **Definition:** The specific physical coordinate of the part within the facility.
*   **Format:** Typically follows a `Site-Zone-Aisle-Rack-Bin` convention (e.g., `WH1-A01-R22-B1`).
*   **Professional Role:** Enables **Path Optimization** for picking and prevents "lost inventory" in large-scale warehouses.

### 3. Lot / Batch Number (`Lot_Batch_No`)
*   **Definition:** A tracking code for a specific production run from a supplier.
*   **Professional Role:** Essential for **Quality Traceability**. If a component fails in the field, this field allows the factory to identify and recall all other units produced in the same batch.

### 4. Serial Number (`Serial_Number`)
*   **Definition:** A unique ID for an individual unit (used for high-value assets).
*   **Professional Role:** Provides **Granular Tracking**. While screws are tracked by batch, expensive items like Motors or PCBs are tracked individually for warranty and maintenance history.

### 5. Quantity On Hand (`Qty_On_Hand`)
*   **Definition:** The total physical count of the part currently sitting in the specified `Location_ID`.
*   **Professional Role:** Represents the **Physical Truth**. This value is updated via barcode scans during receiving or line-side delivery.

### 6. Quantity Reserved (`Qty_Reserved`)
*   **Definition:** Stock that is physically present but allocated to an active Work Order.
*   **Professional Role:** Enables **Availability Logic**. 
    *   *Formula:* `Available to Promise (ATP) = Qty_On_Hand - Qty_Reserved`.
    *   This prevents the system from over-promising materials to multiple production lines.

### 7. Unit of Measure (`UoM`)
*   **Definition:** The standard unit of count (e.g., EA, KG, L, MM).
*   **Professional Role:** Ensures **Transaction Consistency**. It prevents errors such as a warehouse worker issuing 10 *boxes* of parts when the line only requested 10 *individual* pieces.

### 8. Stock Status (`Stock_Status`)
*   **Definition:** The operational state of the inventory.
*   **Standard Values:**
    *   **Available:** Ready for production.
    *   **Quarantine:** Held for Quality Assurance (QA) inspection.
    *   **Scrap:** Defective material awaiting disposal.
*   **Professional Role:** Acts as a **System Gatekeeper**, blocking the use of unverified or damaged materials.

### 9. Last Cycle Count (`Last_Cycle_Count`)
*   **Definition:** The timestamp of the last physical audit of this specific bin.
*   **Professional Role:** Ensures **Data Integrity**. Regular cycle counts reconcile the "Digital Inventory" with the "Physical Reality," identifying shrinkage or scanning errors.



***

# Technical Guide: EBOM to MBOM Manual Conversion Process


## Detailed Step-by-Step Conversion Workflow

### Step 1: Design Integrity Audit (The Handover)
Before conversion begins, the Manufacturing Engineer performs a "sanity check" on the EBOM.
*   **Action:** Verify **Part Numbers** and **Revision Levels** (e.g., Rev A to Rev B).
*   **Detail:** If the CAD model is updated but the EBOM isn't, the factory will build an obsolete version. This step ensures the "As-Designed" data is frozen and accurate.

### Step 2: Sequential Station Mapping (Routing)
Engineering groups parts by function (e.g., all screws in one folder). Manufacturing must group them by **Work Centers**.
*   **Action:** Assign every EBOM line item to a specific **Station ID** (e.g., Station 10: Frame, Station 20: Engine).
*   **Detail:** This creates the "Assembly Sequence." Parts are physically delivered only to the station where they are installed, reducing clutter on the factory floor.

### Step 3: Integration of Consumables (Non-CAD Items)
3D CAD models usually do not include "liquids" or "soft goods."
*   **Action:** Manually add Part Numbers for items like **Grease, Loctite (Glue), Solder, and Tape**.
*   **Detail:** These are essential for production. Without adding them to the MBOM, the ERP system won't trigger a purchase order, and the assembly line will stop for lack of glue or lubricant.

### Step 4: Scrap Factor & Buffer Calculation
In a perfect design, 1 part = 1 product. In a real factory, parts break or get lost.
*   **Action:** Apply a **Scrap Percentage** to high-risk or small components.
*   **Detail:** 
    *   *Formula:* `Required Qty = Design Qty * (1 + Scrap %)`
    *   *Example:* If 100 LEDs are needed and the scrap rate is 5%, the MBOM will request 105 units to ensure the target of 100 finished goods is met.

### Step 5: "Make vs. Buy" BOM Explosion
The engineer must decide how deep the BOM goes based on the factory's capability.
*   **Action:** Define if a sub-assembly is a **Purchased Part** or an **In-House Build**.
*   **Detail:** 
    *   **Buy:** The sub-assembly stays as one line item.
    *   **Make:** The sub-assembly is "exploded," listing every internal nut, bolt, and wire needed to build it on-site.

### Step 6: Packaging, Labeling, and Shipping
The EBOM ends at the product. The MBOM ends at the shipping dock.
*   **Action:** Add Part Numbers for **Product Boxes, Bubble Wrap, Pallets, and Barcode Labels**.
*   **Detail:** This ensures the finished product is protected and traceable during transport to the customer.

### Step 7: Supply Chain Logic (Lead Time & Alternatives)
The final step is adding "Intelligence" to the list to prevent delays.
*   **Action:** Map **Lead Times** and **Alternative Parts**.
*   **Detail:** 
    *   **Lead Time:** If a part takes 5 days to move from the warehouse to the line, the system needs this data to "call" for parts early.
    *   **Alternatives:** If "Part A" is out of stock, the MBOM identifies "Part B" as a pre-approved substitute to keep the line moving.

---

