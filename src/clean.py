import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. FILE & DIRECTORY CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, "mp_full_dataset.csv")
OUTPUT_FILE_GUIDE = os.path.join(DATA_DIR, "cleaned_voltage_dataset.csv")
OUTPUT_FILE_COMPAT = os.path.join(DATA_DIR, "cleaned_dataset.csv")

# -----------------------------------------------------------------------------
# 2. STEP 3: RAW DATASET INSPECTION
# -----------------------------------------------------------------------------
def inspect_and_clean():
    print("=" * 70)
    print("BATTERY VOLTAGE PREDICTION - STEP 3 & STEP 4: INSPECT & CLEAN")
    print("=" * 70)
    print(f"Loading raw dataset from: {INPUT_FILE} ...")
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found at: {INPUT_FILE}")

    df_raw = pd.read_csv(INPUT_FILE, low_memory=False)
    raw_rows, raw_cols = df_raw.shape
    
    print("\n--- STEP 3: RAW DATASET INSPECTION REPORT ---")
    print(f"Total Raw Records       : {raw_rows}")
    print(f"Total Raw Columns       : {raw_cols}")
    
    # Audit empty rows
    completely_empty_rows = int(df_raw.isna().all(axis=1).sum())
    print(f"Completely Empty Rows   : {completely_empty_rows}")
    
    # Audit Working Ion
    raw_ions = df_raw["Working Ion"].value_counts(dropna=False)
    print(f"Blank / Missing Ions    : {df_raw['Working Ion'].isna().sum()}")
    print("\nRaw Working Ion Distribution:")
    for ion, count in raw_ions.items():
        ion_label = "NaN / Missing" if pd.isna(ion) else str(ion)
        print(f"  {ion_label:<15}: {count:>6}")

    # Audit Voltage in Raw Data
    voltage_col = "Average Voltage (V)"
    if voltage_col not in df_raw.columns:
        raise KeyError(f"Target column '{voltage_col}' not found in raw dataset.")
    
    raw_voltages = pd.to_numeric(df_raw[voltage_col], errors="coerce")
    raw_valid_v = raw_voltages.dropna()
    print(f"\nRaw Voltage Range       : Min = {raw_valid_v.min():.4f} V, Max = {raw_valid_v.max():.4f} V")
    print(f"Raw Negative Voltages   : {(raw_valid_v <= 0).sum()} (unphysical for battery intercalation)")
    print(f"Raw Outliers (> 10 V)   : {(raw_valid_v > 10).sum()} (DFT calculation artifacts)")

    # -------------------------------------------------------------------------
    # 3. STEP 4: DATA CLEANING PIPELINE ACCORDING TO PAPER NORMS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("--- STEP 4: DATA CLEANING (BASE PAPER NORMS) ---")
    print("=" * 70)
    
    # Norm 1: Remove completely empty rows
    df = df_raw.dropna(how="all").copy()
    removed_empty = raw_rows - len(df)
    print(f"1. Removed completely empty rows           : {removed_empty:>5} rows removed")

    # Norm 2: Remove rows where Working Ion is blank/missing
    working_ion = df["Working Ion"].astype("string").str.strip()
    valid_ion_mask = working_ion.notna() & (working_ion != "")
    removed_blank_ion = int((~valid_ion_mask).sum())
    df = df[valid_ion_mask].copy()
    print(f"2. Removed rows with blank Working Ion      : {removed_blank_ion:>5} rows removed")

    # Norm 3: Physical Voltage Filtering
    # Insertion battery cathode voltages must be strictly positive (V > 0).
    # Voltages <= 0 V mean plating rather than intercalation (thermodynamically invalid).
    # Voltages > 10 V are DFT artifacts/outliers well beyond electrochemical stability.
    df[voltage_col] = pd.to_numeric(df[voltage_col], errors="coerce")
    null_voltage = int(df[voltage_col].isna().sum())
    df = df.dropna(subset=[voltage_col]).copy()
    
    negative_voltage_mask = df[voltage_col] <= 0.0
    removed_negative_v = int(negative_voltage_mask.sum())
    
    high_voltage_mask = df[voltage_col] > 10.0
    removed_high_v = int(high_voltage_mask.sum())
    
    df = df[(df[voltage_col] > 0.0) & (df[voltage_col] <= 10.0)].copy()
    print(f"3. Removed null voltages                   : {null_voltage:>5} rows removed")
    print(f"4. Removed negative voltages (<= 0 V)      : {removed_negative_v:>5} rows removed")
    print(f"5. Removed extreme voltage outliers (> 10 V): {removed_high_v:>5} rows removed")

    # Norm 4: Handle duplicates while preserving distinct electrode records
    dup_candidates = ["Electrode_data_material_id", "Formula (Charged State)", "Formula (Discharged State)", "Working Ion", voltage_col]
    dup_subset = [c for c in dup_candidates if c in df.columns]
    exact_duplicates = int(df.duplicated(subset=dup_subset).sum())
    if exact_duplicates > 0:
        df = df.drop_duplicates(subset=dup_subset).copy()
    print(f"6. Removed exact duplicate records         : {exact_duplicates:>5} rows removed")

    # Norm 5: Verify valid chemical formulas and host structure
    formula_valid = df["Formula (Charged State)"].notna() & df["Formula (Discharged State)"].notna() & df["Host Structure"].notna()
    invalid_formulas = int((~formula_valid).sum())
    df = df[formula_valid].copy()
    print(f"7. Removed invalid/missing formula/structure: {invalid_formulas:>5} rows removed")

    # Norm 6: Column pruning (ensure crystal and space group info are preserved)
    # Keep core material, electrochemical, and structural columns needed for research
    # and prune the ~340 raw VASP POTCAR metadata columns that clutter the dataset.
    core_columns = [
        "Working Ion",
        "Average Voltage (V)",
        "Max Voltage Step (V)",
        "Number of Voltage Steps",
        "Framework Formula",
        "Formula (Charged State)",
        "Formula (Discharged State)",
        "Internal ID (Charged State)",
        "Internal ID (Discharged State)",
        "Max Volume Change",
        "Gravimetric Capacity",
        "Volumetric Capacity",
        "Gravimetric Energy Density",
        "Volumetric Energy Density",
        "Working-Ion Fraction (Charged)",
        "Working-Ion Fraction (Discharged)",
        "Stability (Charged State)",
        "Stability (Discharged State)",
        "Battery Type",
        "Chemical System",
        "Number of Elements",
        "Anonymous Formula",
        "Elements Present",
        "Framework Composition",
        "Host Structure",
        "Battery Formula",
        "Electrode_data_material_id",
    ]
    structural_extras = [c for c in df.columns if any(k in c.lower() for k in ["space", "lattice", "crystal", "cif"])]
    core_columns.extend(structural_extras)
    
    available_core = [c for c in dict.fromkeys(core_columns) if c in df.columns]
    df = df[available_core].reset_index(drop=True)
    clean_rows, clean_cols = df.shape

    # -------------------------------------------------------------------------
    # 4. SAVE CLEANED DATASETS
    # -------------------------------------------------------------------------
    df.to_csv(OUTPUT_FILE_GUIDE, index=False)
    df.to_csv(OUTPUT_FILE_COMPAT, index=False)
    print(f"\nSaved cleaned datasets:")
    print(f"  -> {OUTPUT_FILE_GUIDE}")
    print(f"  -> {OUTPUT_FILE_COMPAT}")

    # -------------------------------------------------------------------------
    # 5. STEP 3 & 4 SUMMARY & OBSERVATIONS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CLEANING & QUALITY AUDIT SUMMARY")
    print("=" * 70)
    print(f"Total Raw Input Records     : {raw_rows}")
    print(f"Total Rows Removed          : {raw_rows - clean_rows}")
    print(f"Final Clean Records Retained: {clean_rows}")
    print(f"Final Columns Retained      : {clean_cols} (pruned from {raw_cols} raw columns)")

    print("\n--- Target Voltage Statistics (Cleaned Data) ---")
    v_stats = df[voltage_col].describe()
    print(f"  Count : {int(v_stats['count'])}")
    print(f"  Mean  : {v_stats['mean']:.4f} V")
    print(f"  Std   : {v_stats['std']:.4f} V")
    print(f"  Min   : {v_stats['min']:.4f} V")
    print(f"  25%   : {v_stats['25%']:.4f} V")
    print(f"  50%   : {v_stats['50%']:.4f} V (Median)")
    print(f"  75%   : {v_stats['75%']:.4f} V")
    print(f"  Max   : {v_stats['max']:.4f} V")

    print("\n--- Working Ion Distribution (Cleaned Data) ---")
    paper_ions = ["Li", "Mg", "Ca", "Zn", "Al", "Y"]
    ion_counts = df["Working Ion"].value_counts()
    
    print("A. Base Paper Primary Training Ions (Li, Mg, Ca, Zn, Al, Y):")
    base_paper_total = sum(ion_counts.get(ion, 0) for ion in paper_ions)
    for ion in paper_ions:
        count = ion_counts.get(ion, 0)
        pct = (count / clean_rows) * 100
        print(f"  {ion:<4}: {count:>5} ({pct:>5.1f}%)")
    print(f"  Subtotal Base Paper Ions : {base_paper_total} ({(base_paper_total/clean_rows)*100:.1f}%)")

    print("\nB. Extended Ions (Na, K - Evaluated separately in paper):")
    extended_ions = [ion for ion in ion_counts.index if ion not in paper_ions]
    extended_total = sum(ion_counts.get(ion, 0) for ion in extended_ions)
    for ion in extended_ions:
        count = ion_counts.get(ion, 0)
        pct = (count / clean_rows) * 100
        print(f"  {ion:<4}: {count:>5} ({pct:>5.1f}%)")
    print(f"  Subtotal Extended Ions   : {extended_total} ({(extended_total/clean_rows)*100:.1f}%)")

    # Missing value audit on essential columns
    print("\n--- Missing Values in Essential Columns ---")
    essential_cols = ["Working Ion", voltage_col, "Formula (Charged State)", "Formula (Discharged State)", "Host Structure"]
    missing_audit = df[essential_cols].isna().sum()
    for col, n_missing in missing_audit.items():
        print(f"  {col:<30}: {n_missing} missing")

    # -------------------------------------------------------------------------
    # 6. GENERATE VISUALIZATIONS (STEP 3 DELIVERABLE)
    # -------------------------------------------------------------------------
    # Figure 1: Voltage distribution histogram
    v_plot_path = os.path.join(PLOTS_DIR, "voltage_distribution.png")
    plt.figure(figsize=(9, 5))
    plt.hist(df[voltage_col], bins=35, color="#2b5c8f", edgecolor="black", alpha=0.8)
    plt.axvline(v_stats["mean"], color="red", linestyle="--", linewidth=1.8, label=f"Mean: {v_stats['mean']:.2f} V")
    plt.axvline(v_stats["50%"], color="green", linestyle=":", linewidth=1.8, label=f"Median: {v_stats['50%']:.2f} V")
    plt.xlabel("Average Voltage (V)", fontsize=12)
    plt.ylabel("Number of Electrodes", fontsize=12)
    plt.title("Distribution of Average Voltage (Cleaned Dataset)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(v_plot_path, dpi=300)
    plt.close()
    print(f"\nGenerated voltage distribution plot: {v_plot_path}")

    # Figure 2: Working ion distribution bar chart
    ion_plot_path = os.path.join(PLOTS_DIR, "working_ion_distribution.png")
    plt.figure(figsize=(9, 5))
    colors = ["#2b5c8f" if ion in paper_ions else "#e67e22" for ion in ion_counts.index]
    bars = plt.bar(ion_counts.index, ion_counts.values, color=colors, edgecolor="black", alpha=0.85)
    plt.xlabel("Working Ion", fontsize=12)
    plt.ylabel("Number of Electrodes", fontsize=12)
    plt.title("Electrode Count per Working Ion (Blue = Base Paper, Orange = Extended)", fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 30, f"{int(yval)}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(ion_plot_path, dpi=300)
    plt.close()
    print(f"Generated working ion distribution plot: {ion_plot_path}")
    print("=" * 70)

if __name__ == "__main__":
    inspect_and_clean()