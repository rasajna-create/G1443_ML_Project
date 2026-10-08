import os
import time
import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester

load_dotenv()

API_KEY = os.getenv("MP_API_KEY")

if not API_KEY:
    raise ValueError("MP_API_KEY not found in .env")

IONS = ["Li", "Na", "Mg", "Ca", "Zn", "Al", "Y", "K"]

OUTPUT_FILE = "mp_full_dataset.csv"
ION_FOLDER = "ion_data"

MAX_RETRIES = 5
RETRY_DELAY = 10
TIMEOUT = 300

EXCLUDE_STRUCTURE_SITES = True

os.makedirs(ION_FOLDER, exist_ok=True)


SIMPLE_FIELD_LABELS = {
    "battery_id": "Battery/Electrode ID",
    "working_ion": "Working Ion",
    "average_voltage": "Average Voltage (V)",
    "max_voltage_step": "Max Voltage Step (V)",
    "num_steps": "Number of Voltage Steps",
    "framework_formula": "Framework Formula",
    "formula_charge": "Formula (Charged State)",
    "formula_discharge": "Formula (Discharged State)",
    "id_charge": "Internal ID (Charged State)",
    "id_discharge": "Internal ID (Discharged State)",
    "max_delta_volume": "Max Volume Change",
    "capacity_grav": "Gravimetric Capacity",
    "capacity_vol": "Volumetric Capacity",
    "energy_grav": "Gravimetric Energy Density",
    "energy_vol": "Volumetric Energy Density",
    "fracA_charge": "Working-Ion Fraction (Charged)",
    "fracA_discharge": "Working-Ion Fraction (Discharged)",
    "stability_charge": "Stability (Charged State)",
    "stability_discharge": "Stability (Discharged State)",
    "thermo_type": "Thermodynamic Type",
    "battery_type": "Battery Type",
    "chemsys": "Chemical System",
    "nelements": "Number of Elements",
    "formula_anonymous": "Anonymous Formula",
    "elements": "Elements Present",
    "framework": "Framework Composition",
    "host_structure": "Host Structure",
    "battery_formula": "Battery Formula",
    "last_updated": "Last Updated",
    "warnings": "Warnings",
    "material_ids": "Internal Material ID List (not real MP IDs)",
}


def flatten(obj, prefix=""):
    flat = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}_{key}" if prefix else str(key)
            flat.update(flatten(value, new_prefix))

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            new_prefix = f"{prefix}_{i}"
            flat.update(flatten(value, new_prefix))

    else:
        flat[prefix] = obj

    return flat


def download_ion(ion):

    ion_file = os.path.join(
        ION_FOLDER,
        f"{ion}.csv"
    )

    if os.path.exists(ion_file):

        try:
            old_df = pd.read_csv(ion_file)

            if len(old_df) > 0:
                print(
                    f"{ion} already exists "
                    f"({len(old_df)} rows)."
                )
                return old_df

        except Exception:
            pass


    for attempt in range(1, MAX_RETRIES + 1):

        print(
            f"\nDownloading {ion} "
            f"(attempt {attempt}/{MAX_RETRIES})..."
        )

        try:

            with MPRester(
                API_KEY,
                timeout=TIMEOUT
            ) as mpr:

                docs = (
                    mpr.materials
                    .insertion_electrodes
                    .search(
                        working_ion=ion
                    )
                )

            print(
                f"{ion}: {len(docs)} records downloaded"
            )

            all_docs = [
                doc.model_dump()
                for doc in docs
            ]

            output_rows = []
            simple_columns_seen = []
            electrode_columns_seen = []

            for doc in all_docs:

                battery_id = doc.get("battery_id")

                material_id = (
                    str(battery_id).split("_")[0]
                    if battery_id
                    else None
                )

                simple_data = {}

                for raw_key, label in SIMPLE_FIELD_LABELS.items():

                    if raw_key in doc:

                        simple_data[label] = doc[raw_key]

                        if label not in simple_columns_seen:
                            simple_columns_seen.append(label)


                eo = doc.get("electrode_object") or {}

                stable_entries = (
                    eo.get("stable_entries") or []
                )


                if not stable_entries:

                    row = {
                        "Material ID": material_id
                    }

                    row.update(simple_data)

                    output_rows.append(row)

                else:

                    for i, entry in enumerate(stable_entries):

                        row = {
                            "Material ID": material_id
                        }

                        if i == 0:

                            row.update(simple_data)

                        else:

                            for label in simple_columns_seen:
                                row[label] = ""


                        row["Stable Entry #"] = i + 1


                        if isinstance(entry, dict):

                            entry_to_flatten = entry

                            if (
                                EXCLUDE_STRUCTURE_SITES
                                and "structure" in entry_to_flatten
                            ):

                                entry_to_flatten = {
                                    k: v
                                    for k, v in entry.items()
                                    if k != "structure"
                                }


                            flat_entry = flatten(
                                entry_to_flatten,
                                prefix="Electrode"
                            )


                            for col, value in flat_entry.items():

                                row[col] = value

                                if col not in electrode_columns_seen:
                                    electrode_columns_seen.append(col)


                        output_rows.append(row)


                    output_rows.append({})


            column_order = (
                ["Material ID"]
                + simple_columns_seen
                + ["Stable Entry #"]
                + electrode_columns_seen
            )

            df = pd.DataFrame(output_rows)

            for col in column_order:

                if col not in df.columns:
                    df[col] = ""

            df = df[column_order]

            df.to_csv(
                ion_file,
                index=False
            )

            print(
                f"Saved: {ion_file}"
            )

            return df


        except Exception as e:

            print(
                f"{ion} failed: {e}"
            )

            if attempt < MAX_RETRIES:

                print(
                    f"Retrying in {RETRY_DELAY} seconds..."
                )

                time.sleep(RETRY_DELAY)


    print(
        f"{ion} FAILED after {MAX_RETRIES} attempts."
    )

    return None


# ============================================================
# DOWNLOAD
# ============================================================

print("=" * 60)
print("DOWNLOADING MATERIALS PROJECT ELECTRODE DATA")
print("=" * 60)

data = {}

for ion in IONS:

    result = download_ion(ion)

    if result is not None:
        data[ion] = result


# ============================================================
# COMBINE IN ORDER
# ============================================================

print("\nCombining datasets...")

combined = []

for ion in IONS:

    if ion in data:

        print(
            f"{ion}: {len(data[ion])} rows"
        )

        combined.append(data[ion])


if not combined:

    raise RuntimeError(
        "No data was downloaded."
    )


final_df = pd.concat(
    combined,
    ignore_index=True
)


final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(
    f"Final CSV: {OUTPUT_FILE}"
)

print(
    f"Total rows: {len(final_df)}"
)

print(
    f"Total columns: {len(final_df.columns)}"
)

print("\nIon order:")

for ion in IONS:

    if ion in data:
        print(
            f"{ion}: {len(data[ion])}"
        )
    else:
        print(
            f"{ion}: FAILED"
        )

print("=" * 60)