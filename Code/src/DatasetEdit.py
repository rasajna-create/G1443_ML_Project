"""
Keeps ALL original rows (6,683) no matter what. For each row, tries to
extract lattice parameters (a, b, c, alpha, beta, gamma, volume) from the
'Host Structure' column. Where that data exists and parses successfully,
the 7 new columns get real numbers. Where it's missing or fails to parse,
those 7 cells are simply left blank (NaN) for that row - the row itself
is never removed.
"""

import ast
import json
import pandas as pd

INPUT_FILE = r"C:\Users\Srihan Uppala\OneDrive\Desktop\Crystal Domain\Code\Dataset\FinalCleanDataset.csv"
OUTPUT_FILE = INPUT_FILE  # overwrite in place; change this if you want a separate file

# -------------------------------------------------------------------
# 1. Load the dataset and record the starting row count
# -------------------------------------------------------------------
df = pd.read_csv(INPUT_FILE, low_memory=False)
starting_row_count = len(df)
print(f"Loaded {starting_row_count} rows and {df.shape[1]} columns")

if "Host Structure" not in df.columns:
    raise KeyError(
        f"'Host Structure' column not found. Available columns: {list(df.columns)}"
    )

# -------------------------------------------------------------------
# 2. Helper: safely parse a cell's text back into a Python dict.
#    Returns None if the cell is empty or can't be parsed - it does NOT
#    raise an error and does NOT remove the row.
# -------------------------------------------------------------------
def parse_structure(value):
    if pd.isnull(value):
        return None
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None

# -------------------------------------------------------------------
# 3. Go through every row (all 6,683 of them), extracting lattice
#    parameters where possible and leaving None where not possible.
#    The row itself is always kept either way.
# -------------------------------------------------------------------
print("\nExtracting lattice parameters from 'Host Structure'...")

lattice_fields = ["a", "b", "c", "alpha", "beta", "gamma", "volume"]
extracted = {field: [] for field in lattice_fields}

success_count = 0

for value in df["Host Structure"]:
    structure_dict = parse_structure(value)
    lattice = structure_dict.get("lattice") if structure_dict is not None else None

    if lattice is None:
        for field in lattice_fields:
            extracted[field].append(None)
    else:
        success_count += 1
        for field in lattice_fields:
            extracted[field].append(lattice.get(field))

print(f"Successfully extracted lattice data for {success_count} / {starting_row_count} rows")
print(f"Remaining {starting_row_count - success_count} rows will have blank values in the new columns")

# -------------------------------------------------------------------
# 4. Add the new columns - same length as the dataframe, so every
#    original row is preserved no matter what
# -------------------------------------------------------------------
column_rename = {
    "a": "Lattice a (Å)",
    "b": "Lattice b (Å)",
    "c": "Lattice c (Å)",
    "alpha": "Lattice alpha (°)",
    "beta": "Lattice beta (°)",
    "gamma": "Lattice gamma (°)",
    "volume": "Lattice Volume (Å³)",
}

for field, col_name in column_rename.items():
    assert len(extracted[field]) == starting_row_count, "Length mismatch - should never happen"
    df[col_name] = extracted[field]

# -------------------------------------------------------------------
# 5. Confirm no rows were lost before saving
# -------------------------------------------------------------------
ending_row_count = len(df)
print(f"\nRow count before: {starting_row_count}")
print(f"Row count after:  {ending_row_count}")

if ending_row_count != starting_row_count:
    raise AssertionError(
        "Row count changed unexpectedly! This should never happen with this "
        "script - something is wrong. NOT saving the file."
    )

print("Confirmed: no rows were added or removed.")

# -------------------------------------------------------------------
# 6. Save
# -------------------------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to '{OUTPUT_FILE}'")

# -------------------------------------------------------------------
# 7. Sanity check
# -------------------------------------------------------------------
print("\nBlank counts per new column (should all equal the same number):")
print(df[list(column_rename.values())].isnull().sum())

print("\nSample rows with real lattice data:")
has_data = df[df["Lattice a (Å)"].notnull()]
print(has_data[list(column_rename.values())].head())