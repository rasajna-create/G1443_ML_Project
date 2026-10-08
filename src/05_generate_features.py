import ast
import json
import os
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# -----------------------------------------------------------------------------
# 1. SETUP & CONSTANTS
# -----------------------------------------------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.path.abspath(".")

DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "cleaned_voltage_dataset.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "features_raw.csv")

MAGPIE_PROXIES = [
    'Z', 'mendeleev_no', 'atomic_mass', 'melting_point', 'group', 'row',
    'atomic_radius', 'X', 'boiling_point', 'density_of_solid',
    'molar_volume', 'thermal_conductivity', 'max_oxidation_state',
    'min_oxidation_state', 'coefficient_of_linear_thermal_expansion',
    'bulk_modulus', 'youngs_modulus', 'brinell_hardness',
    'rigidity_modulus', 'mineral_hardness', 'vickers_hardness',
    'electrical_resistivity'
]

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
# Cache element properties in memory to avoid redundant pymatgen instantiations
ELEMENT_CACHE = {}

def get_element_property(el_str, prop):
    key = (el_str, prop)
    if key in ELEMENT_CACHE:
        return ELEMENT_CACHE[key]
    try:
        el = Element(el_str)
        val = getattr(el, prop, np.nan)
        if val is None:
            res = np.nan
        elif isinstance(val, (tuple, list)):
            res = float(val[0])
        else:
            res = float(val)
    except Exception:
        res = np.nan
    ELEMENT_CACHE[key] = res
    return res

def generate_composition_features(formulas, prefix):
    """Generates min, max, range, mean, and std for elemental properties."""
    features = []
    
    # Pre-build standardized column names
    col_names = [f"{prefix}_{prop}_{stat}" for prop in MAGPIE_PROXIES for stat in ['min', 'max', 'range', 'mean', 'std']]
    empty_row = {col: np.nan for col in col_names}

    for f in formulas:
        if pd.isna(f) or not str(f).strip():
            features.append(empty_row.copy())
            continue

        try:
            comp = Composition(str(f).strip())
            total_atoms = comp.num_atoms
            if total_atoms <= 0:
                features.append(empty_row.copy())
                continue

            fractions = {el.symbol: amt / total_atoms for el, amt in comp.items()}
            elements = list(fractions.keys())

            row_dict = {}
            for prop in MAGPIE_PROXIES:
                vals = [get_element_property(el, prop) for el in elements]
                fracs = [fractions[el] for el in elements]

                valid_pairs = [(v, fr) for v, fr in zip(vals, fracs) if not pd.isna(v)]
                if not valid_pairs:
                    for stat in ['min', 'max', 'range', 'mean', 'std']:
                        row_dict[f"{prefix}_{prop}_{stat}"] = np.nan
                    continue

                valid_vals = np.array([p[0] for p in valid_pairs])
                valid_fracs = np.array([p[1] for p in valid_pairs])
                norm_fracs = valid_fracs / valid_fracs.sum()

                c_min = float(np.min(valid_vals))
                c_max = float(np.max(valid_vals))
                c_range = c_max - c_min
                c_mean = float(np.sum(valid_vals * norm_fracs))
                variance = float(np.sum(norm_fracs * ((valid_vals - c_mean) ** 2)))
                c_std = float(np.sqrt(max(0.0, variance)))

                row_dict[f"{prefix}_{prop}_min"] = c_min
                row_dict[f"{prefix}_{prop}_max"] = c_max
                row_dict[f"{prefix}_{prop}_range"] = c_range
                row_dict[f"{prefix}_{prop}_mean"] = c_mean
                row_dict[f"{prefix}_{prop}_std"] = c_std

            features.append(row_dict)
        except Exception:
            features.append(empty_row.copy())

    return pd.DataFrame(features)

def extract_working_ion_properties(ions):
    features = []
    for ion in ions:
        ion_str = str(ion).strip() if pd.notna(ion) else ""
        row = {
            'Working_Ion_Atomic_Mass': get_element_property(ion_str, 'atomic_mass'),
            'Working_Ion_Atomic_Radius': get_element_property(ion_str, 'atomic_radius'),
            'Working_Ion_Electronegativity': get_element_property(ion_str, 'X'),
            'Working_Ion_Ionization_Energy': get_element_property(ion_str, 'ionization_energy'),
            'Working_Ion_Average_Ionic_Radius': get_element_property(ion_str, 'average_ionic_radius')
        }
        features.append(row)
    return pd.DataFrame(features)

def extract_structural_features(df):
    """Extracts Space Group Number and One-Hot Crystal Systems efficiently."""
    space_groups = []
    lattice_types = []

    # Check if pre-parsed columns exist in dataframe
    sg_col = next((c for c in df.columns if c.lower() in ["spacegroup", "space_group", "spacegroup.number", "space group"]), None)
    lat_col = next((c for c in df.columns if c.lower() in ["crystal_system", "crystal system", "lattice_type", "lattice type"]), None)

    if sg_col and lat_col:
        print("Using existing structural columns from dataset...")
        space_groups = pd.to_numeric(df[sg_col], errors='coerce').fillna(0).tolist()
        lattice_types = df[lat_col].fillna("Unknown").tolist()
    else:
        print("Extracting from 'Host Structure' dictionaries (this may take a couple of minutes)...")
        for s_val in df.get('Host Structure', []):
            sg_num = np.nan
            lat_type = "Unknown"
            if pd.notna(s_val):
                try:
                    if isinstance(s_val, str):
                        try:
                            s_dict = ast.literal_eval(s_val)
                        except Exception:
                            s_val_fix = s_val.replace('True', 'true').replace('False', 'false').replace('None', 'null').replace("'", '"')
                            s_dict = json.loads(s_val_fix)
                    else:
                        s_dict = s_val
                    struct = Structure.from_dict(s_dict)
                    sga = SpacegroupAnalyzer(struct, symprec=0.1)
                    sg_num = sga.get_space_group_number()
                    lat_type = sga.get_crystal_system()
                except Exception:
                    pass
            space_groups.append(sg_num)
            lattice_types.append(lat_type)

    df_sg = pd.DataFrame({'Space_Group_Number': space_groups})
    df_sg['Space_Group_Number'] = df_sg['Space_Group_Number'].fillna(0).astype(float)
    df_lat_encoded = pd.get_dummies(pd.Series(lattice_types, name="Lattice"), prefix='Lattice', dtype=float)

    return pd.concat([df_sg, df_lat_encoded], axis=1)

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("BATTERY VOLTAGE PREDICTION - STEP 5: GENERATE NUMERICAL FEATURES")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Ensure cleaning step was run.")
        return

    print(f"Loading cleaned dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    # 1. Compositional Properties (Charged & Discharged)
    print("Computing compositional features (Discharged state)...")
    df_discharged = generate_composition_features(df['Formula (Discharged State)'], 'Discharged')

    print("Computing compositional features (Charged state)...")
    df_charged = generate_composition_features(df['Formula (Charged State)'], 'Charged')

    # 2. Working Ion Specific Features
    print("Extracting working ion intrinsic properties...")
    df_ion_props = extract_working_ion_properties(df['Working Ion'])

    # 3. Concentration & Stoichiometry
    print("Extracting working ion concentration & fractions...")
    df_conc = pd.DataFrame()
    df_conc['Fraction_Charged'] = pd.to_numeric(df.get('Working-Ion Fraction (Charged)', 0.0), errors='coerce').fillna(0.0)
    df_conc['Fraction_Discharged'] = pd.to_numeric(df.get('Working-Ion Fraction (Discharged)', 0.0), errors='coerce').fillna(0.0)
    df_conc['Fraction_Interval'] = df_conc['Fraction_Discharged'] - df_conc['Fraction_Charged']

    # 4. Structural Features
    print("Extracting structural features...")
    df_structure = extract_structural_features(df)

    # 5. Assemble Final Feature Matrix X
    print("Assembling numerical feature matrix X...")
    X = pd.concat([
        df_discharged, df_charged,
        df_ion_props, df_conc, df_structure
    ], axis=1)

    # Impute remaining NaN values with column medians (for missing experimental hardness/modulus)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    # 6. Metadata and Target
    target = 'Average Voltage (V)'
    y = pd.to_numeric(df[target], errors='coerce')

    metadata_cols = [
        'Working Ion', 'Battery Formula', 'Formula (Charged State)',
        'Formula (Discharged State)', 'Electrode_data_material_id'
    ]
    metadata_df = df[[c for c in metadata_cols if c in df.columns]]

    # 7. Merge and Save
    final_df = pd.concat([metadata_df, X, y], axis=1)

    print("\n--- VALIDATION ---")
    print(f"Total Samples           : {len(final_df)}")
    print(f"Total Features in X     : {X.shape[1]}")
    print(f"Total NaN values in X   : {X.isna().sum().sum()}")
    print(f"Total Infinite values   : {np.isinf(X.values).sum()}")

    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved raw features dataset to: {OUTPUT_FILE}")
    print("=" * 70)

if __name__ == '__main__':
    main()
