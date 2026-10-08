import os
import pandas as pd
import numpy as np
import ast
import json
from pymatgen.core import Composition, Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "cleaned_dataset.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "dataset_features_Xy.csv")

MAGPIE_PROXIES = [
    'Z', 'mendeleev_no', 'atomic_mass', 'melting_point', 'group', 'row',
    'atomic_radius', 'X', 'boiling_point', 'density_of_solid',
    'molar_volume', 'thermal_conductivity', 'max_oxidation_state',
    'min_oxidation_state', 'coefficient_of_linear_thermal_expansion',
    'bulk_modulus', 'youngs_modulus', 'brinell_hardness',
    'rigidity_modulus', 'mineral_hardness', 'vickers_hardness',
    'electrical_resistivity'
]

ELEMENT_CACHE = {}

def get_element_property(el_str, prop):
    key = (el_str, prop)
    if key in ELEMENT_CACHE: return ELEMENT_CACHE[key]
    try:
        el = Element(el_str)
        val = getattr(el, prop, np.nan)
        res = float(val[0]) if isinstance(val, (tuple, list)) else float(val) if val is not None else np.nan
    except:
        res = np.nan
    ELEMENT_CACHE[key] = res
    return res

def generate_comp_features(formulas, prefix):
    features = []
    col_names = [f"{prefix}_{prop}_{stat}" for prop in MAGPIE_PROXIES for stat in ['min', 'max', 'range', 'mean', 'std']]
    empty_row = {col: np.nan for col in col_names}

    for f in formulas:
        if pd.isna(f) or not str(f).strip():
            features.append(empty_row.copy())
            continue
        try:
            comp = Composition(str(f).strip())
            total = comp.num_atoms
            if total <= 0:
                features.append(empty_row.copy())
                continue
            fracs = {el.symbol: amt / total for el, amt in comp.items()}
            elements = list(fracs.keys())
            row_dict = {}
            for prop in MAGPIE_PROXIES:
                vals = [get_element_property(el, prop) for el in elements]
                fr = [fracs[el] for el in elements]
                valid = [(v, f) for v, f in zip(vals, fr) if not pd.isna(v)]
                if not valid:
                    for stat in ['min', 'max', 'range', 'mean', 'std']:
                        row_dict[f"{prefix}_{prop}_{stat}"] = np.nan
                    continue
                v_arr = np.array([x[0] for x in valid])
                f_arr = np.array([x[1] for x in valid])
                f_arr = f_arr / f_arr.sum()
                
                c_mean = float(np.sum(v_arr * f_arr))
                row_dict.update({
                    f"{prefix}_{prop}_min": float(np.min(v_arr)),
                    f"{prefix}_{prop}_max": float(np.max(v_arr)),
                    f"{prefix}_{prop}_range": float(np.max(v_arr) - np.min(v_arr)),
                    f"{prefix}_{prop}_mean": c_mean,
                    f"{prefix}_{prop}_std": float(np.sqrt(max(0.0, np.sum(f_arr * ((v_arr - c_mean)**2)))))
                })
            features.append(row_dict)
        except:
            features.append(empty_row.copy())
    return pd.DataFrame(features)

def extract_structural(df):
    sg, lat = [], []
    for s_val in df.get('Host Structure', []):
        s, l = 0, "Unknown"
        if pd.notna(s_val):
            try:
                if isinstance(s_val, str):
                    s_val = s_val.replace('True', 'true').replace('False', 'false').replace('None', 'null').replace("'", '"')
                    struct = Structure.from_dict(json.loads(s_val))
                else:
                    struct = Structure.from_dict(s_val)
                sga = SpacegroupAnalyzer(struct, symprec=0.1)
                s = sga.get_space_group_number()
                l = sga.get_crystal_system()
            except:
                pass
        sg.append(s)
        lat.append(l)
    
    df_sg = pd.DataFrame({'Space_Group_Number': sg}).fillna(0).astype(float)
    df_lat = pd.get_dummies(pd.Series(lat, name="Lattice"), prefix='Lattice', dtype=float)
    
    # Force EXACT 7 lattice types
    expected_lats = [f"Lattice_{x}" for x in ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'triclinic', 'trigonal']]
    for c in expected_lats:
        if c not in df_lat.columns: df_lat[c] = 0.0
    return pd.concat([df_sg, df_lat[expected_lats]], axis=1)

def main():
    print("Step 5: Feature Engineering")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    df_dis = generate_comp_features(df['Formula (Discharged State)'], 'Discharged')
    df_chg = generate_comp_features(df['Formula (Charged State)'], 'Charged')
    
    df_ion = pd.DataFrame({
        'Working_Ion_Atomic_Mass': [get_element_property(i, 'atomic_mass') for i in df['Working Ion']],
        'Working_Ion_Atomic_Radius': [get_element_property(i, 'atomic_radius') for i in df['Working Ion']],
        'Working_Ion_Electronegativity': [get_element_property(i, 'X') for i in df['Working Ion']],
        'Working_Ion_Ionization_Energy': [get_element_property(i, 'ionization_energy') for i in df['Working Ion']],
        'Working_Ion_Average_Ionic_Radius': [get_element_property(i, 'average_ionic_radius') for i in df['Working Ion']]
    })
    
    df_conc = pd.DataFrame()
    df_conc['Fraction_Charged'] = pd.to_numeric(df.get('Working-Ion Fraction (Charged)', 0.0), errors='coerce').fillna(0.0)
    df_conc['Fraction_Discharged'] = pd.to_numeric(df.get('Working-Ion Fraction (Discharged)', 0.0), errors='coerce').fillna(0.0)
    df_conc['Fraction_Interval'] = df_conc['Fraction_Discharged'] - df_conc['Fraction_Charged']
    
    df_struct = extract_structural(df)
    
    X = pd.concat([df_dis, df_chg, df_ion, df_conc, df_struct], axis=1)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    
    meta_cols = ['Working Ion', 'Battery Formula', 'Formula (Charged State)', 'Formula (Discharged State)', 'Electrode_data_material_id']
    meta = df[[c for c in meta_cols if c in df.columns]]
    y = df['Average Voltage (V)']
    
    final_df = pd.concat([meta, X, y], axis=1)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Feature dataset saved with {X.shape[1]} features.")

if __name__ == '__main__':
    main()
