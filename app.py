import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pymatgen.core import Composition, Element
import sys

# Ensure src modules can be imported
import importlib
predict_module = importlib.import_module("src.06_predict_pipeline")
BatteryVoltagePredictor = predict_module.BatteryVoltagePredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Constants
MAGPIE_PROXIES = [
    'Z', 'mendeleev_no', 'atomic_mass', 'melting_point', 'group', 'row',
    'atomic_radius', 'X', 'boiling_point', 'density_of_solid',
    'molar_volume', 'thermal_conductivity', 'max_oxidation_state',
    'min_oxidation_state', 'coefficient_of_linear_thermal_expansion',
    'bulk_modulus', 'youngs_modulus', 'brinell_hardness',
    'rigidity_modulus', 'mineral_hardness', 'vickers_hardness',
    'electrical_resistivity'
]
LATTICE_TYPES = ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'triclinic', 'trigonal']

ELEMENT_CACHE = {}

@st.cache_resource
def load_predictor():
    return BatteryVoltagePredictor(MODELS_DIR)

def generate_comp_features(formula, prefix):
    features = {}
    for prop in MAGPIE_PROXIES:
        for stat in ['min', 'max', 'range', 'mean', 'std']:
            features[f"{prefix}_{prop}_{stat}"] = 0.0
    if not formula: return features
    try:
        comp = Composition(formula)
        total = comp.num_atoms
        if total <= 0: return features
        fracs = {el.symbol: amt / total for el, amt in comp.items()}
        elements = list(fracs.keys())
        
        for prop in MAGPIE_PROXIES:
            vals = []
            for el in elements:
                key = (el, prop)
                if key in ELEMENT_CACHE:
                    vals.append(ELEMENT_CACHE[key])
                    continue
                try:
                    val = getattr(Element(el), prop, np.nan)
                    val = float(val[0]) if isinstance(val, (tuple, list)) else float(val) if val is not None else np.nan
                    ELEMENT_CACHE[key] = val
                    vals.append(val)
                except:
                    ELEMENT_CACHE[key] = np.nan
                    vals.append(np.nan)
            
            valid = [(v, f) for v, f in zip(vals, [fracs[e] for e in elements]) if not pd.isna(v)]
            if not valid: continue
            
            v_arr = np.array([x[0] for x in valid])
            f_arr = np.array([x[1] for x in valid])
            f_arr = f_arr / f_arr.sum()
            
            c_mean = float(np.sum(v_arr * f_arr))
            features[f"{prefix}_{prop}_min"] = float(np.min(v_arr))
            features[f"{prefix}_{prop}_max"] = float(np.max(v_arr))
            features[f"{prefix}_{prop}_range"] = features[f"{prefix}_{prop}_max"] - features[f"{prefix}_{prop}_min"]
            features[f"{prefix}_{prop}_mean"] = c_mean
            features[f"{prefix}_{prop}_std"] = float(np.sqrt(max(0.0, np.sum(f_arr * ((v_arr - c_mean)**2)))))
    except:
        pass
    return features

def featurize_input(ion, chg, dis, sg, lat):
    row = {}
    row.update(generate_comp_features(dis, 'Discharged'))
    row.update(generate_comp_features(chg, 'Charged'))
    
    try:
        row['Working_Ion_Atomic_Mass'] = float(getattr(Element(ion), 'atomic_mass'))
        row['Working_Ion_Atomic_Radius'] = float(getattr(Element(ion), 'atomic_radius'))
        row['Working_Ion_Electronegativity'] = float(getattr(Element(ion), 'X'))
        row['Working_Ion_Ionization_Energy'] = float(getattr(Element(ion), 'ionization_energy'))
        row['Working_Ion_Average_Ionic_Radius'] = float(getattr(Element(ion), 'average_ionic_radius'))
    except:
        pass
        
    try:
        row['Fraction_Charged'] = Composition(chg).get_atomic_fraction(ion)
        row['Fraction_Discharged'] = Composition(dis).get_atomic_fraction(ion)
    except:
        row['Fraction_Charged'] = 0.0
        row['Fraction_Discharged'] = 0.0
    row['Fraction_Interval'] = row['Fraction_Discharged'] - row['Fraction_Charged']
    
    row['Space_Group_Number'] = float(sg)
    for l in LATTICE_TYPES:
        row[f"Lattice_{l}"] = 1.0 if l == lat else 0.0
        
    return pd.DataFrame([row])

def main():
    st.set_page_config(page_title="Battery ML", layout="wide")
    st.title("🔋 Battery Voltage ML Predictor")
    
    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Models not found. Please run the training pipeline first. {e}")
        return

    tab1, tab2 = st.tabs(["Single Material", "Dynamic Profile"])
    
    with tab1:
        st.header("Single Material Voltage Estimator")
        col1, col2 = st.columns(2)
        with col1:
            ion = st.selectbox("Working Ion", ['Li', 'Na', 'K', 'Mg', 'Ca', 'Zn', 'Al', 'Y'])
            chg = st.text_input("Charged Formula", "CoO2")
            dis = st.text_input("Discharged Formula", "LiCoO2")
        with col2:
            sg = st.number_input("Space Group", min_value=1, max_value=230, value=166)
            lat = st.selectbox("Lattice", LATTICE_TYPES, index=6)
            model = st.selectbox("Model", ["SVR", "DNN", "KRR"])
            
        if st.button("Predict"):
            df = featurize_input(ion, chg, dis, sg, lat)
            v = predictor.predict_single(df, model)
            st.success(f"Predicted Average Voltage: {v:.3f} V")
            
    with tab2:
        st.header("Dynamic Multi-Step Voltage Profile ($V$ vs $x$)")
        st.write("Simulates voltage dropping as working ion concentration $x$ increases.")
        if st.button("Simulate Profile"):
            df = featurize_input(ion, chg, dis, sg, lat)
            xs = np.linspace(0.0, 1.0, 20)
            vs = predictor.dynamic_profile(df, xs, model)
            chart_data = pd.DataFrame({"x (Ion Fraction)": xs, "Voltage (V)": vs})
            st.line_chart(chart_data, x="x (Ion Fraction)", y="Voltage (V)")

if __name__ == '__main__':
    main()
