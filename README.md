# diel_lake_carbon_phreeqc
This repository contains a fully reproducible Jupyter notebook that simulates diel carbon cycling and carbon isotope dynamics (δ¹³C of DIC) in a shallow, well mixed lake.
The model includes:
- Diel primary production with mixed CO₂ / HCO₃⁻ uptake
- Respiration
- Meteorologically scaled air–water CO₂ exchange
- Explicit δ¹³C(DIC) diagnostics
- Interactive sliders for key parameters

## ▶ Run in your browser (no installation required)

Click the Binder badge below:

[![Binder](https://mybinder.org/badge_logo.svg)](
https://mybinder.org/v2/gh/johncayers/diel_lake_carbon_phreeqc/HEAD
)

Binder will:
1. Build the environment (~2–4 minutes)
2. Launch JupyterLab
3. Open the notebook in your browser

## 🧪 Using the notebook

1. Run cells from top to bottom
2. Use the sliders to adjust:
   - Photosynthetic fractionation (εₚ)
   - Gas transfer velocity (k₆₀₀)
3. Observe changes in:
   - pH
   - pCO₂
   - δ¹³C(DIC)

The default parameter set reproduces observed summer diel δ¹³C(DIC)
variability (−6.7 to −8.2‰).

## 📦 Outputs

The notebook automatically:
- Exports model results to CSV
- Produces publication ready figures

## 📖 Citation

If you use this model, please cite the associated manuscript and PHREEQC.
``
