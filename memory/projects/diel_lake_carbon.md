---
name: DielLakeCarbon
description: Diel carbon isotope model for Stephens Lake, TN — current version v9
type: project
---

# DielLakeCarbon Project

Python-based diel model that simulates daily cycling of CO2, dissolved inorganic carbon (DIC), carbon-13 isotopes (d13C-DIC), gross primary production (GPP), ecosystem respiration (ER), and calcite saturation index (SI) in Stephens Lake, Tennessee.

**Why:** Understanding diel carbon dynamics in freshwater lakes; manuscript in preparation.

**How to apply:** This is the active research project. Code lives in `diel_model_v9.py`. Results in `Diel_v9_results_allseasons.csv`. When helping with tasks, assume geochemical context and that model outputs/plots need to match publication standards.

## Study Site
- Stephens Lake, TN — LATITUDE = 35.939311, LONGITUDE = -87.015833

## Key Files
- `diel_model_v9.py` — main model
- `diel_model_v8.py` — previous version (reference)
- `stephens_lake_SI.py` / `.ipynb` — Phreeqc saturation index calculations
- `Diel_v9_results_allseasons.csv` — combined seasonal output
- `phreeqc.dat` — thermodynamic database
- `input/` — seasonal input files (include Na, Cl for charge balance)
- `output/` — model output CSVs by season
- `plots/` — figures

## Model Structure (V9)
- Objective: minimize errors in ER, d13C-DIC, and GPP simultaneously
- Geochemical backend: Phreeqc (not GWB)
- Calcite kinetics being evaluated for inclusion
- Helper functions needed: Sc_O2, Sc_CO2, henry_CO2

## Manuscript Status
- Results and Discussion sections drafted; need integration across all four seasons
- Fig. 3: Ca2+ concentration — needs measured values overlaid + Phreeqc SI values
- Summer nights: measured calcite SI << model SI (open question)

## Seasons Modeled
Spring, Summer, Fall, Winter — combined in `Diel_v9_results_allseasons.csv`
