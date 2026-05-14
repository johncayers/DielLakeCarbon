---
name: Glossary
description: Geochemical and modeling terms used in the DielLakeCarbon project
type: reference
---

# Glossary

## Acronyms & Abbreviations

| Term | Meaning |
|------|---------|
| diel | 24-hour daily cycle (derived from "dies" = day) |
| DIC | Dissolved Inorganic Carbon (CO2 + HCO3- + CO3 2-) |
| d13C-DIC | Delta-13C isotope ratio of dissolved inorganic carbon |
| GPP | Gross Primary Production — photosynthetic C fixation rate |
| ER | Ecosystem Respiration — total ecosystem CO2 release rate |
| SI | Saturation Index — log(IAP/Ksp); >0 = supersaturated, <0 = undersaturated |
| Sic | Saturation Index of calcite specifically |
| Ca2+ | Calcium ion (key for calcite saturation) |
| Sc_O2 | Schmidt number for oxygen (controls gas transfer) |
| Sc_CO2 | Schmidt number for CO2 (controls gas transfer) |
| henry_CO2 | Henry's law constant for CO2 (solubility) |

## Software

| Tool | What |
|------|------|
| Phreeqc | Free USGS geochemical speciation/reaction modeling software; used for saturation indices |
| GWB | Geochemist's Workbench — commercial geochemical software (NOT used; Phreeqc preferred) |
| Spec8 | GWB module for aqueous speciation |
| phreeqc.dat | Phreeqc thermodynamic database file (in project root) |

## Model Versions
| Version | Notes |
|---------|-------|
| V8 | Previous model version — `diel_model_v8.py` |
| V9 | Current — minimizes errors in ER + d13C-DIC + GPP; `diel_model_v9.py` |
