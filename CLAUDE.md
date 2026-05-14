# Memory

## Me
John Ayers, researcher at Vanderbilt University. Working on diel carbon isotope modeling for lakes.
Preferred tools: Python, R Tidyverse.

## Projects
| Name | What |
|------|------|
| **DielLakeCarbon** | Diel carbon isotope model (v9) for Stephens Lake, TN. Models CO2/DIC dynamics, GPP, ER, calcite saturation, and d13C-DIC over daily cycles across four seasons. |

## Study Site
- **Stephens Lake** — field site; LATITUDE = 35.939311, LONGITUDE = -87.015833 (central TN)

## Key Terms
| Term | Meaning |
|------|---------|
| diel | 24-hour daily cycle |
| DIC | Dissolved Inorganic Carbon |
| d13C-DIC | delta-13C isotope ratio of DIC |
| GPP | Gross Primary Production |
| ER | Ecosystem Respiration |
| SI / Sic | Saturation Index (of calcite) |
| Ca2+ | Calcium ion |
| Sc_O2 | Schmidt number for O2 |
| Sc_CO2 | Schmidt number for CO2 |
| henry_CO2 | Henry's law constant for CO2 |
| Phreeqc | Free USGS geochemical modeling software (used for saturation indices) |
| GWB | Geochemist's Workbench (commercial geochemical software) |
| Spec8 | GWB module for speciation calculations |
| V9 | Version 9 of the diel model |

## Key Files
| File | Purpose |
|------|---------|
| `diel_model_v9.py` | Main diel model script |
| `Diel_v9_results_allseasons.csv` | Combined output for all four seasons |
| `phreeqc.dat` | Phreeqc thermodynamic database |
| `output/Stephens_Lake_Summer_diel_SI.csv` | Measured summer saturation indices |
| `output/summer/output_v9/diel_v9_results.csv` | Model summer output |

## Preferences
- Uses Python and R Tidyverse for coding
- Prefers Claude to write/modify code directly
- Model outputs go in `output/` directory
- Plots go in `plots/` directory
