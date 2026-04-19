#!/usr/bin/env python3
"""
Diel Carbon Isotope Model v7
=============================
• Isotope-enabled PHREEQC ([13C] as separate master species)
• Measured Ca²⁺ concentrations for accurate Calcite SI
• PAR-driven GPP (Michaelis–Menten × Q10)
• Multi-objective: minimise d13C-DIC AND GPP residuals
• Bounded differential evolution with Bayesian penalty priors
• 10-panel publication figure

Usage:
    python diel_model_v7.py                  # uses defaults
    python diel_model_v7.py path/to/data.csv # custom CSV
"""

import numpy as np
import pandas as pd
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc
from astral import LocationInfo
from astral.sun import sun, elevation as solar_elevation
from scipy.optimize import differential_evolution
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings('ignore')

###############################################################################
# CONFIGURATION
###############################################################################
LATITUDE      = 35.939311
LONGITUDE     = -87.015833
TZ_OFF        = -5            # UTC offset (CDT)
LAKE_DEPTH    = 1.0           # metres
ATM_d13C      = -8.5          # ‰ VPDB (atmospheric CO₂)
ATM_CO2_PPM   = 425.0         # ppm
R_VPDB        = 0.011237      # ¹³C/¹²C VPDB standard ratio

CSV_PATH      = '/mnt/user-data/uploads/SL20250718.csv'
OUTPUT_DIR    = 'output_v9'

# Override CSV path from command line if provided
if len(sys.argv) > 1:
    CSV_PATH = sys.argv[1]

os.makedirs(OUTPUT_DIR, exist_ok=True)

###############################################################################
# ISOTOPE-ENABLED PHREEQC DATABASE  (13C + Calcite)
###############################################################################
# Fractionation:
#   HCO3⁻/CO3²⁻ : ε ≈ −0.6 ‰ (constant) → ΔA₁ = −0.000261
#   CO2(aq)/CO3²⁻: ε(T) = −9866/T + 23.52 ‰
#       → ΔA₁ = +0.01021, ΔA₃ = −4.284
# References: Mook et al. (1974), Deines et al. (1974), Zhang et al. (1995)
###############################################################################
PHREEQC_DB = """
SOLUTION_MASTER_SPECIES
H        H+             -1.     H               1.008
H(0)     H2             0.0     H
H(1)     H+             -1.     0.0
E        e-             0.0     0.0             0.0
O        H2O            0.0     O               16.00
O(0)     O2             0.0     O
O(-2)    H2O            0.0     0.0
C        CO3-2          2.0     C               12.011
C(4)     CO3-2          2.0     CO3             12.011
Ca       Ca+2           0.0     Ca              40.08
Na       Na+            0.0     Na              22.99
Cl       Cl-            0.0     Cl              35.45
Alkalinity CO3-2        1.0     Ca              50.04
[13C]     [13C]O3-2      2.0     [13C]           13.003
[13C](4)  [13C]O3-2      2.0     [13C]O3         13.003

SOLUTION_SPECIES
H+ = H+
    log_k 0.0
e- = e-
    log_k 0.0
H2O = H2O
    log_k 0.0
CO3-2 = CO3-2
    log_k 0.0
Ca+2 = Ca+2
    log_k 0.0
Na+ = Na+
    log_k 0.0
Cl- = Cl-
    log_k 0.0
CO3-2 + H+ = HCO3-
    log_k 10.3288
    delta_h -3.561 kcal
    -analytic 107.8871 0.03252849 -5151.79 -38.92561 563713.9
CO3-2 + 2H+ = CO2 + H2O
    log_k 16.6809
    delta_h -5.738 kcal
    -analytic 464.1965 0.09344813 -26986.16 -165.75951 2248628.9
H2O = OH- + H+
    log_k -14.0
    delta_h 13.362 kcal
    -analytic -283.9710 -0.05069842 13323.0 102.24447 -1119669.0
2H2O = O2 + 4H+ + 4e-
    log_k -86.08
    delta_h 134.79 kcal
2H+ + 2e- = H2
    log_k -3.15
    delta_h -1.759 kcal
Ca+2 + CO3-2 = CaCO3
    log_k 3.224
Ca+2 + CO3-2 + H+ = CaHCO3+
    log_k 11.435
Ca+2 + H2O = CaOH+ + H+
    log_k -12.78
Na+ + CO3-2 = NaCO3-
    log_k 1.27
Na+ + CO3-2 + H+ = NaHCO3
    log_k 10.079
# ---- 13C species with T-dependent fractionation ----
[13C]O3-2 = [13C]O3-2
    log_k 0.0
[13C]O3-2 + H+ = H[13C]O3-
    log_k 10.328539
    delta_h -3.561 kcal
    -analytic 107.886839 0.03252849 -5151.79 -38.92561 563713.9
[13C]O3-2 + 2H+ = [13C]O2 + H2O
    log_k 16.677444
    delta_h -5.738 kcal
    -analytic 464.2067 0.09344813 -26990.444 -165.75951 2248628.9
Ca+2 + [13C]O3-2 = Ca[13C]O3
    log_k 3.224
Ca+2 + [13C]O3-2 + H+ = CaH[13C]O3+
    log_k 11.4347
Na+ + [13C]O3-2 = Na[13C]O3-
    log_k 1.27
Na+ + [13C]O3-2 + H+ = NaH[13C]O3
    log_k 10.0787

PHASES
CO2(g)
    CO2 = CO2 + H2O - H2O
    log_k -1.468
    delta_h -4.776 kcal
    -analytic 108.3865 0.01985076 -6919.53 -40.45154 669365.0
Calcite
    CaCO3 = Ca+2 + CO3-2
    log_k -8.48
    delta_h -2.297 kcal
    -analytic -171.9065 -0.077993 2839.319 71.595 0.0
END
"""

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def Sc_o2(T):
    """Schmidt number for O₂ in freshwater (Wanninkhof, 2014)."""
    return 1745.1 - 124.34*T + 4.8055*T**2 - 0.10115*T**3 + 0.00086842*T**4

def Sc_co2(T):
    """Schmidt number for CO₂ in freshwater (Wanninkhof, 2014)."""
    return 1923.6 - 125.06*T + 4.3773*T**2 - 0.085681*T**3 + 0.00070284*T**4

def ko2_to_k600(ko2, T):
    """Convert KO₂ (m d⁻¹) to k₆₀₀ (m d⁻¹) via Schmidt number ratio."""
    return ko2 * (600.0 / Sc_o2(T))**(-0.5)

def k600_to_kco2(k600, T):
    """Convert k₆₀₀ (m d⁻¹) to kCO₂ (m d⁻¹) via Schmidt number ratio."""
    return k600 * (Sc_co2(T) / 600.0)**(-0.5)

def henry_co2(T):
    """Henry's law constant for CO₂ (mol kg⁻¹ atm⁻¹; Weiss, 1974)."""
    TK = T + 273.15
    return np.exp(-58.0931 + 90.5069*(100.0/TK) + 22.294*np.log(TK/100.0))

def d13c_to_R(d13c):
    """δ¹³C (‰ VPDB) → ¹³C/¹²C ratio."""
    return R_VPDB * (1.0 + d13c / 1000.0)

def R_to_d13c(R):
    """¹³C/¹²C ratio → δ¹³C (‰ VPDB)."""
    return (R / R_VPDB - 1.0) * 1000.0

def calc_par(observer, dt):
    """Estimate clear-sky PAR (W m⁻²) from solar geometry.
    Uses Kasten & Young (1989) air mass formula."""
    elev = solar_elevation(observer, dt)
    if elev <= 0:
        return 0.0
    elev_rad = np.radians(elev)
    am = 1.0 / (np.sin(elev_rad) + 0.50572*(elev + 6.07995)**(-1.6364))
    am = min(am, 38.0)
    return max(1361.0 * 0.45 * (0.72**am) * np.sin(elev_rad), 0.0)


def create_phreeqc(output_dir):
    """Write the inline database to disk and load it into a PHREEQC instance."""
    db_path = os.path.join(output_dir, 'phreeqc_v9.dat')
    with open(db_path, 'w') as f:
        f.write(PHREEQC_DB)
    p = IPhreeqc()
    p.load_database(db_path)
    if p.phc_database_error_count > 0:
        raise RuntimeError(f"PHREEQC DB error: {p.get_error_string()}")
    return p


def run_speciation(phreeqc, T, pH, DIC_mgC, d13c, Ca_mgL, Na_mgL, Cl_mgL):
    """Full isotope + calcite speciation using MEASURED Ca, Na, and Cl.

    Parameters
    ----------
    T       : temperature (°C)
    pH      : measured pH
    DIC_mgC : DIC as mg C L⁻¹
    d13c    : δ¹³C-DIC (‰ VPDB)
    Ca_mgL  : dissolved calcium (mg L⁻¹)
    Na_mgL  : dissolved sodium (mg L⁻¹)
    Cl_mgL  : dissolved chloride (mg L⁻¹)

    Returns
    -------
    dict with speciation, pCO₂, SI_Calcite, species-specific δ¹³C, etc.

    Notes
    -----
    Measured Na⁺ is used for charge balance (adjusted to absorb unmeasured
    ions such as K⁺, Mg²⁺, SO₄²⁻).  Ca²⁺ and Cl⁻ are fixed at measured
    values.
    """
    R = d13c_to_R(d13c)
    total_mmol = DIC_mgC / 12.011
    c12 = total_mmol / (1.0 + R)
    c13 = total_mmol * R / (1.0 + R)
    Ca_mmol = Ca_mgL / 40.08
    Na_mmol = Na_mgL / 22.99
    Cl_mmol = Cl_mgL / 35.45

    phreeqc.run_string(
        f"SOLUTION 1\n"
        f"    temp {T}\n"
        f"    pH {pH}\n"
        f"    C(4) {c12}\n"
        f"    [13C](4) {c13}\n"
        f"    Ca {Ca_mmol}\n"
        f"    Na {Na_mmol} charge\n"
        f"    Cl {Cl_mmol}\n"
        f"SELECTED_OUTPUT\n"
        f"    -reset false\n"
        f"    -pH\n"
        f"    -totals C(4) [13C](4) Ca Na Cl\n"
        f"    -molalities CO2 HCO3- CO3-2 [13C]O2 H[13C]O3- [13C]O3-2 Ca+2\n"
        f"    -si CO2(g) Calcite\n"
        f"END\n"
    )
    v = phreeqc.get_selected_output_array()[1]
    # Indices: 0=pH, 1=C(4), 2=[13C](4), 3=Ca, 4=Na, 5=Cl,
    #          6=CO2, 7=HCO3, 8=CO3,
    #          9=[13C]O2, 10=H[13C]O3, 11=[13C]O3,
    #          12=Ca+2, 13=si_CO2g, 14=si_Calcite
    t12 = v[1]; t13 = v[2]
    CO2_12 = v[6]; HCO3_12 = v[7]; CO3_12 = v[8]
    CO2_13 = v[9]; HCO3_13 = v[10]; CO3_13 = v[11]
    return {
        'pH': v[0],
        'pCO2': 10**v[13] * 1e6,
        'SI_Calcite': v[14],
        'Ca_free': v[12],      # mol/kgw free Ca²⁺
        'Ca_total': v[3],      # mol/kgw total Ca
        'Na_total': v[4],      # mol/kgw total Na (charge-balanced)
        'Cl_total': v[5],      # mol/kgw total Cl
        'CO2_12': CO2_12, 'HCO3_12': HCO3_12, 'CO3_12': CO3_12,
        'CO2_13': CO2_13, 'HCO3_13': HCO3_13, 'CO3_13': CO3_13,
        'tot12': t12, 'tot13': t13,
        'd13C_DIC':  R_to_d13c(t13/t12) if t12 > 0 else -999,
        'd13C_CO2':  R_to_d13c(CO2_13/CO2_12) if CO2_12 > 0 else -999,
        'd13C_HCO3': R_to_d13c(HCO3_13/HCO3_12) if HCO3_12 > 0 else -999,
        'fCO2':  CO2_12/t12 if t12 > 0 else 0,
        'fHCO3': HCO3_12/t12 if t12 > 0 else 0,
    }


###############################################################################
# MAIN MODEL
###############################################################################

def main():
    print("=" * 70)
    print("DIEL 13C-DIC MODEL v9 — Tri-Objective (d13C + GPP + ER), T-dependent ER")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    df_raw = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    df_raw.columns = df_raw.columns.str.strip()

    # Detect date format: 2-digit vs 4-digit year
    sample = df_raw['DateTime'].iloc[0]
    fmt = '%m/%d/%y %H:%M' if len(sample.split('/')[-1].split()[0]) == 2 else '%m/%d/%Y %H:%M'
    df_raw['DateTime'] = pd.to_datetime(df_raw['DateTime'], format=fmt)

    df_raw = df_raw.rename(columns={
        'DO (mg L-1)': 'DO', 'pH': 'pH_meas', 'Temperature (°C)': 'temp',
        'Calcium (mg L-1)': 'Ca_mgL',
        'Sodium (mg L-1)': 'Na_mgL',
        'Chloride (mg L-1)': 'Cl_mgL',
        'd13C-DIC (‰)': 'd13C_DIC', 'DIC  (mg L-1)': 'DIC_mgL',
        'GPP (mg O2 L-1 h-1)': 'GPP', 'R (mg O2 L-1 h-1)': 'ER',
        'KO2 (m d-1)': 'KO2',
    }).rename(columns={'DateTime': 'datetime'})

    df_obs = df_raw.copy()
    obs_times = df_obs['datetime'].values
    obs_d13c  = df_obs['d13C_DIC'].values
    obs_gpp   = df_obs['GPP'].values
    obs_er    = df_obs['ER'].values

    print(f"\n  CSV:  {CSV_PATH}")
    print(f"  Obs:  {len(df_raw)}")
    print(f"  Date: {df_raw['datetime'].min()} → {df_raw['datetime'].max()}")
    print(f"  δ13C: [{obs_d13c.min():.2f}, {obs_d13c.max():.2f}] ‰")
    print(f"  DIC:  [{df_raw['DIC_mgL'].min():.2f}, {df_raw['DIC_mgL'].max():.2f}] mg/L")
    print(f"  Ca:   [{df_raw['Ca_mgL'].min():.1f}, {df_raw['Ca_mgL'].max():.1f}] mg/L")
    print(f"  Na:   [{df_raw['Na_mgL'].min():.3f}, {df_raw['Na_mgL'].max():.3f}] mg/L")
    print(f"  Cl:   [{df_raw['Cl_mgL'].min():.3f}, {df_raw['Cl_mgL'].max():.3f}] mg/L")
    print(f"  Temp: [{df_raw['temp'].min():.1f}, {df_raw['temp'].max():.1f}] °C")
    print(f"  pH:   [{df_raw['pH_meas'].min():.2f}, {df_raw['pH_meas'].max():.2f}]")

    # Hourly interpolation
    df = df_raw.set_index('datetime').resample('1h').interpolate('linear').reset_index()
    n = len(df)
    print(f"  Hourly: {n} points")

    # ── Solar geometry + PAR ─────────────────────────────────────────────
    tz  = timezone(timedelta(hours=TZ_OFF))
    loc = LocationInfo("Lake", "Region", "UTC", LATITUDE, LONGITUDE)
    sun_info = {}
    for d in df['datetime'].dt.date.unique():
        s = sun(loc.observer, date=d, tzinfo=tz)
        sun_info[d] = (s['sunrise'], s['sunset'])
        print(f"  {d}: Rise {s['sunrise'].strftime('%H:%M')}, Set {s['sunset'].strftime('%H:%M')}")

    PAR_arr = np.zeros(n)
    for i in range(n):
        PAR_arr[i] = calc_par(loc.observer, df['datetime'].iloc[i].replace(tzinfo=tz))
    PAR_max = PAR_arr.max()
    print(f"  PAR peak: {PAR_max:.1f} W/m²")

    # ── Pre-compute PHREEQC speciation (with MEASURED Ca) ────────────────
    print("\nPre-computing PHREEQC speciation (measured Ca for Calcite SI)...")
    phreeqc = create_phreeqc(OUTPUT_DIR)

    spec = []
    for i in range(n):
        s = run_speciation(
            phreeqc, df['temp'].iloc[i], df['pH_meas'].iloc[i],
            df['DIC_mgL'].iloc[i], df['d13C_DIC'].iloc[i],
            df['Ca_mgL'].iloc[i], df['Na_mgL'].iloc[i], df['Cl_mgL'].iloc[i],
        )
        spec.append(s)

    pH_mod       = np.array([s['pH'] for s in spec])
    pCO2_arr     = np.array([s['pCO2'] for s in spec])
    SI_calc      = np.array([s['SI_Calcite'] for s in spec])
    CO2_mg       = np.array([s['CO2_12'] * 44010 for s in spec])
    HCO3_mg      = np.array([s['HCO3_12'] * 61017 for s in spec])
    CO3_mg       = np.array([s['CO3_12'] * 60009 for s in spec])
    d13C_CO2_eq  = np.array([s['d13C_CO2'] for s in spec])
    d13C_HCO3_eq = np.array([s['d13C_HCO3'] for s in spec])
    fCO2_12      = np.array([s['fCO2'] for s in spec])
    fHCO3_12     = np.array([s['fHCO3'] for s in spec])

    print(f"  Init: pCO2={pCO2_arr[0]:.0f} µatm, SI_Calcite={SI_calc[0]:.3f}")
    print(f"  SI_Calcite range: [{SI_calc.min():.3f}, {SI_calc.max():.3f}]")
    print(f"  Fractionation CO2−HCO3 = {d13C_CO2_eq[0] - d13C_HCO3_eq[0]:.2f} ‰")

    # Derived arrays
    temp    = df['temp'].values;    TK = temp + 273.15
    DIC_mgL = df['DIC_mgL'].values; ER_raw = df['ER'].values
    KO2_raw = df['KO2'].values;     RQ = 12.011 / 32.0
    dDIC_dt = np.diff(DIC_mgL, prepend=DIC_mgL[0])
    T_ref   = np.mean(temp)

    k600_b     = np.array([ko2_to_k600(KO2_raw[i], temp[i]) for i in range(n)])
    kco2_b     = np.array([k600_to_kco2(k600_b[i], temp[i]) for i in range(n)])
    CO2_eq_atm = np.array([henry_co2(temp[i]) * ATM_CO2_PPM * 1e-6 for i in range(n)])
    CO2_mol    = np.array([s['CO2_12'] for s in spec])
    flux_mmol  = kco2_b * (CO2_mol - CO2_eq_atm) * 1e6

    # ── Model function ──────────────────────────────────────────────────
    def run_model(pvec):
        GPP_max, Ik, Q10_gpp, ER_base, Q10_er, pf, d13c_org, k_gas_iso = pvec

        # GPP: Michaelis-Menten light x Q10 temperature
        GPP_t = np.zeros(n)
        for i in range(n):
            if PAR_arr[i] > 0:
                GPP_t[i] = GPP_max * (PAR_arr[i] / (Ik + PAR_arr[i])) \
                           * Q10_gpp**((temp[i] - T_ref) / 10.0)

        # ER: Q10 temperature dependence (ER_base is negative at T_ref)
        ER_t = np.array([ER_base * Q10_er**((temp[i] - T_ref) / 10.0)
                         for i in range(n)])

        d13C = np.zeros(n)
        d13C[0] = df['d13C_DIC'].iloc[0]
        for i in range(1, n):
            gpp_C = GPP_t[i] * RQ
            er_C  = -ER_t[i] * RQ
            gas_C = er_C - gpp_C - dDIC_dt[i]

            shift       = d13C[i-1] - df['d13C_DIC'].iloc[i-1]
            d13C_CO2_s  = d13C_CO2_eq[i-1] + shift
            d13C_HCO3_s = d13C_HCO3_eq[i-1] + shift

            iso_er  = er_C * d13c_org
            d13_gpp = fCO2_12[i-1]*(d13C_CO2_s + pf) + fHCO3_12[i-1]*(d13C_HCO3_s + pf)
            iso_gpp = gpp_C * d13_gpp

            if gas_C > 0:
                iso_gas = gas_C * (d13C_CO2_s - 0.8) * k_gas_iso
            else:
                iso_gas = gas_C * (ATM_d13C - (-0.373*1000/TK[i-1] + 0.19)) * k_gas_iso

            d13C[i] = (DIC_mgL[i-1]*d13C[i-1] + iso_er - iso_gpp - iso_gas) \
                       / DIC_mgL[i] if DIC_mgL[i] > 0.5 else d13C[i-1]

        return d13C, GPP_t, ER_t

    # ── Objective with Bayesian priors ──────────────────────────────────
    def objective(pvec):
        try:
            d13C, GPP_t, ER_t = run_model(pvec)
        except Exception:
            return 1e6

        mod_d13c = np.interp(obs_times.astype(np.float64),
                             df['datetime'].values.astype(np.float64), d13C)
        mod_gpp  = np.interp(obs_times.astype(np.float64),
                             df['datetime'].values.astype(np.float64), GPP_t)
        mod_er   = np.interp(obs_times.astype(np.float64),
                             df['datetime'].values.astype(np.float64), ER_t)
        if not np.all(np.isfinite(mod_d13c)):
            return 1e6

        # --- Tri-objective: d13C + GPP + ER ---
        rmse_d13c = np.sqrt(np.mean((obs_d13c - mod_d13c)**2))

        hours    = np.array([pd.Timestamp(t).hour for t in obs_times])
        gpp_mask = hours != 18   # exclude twilight-averaged GPP
        rmse_gpp = np.sqrt(np.mean((obs_gpp[gpp_mask] - mod_gpp[gpp_mask])**2)) \
                   if gpp_mask.sum() > 0 else 0.0

        rmse_er  = np.sqrt(np.mean((obs_er - mod_er)**2))

        # Weights normalise each objective to the d13C scale
        # d13C range ~ 1.4, GPP range ~ 30, ER range ~ 1.1
        w_gpp = 0.05    # ≈ 1.4/30
        w_er  = 1.0     # ≈ 1.4/1.1 ≈ 1.3 → use 1.0 for equal weight
        cost  = rmse_d13c + w_gpp * rmse_gpp + w_er * rmse_er

        # Bayesian Gaussian penalty priors
        GPP_max, Ik, Q10_gpp, ER_base, Q10_er, pf, d13c_org, k_gas_iso = pvec
        cost += 0.02 * ((pf        - (-20)) / 5.0)**2
        cost += 0.02 * ((d13c_org  - (-28)) / 3.0)**2
        cost += 0.02 * ((k_gas_iso - 1.0)   / 0.5)**2
        cost += 0.01 * ((Q10_gpp   - 2.0)   / 0.5)**2
        cost += 0.01 * ((Q10_er    - 2.0)   / 0.5)**2
        cost += 0.01 * ((ER_base   - np.mean(ER_raw)) / 1.0)**2

        return cost

    # ── Optimise ─────────────────────────────────────────────────────────
    print("\nOptimising (bounded DE + Bayesian priors)...")
    bounds = [
        (5.0, 50.0),     # GPP_max  (mg O₂ L⁻¹ h⁻¹)
        (20.0, 500.0),   # Ik       (W m⁻²)
        (1.2, 3.5),      # Q10_gpp
        (-15.0, -5.0),   # ER_base  (mg O₂ L⁻¹ h⁻¹ at T_ref, negative)
        (1.2, 3.5),      # Q10_er
        (-30.0, -10.0),  # PF       (‰)
        (-34.0, -22.0),  # δ¹³C_org (‰)
        (0.3, 2.5),      # k_gas_iso
    ]

    result = differential_evolution(
        objective, bounds, seed=42,
        maxiter=500, tol=1e-7,
        popsize=30, mutation=(0.5, 1.5), recombination=0.9,
        polish=True,
    )
    bx = result.x

    plabels = ['GPP_max (mgO2/L/h)', 'Ik (W/m²)', 'Q10_GPP',
               'ER_base (mgO2/L/h)', 'Q10_ER', 'PF (‰)', 'δ13C_org (‰)', 'k_gas_iso']
    print(f"\n  OPTIMISED (cost = {result.fun:.5f}):")
    for lab, val in zip(plabels, bx):
        print(f"    {lab:>22s}: {val:9.3f}")

    # ── Final run ────────────────────────────────────────────────────────
    d13C_final, GPP_final, ER_final = run_model(bx)

    mod_d13c = np.interp(obs_times.astype(np.float64),
                         df['datetime'].values.astype(np.float64), d13C_final)
    mod_gpp  = np.interp(obs_times.astype(np.float64),
                         df['datetime'].values.astype(np.float64), GPP_final)

    rmse_d13c = np.sqrt(np.mean((obs_d13c - mod_d13c)**2))
    r2_d13c   = 1 - np.sum((obs_d13c - mod_d13c)**2) / \
                    np.sum((obs_d13c - np.mean(obs_d13c))**2)

    hours    = np.array([pd.Timestamp(t).hour for t in obs_times])
    gpp_mask = hours != 18
    rmse_gpp = np.sqrt(np.mean((obs_gpp[gpp_mask] - mod_gpp[gpp_mask])**2))
    r2_gpp   = 1 - np.sum((obs_gpp[gpp_mask] - mod_gpp[gpp_mask])**2) / \
                   np.sum((obs_gpp[gpp_mask] - np.mean(obs_gpp[gpp_mask]))**2)

    mod_er   = np.interp(obs_times.astype(np.float64),
                         df['datetime'].values.astype(np.float64), ER_final)
    rmse_er  = np.sqrt(np.mean((obs_er - mod_er)**2))
    r2_er    = 1 - np.sum((obs_er - mod_er)**2) / \
                   np.sum((obs_er - np.mean(obs_er))**2)

    print(f"\n  δ13C-DIC : RMSE = {rmse_d13c:.4f} ‰,  R² = {r2_d13c:.4f}")
    print(f"  GPP (ex18): RMSE = {rmse_gpp:.3f} mg O₂/L/h,  R² = {r2_gpp:.4f}")
    print(f"  ER:         RMSE = {rmse_er:.3f} mg O₂/L/h,  R² = {r2_er:.4f}")

    print(f"\n  {'Time':>16s} {'d13C_o':>7s} {'d13C_m':>7s} {'Δ':>6s}"
          f"  {'GPP_o':>6s} {'GPP_m':>6s}  {'ER_o':>6s} {'ER_m':>6s}  {'SI':>5s}")
    for i in range(len(obs_d13c)):
        ts = pd.Timestamp(obs_times[i]).strftime('%m/%d %H:%M')
        si_i = np.interp(obs_times[i].astype(np.float64),
                         df['datetime'].values.astype(np.float64), SI_calc)
        print(f"  {ts:>16s} {obs_d13c[i]:7.3f} {mod_d13c[i]:7.3f}"
              f" {obs_d13c[i]-mod_d13c[i]:+6.3f}"
              f"  {obs_gpp[i]:6.2f} {mod_gpp[i]:6.2f}"
              f"  {obs_er[i]:6.2f} {mod_er[i]:6.2f}"
              f"  {si_i:5.3f}")

    # ── CSV export ───────────────────────────────────────────────────────
    results = pd.DataFrame({
        'DateTime':              df['datetime'],
        'Temperature_C':         temp,
        'pH_measured':           df['pH_meas'].values,
        'pH_modeled':            pH_mod,
        'pCO2_uatm':            pCO2_arr,
        'Ca_measured_mgL':       df['Ca_mgL'].values,
        'Na_measured_mgL':       df['Na_mgL'].values,
        'Cl_measured_mgL':       df['Cl_mgL'].values,
        'SI_Calcite':            SI_calc,
        'PAR_Wm2':              PAR_arr,
        'GPP_observed_mgO2Lh':  df['GPP'].values,
        'GPP_modeled_mgO2Lh':   GPP_final,
        'ER_observed_mgO2Lh':   ER_raw,
        'ER_modeled_mgO2Lh':    ER_final,
        'NEP_mgO2Lh':           GPP_final + ER_final,
        'KO2_m_d':              KO2_raw,
        'k600_m_d':             k600_b,
        'CO2flux_mmol_m2_d':    flux_mmol,
        'CO2aq_mgL':            CO2_mg,
        'HCO3_mgL':             HCO3_mg,
        'CO3_mgL':              CO3_mg,
        'DIC_mgL':              DIC_mgL,
        'd13C_DIC_measured':     df['d13C_DIC'].values,
        'd13C_DIC_modeled':      d13C_final,
        'd13C_residual':         df['d13C_DIC'].values - d13C_final,
        'd13C_CO2aq_eq':         d13C_CO2_eq,
        'd13C_HCO3_eq':          d13C_HCO3_eq,
        'Daylight_flag':         (PAR_arr > 0).astype(int),
    })
    csv_path = os.path.join(OUTPUT_DIR, 'diel_v9_results.csv')
    results.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n  CSV: {csv_path}")

    # ── 10-panel figure ──────────────────────────────────────────────────
    print("\nPlotting...")
    c_obs, c_mod = '#1a1a1a', '#2166ac'
    c_gpp, c_er, c_flux = '#2ca02c', '#d62728', '#ff7f0e'

    night_spans = []
    for d in df['datetime'].dt.date.unique():
        sr, ss = sun_info[d]
        t0 = pd.Timestamp(datetime.combine(d, datetime.min.time()))
        night_spans.append((t0, pd.Timestamp(sr.replace(tzinfo=None))))
        night_spans.append((pd.Timestamp(ss.replace(tzinfo=None)),
                            t0 + pd.Timedelta(days=1)))

    def shade(ax):
        for s, e in night_spans:
            ax.axvspan(s, e, alpha=0.08, color='gray', zorder=0)

    tt  = results['DateTime']
    odt = df_obs['datetime']

    fig, axes = plt.subplots(10, 1, figsize=(14, 36), sharex=True)

    # 1 ─ PAR
    ax = axes[0]
    ax.fill_between(tt, 0, PAR_arr, color='#f4a460', alpha=0.4)
    ax.plot(tt, PAR_arr, '-', color='#d2691e', lw=1.5, label='PAR (clear-sky)')
    ax2 = ax.twinx()
    ax2.plot(tt, PAR_arr*4.57, '--', color='#8b6914', lw=0.8, alpha=0.5)
    ax2.set_ylabel('µmol m⁻² s⁻¹', fontsize=8, color='#8b6914')
    ax2.tick_params(axis='y', labelcolor='#8b6914', labelsize=8)
    shade(ax); ax.set_ylabel('PAR (W m⁻²)'); ax.legend(fontsize=9)
    ax.set_title(f'PAR at {LATITUDE:.4f}°N, {LONGITUDE:.4f}°W', fontweight='bold')
    ax.grid(alpha=0.3)

    # 2 ─ pH
    ax = axes[1]
    ax.plot(odt, df_obs['pH_meas'], 'ko', ms=5, label='Measured', zorder=3)
    ax.plot(tt, pH_mod, '-', color=c_mod, lw=1.5, label='PHREEQC')
    shade(ax); ax.set_ylabel('pH'); ax.legend(fontsize=9)
    ax.set_title('pH', fontweight='bold'); ax.grid(alpha=0.3)

    # 3 ─ pCO₂
    ax = axes[2]
    ax.plot(tt, pCO2_arr, '-', color=c_mod, lw=1.5)
    ax.axhline(ATM_CO2_PPM, color='gray', ls='--', lw=1,
               label=f'Atmospheric ({ATM_CO2_PPM})')
    shade(ax); ax.set_ylabel('pCO₂ (µatm)'); ax.legend(fontsize=9)
    ax.set_title('pCO₂', fontweight='bold'); ax.grid(alpha=0.3)

    # 4 ─ GPP
    ax = axes[3]
    ax.fill_between(tt, 0, GPP_final, color=c_gpp, alpha=0.2)
    ax.plot(tt, GPP_final, '-', color=c_gpp, lw=2,
            label=f'Model (max={bx[0]:.1f}, Ik={bx[1]:.0f}, Q10g={bx[2]:.2f})')
    ax.plot(odt, obs_gpp, 'o', color='darkgreen', ms=6, zorder=3, label='Observed')
    shade(ax); ax.set_ylabel('GPP (mg O₂ L⁻¹ h⁻¹)'); ax.legend(fontsize=8)
    ax.set_title(f'GPP: PAR × Michaelis-Menten × Q10 | RMSE={rmse_gpp:.2f}, '
                 f'R²={r2_gpp:.3f}', fontweight='bold')
    ax.grid(alpha=0.3)

    # 5 ─ ER (T-dependent)
    ax = axes[4]
    ax.fill_between(tt, 0, ER_final, color=c_er, alpha=0.2)
    ax.plot(tt, ER_final, '-', color=c_er, lw=2,
            label=f'Model (base={bx[3]:.1f}, Q10={bx[4]:.2f})')
    ax.plot(odt, df_obs['ER'], 'o', color='darkred', ms=5,
            label='Observed', zorder=3)
    shade(ax); ax.set_ylabel('ER (mg O₂ L⁻¹ h⁻¹)'); ax.legend(fontsize=9)
    ax.set_title(f'Ecosystem Respiration (Q10 model) | RMSE={rmse_er:.3f}, '
                 f'R²={r2_er:.3f}', fontweight='bold')
    ax.grid(alpha=0.3)

    # 6 ─ CO₂ flux
    ax = axes[5]
    ax.fill_between(tt, 0, flux_mmol, where=flux_mmol >= 0,
                    color=c_flux, alpha=0.3, label='Outgassing')
    ax.fill_between(tt, 0, flux_mmol, where=flux_mmol < 0,
                    color='steelblue', alpha=0.3, label='Ingassing')
    ax.plot(tt, flux_mmol, '-', color=c_flux, lw=1.5)
    ax.axhline(0, color='k', lw=0.5)
    shade(ax); ax.set_ylabel('CO₂ Flux (mmol m⁻² d⁻¹)'); ax.legend(fontsize=9)
    ax.set_title('Air-Water CO₂ Exchange', fontweight='bold'); ax.grid(alpha=0.3)

    # 7 ─ Calcite SI  (with Ca overlay)
    ax = axes[6]
    ax.plot(tt, SI_calc, '-', color='#8856a7', lw=2, label='SI Calcite')
    ax.axhline(0, color='k', lw=1, ls='--', label='Equilibrium')
    ax.fill_between(tt, SI_calc, 0, where=SI_calc > 0,
                    color='#8856a7', alpha=0.15, label='Supersaturated')
    ax.fill_between(tt, SI_calc, 0, where=SI_calc < 0,
                    color='#e0ecf4', alpha=0.5, label='Undersaturated')
    shade(ax); ax.set_ylabel('SI Calcite'); ax.legend(fontsize=8, loc='upper left')
    ax3 = ax.twinx()
    ax3.plot(tt, df['Ca_mgL'].values, 's-', color='#e6550d', ms=3, lw=1,
             alpha=0.7, label='Ca²⁺ (mg/L)')
    ax3.set_ylabel('Ca²⁺ (mg L⁻¹)', fontsize=9, color='#e6550d')
    ax3.tick_params(axis='y', labelcolor='#e6550d', labelsize=8)
    ax3.legend(fontsize=8, loc='upper right')
    ax.set_title('Calcite Saturation (measured Ca²⁺)', fontweight='bold')
    ax.grid(alpha=0.3)

    # 8 ─ DIC species
    ax = axes[7]
    ax.plot(tt, HCO3_mg, '-', color='#1b9e77', lw=1.5, label='HCO₃⁻')
    ax.plot(tt, CO2_mg, '-', color='#d95f02', lw=1.5, label='CO₂(aq)')
    ax.plot(tt, CO3_mg, '-', color='#7570b3', lw=1.5, label='CO₃²⁻')
    shade(ax); ax.set_ylabel('mg L⁻¹'); ax.legend(fontsize=9)
    ax.set_title('DIC Speciation', fontweight='bold'); ax.grid(alpha=0.3)

    # 9 ─ Species δ¹³C
    ax = axes[8]
    ax.plot(tt, d13C_CO2_eq, '-', color='#d95f02', lw=1.5, label='δ¹³C-CO₂(aq)')
    ax.plot(tt, d13C_HCO3_eq, '-', color='#1b9e77', lw=1.5, label='δ¹³C-HCO₃⁻')
    shade(ax); ax.set_ylabel('δ¹³C (‰)'); ax.legend(fontsize=9)
    ax.set_title('Equilibrium ¹³C Fractionation (PHREEQC)', fontweight='bold')
    ax.grid(alpha=0.3)

    # 10 ─ δ¹³C-DIC
    ax = axes[9]
    ax.plot(odt, obs_d13c, 'ko', ms=7, zorder=3, label='Measured')
    ax.plot(tt, d13C_final, '-', color=c_mod, lw=2, zorder=2, label='Modeled')
    ax.fill_between(tt, d13C_final - rmse_d13c, d13C_final + rmse_d13c,
                    color=c_mod, alpha=0.15, label=f'±1σ ({rmse_d13c:.2f}‰)')
    shade(ax); ax.set_ylabel('δ¹³C-DIC (‰ VPDB)'); ax.set_xlabel('Date / Time')
    ax.legend(fontsize=9)
    ax.set_title(f'δ¹³C-DIC | R² = {r2_d13c:.3f}, RMSE = {rmse_d13c:.3f} ‰',
                 fontweight='bold')
    ax.grid(alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))

    plt.tight_layout()
    for fmt in ['svg', 'png']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'diel_v9_results.{fmt}'),
                    format=fmt, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Figures: {OUTPUT_DIR}/")
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
