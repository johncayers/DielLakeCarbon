"""
Diel carbonate–isotope model (freshwater) with wind-dependent k and HCO3- uptake.

Features
--------
- Tracks 12C and 13C in CO2(aq), HCO3-, CO3-- (mmol m^-2 inventories)
- Solves pH and carbonate speciation each step from DIC and TA (freshwater TA definition)
- Enforces equilibrium isotopic partition among species (alpha_eq)
- Gas exchange on CO2(aq) using wind-dependent k and Henry’s law; kinetic isotope effects
- Respiration adds CO2; photosynthesis removes CO2 and/or HCO3- with distinct fractionations
- TA held constant (first-order diel assumption)

References / defaults (didactic values)
---------------------------------------
- Weiss (1974) CO2 solubility (K0; mol L^-1 atm^-1)
- Freshwater carbonate: pK1~6.3, pK2~10.3, pKw~14 at 25 °C
- Equilibrium isotope fractionation (25 °C): HCO3- ~ +9 ‰, CO3-- ~ +18 ‰ vs CO2
- Gas-exchange kinetic fractionation: invasion ~ -1 ‰; evasion ~ +9 ‰
- k600 parameterizations:
    Cole & Caraco (1998): k600 = 2.07 + 0.215 u10^1.7 (cm h^-1)
    Wanninkhof (1992):    k600 = 0.31 u10^2       (cm h^-1)
- Schmidt number scaling: k = k600 * (Sc/600)^(-n), n~0.5 typical for smooth flow

Author: (translated for Python)
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 0) d <-> R (13C/12C) helpers
# ---------------------------

R_STD = 0.0111802  # VPDB

def delta_to_R(delta_permil: float) -> float:
    return R_STD * (1.0 + delta_permil / 1000.0)

def R_to_delta(R: float) -> float:
    return 1000.0 * (R / R_STD - 1.0)

def f13_from_R(R: float) -> float:
    """Fraction of 13C given 13C/12C = R."""
    return R / (1.0 + R)

# ---------------------------
# 1) Physicochemical constants
# ---------------------------

def K0_CO2_Weiss(T_C: float, S: float = 0.0) -> float:
    """
    Weiss (1974) CO2 solubility in water/seawater (mol L^-1 atm^-1).
    T in °C; S in PSU (0 for freshwater).
    """
    T_K = T_C + 273.15
    A1, A2, A3 = -58.0931, 90.5069, 22.2940
    B1, B2, B3 = 0.027766, -0.025888, 0.0050578
    lnK0 = (A1 + A2 * (100.0 / T_K) + A3 * math.log(T_K / 100.0) +
            S * (B1 + B2 * (T_K / 100.0) + B3 * (T_K / 100.0) ** 2))
    return math.exp(lnK0)

# Freshwater carbonate constants at ~25 °C (didactic)
pK1, pK2, pKw = 6.3, 10.3, 14.0
K1, K2, Kw = 10 ** (-pK1), 10 ** (-pK2), 10 ** (-pKw)

# Equilibrium isotope fractionation among species (CO2 basis)
alpha_eq_HCO3 = 1.009   # +9 ‰ vs CO2
alpha_eq_CO3  = 1.018   # +18 ‰ vs CO2

# Gas-exchange kinetic isotope effects
eps_inv = -1.0; alpha_inv = 1.0 + eps_inv / 1000.0   # invasion
eps_eva = +9.0; alpha_eva = 1.0 + eps_eva / 1000.0   # evasion

# Photosynthetic fractionations (set separately for species)
eps_p_CO2  = -20.0; alpha_p_CO2  = 1.0 + eps_p_CO2  / 1000.0
eps_p_HCO3 = -10.0; alpha_p_HCO3 = 1.0 + eps_p_HCO3 / 1000.0

# ---------------------------
# 2) Speciation solver (freshwater TA definition)
# ---------------------------

def speciate_carbonate(DIC_mol_L: float, TA_mol_L: float, K1: float, K2: float, Kw: float):
    """
    Solve for pH given DIC and TA using freshwater TA definition:
        TA = [HCO3-] + 2[CO3--] + [OH-] - [H+]
    and carbonate speciation:
        alpha0 = 1 / (1 + K1/H + K1*K2/H^2)
        alpha1 = (K1/H)*alpha0
        alpha2 = (K1*K2/H^2)*alpha0
    Returns pH, species concentrations (mol L^-1), and alpha fractions.
    """
    def TA_residual(logH: float) -> float:
        H = 10 ** logH
        alpha0 = 1.0 / (1.0 + K1 / H + (K1 * K2) / (H ** 2))
        alpha1 = (K1 / H) * alpha0
        alpha2 = (K1 * K2) / (H ** 2) * alpha0
        CO2  = alpha0 * DIC_mol_L
        HCO3 = alpha1 * DIC_mol_L
        CO3  = alpha2 * DIC_mol_L
        OH   = Kw / H
        TA_calc = HCO3 + 2.0 * CO3 + OH - H
        return TA_calc - TA_mol_L

    # Bisection in log10(H+) over a wide range (pH 4–10)
    lo, hi = -10.0, -4.0
    f_lo, f_hi = TA_residual(lo), TA_residual(hi)
    if f_lo * f_hi > 0:
        raise RuntimeError("Speciation root not bracketed. Adjust TA/DIC or search bounds.")

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = TA_residual(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
        if abs(hi - lo) < 1e-10:
            break

    logH = 0.5 * (lo + hi)
    H = 10 ** logH
    pH = -math.log10(H)
    alpha0 = 1.0 / (1.0 + K1 / H + (K1 * K2) / (H ** 2))
    alpha1 = (K1 / H) * alpha0
    alpha2 = (K1 * K2) / (H ** 2) * alpha0
    CO2  = alpha0 * DIC_mol_L
    HCO3 = alpha1 * DIC_mol_L
    CO3  = alpha2 * DIC_mol_L
    return {
        "pH": pH,
        "CO2": CO2, "HCO3": HCO3, "CO3": CO3,
        "alpha": {"alpha0": alpha0, "alpha1": alpha1, "alpha2": alpha2},
    }

# ---------------------------
# 3) Isotope equilibrium partition across species
# ---------------------------

def isotope_equilibrium_partition(N12_total: float, N13_total: float,
                                  f_CO2: float, f_HCO3: float, f_CO3: float,
                                  alpha_HCO3: float, alpha_CO3: float):
    """
    Given total 12C and 13C inventories and species mole fractions f_i,
    find R_CO2 such that R_HCO3 = R_CO2 * a_HCO3, R_CO3 = R_CO2 * a_CO3
    and total N13 matches.
    Returns species 12C/13C inventories and species R values.
    """
    N_total = N12_total + N13_total
    N_CO2  = f_CO2  * N_total
    N_HCO3 = f_HCO3 * N_total
    N_CO3  = f_CO3  * N_total

    def target(R_CO2: float) -> float:
        R_HCO3 = R_CO2 * alpha_HCO3
        R_CO3  = R_CO2 * alpha_CO3
        N13_sum = (f13_from_R(R_CO2)  * N_CO2 +
                   f13_from_R(R_HCO3) * N_HCO3 +
                   f13_from_R(R_CO3)  * N_CO3)
        return N13_sum - N13_total

    # Bisection around R_STD
    lo, hi = R_STD * 0.5, R_STD * 1.5
    f_lo, f_hi = target(lo), target(hi)
    if f_lo * f_hi > 0:
        # widen search if not bracketed
        lo, hi = R_STD * 0.1, R_STD * 10.0
        f_lo, f_hi = target(lo), target(hi)
        if f_lo * f_hi > 0:
            raise RuntimeError("Isotope partition root not bracketed.")

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = target(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
        if abs(hi - lo) < 1e-14:
            break

    R_CO2 = 0.5 * (lo + hi)
    R_HCO3 = R_CO2 * alpha_HCO3
    R_CO3  = R_CO2 * alpha_CO3

    f13_CO2  = f13_from_R(R_CO2)
    f13_HCO3 = f13_from_R(R_HCO3)
    f13_CO3  = f13_from_R(R_CO3)

    N13_CO2  = f13_CO2  * N_CO2
    N13_HCO3 = f13_HCO3 * N_HCO3
    N13_CO3  = f13_CO3  * N_CO3

    N12_CO2  = N_CO2  - N13_CO2
    N12_HCO3 = N_HCO3 - N13_HCO3
    N12_CO3  = N_CO3  - N13_CO3

    return {
        "R_CO2": R_CO2, "R_HCO3": R_HCO3, "R_CO3": R_CO3,
        "N12": {"CO2": N12_CO2, "HCO3": N12_HCO3, "CO3": N12_CO3},
        "N13": {"CO2": N13_CO2, "HCO3": N13_HCO3, "CO3": N13_CO3},
    }

# ---------------------------
# 4) Wind -> k: Schmidt & parameterizations
# ---------------------------

def schmidt_CO2_fresh(T_C: float) -> float:
    """Schmidt number for CO2 in freshwater (dimensionless) vs T (°C)."""
    return 1911.0 - 118.11 * T_C + 3.4527 * T_C**2 - 0.04132 * T_C**3

def k600_from_wind(u10: float, method: str = "ColeCaraco1998") -> float:
    """
    Return k600 in cm h^-1 from u10 (m s^-1).
    method: 'ColeCaraco1998' or 'Wanninkhof1992'
    """
    method = method.lower()
    if method == "colecaraco1998":
        return 2.07 + 0.215 * (u10 ** 1.7)
    elif method == "wanninkhof1992":
        return 0.31 * (u10 ** 2.0)
    else:
        raise ValueError("Unknown k600 method.")

def k_from_u10(u10: float, T_C: float, method: str = "ColeCaraco1998", n_exp: float = 0.5) -> float:
    """
    Compute k (m h^-1) from u10 (m s^-1) using k600 parameterization and Schmidt scaling.
    """
    k600_cm_h = k600_from_wind(u10, method=method)
    Sc = schmidt_CO2_fresh(T_C)
    k_cm_h = k600_cm_h * (Sc / 600.0) ** (-n_exp)
    k_m_h = k_cm_h / 100.0
    return k_m_h

# ---------------------------
# 5) Main simulation
# ---------------------------

def run_simulation():
    # Time grid
    hours = np.arange(0, 25)  # 0..24
    dt_h = 1.0

    # Physical
    depth_m = 2.0
    vol_m3_per_m2 = depth_m

    # Temperature & air
    T_C = 25.0
    pCO2_uatm = 420.0
    pCO2_atm = pCO2_uatm * 1e-6
    K0 = K0_CO2_Weiss(T_C, S=0.0)                 # mol L^-1 atm^-1
    CO2_eq_mol_L = K0 * pCO2_atm                  # mol L^-1
    CO2_eq_mmol_m3 = CO2_eq_mol_L * 1e6           # mmol m^-3

    # Wind (u10; m s^-1): calm night, breezy midday
    u10_hour = np.full(24, 1.5)
    u10_hour[6:10] = 3.0    # 06–10h
    u10_hour[10:15] = 4.0   # 10–15h
    u10_hour[15:18] = 2.5   # 15–18h
    k_method = "ColeCaraco1998"
    n_exp = 0.5

    # Biogeochemistry: initial conditions
    DIC0_mmol_m3 = 2000.0           # 2 mmol L^-1
    TA0_mmol_m3  = 2200.0
    delta_DIC0   = -6.5

    DIC0_mmol_m2 = DIC0_mmol_m3 * vol_m3_per_m2
    TA0_mol_L    = TA0_mmol_m3 / 1000.0
    DIC0_mol_L   = DIC0_mmol_m3 / 1000.0

    # Initial isotope inventories from d
    R_DIC0 = delta_to_R(delta_DIC0)
    f13_DIC0 = f13_from_R(R_DIC0)
    N_total0 = DIC0_mmol_m2
    N13_total = f13_DIC0 * N_total0
    N12_total = N_total0 - N13_total

    # Initial speciation & isotope partitioning
    sp0 = speciate_carbonate(DIC0_mol_L, TA0_mol_L, K1, K2, Kw)
    f_CO2  = sp0["alpha"]["alpha0"]
    f_HCO3 = sp0["alpha"]["alpha1"]
    f_CO3  = sp0["alpha"]["alpha2"]

    iso0 = isotope_equilibrium_partition(N12_total, N13_total,
                                         f_CO2, f_HCO3, f_CO3,
                                         alpha_eq_HCO3, alpha_eq_CO3)

    # Inventories (mmol m^-2)
    N12 = {k: float(v) for k, v in iso0["N12"].items()}
    N13 = {k: float(v) for k, v in iso0["N13"].items()}

    # Metabolism (mmol C m^-2 h^-1)
    R_hour   = np.full(24, 4.0)
    GPP_hour = np.zeros(24)
    GPP_hour[6:9]   = 4.0
    GPP_hour[9:15]  = 10.0
    GPP_hour[15:18] = 4.0

    # Split of GPP between HCO3- vs CO2 uptake
    phi_HCO3 = 0.4  # 40% from HCO3-, 60% from CO2

    # Source/sink isotopic signatures
    delta_atm = -8.5; R_atm = delta_to_R(delta_atm)
    delta_R_OM = -26.0; R_resp = delta_to_R(delta_R_OM)

    # Storage
    out = {
        "hour": hours.astype(float),
        "u10": np.full(25, np.nan),
        "k_m_h": np.full(25, np.nan),
        "pH": np.full(25, np.nan),
        "DIC": np.full(25, np.nan),
        "CO2": np.full(25, np.nan),
        "HCO3": np.full(25, np.nan),
        "CO3": np.full(25, np.nan),
        "delta_DIC": np.full(25, np.nan),
        "delta_CO2": np.full(25, np.nan),
        "delta_HCO3": np.full(25, np.nan),
        "delta_CO3": np.full(25, np.nan),
        "F_gas": np.full(25, np.nan),
        "F_GPP_CO2": np.full(25, np.nan),
        "F_GPP_HCO3": np.full(25, np.nan),
        "F_R": np.full(25, np.nan),
    }

    # Time stepping
    for t in range(24):
        # Current totals
        N12_total = sum(N12.values())
        N13_total = sum(N13.values())
        N_total   = N12_total + N13_total
        DIC_mmol_m3 = N_total / vol_m3_per_m2
        DIC_mol_L   = DIC_mmol_m3 / 1000.0

        # Speciation & equilibrium isotope partition
        sp = speciate_carbonate(DIC_mol_L, TA0_mol_L, K1, K2, Kw)
        f_CO2  = sp["alpha"]["alpha0"]
        f_HCO3 = sp["alpha"]["alpha1"]
        f_CO3  = sp["alpha"]["alpha2"]
        pH = sp["pH"]

        iso = isotope_equilibrium_partition(N12_total, N13_total,
                                            f_CO2, f_HCO3, f_CO3,
                                            alpha_eq_HCO3, alpha_eq_CO3)
        N12 = {k: float(v) for k, v in iso["N12"].items()}
        N13 = {k: float(v) for k, v in iso["N13"].items()}
        R_CO2  = iso["R_CO2"]
        R_HCO3 = iso["R_HCO3"]
        R_CO3  = iso["R_CO3"]

        delta_CO2  = R_to_delta(R_CO2)
        delta_HCO3 = R_to_delta(R_HCO3)
        delta_CO3  = R_to_delta(R_CO3)
        delta_DIC  = R_to_delta(N13_total / N12_total)

        # Concentrations for reporting (mmol m^-3)
        CO2_conc = (N12["CO2"]  + N13["CO2"])  / vol_m3_per_m2
        HCO3_conc= (N12["HCO3"] + N13["HCO3"]) / vol_m3_per_m2
        CO3_conc = (N12["CO3"]  + N13["CO3"])  / vol_m3_per_m2

        # Wind-dependent k and gas exchange on CO2
        u10 = u10_hour[t]
        k_m_h = k_from_u10(u10, T_C, method=k_method, n_exp=n_exp)
        CO2_eq = CO2_eq_mmol_m3
        F_gas = k_m_h * (CO2_conc - CO2_eq)  # + = evasion

        if F_gas > 0:
            # Evasion: emitted CO2 lighter than aqueous CO2
            R_emit = R_CO2 / alpha_eva
            f13_emit = f13_from_R(R_emit)
            F13_emit = F_gas * f13_emit
            F12_emit = F_gas - F13_emit
            # Remove from aqueous CO2 pool, limit by availability
            take12 = min(F12_emit, N12["CO2"])
            take13 = min(F13_emit, N13["CO2"])
            N12["CO2"] -= take12
            N13["CO2"] -= take13
        elif F_gas < 0:
            # Invasion: incoming CO2 carries atmospheric ratio * alpha_inv
            R_in = R_atm * alpha_inv
            f13_in = f13_from_R(R_in)
            F_in = -F_gas
            F13_in = F_in * f13_in
            F12_in = F_in - F13_in
            N12["CO2"] += F12_in
            N13["CO2"] += F13_in

        # Respiration adds CO2
        F_R = float(R_hour[t])
        if F_R > 0:
            f13_R = f13_from_R(R_resp)
            N13["CO2"] += F_R * f13_R
            N12["CO2"] += F_R * (1.0 - f13_R)

        # Photosynthesis: split between CO2 and HCO3-
        F_GPP = float(GPP_hour[t])
        F_GPP_CO2  = (1.0 - phi_HCO3) * F_GPP
        F_GPP_HCO3 = phi_HCO3 * F_GPP

        # CO2 uptake with ep_CO2
        if F_GPP_CO2 > 0:
            R_bio_CO2 = R_CO2 / alpha_p_CO2
            f13_bio_CO2 = f13_from_R(R_bio_CO2)
            rem13 = min(F_GPP_CO2 * f13_bio_CO2, N13["CO2"])
            rem12 = min(F_GPP_CO2 * (1.0 - f13_bio_CO2), N12["CO2"])
            N13["CO2"] -= rem13
            N12["CO2"] -= rem12

        # HCO3- uptake with ep_HCO3
        if F_GPP_HCO3 > 0:
            R_bio_HCO3 = R_HCO3 / alpha_p_HCO3
            f13_bio_HCO3 = f13_from_R(R_bio_HCO3)
            rem13 = min(F_GPP_HCO3 * f13_bio_HCO3, N13["HCO3"])
            rem12 = min(F_GPP_HCO3 * (1.0 - f13_bio_HCO3), N12["HCO3"])
            N13["HCO3"] -= rem13
            N12["HCO3"] -= rem12
            # Note: TA held constant. For research on TA dynamics, extend model.

        # Store AFTER fluxes
        N12_total = sum(N12.values())
        N13_total = sum(N13.values())
        N_total   = N12_total + N13_total
        DIC_mmol_m3 = N_total / vol_m3_per_m2

        out["hour"][t]        = t
        out["u10"][t]         = u10
        out["k_m_h"][t]       = k_m_h
        out["pH"][t]          = pH
        out["DIC"][t]         = DIC_mmol_m3
        out["CO2"][t]         = CO2_conc
        out["HCO3"][t]        = HCO3_conc
        out["CO3"][t]         = CO3_conc
        out["delta_DIC"][t]   = delta_DIC
        out["delta_CO2"][t]   = delta_CO2
        out["delta_HCO3"][t]  = delta_HCO3
        out["delta_CO3"][t]   = delta_CO3
        out["F_gas"][t]       = F_gas
        out["F_GPP_CO2"][t]   = F_GPP_CO2
        out["F_GPP_HCO3"][t]  = F_GPP_HCO3
        out["F_R"][t]         = F_R

    # Repeat last row for hour=24 continuity
    for key in out.keys():
        if key == "hour":
            out[key][-1] = 24.0
        else:
            out[key][-1] = out[key][-2]

    # Diagnostics
    print(f"\nDaily totals: GPP = {np.sum(GPP_hour):.1f}, R = {np.sum(R_hour):.1f} mmol m^-2 d^-1")
    print(f"Mean u10 = {np.nanmean(out['u10'][:-1]):.2f} m s^-1 | Mean k = {np.nanmean(out['k_m_h'][:-1]):.4f} m h^-1")
    print(f"Mean pH: {np.nanmean(out['pH'][:-1]):.2f} | Mean d13C-DIC: {np.nanmean(out['delta_DIC'][:-1]):.2f} ‰")
    print(f"Equilibrium [CO2] at {pCO2_uatm:.0f} µatm: {CO2_eq_mmol_m3:.1f} mmol m^-3")

    return out

# ---------------------------
# 6) Plotting
# ---------------------------

def plot_outputs(out):
    fig, axs = plt.subplots(3, 2, figsize=(11, 10))
    t = out["hour"]

    axs[0,0].plot(t, out["u10"], color="gray", lw=2)
    axs[0,0].set_xlabel("Hour"); axs[0,0].set_ylabel(r"$u_{10}$ (m s$^{-1}$)")
    axs[0,0].set_title("Wind (10 m)")

    axs[0,1].plot(t, out["k_m_h"], color="navy", lw=2)
    axs[0,1].set_xlabel("Hour"); axs[0,1].set_ylabel(r"$k$ (m h$^{-1}$)")
    axs[0,1].set_title("Gas Transfer Velocity")

    axs[1,0].plot(t, out["pH"], color="steelblue", lw=2)
    axs[1,0].set_xlabel("Hour"); axs[1,0].set_ylabel("pH"); axs[1,0].set_title("pH")

    axs[1,1].plot(t, out["DIC"], color="black", lw=2)
    axs[1,1].set_xlabel("Hour"); axs[1,1].set_ylabel(r"DIC (mmol m$^{-3}$)")
    axs[1,1].set_title("DIC")

    axs[2,0].plot(t, out["CO2"], color="firebrick", lw=2, label="CO$_2$")
    axs[2,0].plot(t, out["HCO3"], color="darkgreen", lw=2, label="HCO$_3^-$")
    axs[2,0].plot(t, out["CO3"], color="goldenrod", lw=2, label="CO$_3^{2-}$")
    axs[2,0].set_xlabel("Hour"); axs[2,0].set_ylabel(r"Species (mmol m$^{-3}$)")
    axs[2,0].set_title("Carbonate Species")
    axs[2,0].legend()

    # New figure for isotopes and gas flux for clarity
    fig2, axs2 = plt.subplots(2, 2, figsize=(11, 8))
    axs2[0,0].plot(t, out["delta_DIC"], color="black", lw=2)
    axs2[0,0].set_xlabel("Hour"); axs2[0,0].set_ylabel(r"$\delta^{13}$C DIC (‰)")
    axs2[0,0].set_title(r"$\delta^{13}$C of DIC")

    axs2[0,1].plot(t, out["delta_CO2"], color="firebrick", lw=2, label="CO$_2$")
    axs2[0,1].plot(t, out["delta_HCO3"], color="darkgreen", lw=2, label="HCO$_3^-$")
    axs2[0,1].plot(t, out["delta_CO3"], color="goldenrod", lw=2, label="CO$_3^{2-}$")
    axs2[0,1].set_xlabel("Hour"); axs2[0,1].set_ylabel(r"$\delta^{13}$C (‰)")
    axs2[0,1].set_title(r"$\delta^{13}$C of species")
    axs2[0,1].legend()

    axs2[1,0].stem(t, out["F_gas"], linefmt="gray", markerfmt=" ", basefmt=" ")
    axs2[1,0].set_xlabel("Hour"); axs2[1,0].set_ylabel(r"$F_{gas}$ (mmol m$^{-2}$ h$^{-1}$)")
    axs2[1,0].set_title("Gas Exchange (+ out)")

    # Hide unused subplot
    axs2[1,1].axis("off")

    plt.tight_layout()
    plt.show()

# ---------------------------
# 7) Entrypoint
# ---------------------------

if __name__ == "__main__":
    out = run_simulation()
    plot_outputs(out)