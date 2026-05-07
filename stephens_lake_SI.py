#!/usr/bin/env python3
"""
Stephens Lake – Diel Water Chemistry: PHREEQC Saturation Index Analysis
========================================================================
Calculates saturation indices (SI) for Calcite, Dolomite, Goethite, and
Pyrolusite for each water sample using PhreeqPy / IPhreeqc with the
phreeqc.dat thermodynamic database (inlined PHREEQC input template).

Outputs
-------
  Stephens_Lake_{Season}_diel_SI.csv  – full diel time series per season
  Stephens_Lake_SI_panel.png / .svg   – 2×2 season panel plot

Requirements
------------
  pip install phreeqpy pandas matplotlib
  IPhreeqc shared library is bundled with phreeqpy on Windows / macOS / Linux.
  phreeqc.dat must be in the same directory as this script.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    import phreeqpy.iphreeqc.phreeqc_dll as _phreeqc_dll
except ImportError as exc:
    raise ImportError(
        "phreeqpy is required. Install with:\n"
        "  pip install phreeqpy\n"
        "The IPhreeqc shared library is bundled with phreeqpy on most platforms."
    ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).parent
DB_PATH  = _HERE / "phreeqc.dat"
CSV_PATH = _HERE / "StephensLake_AllSeasons_Chemistry.csv"
OUT_DIR  = _HERE

MINERALS = ["Calcite", "Dolomite", "Goethite", "Pyrolusite", "Hydroxyapatite"]
SEASONS  = ["Fall", "Winter", "Spring", "Summer"]

# Visual style for each mineral in the time-series plots
MINERAL_STYLE: dict[str, dict] = {
    "Calcite":        {"color": "#1f77b4", "ls": "-",  "lw": 1.8},
    "Dolomite":       {"color": "#ff7f0e", "ls": "--", "lw": 1.8},
    "Goethite":       {"color": "#8B0000", "ls": "-.", "lw": 1.8},
    "Pyrolusite":     {"color": "#9467bd", "ls": ":",  "lw": 2.2},
    "Hydroxyapatite": {"color": "#2ca02c", "ls": (0,(3,1,1,1)), "lw": 1.8},
}

# Molecular weights (g/mol) for unit-conversion comments in build_phreeqc_input
_MW = {
    "Si":    28.086,   # Si element
    "SiO2":  60.084,   # SiO2  (phreeqc.dat gfw_formula for Si)
    "NO3":   62.004,   # NO3-
    "N":     14.007,   # N element (phreeqc.dat gfw_formula for N(5))
    "C":     12.011,   # C element
    "HCO3":  61.016,   # HCO3- (phreeqc.dat gfw_formula for C / C(4))
}

# Raw CSV columns required to compute each mineral's SI.
# If any listed column is NaN for a sample, that mineral's SI is set to NaN
# so it is not plotted for that sample.
MINERAL_REQUIRED_COLS: dict[str, list[str]] = {
    "Calcite":        ["Calcium_mg_L-1",   "DIC_mg_L-1"],
    "Dolomite":       ["Calcium_mg_L-1",   "Magnesium_mg_L-1", "DIC_mg_L-1"],
    "Goethite":       ["Iron_ug_L-1",      "DO_mg_L-1"],
    "Pyrolusite":     ["Manganese_ug_L-1", "DO_mg_L-1"],
    "Hydroxyapatite": ["Calcium_mg_L-1",   "Phosphorus_mg_L-1"],
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: Path) -> pd.DataFrame:
    """
    Read the seasonal chemistry CSV, parse datetimes, and sanitise
    concentration columns by clipping physically impossible negative values
    (below-detection-limit artefacts) to zero.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])

    conc_cols = [c for c in df.columns
                 if any(tag in c for tag in ("_mg_L", "_ug_L"))]
    for col in conc_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PHREEQC input template (inlined)
# ─────────────────────────────────────────────────────────────────────────────
def _safe(val, default: float = 0.0, minimum: float = 0.0) -> float:
    """Return a finite, clamped float safe for use in a PHREEQC input string."""
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else max(v, minimum)
    except (TypeError, ValueError):
        return default


def build_phreeqc_input(row: pd.Series, sol_num: int = 1) -> str:
    """
    Construct and return the complete PHREEQC input string for one water
    sample.  The SOLUTION block, SELECTED_OUTPUT block, and END keyword are
    all included in the returned string (everything in one file).

    Unit-conversion rationale
    -------------------------
    PHREEQC converts mg/L → mol/kgw using the gram-formula-weight (gfw) of
    the *gfw_formula* field in SOLUTION_MASTER_SPECIES of phreeqc.dat.
    The conversions below ensure the correct molar amounts are passed:

      Element   gfw_formula   gfw (g/mol)  Data column          Conversion
      Ca/Mg/Na  element       atomic mass  mg/L as element      none needed
      K/Fe/Mn/P element       atomic mass  mg/L as element      none needed
      S(6)      SO4           96.06        Sulfate_mg_L (SO4)   none needed
      N(5)      N             14.007       Nitrate_mg_L (NO3)   × 14.007/62.004
      Si        SiO2          60.084       Silicon_mg_L (Si)    × 60.084/28.086
      C(4)      HCO3          61.016       DIC_mg_L (C)         × 61.016/12.011
      O(0)      O             15.999       DO_mg_L (O2)         none needed *

    * O(0): PHREEQC divides input by MW_O = 15.999 g/mol.
      8 mg/L input → 8/15999 mol O/kgw = 5.0×10⁻⁴ mol O/kgw.
      Since O₂ = 2 O(0): 2.5×10⁻⁴ mol O₂/kgw = 8 mg/L O₂.  ✓

    Pyrolusite SI depends on pe (redox); O(0) from dissolved oxygen sets the
    redox state.  Goethite SI depends on Fe(III); with positive pe, PHREEQC
    partitions total Fe between Fe²⁺ and Fe³⁺ automatically.
    """
    temp  = _safe(row.get("TEMP_degC"),              default=25.0)
    pH    = _safe(row.get("PH"),                     default=7.0)
    Ca    = _safe(row.get("Calcium_mg_L-1"))
    Mg    = _safe(row.get("Magnesium_mg_L-1"))
    Na    = _safe(row.get("Sodium_mg_L-1"))
    K     = _safe(row.get("Potassium_mg_L-1"))
    Cl    = _safe(row.get("Chloride_mg_L-1"))
    SO4   = _safe(row.get("Sulfate_mg_L-1"))                     # mg/L as SO4
    NO3   = _safe(row.get("Nitrate_mg_L-1"))                     # mg/L as NO3
    P     = _safe(row.get("Phosphorus_mg_L-1"), minimum=1e-9)    # mg/L as P
    Si_el = _safe(row.get("Silicon_mg_L-1"))                     # mg/L as Si
    Fe_ug = _safe(row.get("Iron_ug_L-1"))                        # µg/L as Fe
    Mn_ug = _safe(row.get("Manganese_ug_L-1"))                   # µg/L as Mn
    DIC   = _safe(row.get("DIC_mg_L-1"), minimum=0.01)           # mg/L as C
    DO    = _safe(row.get("DO_mg_L-1"), default=8.0, minimum=0.01)  # mg/L as O2

    # Unit conversions (see docstring table above)
    NO3_as_N   = NO3   * (_MW["N"] / _MW["NO3"])        # → mg/L as N
    Si_as_SiO2 = Si_el * (_MW["SiO2"] / _MW["Si"])      # → mg/L as SiO2
    Fe_mg      = max(Fe_ug / 1_000.0, 1e-9)             # µg/L → mg/L
    Mn_mg      = max(Mn_ug / 1_000.0, 1e-9)             # µg/L → mg/L
    DIC_as_HCO3 = DIC  * (_MW["HCO3"] / _MW["C"])       # mg/L as C → mg/L as HCO3

    # ── Inlined PHREEQC template ──────────────────────────────────────────────
    return (
        f"SOLUTION {sol_num}\n"
        f"    temp    {temp:.4f}\n"
        f"    pH      {pH:.4f}\n"
        f"    units   mg/L\n"
        f"    density 1.0\n"
        f"    Ca      {Ca:.6f}\n"
        f"    Mg      {Mg:.6f}\n"
        f"    Na      {Na:.6f}\n"
        f"    K       {K:.6f}\n"
        f"    Cl      {Cl:.6f}\n"
        f"    S(6)    {SO4:.6f}\n"
        f"    N(5)    {NO3_as_N:.9f}\n"
        f"    P       {P:.10f}\n"
        f"    Si      {Si_as_SiO2:.6f}\n"
        f"    Fe      {Fe_mg:.9f}\n"
        f"    Mn      {Mn_mg:.9f}\n"
        f"    C(4)    {DIC_as_HCO3:.6f}\n"
        f"    O(0)    {DO:.6f}\n"
        f"\n"
        f"SELECTED_OUTPUT\n"
        f"    -si     Calcite Dolomite Goethite Pyrolusite Hydroxyapatite\n"
        f"\n"
        f"END\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  IPhreeqc runner
# ─────────────────────────────────────────────────────────────────────────────
def run_iphreeqc(input_str: str, db_path: Path) -> dict[str, float]:
    """
    Execute IPhreeqc on *input_str* using *db_path* as the thermodynamic
    database and return {mineral_name: SI_value}.

    Returns NaN for any mineral whose SI cannot be retrieved (e.g. very low
    dissolved concentrations that prevent convergence).
    """
    ip = _phreeqc_dll.IPhreeqc()
    ip.load_database(str(db_path))
    rc = ip.run_string(input_str)
    if rc != 0:
        warnings.warn(f"IPhreeqc returned code {rc}: {ip.get_error_string()}")

    arr = ip.get_selected_output_array()
    if len(arr) < 2:
        return {m: np.nan for m in MINERALS}

    header = [str(h).strip() for h in arr[0]]
    data   = list(arr[1])

    result: dict[str, float] = {}
    for mineral in MINERALS:
        key = f"si_{mineral}"
        try:
            v = float(data[header.index(key)])
            result[mineral] = v if np.isfinite(v) else np.nan
        except (ValueError, IndexError, TypeError):
            result[mineral] = np.nan
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Batch saturation-index calculation
# ─────────────────────────────────────────────────────────────────────────────
def calculate_si(df: pd.DataFrame, db_path: Path) -> pd.DataFrame:
    """
    Run IPhreeqc for every row in *df* and append five SI columns
    (Calcite, Dolomite, Goethite, Pyrolusite, Hydroxyapatite).

    Returns the original DataFrame extended with the SI columns.
    """
    records: list[dict] = []
    n = len(df)
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        if i % 5 == 0 or i == n:
            print(f"  {i}/{n}", end="\r", flush=True)
        inp = build_phreeqc_input(row, sol_num=i)
        try:
            si = run_iphreeqc(inp, db_path)
        except Exception as exc:
            warnings.warn(f"Row {idx} ({row.get('DATETIME', '?')}): {exc}")
            si = {m: np.nan for m in MINERALS}
        # Mask SI to NaN where a required species was not measured
        for mineral, req_cols in MINERAL_REQUIRED_COLS.items():
            if any(pd.isna(row.get(col)) for col in req_cols):
                si[mineral] = np.nan
        si["_idx"] = idx
        records.append(si)

    print()  # newline after in-place progress counter
    si_df = pd.DataFrame(records).set_index("_idx")
    return df.join(si_df)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CSV export (diel time series for all parameters)
# ─────────────────────────────────────────────────────────────────────────────
def export_csv(df: pd.DataFrame, out_dir: Path) -> None:
    """Write a complete diel time-series CSV (all parameters + SI) per season."""
    for season in SEASONS:
        sub = df[df["Season"] == season]
        if sub.empty:
            continue
        fpath = out_dir / f"Stephens_Lake_{season}_diel_SI.csv"
        sub.to_csv(fpath, index=False)
        print(f"  {fpath.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  2×2 season panel plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_panel(df: pd.DataFrame, out_dir: Path) -> None:
    """
    2x2 panel plot: one subplot per season, each showing the diel time series
    of SI for all five minerals.  A single shared legend sits below the panels.
    Saved as PNG (150 dpi) and SVG.

    The horizontal line at SI = 0 marks the mineral-solution equilibrium
    boundary (SI > 0 supersaturated; SI < 0 undersaturated).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Shared y-axis range across all seasons
    si_vals = df[MINERALS].to_numpy(dtype=float)
    finite  = si_vals[np.isfinite(si_vals)]
    if finite.size:
        pad   = 0.5
        y_min = finite.min() - pad
        y_max = finite.max() + pad
    else:
        y_min, y_max = -5.0, 5.0

    legend_handles = []
    for ax, season in zip(axes.flat, SEASONS):
        sub = df[df["Season"] == season].sort_values("DATETIME")
        if sub.empty:
            ax.set_title(f"{season} - no data", fontsize=12)
            continue

        for mineral in MINERALS:
            st = MINERAL_STYLE[mineral]
            line, = ax.plot(
                sub["DATETIME"], sub[mineral],
                color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
                marker="o", markersize=4, label=mineral,
            )
            if season == SEASONS[0]:
                legend_handles.append(line)

        ax.axhline(0, color="k", lw=0.9, alpha=0.55)
        ax.set_ylim(y_min, y_max)
        ax.set_title(season, fontsize=13, fontweight="bold")
        ax.set_ylabel("Saturation Index", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, alpha=0.3, lw=0.5)

    for ax in axes.flat:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Reserve space at the bottom for the shared legend, then place it
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.legend(
        handles=legend_handles,
        labels=MINERALS,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(MINERALS),
        fontsize=10,
        framealpha=0.9,
        edgecolor="0.7",
    )

    for ext in ("png", "svg"):
        fpath = out_dir / f"Stephens_Lake_SI_panel.{ext}"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"  {fpath.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Database : {DB_PATH}")
    print(f"Data     : {CSV_PATH}\n")

    df = load_data(CSV_PATH)
    print(f"Loaded {len(df)} samples")
    print(f"Seasons  : {df['Season'].value_counts().to_dict()}\n")

    print("Calculating saturation indices …")
    df_si = calculate_si(df, DB_PATH)

    print("\nMean SI by season:")
    print(df_si.groupby("Season")[MINERALS].mean().round(3).to_string())

    print("\nExporting CSVs:")
    export_csv(df_si, OUT_DIR)

    print("\nGenerating panel plot:")
    plot_panel(df_si, OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
