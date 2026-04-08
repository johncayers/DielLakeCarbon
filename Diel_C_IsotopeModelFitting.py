# ============================================================================
# DIEL CARBON ISOTOPE CYCLING MODEL FOR FRESHWATER LAKES
# Python-PHREEQC Coupled Biogeochemical Simulator
# ============================================================================

# %% [markdown]
# # Diel Carbon Isotope Cycling Model: Lake Biogeochemistry Simulator
# 
# ## Overview
# This notebook implements a coupled Python-PHREEQC model to simulate 24-hour variations 
# in dissolved inorganic carbon (DIC) and its carbon-13 isotope composition (d13C-DIC) 
# in shallow lake systems.
# 
# **Key Processes:**
# 1. Atmospheric CO2 exchange (gas transfer across air-water interface)
# 2. Gross Primary Productivity (GPP) - photosynthetic carbon uptake
# 3. Ecosystem Respiration (ER) - respiratory CO2 production
# 
# **Scientific Foundation:**
# - Carbonate chemistry calculations via PHREEQC
# - Photosynthetic fractionation of stable isotopes
# - Temperature-dependent gas exchange kinetics
# - Diel light-dark cycles based on solar geometry

# %% [markdown]
# ## Installation Requirements
# 
# Run this cell once to install required packages:
# 
# ```bash
# pip install numpy pandas matplotlib seaborn scipy scikit-learn astral phreeqpy
# # Alternative if phreeqpy unavailable: pip install phreeqc
# ```

# %% [markdown]
# ## Cell 1: Imports and Configuration

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.optimize import minimize, curve_fit
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

# For solar calculations
try:
    from astral import LocationInfo
    from astral.sun import sun
except ImportError:
    print("Warning: astral not available. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'astral'])
    from astral import LocationInfo
    from astral.sun import sun

# For PHREEQC integration
try:
    from phreeqpy.iphreeqc import IPhreeqc
    PHREEQC_AVAILABLE = True
except ImportError:
    print("Note: phreeqpy not available. Will use analytical carbonate chemistry instead.")
    PHREEQC_AVAILABLE = False

# %% [markdown]
# ## Cell 2: User Configuration
# 
# **Edit these parameters for your specific site and dataset:**

# %%
# ============================================================================
# USER CONFIGURATION SECTION
# ============================================================================

# Site Information
LATITUDE = 45.5  # degrees North (example: Lake Tahoe region)
LONGITUDE = -120.5  # degrees West
SITE_NAME = "Lake_Site_A"
SIMULATION_DATE = "2025-07-18"  # Format: YYYY-MM-DD
LAKE_DEPTH = 2.0  # meters (well-mixed epilimnion)

# File paths
INPUT_CSV = "SL-20250718.csv"  # Your data file
OUTPUT_DIR = "./model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Biogeochemical Parameters
ATMOSPHERIC_d13C_CO2 = -8.0  # ‰ VPDB (typical atmospheric CO2)
ORGANIC_MATTER_d13C = -28.0  # ‰ VPDB (respired OM, typical freshwater)
PHOTOSYNTHETIC_FRACTIONATION_RANGE = (-15, -30)  # ‰ (will be calibrated)

# Model Configuration
TIME_STEP_HOURS = 1  # Integration time step
OPTIMIZATION_ENABLED = True  # Enable parameter calibration
CONFIDENCE_INTERVAL = 95  # % for error bands

print(f"Configuration Set:")
print(f"  Site: {SITE_NAME}")
print(f"  Date: {SIMULATION_DATE}")
print(f"  Latitude: {LATITUDE}°N")
print(f"  Atmospheric d13C-CO2: {ATMOSPHERIC_d13C_CO2}‰")
print(f"  Organic matter d13C: {ORGANIC_MATTER_d13C}‰")

# %% [markdown]
# ## Cell 3: Data Loading and Validation

# %%
import os

# Load input data
df = pd.read_csv(INPUT_CSV)

# Convert TIME and DATE to datetime
df['DateTime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], format='%m/%d/%Y %I:%M:%S %p')
df = df.sort_values('DateTime').reset_index(drop=True)

# Display data overview
print("=" * 70)
print("DATA VALIDATION REPORT")
print("=" * 70)
print(f"\nTotal records: {len(df)}")
print(f"Time span: {df['DateTime'].min()} to {df['DateTime'].max()}")
print(f"Duration: {(df['DateTime'].max() - df['DateTime'].min()).total_seconds() / 3600:.1f} hours")

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Data summary statistics
print("\nData Summary Statistics:")
print(df[['DO (mg L-1)', 'pH', 'Temperature (°C)', 'DIC  (mg L-1)', '?¹³C-DIC (‰)']].describe())

# Rename columns for convenience
df = df.rename(columns={
    'DO (mg L-1)': 'DO',
    'pH': 'pH_obs',
    'Temperature (°C)': 'Temp',
    'DIC  (mg L-1)': 'DIC',
    '?¹³C-DIC (‰)': 'd13C_DIC_obs',
    'GPP (mg O? L-1 h-1)': 'GPP_obs',
    'R (mg O? L-1 h-1)': 'ER_obs',
    'KO? (m d-1)': 'K02_obs'
})

print("\nData preparation complete. First few rows:")
print(df[['DateTime', 'Temp', 'pH_obs', 'DIC', 'd13C_DIC_obs', 'GPP_obs', 'ER_obs', 'K02_obs']].head())

# %% [markdown]
# ## Cell 4: Solar Radiation and Photoperiod Calculations

# %%
def calculate_solar_times(latitude, longitude, date_str):
    """
    Calculate sunrise/sunset times and solar noon for given date and location.
    
    Parameters:
    -----------
    latitude : float
        Latitude in decimal degrees (North positive)
    longitude : float
        Longitude in decimal degrees (West positive, as -1 * value)
    date_str : str
        Date string in format 'YYYY-MM-DD'
    
    Returns:
    --------
    dict : Dictionary with sunrise, sunset, and solar_noon times (UTC)
    """
    from astral import LocationInfo
    from astral.sun import sun
    
    location = LocationInfo(
        name=SITE_NAME,
        region="USA",
        latitude=latitude,
        longitude=-abs(longitude),  # Astral uses negative for West
        tzinfo='UTC'
    )
    
    date_obj = pd.to_datetime(date_str).date()
    s = sun(location.observer, date=date_obj)
    
    return {
        'sunrise': s['sunrise'],
        'sunset': s['sunset'],
        'solar_noon': s['noon']
    }

def generate_daylight_flag(datetime_series, latitude, longitude, date_str):
    """
    Generate binary daylight flag for each timestamp.
    1 = daylight, 0 = night
    """
    solar_times = calculate_solar_times(latitude, longitude, date_str)
    sunrise = solar_times['sunrise']
    sunset = solar_times['sunset']
    
    # Adjust to local time if needed (assume UTC here)
    daylight = np.zeros(len(datetime_series))
    for i, dt in enumerate(datetime_series):
        dt_utc = pd.to_datetime(dt, utc=True)
        if sunrise <= dt_utc <= sunset:
            daylight[i] = 1
    
    return daylight

# Calculate solar times
solar_times = calculate_solar_times(LATITUDE, LONGITUDE, SIMULATION_DATE)
print("=" * 70)
print("SOLAR RADIATION AND PHOTOPERIOD")
print("=" * 70)
print(f"Sunrise (UTC):    {solar_times['sunrise']}")
print(f"Solar Noon (UTC): {solar_times['solar_noon']}")
print(f"Sunset (UTC):     {solar_times['sunset']}")

# Generate daylight flag
df['Daylight'] = generate_daylight_flag(df['DateTime'], LATITUDE, LONGITUDE, SIMULATION_DATE)

print(f"\nDaylight hours: {df['Daylight'].sum()} / {len(df)}")

# %% [markdown]
# ## Cell 5: Inline PHREEQC Template and Carbonate Chemistry

# %%
def get_phreeqc_input_template(ph, temp_c, dic_mg_l, d13c_dic_permil):
    """
    Generate PHREEQC input for carbonate system calculation.
    
    This template includes:
    - Definition of 13C as a master species
    - Calculation of CO2, HCO3?, CO3²? distribution
    - pCO2 calculation
    - Isotope ratio calculations
    
    Parameters:
    -----------
    ph : float
        pH (dimensionless)
    temp_c : float
        Temperature (°C)
    dic_mg_l : float
        Dissolved inorganic carbon (mg/L)
    d13c_dic_permil : float
        d13C of DIC (‰ VPDB)
    
    Returns:
    --------
    str : Complete PHREEQC input string
    """
    
    # Convert DIC from mg/L to mol/L
    dic_mol_l = dic_mg_l / 12.01  # Divide by atomic mass of C
    
    # Convert d13C to R (ratio of 13C/12C)
    VPDB_R = 0.0112372  # Standard VPDB ratio
    d13c_decimal = d13c_dic_permil / 1000.0
    r_dic = VPDB_R * (1 + d13c_decimal)
    
    phreeqc_input = f"""TITLE Carbonate system calculation with isotope tracking

SOLUTION 1
    pH {ph}
    Temperature {temp_c}
    C {dic_mol_l} charge
    
EQUILIBRIUM_PHASES 1
    CO2(g)  -3.5  10   # Reasonable pCO2 constraint (~300 ppm)

KNOBS
    -tolerance 1e-9

USER_PRINT
    -selected_output false
    10 PRINT "pH = ", -LA("H+")
    20 PRINT "pCO2 (atm) = ", CALC_VALUE("logSI_CO2(g)") * -1 / 2.303
    30 PRINT "[CO2(aq)] = ", TOT("C") * 10^(PA("H2CO3") + LA("H+") - 0)
    40 PRINT "[HCO3-] = ", TOT("C") * 10^(PA("HCO3-") + LA("H+") - 1)
    50 PRINT "[CO3--] = ", TOT("C") * 10^(PA("CO3-2") + LA("H+") - 2)
    60 PRINT "SI_CO2(g) = ", SI("CO2(g)")

SELECTED_OUTPUT
    -reset false
    -pH
    -saturation_index CO2(g)
    -molalities H+ OH- HCO3- CO3-2 H2CO3 CO2

END"""
    
    return phreeqc_input

# Test PHREEQC input generation
test_input = get_phreeqc_input_template(
    ph=7.5,
    temp_c=25.0,
    dic_mg_l=20.0,
    d13c_dic_permil=-8.0
)

print("=" * 70)
print("PHREEQC INPUT TEMPLATE (SAMPLE)")
print("=" * 70)
print(test_input[:500] + "\n... [truncated for display]")

# %% [markdown]
# ## Cell 6: Analytical Carbonate Chemistry (Backup if PHREEQC unavailable)

# %%
def calculate_carbonate_speciation_analytical(ph, temp_c, dic_total_mol_l):
    """
    Calculate CO2, HCO3?, CO3²? speciation analytically using Ka1 and Ka2.
    
    Temperature-dependent Ka values from Millero (2010).
    This is a backup if PHREEQC is unavailable.
    """
    
    # Temperature-dependent Ka values (freshwater, 25°C reference)
    temp_factor = (temp_c - 25.0) / 25.0
    
    # Ka1 (H2CO3 ? H? + HCO3?) - pKa1 at 25°C ˜ 6.35
    pKa1_25 = 6.35
    pKa1 = pKa1_25 - 0.015 * temp_factor  # Temperature correction
    Ka1 = 10**(-pKa1)
    
    # Ka2 (HCO3? ? H? + CO3²?) - pKa2 at 25°C ˜ 10.33
    pKa2_25 = 10.33
    pKa2 = pKa2_25 - 0.020 * temp_factor
    Ka2 = 10**(-pKa2)
    
    # Calculate H? concentration
    h_conc = 10**(-ph)
    
    # Speciation equations using alpha fractions
    alpha0 = h_con