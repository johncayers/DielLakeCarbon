#!/usr/bin/env python3
"""
Diel Carbon Isotope Model v5 — Full Feature Set
=================================================
• Isotope-enabled PHREEQC (13C as separate master species)
• PAR-driven GPP from solar geometry (Kasten & Young air mass model)
• Calcite dissolution/precipitation with saturation index
• 10-panel publication figure
"""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc
from astral import LocationInfo
from astral.sun import sun, elevation as solar_elevation
from scipy.optimize import minimize
from datetime import datetime, timezone, timedelta
import warnings; warnings.filterwarnings('ignore')

# ========================= CONFIG =========================
LATITUDE, LONGITUDE, TZ_OFF = 36.1627, -86.7816, -5
LAKE_DEPTH    = 1.0       # m
ATM_d13C      = -8.5      # ‰ VPDB
ATM_CO2_PPM   = 425.0
R_VPDB        = 0.011237
CA_MMOL       = 1.0       # Ca concentration (mmol/kgw) — typical for shallow lake
CSV_PATH      = '/mnt/user-data/uploads/SL20241020.csv'
OUTPUT_DIR    = '/home/claude/output_v5'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============= ISOTOPE-ENABLED PHREEQC DATABASE =============
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
# ---- 13C species (T-dependent fractionation) ----
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

# ========================= HELPERS =========================
def Sc_o2(T):  return 1745.1-124.34*T+4.8055*T**2-0.10115*T**3+0.00086842*T**4
def Sc_co2(T): return 1923.6-125.06*T+4.3773*T**2-0.085681*T**3+0.00070284*T**4
def ko2_to_k600(ko2,T): return ko2*(600/Sc_o2(T))**(-0.5)
def k600_to_kco2(k600,T): return k600*(Sc_co2(T)/600)**(-0.5)
def henry_co2(T):
    TK=T+273.15; return np.exp(-58.0931+90.5069*(100/TK)+22.294*np.log(TK/100))
def d13c_to_R(d): return R_VPDB*(1+d/1000)
def R_to_d13c(R): return (R/R_VPDB-1)*1000

def calc_par(observer, dt):
    """Estimate PAR (W/m²) from solar geometry using Kasten & Young (1989) air mass."""
    elev = solar_elevation(observer, dt)
    if elev <= 0: return 0.0
    elev_rad = np.radians(elev)
    am = 1.0/(np.sin(elev_rad)+0.50572*(elev+6.07995)**(-1.6364))
    am = min(am, 38)
    PAR = 1361*0.45*(0.72**am)*np.sin(elev_rad)
    return max(PAR, 0.0)

def create_phreeqc():
    db_path = os.path.join(OUTPUT_DIR, 'phreeqc_iso13c_calcite.dat')
    with open(db_path,'w') as f: f.write(PHREEQC_DB)
    p = IPhreeqc(); p.load_database(db_path)
    if p.phc_database_error_count > 0: raise RuntimeError(p.get_error_string())
    return p

def run_spec_iso(phreeqc, T, pH, DIC_mgC, d13c, Ca_mmol=CA_MMOL):
    """Full isotope+calcite speciation."""
    R = d13c_to_R(d13c); total_mmol = DIC_mgC/12.011
    c12 = total_mmol/(1+R); c13 = total_mmol*R/(1+R)
    phreeqc.run_string(
        f"SOLUTION 1\n    temp {T}\n    pH {pH}\n"
        f"    C(4) {c12}\n    [13C](4) {c13}\n"
        f"    Ca {Ca_mmol}\n    Cl 5.0 charge\n"
        f"SELECTED_OUTPUT\n    -reset false\n    -pH\n"
        f"    -totals C(4) [13C](4)\n"
        f"    -molalities CO2 HCO3- CO3-2 [13C]O2 H[13C]O3- [13C]O3-2\n"
        f"    -si CO2(g) Calcite\nEND\n")
    v = phreeqc.get_selected_output_array()[1]
    tot12=v[1];tot13=v[2]
    CO2_12=v[3];HCO3_12=v[4];CO3_12=v[5]
    CO2_13=v[6];HCO3_13=v[7];CO3_13=v[8]
    return {
        'pH':v[0], 'pCO2':10**v[9]*1e6, 'SI_Calcite':v[10],
        'CO2_12':CO2_12,'HCO3_12':HCO3_12,'CO3_12':CO3_12,
        'CO2_13':CO2_13,'HCO3_13':HCO3_13,'CO3_13':CO3_13,
        'tot12':tot12,'tot13':tot13,
        'd13C_DIC': R_to_d13c(tot13/tot12) if tot12>0 else -999,
        'd13C_CO2': R_to_d13c(CO2_13/CO2_12) if CO2_12>0 else -999,
        'd13C_HCO3':R_to_d13c(HCO3_13/HCO3_12) if HCO3_12>0 else -999,
        'fCO2':CO2_12/tot12 if tot12>0 else 0,
        'fHCO3':HCO3_12/tot12 if tot12>0 else 0,
    }

# ========================= LOAD DATA =========================
print("="*70)
print("DIEL 13C-DIC MODEL v5 — PAR + Calcite + Isotope PHREEQC")
print("="*70)

df_raw = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
df_raw.columns = df_raw.columns.str.strip()
df_raw['DateTime'] = pd.to_datetime(df_raw['DateTime'], format='%m/%d/%y %H:%M')
df_raw = df_raw.rename(columns={
    'DO (mg L-1)':'DO','pH':'pH_meas','Temperature (°C)':'temp',
    'd13C-DIC (‰)':'d13C_DIC','DIC  (mg L-1)':'DIC_mgL',
    'GPP (mg O2 L-1 h-1)':'GPP','R (mg O2 L-1 h-1)':'ER','KO2 (m d-1)':'KO2',
}).rename(columns={'DateTime':'datetime'})
df_obs = df_raw.copy()
obs_times = df_obs['datetime'].values; obs_d13c = df_obs['d13C_DIC'].values
print(f"  Data: {len(df_raw)} obs, δ13C: {obs_d13c.min():.2f} to {obs_d13c.max():.2f}")

df = df_raw.set_index('datetime').resample('1h').interpolate('linear').reset_index()
n = len(df); print(f"  Hourly: {n} points")

# ========================= SOLAR + PAR =========================
tz = timezone(timedelta(hours=TZ_OFF))
loc = LocationInfo("Lake","Region","UTC",LATITUDE,LONGITUDE)
sun_info = {}
for d in df['datetime'].dt.date.unique():
    s = sun(loc.observer, date=d, tzinfo=tz)
    sun_info[d] = (s['sunrise'], s['sunset'])
    print(f"  {d}: Rise {s['sunrise'].strftime('%H:%M')}, Set {s['sunset'].strftime('%H:%M')}")

# Compute PAR for every hourly time step
PAR_arr = np.zeros(n)
for i in range(n):
    dt = df['datetime'].iloc[i]
    PAR_arr[i] = calc_par(loc.observer, dt.replace(tzinfo=tz))
PAR_max = PAR_arr.max()
print(f"  PAR range: 0 – {PAR_max:.1f} W/m²")

# Compute PAR-weighted GPP: scale the observed GPP curve by normalized PAR
# GPP_PAR(t) = GPP_max_obs * PAR(t) / PAR_max  (linear light model)
# But we need GPP_max_obs from the data — use the maximum observed GPP
GPP_obs_max = df_raw['GPP'].max()
GPP_par_raw = GPP_obs_max * PAR_arr / PAR_max if PAR_max > 0 else np.zeros(n)
print(f"  GPP_max (observed): {GPP_obs_max:.2f} mg O2/L/h → PAR-modeled peak: {GPP_par_raw.max():.2f}")

day_flag = PAR_arr > 0  # daylight = any PAR > 0

# ========================= PHREEQC SPECIATION =========================
print("\nPre-computing isotope-enabled PHREEQC speciation (with Calcite SI)...")
phreeqc = create_phreeqc()

spec_data = []
for i in range(n):
    s = run_spec_iso(phreeqc, df['temp'].iloc[i], df['pH_meas'].iloc[i],
                     df['DIC_mgL'].iloc[i], df['d13C_DIC'].iloc[i])
    spec_data.append(s)

pH_mod      = np.array([s['pH'] for s in spec_data])
pCO2_arr    = np.array([s['pCO2'] for s in spec_data])
SI_calc_arr = np.array([s['SI_Calcite'] for s in spec_data])
CO2_mg      = np.array([s['CO2_12']*44010 for s in spec_data])
HCO3_mg     = np.array([s['HCO3_12']*61017 for s in spec_data])
CO3_mg      = np.array([s['CO3_12']*60009 for s in spec_data])
d13C_CO2_eq = np.array([s['d13C_CO2'] for s in spec_data])
d13C_HCO3_eq= np.array([s['d13C_HCO3'] for s in spec_data])
fCO2_12     = np.array([s['fCO2'] for s in spec_data])
fHCO3_12    = np.array([s['fHCO3'] for s in spec_data])

print(f"  Init: pCO2={pCO2_arr[0]:.0f} µatm, SI_Calcite={SI_calc_arr[0]:.3f}")
print(f"  Fractionation CO2-HCO3 = {d13C_CO2_eq[0]-d13C_HCO3_eq[0]:.2f} ‰")

# Derived
temp    = df['temp'].values; TK = temp+273.15
DIC_mgL = df['DIC_mgL'].values; ER_raw = df['ER'].values
KO2_raw = df['KO2'].values; RQ = 12.011/32.0
dDIC_dt = np.diff(DIC_mgL, prepend=DIC_mgL[0])
k600_base = np.array([ko2_to_k600(KO2_raw[i],temp[i]) for i in range(n)])
kco2_base = np.array([k600_to_kco2(k600_base[i],temp[i]) for i in range(n)])
CO2_eq_atm= np.array([henry_co2(temp[i])*ATM_CO2_PPM*1e-6 for i in range(n)])
CO2_mol   = np.array([s['CO2_12'] for s in spec_data])
flux_mmol = kco2_base*(CO2_mol-CO2_eq_atm)*1e6

# ========================= ISOTOPE MODEL =========================
def run_model(pvec):
    """
    pvec: [er_sc, pf, gpp_max_scale, d13c_org, k_iso]
    GPP is now driven by PAR: GPP(t) = gpp_max_scale * GPP_par_raw(t)
    """
    er_sc, pf, gpp_max_sc, d13c_org, k_iso = pvec
    d13C = np.zeros(n); d13C[0] = df['d13C_DIC'].iloc[0]
    for i in range(1,n):
        gpp_eff = GPP_par_raw[i] * gpp_max_sc   # PAR-driven GPP
        er_eff  = ER_raw[i] * er_sc               # ER (negative)
        gpp_C = gpp_eff*RQ; er_C = -er_eff*RQ
        gas_C = er_C - gpp_C - dDIC_dt[i]

        shift = d13C[i-1] - df['d13C_DIC'].iloc[i-1]
        d13C_CO2_s  = d13C_CO2_eq[i-1]  + shift
        d13C_HCO3_s = d13C_HCO3_eq[i-1] + shift

        iso_er  = er_C * d13c_org
        d13_gpp = fCO2_12[i-1]*(d13C_CO2_s+pf) + fHCO3_12[i-1]*(d13C_HCO3_s+pf)
        iso_gpp = gpp_C * d13_gpp

        if gas_C > 0:
            iso_gas = gas_C*(d13C_CO2_s-0.8)*k_iso
        else:
            iso_gas = gas_C*(ATM_d13C-(-0.373*1000/TK[i-1]+0.19))*k_iso

        d13C[i] = (DIC_mgL[i-1]*d13C[i-1]+iso_er-iso_gpp-iso_gas)/DIC_mgL[i] \
                  if DIC_mgL[i]>0.5 else d13C[i-1]
    return d13C

def objective(pvec):
    d = run_model(pvec)
    m = np.interp(obs_times.astype(np.float64), df['datetime'].values.astype(np.float64), d)
    return np.sqrt(np.mean((obs_d13c-m)**2)) if np.all(np.isfinite(m)) else 100.0

# ========================= CALIBRATE =========================
print("\nCalibrating (multi-start)...")
best_rmse, best_x = 999, None
for j,x0 in enumerate([
    [1.0,-20,1.0,-28,1.0],[1.0,-15,1.0,-28,1.0],[1.0,-25,1.0,-28,1.0],
    [0.8,-20,1.0,-26,1.0],[1.2,-20,1.0,-30,1.5],[1.0,-18,0.9,-27,0.7],
    [0.9,-22,1.1,-29,1.2],[1.1,-16,0.95,-25,0.8]]):
    r=minimize(objective,x0,method='Nelder-Mead',options={'maxiter':3000,'xatol':1e-5,'fatol':1e-6})
    if r.fun<best_rmse: best_rmse=r.fun; best_x=r.x.copy()
    print(f"  Start {j+1}/8: RMSE={r.fun:.4f}")
res=minimize(objective,best_x,method='Nelder-Mead',options={'maxiter':5000,'xatol':1e-7,'fatol':1e-8})
best_x=res.x

pn=['ER_scale','PF (‰)','GPP_max_scale','δ13C_org (‰)','k_gas_iso']
print(f"\n  OPTIMIZED (RMSE={res.fun:.4f} ‰):")
for nm,v in zip(pn,best_x): print(f"    {nm:>16s}: {v:.4f}")

# ========================= FINAL RUN =========================
d13C_final = run_model(best_x)
mod_at_obs = np.interp(obs_times.astype(np.float64),df['datetime'].values.astype(np.float64),d13C_final)
rmse = np.sqrt(np.mean((obs_d13c-mod_at_obs)**2))
r2   = 1-np.sum((obs_d13c-mod_at_obs)**2)/np.sum((obs_d13c-np.mean(obs_d13c))**2)
print(f"\n  FIT: RMSE={rmse:.4f} ‰, R²={r2:.4f}")
for i in range(len(obs_d13c)):
    ts=pd.Timestamp(obs_times[i]).strftime('%m/%d %H:%M')
    print(f"    {ts} obs={obs_d13c[i]:7.3f} mod={mod_at_obs[i]:7.3f} Δ={obs_d13c[i]-mod_at_obs[i]:+7.3f}")

# ========================= BUILD RESULTS =========================
gpp_opt = GPP_par_raw * best_x[2]
er_opt  = ER_raw * best_x[0]

results = pd.DataFrame({
    'DateTime':df['datetime'],'Temperature_C':temp,
    'pH_measured':df['pH_meas'].values,'pH_modeled':pH_mod,
    'pCO2_uatm':pCO2_arr,'SI_Calcite':SI_calc_arr,
    'PAR_Wm2':PAR_arr,
    'GPP_input_mgO2Lh':df['GPP'].values if 'GPP' in df.columns else GPP_par_raw,
    'GPP_PAR_modeled_mgO2Lh':gpp_opt,
    'ER_input_mgO2Lh':ER_raw,'ER_optimized_mgO2Lh':er_opt,
    'NEP_mgO2Lh':gpp_opt+er_opt,'KO2_m_d':KO2_raw,'k600_m_d':k600_base,
    'CO2flux_mmol_m2_d':flux_mmol,
    'CO2aq_mgL':CO2_mg,'HCO3_mgL':HCO3_mg,'CO3_mgL':CO3_mg,'DIC_mgL':DIC_mgL,
    'd13C_DIC_measured':df['d13C_DIC'].values,'d13C_DIC_modeled':d13C_final,
    'd13C_residual':df['d13C_DIC'].values-d13C_final,
    'd13C_CO2aq_eq':d13C_CO2_eq,'d13C_HCO3_eq':d13C_HCO3_eq,
    'Daylight_flag':day_flag.astype(int),
})
csv_path=os.path.join(OUTPUT_DIR,'diel_v5_results.csv')
results.to_csv(csv_path,index=False,float_format='%.6f')
print(f"\n  CSV: {csv_path}")

# ========================= 10-PANEL FIGURE =========================
print("\nGenerating 10-panel figure...")
c_obs,c_mod='#1a1a1a','#2166ac'; c_gpp,c_er,c_flux='#2ca02c','#d62728','#ff7f0e'
night_spans=[]
for d in df['datetime'].dt.date.unique():
    sr,ss=sun_info[d]; t0=pd.Timestamp(datetime.combine(d,datetime.min.time()))
    night_spans.append((t0,pd.Timestamp(sr.replace(tzinfo=None))))
    night_spans.append((pd.Timestamp(ss.replace(tzinfo=None)),t0+pd.Timedelta(days=1)))
def shade(ax):
    for s,e in night_spans: ax.axvspan(s,e,alpha=0.08,color='gray',zorder=0)

tt=results['DateTime']; odt=df_obs['datetime']
fig,axes=plt.subplots(10,1,figsize=(14,35),sharex=True)

# 1 — PAR / Sunlight intensity (NEW)
ax=axes[0]
ax.fill_between(tt,0,PAR_arr,color='#f4a460',alpha=0.4)
ax.plot(tt,PAR_arr,'-',color='#d2691e',lw=1.5,label='PAR (clear-sky)')
ax2=ax.twinx()
ax2.plot(tt,PAR_arr*4.57,'--',color='#8b6914',lw=0.8,alpha=0.6)
ax2.set_ylabel('µmol m⁻² s⁻¹',fontsize=8,color='#8b6914')
ax2.tick_params(axis='y',labelcolor='#8b6914',labelsize=8)
shade(ax);ax.set_ylabel('PAR (W m⁻²)');ax.legend(fontsize=9)
ax.set_title('Photosynthetically Active Radiation (solar geometry model)',fontweight='bold');ax.grid(alpha=0.3)

# 2 — pH
ax=axes[1]
ax.plot(odt,df_obs['pH_meas'],'ko',ms=5,label='Measured',zorder=3)
ax.plot(tt,pH_mod,'-',color=c_mod,lw=1.5,label='PHREEQC')
shade(ax);ax.set_ylabel('pH');ax.legend(fontsize=9);ax.set_title('pH',fontweight='bold');ax.grid(alpha=0.3)

# 3 — pCO2
ax=axes[2]
ax.plot(tt,pCO2_arr,'-',color=c_mod,lw=1.5)
ax.axhline(ATM_CO2_PPM,color='gray',ls='--',lw=1,label=f'Atm ({ATM_CO2_PPM})')
shade(ax);ax.set_ylabel('pCO₂ (µatm)');ax.legend(fontsize=9);ax.set_title('pCO₂',fontweight='bold');ax.grid(alpha=0.3)

# 4 — GPP (PAR-driven)
ax=axes[3]
ax.fill_between(tt,0,gpp_opt,color=c_gpp,alpha=0.3)
ax.plot(tt,gpp_opt,'-',color=c_gpp,lw=1.5,label='GPP (PAR-modeled)')
ax.plot(odt,df_obs['GPP'],'o',color='darkgreen',ms=5,label='GPP (observed)')
shade(ax);ax.set_ylabel('GPP (mg O₂ L⁻¹ h⁻¹)');ax.legend(fontsize=9)
ax.set_title('GPP: PAR-Driven vs Observed',fontweight='bold');ax.grid(alpha=0.3)

# 5 — ER
ax=axes[4]
ax.fill_between(tt,0,er_opt,color=c_er,alpha=0.3)
ax.plot(tt,er_opt,'-',color=c_er,lw=1.5,label='ER (opt)')
ax.plot(odt,df_obs['ER'],'o',color='darkred',ms=5,label='ER (input)')
shade(ax);ax.set_ylabel('ER (mg O₂ L⁻¹ h⁻¹)');ax.legend(fontsize=9)
ax.set_title('Ecosystem Respiration',fontweight='bold');ax.grid(alpha=0.3)

# 6 — CO2 Flux
ax=axes[5]
ax.fill_between(tt,0,flux_mmol,where=flux_mmol>=0,color=c_flux,alpha=0.3,label='Outgas')
ax.fill_between(tt,0,flux_mmol,where=flux_mmol<0,color='steelblue',alpha=0.3,label='Ingas')
ax.plot(tt,flux_mmol,'-',color=c_flux,lw=1.5); ax.axhline(0,color='k',lw=0.5)
shade(ax);ax.set_ylabel('CO₂ Flux (mmol m⁻² d⁻¹)');ax.legend(fontsize=9)
ax.set_title('Air-Water CO₂ Exchange',fontweight='bold');ax.grid(alpha=0.3)

# 7 — Calcite Saturation Index (NEW)
ax=axes[6]
ax.plot(tt,SI_calc_arr,'-',color='#8856a7',lw=2)
ax.axhline(0,color='k',lw=1,ls='--',label='Equilibrium (SI=0)')
ax.fill_between(tt,SI_calc_arr,0,where=SI_calc_arr>0,color='#8856a7',alpha=0.15,label='Supersaturated')
ax.fill_between(tt,SI_calc_arr,0,where=SI_calc_arr<0,color='#e0ecf4',alpha=0.5,label='Undersaturated')
shade(ax);ax.set_ylabel('SI Calcite');ax.legend(fontsize=9)
ax.set_title('Calcite Saturation Index (PHREEQC)',fontweight='bold');ax.grid(alpha=0.3)

# 8 — DIC species
ax=axes[7]
ax.plot(tt,HCO3_mg,'-',color='#1b9e77',lw=1.5,label='HCO₃⁻')
ax.plot(tt,CO2_mg,'-',color='#d95f02',lw=1.5,label='CO₂(aq)')
ax.plot(tt,CO3_mg,'-',color='#7570b3',lw=1.5,label='CO₃²⁻')
shade(ax);ax.set_ylabel('mg L⁻¹');ax.legend(fontsize=9);ax.set_title('DIC Speciation',fontweight='bold');ax.grid(alpha=0.3)

# 9 — Species d13C from PHREEQC
ax=axes[8]
ax.plot(tt,d13C_CO2_eq,'-',color='#d95f02',lw=1.5,label='δ¹³C-CO₂(aq)')
ax.plot(tt,d13C_HCO3_eq,'-',color='#1b9e77',lw=1.5,label='δ¹³C-HCO₃⁻')
shade(ax);ax.set_ylabel('δ¹³C (‰)');ax.legend(fontsize=9)
ax.set_title('Equilibrium ¹³C Fractionation (PHREEQC)',fontweight='bold');ax.grid(alpha=0.3)

# 10 — d13C-DIC (primary target)
ax=axes[9]
ax.plot(odt,obs_d13c,'ko',ms=7,zorder=3,label='Measured')
ax.plot(tt,d13C_final,'-',color=c_mod,lw=2,zorder=2,label='Modeled')
ax.fill_between(tt,d13C_final-rmse,d13C_final+rmse,color=c_mod,alpha=0.15,label=f'±1σ ({rmse:.2f}‰)')
shade(ax);ax.set_ylabel('δ¹³C-DIC (‰ VPDB)');ax.set_xlabel('Date/Time')
ax.legend(fontsize=9);ax.set_title(f'δ¹³C-DIC | R²={r2:.3f}, RMSE={rmse:.3f}‰',fontweight='bold');ax.grid(alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
plt.tight_layout()
for fmt in ['svg','png']:
    fig.savefig(os.path.join(OUTPUT_DIR,f'diel_v5_results.{fmt}'),format=fmt,dpi=150,bbox_inches='tight')
plt.close()
print(f"  Figures saved to {OUTPUT_DIR}/")
print("="*70);print("COMPLETE");print("="*70)
