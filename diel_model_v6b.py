#!/usr/bin/env python3
"""
Diel Carbon Isotope Model v6b — T-dependent GPP, multi-objective, bounded
"""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc
from astral import LocationInfo
from astral.sun import sun, elevation as solar_elevation
from scipy.optimize import differential_evolution
from datetime import datetime, timezone, timedelta
import warnings; warnings.filterwarnings('ignore')

LATITUDE, LONGITUDE, TZ_OFF = 35.939311, -87.015833, -5
LAKE_DEPTH=1.0; ATM_d13C=-8.5; ATM_CO2_PPM=425.0; R_VPDB=0.011237; CA_MMOL=1.0
CSV_PATH='input\\SL20250718.csv'
OUTPUT_DIR='output\\summer\\v6b'; os.makedirs(OUTPUT_DIR,exist_ok=True)

PHREEQC_DB = open('input\\phreeqc_v6.dat').read()

def Sc_o2(T):  return 1745.1-124.34*T+4.8055*T**2-0.10115*T**3+0.00086842*T**4
def Sc_co2(T): return 1923.6-125.06*T+4.3773*T**2-0.085681*T**3+0.00070284*T**4
def ko2_to_k600(ko2,T): return ko2*(600/Sc_o2(T))**(-0.5)
def k600_to_kco2(k600,T): return k600*(Sc_co2(T)/600)**(-0.5)
def henry_co2(T):
    TK=T+273.15; return np.exp(-58.0931+90.5069*(100/TK)+22.294*np.log(TK/100))
def d13c_to_R(d): return R_VPDB*(1+d/1000)
def R_to_d13c(R): return (R/R_VPDB-1)*1000
def calc_par(observer,dt):
    elev=solar_elevation(observer,dt)
    if elev<=0: return 0.0
    er=np.radians(elev)
    am=1.0/(np.sin(er)+0.50572*(elev+6.07995)**(-1.6364)); am=min(am,38)
    return max(1361*0.45*(0.72**am)*np.sin(er),0.0)

def create_phreeqc():
    db=os.path.join(OUTPUT_DIR,'phreeqc.dat')
    with open(db,'w') as f: f.write(PHREEQC_DB)
    p=IPhreeqc(); p.load_database(db)
    if p.phc_database_error_count>0: raise RuntimeError(p.get_error_string())
    return p

def run_spec(phreeqc,T,pH,DIC_mgC,d13c,Ca=CA_MMOL):
    R=d13c_to_R(d13c);tot=DIC_mgC/12.011;c12=tot/(1+R);c13=tot*R/(1+R)
    phreeqc.run_string(
        f"SOLUTION 1\n temp {T}\n pH {pH}\n C(4) {c12}\n [13C](4) {c13}\n"
        f" Ca {Ca}\n Cl 5.0 charge\n"
        f"SELECTED_OUTPUT\n -reset false\n -pH\n -totals C(4) [13C](4)\n"
        f" -molalities CO2 HCO3- CO3-2 [13C]O2 H[13C]O3- [13C]O3-2\n"
        f" -si CO2(g) Calcite\nEND\n")
    v=phreeqc.get_selected_output_array()[1]
    t12=v[1];t13=v[2]
    return {
        'pH':v[0],'pCO2':10**v[9]*1e6,'SI_Calcite':v[10],
        'CO2_12':v[3],'HCO3_12':v[4],'CO3_12':v[5],
        'CO2_13':v[6],'HCO3_13':v[7],'CO3_13':v[8],
        'tot12':t12,'tot13':t13,
        'd13C_DIC':R_to_d13c(t13/t12) if t12>0 else -999,
        'd13C_CO2':R_to_d13c(v[6]/v[3]) if v[3]>0 else -999,
        'd13C_HCO3':R_to_d13c(v[7]/v[4]) if v[4]>0 else -999,
        'fCO2':v[3]/t12 if t12>0 else 0,
        'fHCO3':v[4]/t12 if t12>0 else 0,
    }

# ========================= LOAD =========================
print("="*70); print("DIEL MODEL v6b — T-dependent GPP, Bayesian priors"); print("="*70)

df_raw=pd.read_csv(CSV_PATH,encoding='utf-8-sig'); df_raw.columns=df_raw.columns.str.strip()
df_raw['DateTime']=pd.to_datetime(df_raw['DateTime'],format='%m/%d/%y %H:%M')
df_raw=df_raw.rename(columns={
    'DO (mg L-1)':'DO','pH':'pH_meas','Temperature (°C)':'temp',
    'd13C-DIC (‰)':'d13C_DIC','DIC  (mg L-1)':'DIC_mgL',
    'GPP (mg O2 L-1 h-1)':'GPP','R (mg O2 L-1 h-1)':'ER','KO2 (m d-1)':'KO2',
}).rename(columns={'DateTime':'datetime'})
df_obs=df_raw.copy(); obs_times=df_obs['datetime'].values
obs_d13c=df_obs['d13C_DIC'].values; obs_gpp=df_obs['GPP'].values; obs_er=df_obs['ER'].values
print(f"  Obs={len(df_raw)}, δ13C=[{obs_d13c.min():.2f},{obs_d13c.max():.2f}]")

df=df_raw.set_index('datetime').resample('1h').interpolate('linear').reset_index()
n=len(df); print(f"  Hourly={n}")

# Solar + PAR
tz=timezone(timedelta(hours=TZ_OFF))
loc=LocationInfo("Lake","Region","UTC",LATITUDE,LONGITUDE)
sun_info={}
for d in df['datetime'].dt.date.unique():
    s=sun(loc.observer,date=d,tzinfo=tz); sun_info[d]=(s['sunrise'],s['sunset'])
    print(f"  {d}: Rise {s['sunrise'].strftime('%H:%M')}, Set {s['sunset'].strftime('%H:%M')}")

PAR_arr=np.zeros(n)
for i in range(n):
    PAR_arr[i]=calc_par(loc.observer,df['datetime'].iloc[i].replace(tzinfo=tz))
print(f"  PAR peak: {PAR_arr.max():.1f} W/m²")

# PHREEQC speciation
print("\nPHREEQC speciation...")
phreeqc=create_phreeqc()
spec=[run_spec(phreeqc,df['temp'].iloc[i],df['pH_meas'].iloc[i],
               df['DIC_mgL'].iloc[i],df['d13C_DIC'].iloc[i]) for i in range(n)]

pH_mod=np.array([s['pH'] for s in spec])
pCO2_arr=np.array([s['pCO2'] for s in spec])
SI_calc=np.array([s['SI_Calcite'] for s in spec])
CO2_mg=np.array([s['CO2_12']*44010 for s in spec])
HCO3_mg=np.array([s['HCO3_12']*61017 for s in spec])
CO3_mg=np.array([s['CO3_12']*60009 for s in spec])
d13C_CO2_eq=np.array([s['d13C_CO2'] for s in spec])
d13C_HCO3_eq=np.array([s['d13C_HCO3'] for s in spec])
fCO2_12=np.array([s['fCO2'] for s in spec])
fHCO3_12=np.array([s['fHCO3'] for s in spec])
print(f"  pCO2={pCO2_arr[0]:.0f}, SI_Cc={SI_calc[0]:.3f}, CO2-HCO3 frac={d13C_CO2_eq[0]-d13C_HCO3_eq[0]:.2f}‰")

temp=df['temp'].values; TK=temp+273.15; DIC_mgL=df['DIC_mgL'].values
ER_raw=df['ER'].values; KO2_raw=df['KO2'].values; RQ=12.011/32.0
dDIC_dt=np.diff(DIC_mgL,prepend=DIC_mgL[0])
k600_b=np.array([ko2_to_k600(KO2_raw[i],temp[i]) for i in range(n)])
kco2_b=np.array([k600_to_kco2(k600_b[i],temp[i]) for i in range(n)])
CO2_eq_atm=np.array([henry_co2(temp[i])*ATM_CO2_PPM*1e-6 for i in range(n)])
CO2_mol=np.array([s['CO2_12'] for s in spec])
flux_mmol=kco2_b*(CO2_mol-CO2_eq_atm)*1e6

# T-ref for Q10
T_ref = np.mean(temp)  # ~15.8 C

# ========================= MODEL =========================
# Parameters:
#   GPP_max  : mg O2/L/h at T_ref (peak, if PAR→∞)
#   Ik       : W/m2 (half-saturation PAR)
#   Q10_gpp  : temperature sensitivity of GPP
#   ER_base  : mg O2/L/h (constant, negative)
#   pf       : ‰ (photosynthetic fractionation)
#   d13c_org : ‰ (d13C of respired OM)
#   k_gas_iso: gas exchange isotope scaling

def run_model(pvec):
    GPP_max, Ik, Q10_gpp, ER_base, pf, d13c_org, k_gas_iso = pvec
    
    # GPP: Michaelis-Menten light response × Q10 temperature dependence
    GPP_t = np.zeros(n)
    for i in range(n):
        if PAR_arr[i] > 0:
            f_light = PAR_arr[i] / (Ik + PAR_arr[i])
            f_temp  = Q10_gpp ** ((temp[i] - T_ref) / 10.0)
            GPP_t[i] = GPP_max * f_light * f_temp
    
    # ER: constant rate
    ER_t = np.full(n, ER_base)
    
    # d13C evolution
    d13C = np.zeros(n); d13C[0] = df['d13C_DIC'].iloc[0]
    for i in range(1, n):
        gpp_C = GPP_t[i]*RQ; er_C = -ER_t[i]*RQ
        gas_C = er_C - gpp_C - dDIC_dt[i]
        shift = d13C[i-1] - df['d13C_DIC'].iloc[i-1]
        d13C_CO2_s = d13C_CO2_eq[i-1] + shift
        d13C_HCO3_s = d13C_HCO3_eq[i-1] + shift
        iso_er = er_C * d13c_org
        d13_gpp = fCO2_12[i-1]*(d13C_CO2_s+pf) + fHCO3_12[i-1]*(d13C_HCO3_s+pf)
        iso_gpp = gpp_C * d13_gpp
        if gas_C > 0:
            iso_gas = gas_C*(d13C_CO2_s-0.8)*k_gas_iso
        else:
            iso_gas = gas_C*(ATM_d13C-(-0.373*1000/TK[i-1]+0.19))*k_gas_iso
        d13C[i] = (DIC_mgL[i-1]*d13C[i-1]+iso_er-iso_gpp-iso_gas)/DIC_mgL[i] \
                  if DIC_mgL[i]>0.5 else d13C[i-1]
    return d13C, GPP_t, ER_t

def objective(pvec):
    try: d13C, GPP_t, ER_t = run_model(pvec)
    except: return 1e6
    
    mod_d13c = np.interp(obs_times.astype(np.float64),
                         df['datetime'].values.astype(np.float64), d13C)
    mod_gpp = np.interp(obs_times.astype(np.float64),
                        df['datetime'].values.astype(np.float64), GPP_t)
    if not np.all(np.isfinite(mod_d13c)): return 1e6
    
    # d13C RMSE
    rmse_d13c = np.sqrt(np.mean((obs_d13c - mod_d13c)**2))
    
    # GPP RMSE — exclude 18:00 obs (edge of daylight, 4h-average artifact)
    # Use only observations where time is NOT 18:00
    hours = np.array([pd.Timestamp(t).hour for t in obs_times])
    gpp_mask = hours != 18  # exclude twilight-averaged GPP
    if gpp_mask.sum() > 0:
        rmse_gpp = np.sqrt(np.mean((obs_gpp[gpp_mask] - mod_gpp[gpp_mask])**2))
    else:
        rmse_gpp = 0
    
    # Weight: normalise so d13C range (~0.66) ≈ GPP range (~10)
    w_gpp = 0.066
    cost = rmse_d13c + w_gpp * rmse_gpp
    
    # Bayesian Gaussian penalty priors
    GPP_max, Ik, Q10_gpp, ER_base, pf, d13c_org, k_gas_iso = pvec
    cost += 0.02 * ((pf - (-20)) / 5)**2           # PF ~ -20 ± 5 ‰
    cost += 0.02 * ((d13c_org - (-28)) / 3)**2      # d13C_org ~ -28 ± 3 ‰
    cost += 0.02 * ((k_gas_iso - 1.0) / 0.5)**2     # k_gas ~ 1.0 ± 0.5
    cost += 0.01 * ((Q10_gpp - 2.0) / 0.5)**2       # Q10 ~ 2.0 ± 0.5
    cost += 0.01 * ((ER_base - (-5.5)) / 1.0)**2    # ER ~ -5.5 ± 1
    
    return cost

# ========================= OPTIMIZE =========================
print("\nOptimising (bounded DE + Bayesian priors)...")
bounds = [
    (5.0, 30.0),    # GPP_max
    (20.0, 500.0),  # Ik
    (1.2, 3.5),     # Q10_gpp
    (-10.0, -2.0),  # ER_base
    (-30.0, -10.0), # pf
    (-34.0, -22.0), # d13c_org
    (0.3, 2.5),     # k_gas_iso
]

result = differential_evolution(
    objective, bounds, seed=42,
    maxiter=500, tol=1e-7,
    popsize=30, mutation=(0.5, 1.5), recombination=0.9,
    polish=True
)
bx = result.x
GPP_max_opt, Ik_opt, Q10_opt, ER_opt, pf_opt, d13c_org_opt, kgas_opt = bx
print(f"\n  OPTIMISED (cost={result.fun:.5f}):")
for l,v in zip(['GPP_max (mgO2/L/h)','Ik (W/m²)','Q10_GPP','ER_base (mgO2/L/h)',
                'PF (‰)','δ13C_org (‰)','k_gas_iso'], bx):
    print(f"    {l:>22s}: {v:8.3f}")

# ========================= FINAL RUN =========================
d13C_final, GPP_final, ER_final = run_model(bx)
mod_d13c = np.interp(obs_times.astype(np.float64),
                     df['datetime'].values.astype(np.float64), d13C_final)
mod_gpp = np.interp(obs_times.astype(np.float64),
                    df['datetime'].values.astype(np.float64), GPP_final)

rmse_d13c = np.sqrt(np.mean((obs_d13c-mod_d13c)**2))
r2_d13c = 1-np.sum((obs_d13c-mod_d13c)**2)/np.sum((obs_d13c-np.mean(obs_d13c))**2)
hours = np.array([pd.Timestamp(t).hour for t in obs_times])
gpp_mask = hours != 18
rmse_gpp = np.sqrt(np.mean((obs_gpp[gpp_mask]-mod_gpp[gpp_mask])**2))
r2_gpp = 1-np.sum((obs_gpp[gpp_mask]-mod_gpp[gpp_mask])**2)/np.sum((obs_gpp[gpp_mask]-np.mean(obs_gpp[gpp_mask]))**2)

print(f"\n  δ13C-DIC: RMSE={rmse_d13c:.4f} ‰, R²={r2_d13c:.4f}")
print(f"  GPP (excl 18h): RMSE={rmse_gpp:.4f} mg O₂/L/h, R²={r2_gpp:.4f}")
print(f"\n  {'Time':>16s} {'d13C_o':>7s} {'d13C_m':>7s} {'Δ':>6s}  {'GPP_o':>6s} {'GPP_m':>6s} {'Δ':>6s}")
for i in range(len(obs_d13c)):
    ts=pd.Timestamp(obs_times[i]).strftime('%m/%d %H:%M')
    print(f"  {ts:>16s} {obs_d13c[i]:7.3f} {mod_d13c[i]:7.3f} {obs_d13c[i]-mod_d13c[i]:+6.3f}"
          f"  {obs_gpp[i]:6.2f} {mod_gpp[i]:6.2f} {obs_gpp[i]-mod_gpp[i]:+6.2f}")

# ========================= CSV =========================
results = pd.DataFrame({
    'DateTime':df['datetime'],'Temperature_C':temp,
    'pH_measured':df['pH_meas'].values,'pH_modeled':pH_mod,
    'pCO2_uatm':pCO2_arr,'SI_Calcite':SI_calc,
    'PAR_Wm2':PAR_arr,
    'GPP_observed_mgO2Lh':df['GPP'].values if 'GPP' in df.columns else np.nan,
    'GPP_modeled_mgO2Lh':GPP_final,
    'ER_observed_mgO2Lh':ER_raw,'ER_modeled_mgO2Lh':ER_final,
    'NEP_mgO2Lh':GPP_final+ER_final,
    'KO2_m_d':KO2_raw,'k600_m_d':k600_b,
    'CO2flux_mmol_m2_d':flux_mmol,
    'CO2aq_mgL':CO2_mg,'HCO3_mgL':HCO3_mg,'CO3_mgL':CO3_mg,'DIC_mgL':DIC_mgL,
    'd13C_DIC_measured':df['d13C_DIC'].values,'d13C_DIC_modeled':d13C_final,
    'd13C_residual':df['d13C_DIC'].values-d13C_final,
    'd13C_CO2aq_eq':d13C_CO2_eq,'d13C_HCO3_eq':d13C_HCO3_eq,
    'Daylight_flag':(PAR_arr>0).astype(int),
})
results.to_csv(os.path.join(OUTPUT_DIR,'diel_v6b_results.csv'),index=False,float_format='%.6f')

# ========================= 10-PANEL FIGURE =========================
print("\nPlotting...")
c_obs,c_mod='#1a1a1a','#2166ac'; c_gpp,c_er,c_flux='#2ca02c','#d62728','#ff7f0e'
night_spans=[]
for d in df['datetime'].dt.date.unique():
    sr,ss=sun_info[d]; t0=pd.Timestamp(datetime.combine(d,datetime.min.time()))
    night_spans.append((t0,pd.Timestamp(sr.replace(tzinfo=None))))
    night_spans.append((pd.Timestamp(ss.replace(tzinfo=None)),t0+pd.Timedelta(days=1)))
def shade(ax):
    for s,e in night_spans: ax.axvspan(s,e,alpha=0.08,color='gray',zorder=0)
tt=results['DateTime']; odt=df_obs['datetime']

fig,axes=plt.subplots(10,1,figsize=(14,36),sharex=True)

ax=axes[0]
ax.fill_between(tt,0,PAR_arr,color='#f4a460',alpha=0.4)
ax.plot(tt,PAR_arr,'-',color='#d2691e',lw=1.5,label='PAR (clear-sky)')
ax2=ax.twinx(); ax2.plot(tt,PAR_arr*4.57,'--',color='#8b6914',lw=0.8,alpha=0.5)
ax2.set_ylabel('µmol m⁻² s⁻¹',fontsize=8,color='#8b6914')
ax2.tick_params(axis='y',labelcolor='#8b6914',labelsize=8)
shade(ax);ax.set_ylabel('PAR (W m⁻²)');ax.legend(fontsize=9)
ax.set_title(f'PAR at {LATITUDE:.4f}°N, {LONGITUDE:.4f}°W',fontweight='bold');ax.grid(alpha=0.3)

ax=axes[1]
ax.plot(odt,df_obs['pH_meas'],'ko',ms=5,label='Measured',zorder=3)
ax.plot(tt,pH_mod,'-',color=c_mod,lw=1.5,label='PHREEQC')
shade(ax);ax.set_ylabel('pH');ax.legend(fontsize=9);ax.set_title('pH',fontweight='bold');ax.grid(alpha=0.3)

ax=axes[2]
ax.plot(tt,pCO2_arr,'-',color=c_mod,lw=1.5)
ax.axhline(ATM_CO2_PPM,color='gray',ls='--',lw=1,label=f'Atm ({ATM_CO2_PPM})')
shade(ax);ax.set_ylabel('pCO₂ (µatm)');ax.legend(fontsize=9);ax.set_title('pCO₂',fontweight='bold');ax.grid(alpha=0.3)

ax=axes[3]
ax.fill_between(tt,0,GPP_final,color=c_gpp,alpha=0.2)
ax.plot(tt,GPP_final,'-',color=c_gpp,lw=2,
        label=f'Model (max={GPP_max_opt:.1f}, Ik={Ik_opt:.0f}, Q10={Q10_opt:.2f})')
ax.plot(odt,obs_gpp,'o',color='darkgreen',ms=6,zorder=3,label='Observed')
shade(ax);ax.set_ylabel('GPP (mg O₂ L⁻¹ h⁻¹)');ax.legend(fontsize=8)
ax.set_title(f'GPP: PAR×Michaelis-Menten×Q10 | RMSE={rmse_gpp:.2f} (excl 18h), R²={r2_gpp:.3f}',
             fontweight='bold');ax.grid(alpha=0.3)

ax=axes[4]
ax.axhline(ER_opt,color=c_er,lw=2,label=f'Model ({ER_opt:.2f} mg O₂/L/h)')
ax.plot(odt,obs_er,'o',color='darkred',ms=5,label='Observed',zorder=3)
shade(ax);ax.set_ylabel('ER (mg O₂ L⁻¹ h⁻¹)');ax.legend(fontsize=9)
ax.set_title('Ecosystem Respiration',fontweight='bold');ax.grid(alpha=0.3)

ax=axes[5]
ax.fill_between(tt,0,flux_mmol,where=flux_mmol>=0,color=c_flux,alpha=0.3,label='Outgas')
ax.fill_between(tt,0,flux_mmol,where=flux_mmol<0,color='steelblue',alpha=0.3,label='Ingas')
ax.plot(tt,flux_mmol,'-',color=c_flux,lw=1.5); ax.axhline(0,color='k',lw=0.5)
shade(ax);ax.set_ylabel('CO₂ Flux (mmol m⁻² d⁻¹)');ax.legend(fontsize=9)
ax.set_title('Air-Water CO₂ Exchange',fontweight='bold');ax.grid(alpha=0.3)

ax=axes[6]
ax.plot(tt,SI_calc,'-',color='#8856a7',lw=2)
ax.axhline(0,color='k',lw=1,ls='--',label='Equilibrium')
ax.fill_between(tt,SI_calc,0,where=SI_calc>0,color='#8856a7',alpha=0.15,label='Supersaturated')
ax.fill_between(tt,SI_calc,0,where=SI_calc<0,color='#e0ecf4',alpha=0.5,label='Undersaturated')
shade(ax);ax.set_ylabel('SI Calcite');ax.legend(fontsize=9)
ax.set_title('Calcite Saturation Index',fontweight='bold');ax.grid(alpha=0.3)

ax=axes[7]
ax.plot(tt,HCO3_mg,'-',color='#1b9e77',lw=1.5,label='HCO₃⁻')
ax.plot(tt,CO2_mg,'-',color='#d95f02',lw=1.5,label='CO₂(aq)')
ax.plot(tt,CO3_mg,'-',color='#7570b3',lw=1.5,label='CO₃²⁻')
shade(ax);ax.set_ylabel('mg L⁻¹');ax.legend(fontsize=9);ax.set_title('DIC Speciation',fontweight='bold');ax.grid(alpha=0.3)

ax=axes[8]
ax.plot(tt,d13C_CO2_eq,'-',color='#d95f02',lw=1.5,label='δ¹³C-CO₂(aq)')
ax.plot(tt,d13C_HCO3_eq,'-',color='#1b9e77',lw=1.5,label='δ¹³C-HCO₃⁻')
shade(ax);ax.set_ylabel('δ¹³C (‰)');ax.legend(fontsize=9)
ax.set_title('Equilibrium ¹³C Fractionation (PHREEQC)',fontweight='bold');ax.grid(alpha=0.3)

ax=axes[9]
ax.plot(odt,obs_d13c,'ko',ms=7,zorder=3,label='Measured')
ax.plot(tt,d13C_final,'-',color=c_mod,lw=2,zorder=2,label='Modeled')
ax.fill_between(tt,d13C_final-rmse_d13c,d13C_final+rmse_d13c,
                color=c_mod,alpha=0.15,label=f'±1σ ({rmse_d13c:.2f}‰)')
shade(ax);ax.set_ylabel('δ¹³C-DIC (‰ VPDB)');ax.set_xlabel('Date / Time')
ax.legend(fontsize=9)
ax.set_title(f'δ¹³C-DIC | R²={r2_d13c:.3f}, RMSE={rmse_d13c:.3f}‰',fontweight='bold')
ax.grid(alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
plt.tight_layout()
for fmt in ['svg','png']:
    fig.savefig(os.path.join(OUTPUT_DIR,f'diel_v6b_results.{fmt}'),format=fmt,dpi=150,bbox_inches='tight')
plt.close()
print(f"  Done: {OUTPUT_DIR}/")
print("="*70);print("COMPLETE");print("="*70)
