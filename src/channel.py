"""channel.py : bilan de liaison descendante et SINR (modèle inspiré du HHS)."""
import warnings
import numpy as np

C = 299792458.0

# Laws et Parsons (tableau du cours) : f (GHz), kh, kv, ah, av
LP = np.array([
    [1,  0.0000387, 0.0000352, 0.912, 0.880],
    [2,  0.000154,  0.000138,  0.963, 0.923],
    [4,  0.000650,  0.000591,  1.121, 1.075],
    [6,  0.00175,   0.00155,   1.308, 1.265],
    [8,  0.00454,   0.00395,   1.327, 1.310],
    [10, 0.0101,    0.00887,   1.276, 1.264],
    [12, 0.0188,    0.0168,    1.217, 1.200],
    [15, 0.0367,    0.0335,    1.154, 1.128],
    [20, 0.0751,    0.0691,    1.099, 1.065],
    [25, 0.124,     0.113,     1.061, 1.030],
    [30, 0.187,     0.167,     1.021, 1.000],
    [35, 0.263,     0.233,     0.979, 0.963],
    [40, 0.350,     0.310,     0.939, 0.929],
    [50, 0.536,     0.479,     0.873, 0.868],
    [60, 0.707,     0.642,     0.826, 0.824],
])


def rain_coeffs(f_ghz, pol="c"):
    """k et a interpolés (k en log, a linéaire, en fonction de log f). pol : h, v ou c (circulaire)."""
    lf, x = np.log10(LP[:, 0]), np.log10(f_ghz)
    kh = 10 ** np.interp(x, lf, np.log10(LP[:, 1]))
    kv = 10 ** np.interp(x, lf, np.log10(LP[:, 2]))
    ah, av = np.interp(x, lf, LP[:, 3]), np.interp(x, lf, LP[:, 4])
    if pol == "h":
        return kh, ah
    if pol == "v":
        return kv, av
    k = (kh + kv) / 2
    return k, (kh * ah + kv * av) / (2 * k)


def rain_specific_atten(f_ghz, rain_mm_h, pol="c"):
    """Affaiblissement linéique de la pluie (dB/km) : gamma = k * R^a."""
    if rain_mm_h <= 0:
        return 0.0
    k, a = rain_coeffs(f_ghz, pol)
    return k * rain_mm_h ** a


def fspl_db(range_km, f_ghz):
    return 20 * np.log10(4 * np.pi * np.asarray(range_km) * 1e3 * f_ghz * 1e9 / C)


def noise_dbm(bw_hz, nf_db):
    return -174.0 + 10 * np.log10(bw_hz) + nf_db


def link_budget(range_km, el_deg, eirp_g_dbm, f_ghz=10.75, bw_hz=250e6, nf_db=0.8,
                rain_mm_h=0.0, pol="c", gas_zenith_db=0.1, rain_height_km=3.0,
                station_alt_km=0.0, interference_dbm=None):
    """SINR (dB) pour des tableaux de distance et d'élévation (mêmes dimensions).
    eirp_g_dbm = EIRP du satellite + gain du terminal (dBm), à caler."""
    with np.errstate(invalid="ignore", divide="ignore"):
        sin_el = np.sin(np.radians(el_deg))
        sin_el = np.where(sin_el > 0.02, sin_el, np.nan)          # élévation > ~1,1°
        l_fspl = fspl_db(range_km, f_ghz)
        a_gas = gas_zenith_db / sin_el
        l_rain_km = max(rain_height_km - station_alt_km, 0.0) / sin_el
        a_rain = rain_specific_atten(f_ghz, rain_mm_h, pol) * l_rain_km
        prx = eirp_g_dbm - l_fspl - a_gas - a_rain                # dBm
        n_mw = 10 ** (noise_dbm(bw_hz, nf_db) / 10)
        i_mw = 0.0 if interference_dbm is None else 10 ** (interference_dbm / 10)
        sinr = prx - 10 * np.log10(n_mw + i_mw)
    return {"sinr_db": sinr, "prx_dbm": prx, "fspl_db": l_fspl, "a_gas_db": a_gas, "a_rain_db": a_rain}


def calibrate_eirp_g(range_km, el_deg, visible, target_best_sinr_db=14.3, **kw):
    """EIRP + Gt (dBm) tel que le SINR du meilleur satellite visible vaille en moyenne la cible."""
    s = link_budget(range_km, el_deg, 0.0, **kw)["sinr_db"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best = np.nanmax(np.where(visible, s, np.nan), axis=0)
    return float(target_best_sinr_db - np.nanmean(best))
