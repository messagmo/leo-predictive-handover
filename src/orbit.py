"""orbit.py : TLE, propagation SGP4 et visibilité depuis un utilisateur (fixe ou mobile)."""
import math, os, urllib.request
import numpy as np
from sgp4.api import Satrec, SatrecArray, jday

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
A_WGS84_KM = 6378.137
E2 = (1 / 298.257223563) * (2 - 1 / 298.257223563)   # excentricité² WGS84


def download_tle(path="data/starlink.tle", group="starlink"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    urllib.request.urlretrieve(CELESTRAK_URL.format(group=group), path)
    return path


def load_tle(path):
    """Lit un fichier TLE (3 lignes par satellite) -> liste de (nom, Satrec)."""
    lines = [l.rstrip() for l in open(path) if l.strip()]
    return [(lines[i].strip(), Satrec.twoline2rv(lines[i + 1], lines[i + 2]))
            for i in range(0, len(lines) - 2, 3)]


def filter_tle(sats, ref_jd, mm_min=14.9, mm_max=15.2, bstar_max=1e-4, max_age_days=30):
    """Filtre du HHS : mouvement moyen (tours/jour), |B*| et âge du TLE."""
    kept = []
    for name, s in sats:
        mm = s.no_kozai * 1440.0 / (2 * math.pi)             # rad/min -> tours/jour
        age = ref_jd - (s.jdsatepoch + s.jdsatepochF)
        if mm_min <= mm <= mm_max and abs(s.bstar) <= bstar_max and abs(age) <= max_age_days:
            kept.append((name, s))
    return kept


def time_grid(start, duration_s, step_s):
    """Grille de temps (jd, fr) à partir d'un datetime UTC."""
    jd0, fr0 = jday(start.year, start.month, start.day, start.hour, start.minute,
                    start.second + start.microsecond / 1e6)
    t = fr0 + np.arange(0, duration_s + 1e-9, step_s) / 86400.0
    return jd0 + np.floor(t), t - np.floor(t)


def gmst_rad(jd, fr):
    """Angle sidéral de Greenwich (UT1 ≈ UTC)."""
    T = ((jd - 2451545.0) + fr) / 36525.0
    sec = 67310.54841 + (876600 * 3600 + 8640184.812866) * T + 0.093104 * T**2 - 6.2e-6 * T**3
    return np.deg2rad((sec % 86400.0) / 240.0)


def propagate_ecef(sats, jd, fr):
    """Positions ECEF (km) de tous les satellites : tableau (n_sat, n_temps, 3)."""
    err, r, _ = SatrecArray([s for _, s in sats]).sgp4(jd, fr)   # r en TEME
    r = np.array(r)
    r[err != 0] = np.nan                                          # satellites en erreur
    th = gmst_rad(jd, fr)
    c, s = np.cos(th), np.sin(th)
    x = r[..., 0] * c + r[..., 1] * s
    y = -r[..., 0] * s + r[..., 1] * c
    return np.stack([x, y, r[..., 2]], axis=-1)


def geodetic_to_ecef(lat_deg, lon_deg, alt_m=0.0):
    """Position ECEF (km) ; accepte des scalaires ou des tableaux (trajectoire)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    h = np.asarray(alt_m, dtype=float) / 1000.0
    N = A_WGS84_KM / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.stack([(N + h) * np.cos(lat) * np.cos(lon),
                     (N + h) * np.cos(lat) * np.sin(lon),
                     (N * (1 - E2) + h) * np.sin(lat)], axis=-1)


def up_vector(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def look_angles(sat_ecef, obs_ecef, up):
    """Élévation (degrés) et distance oblique (km), tableaux (n_sat, n_temps)."""
    n_t = sat_ecef.shape[1]
    obs = np.broadcast_to(obs_ecef, (n_t, 3))
    up = np.broadcast_to(up, (n_t, 3))
    with np.errstate(invalid="ignore"):
        rho = sat_ecef - obs[None, :, :]
        rng = np.linalg.norm(rho, axis=-1)
        sin_el = np.einsum("stk,tk->st", rho, up) / rng
        el = np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))
    return el, rng


def remaining_visibility(visible, step_s):
    """Temps de visibilité restant (s) à chaque instant (limité à la fin de la fenêtre)."""
    rem = np.zeros(visible.shape)
    for t in range(visible.shape[1] - 1, -1, -1):
        nxt = rem[:, t + 1] if t + 1 < visible.shape[1] else 0.0
        rem[:, t] = np.where(visible[:, t], nxt + step_s, 0.0)
    return rem


def compute_visibility(sats, start, duration_s, step_s, lat, lon, alt_m=0.0, el_min=5.0):
    """Pipeline complet. lat/lon/alt_m : scalaires (terminal fixe) ou tableaux de
    longueur n_temps (terminal mobile)."""
    jd, fr = time_grid(start, duration_s, step_s)
    sat_ecef = propagate_ecef(sats, jd, fr)
    el, rng = look_angles(sat_ecef, geodetic_to_ecef(lat, lon, alt_m), up_vector(lat, lon))
    visible = el >= el_min                                        # NaN -> False
    return {"names": [n for n, _ in sats], "jd": jd, "fr": fr, "step_s": step_s,
            "el": el, "range_km": rng, "visible": visible,
            "remaining_s": remaining_visibility(visible, step_s)}
