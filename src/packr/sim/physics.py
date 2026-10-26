"""Phase 3: physics-based Monte Carlo shelf-life simulation (first pass, UNCALIBRATED).

One call simulates many (scenario, commodity, package) rows at once, each for `n_trials` Monte Carlo draws, and
returns the success rate (share of draws whose shelf life reaches the target and that hit no hard failure),
shelf-life percentiles and the share of failures by cause.

Two models, chosen by `category`:

  dry_snack     failure = moisture uptake past the critical water activity (texture) OR rancidity
                  moisture:  linear isotherm, dM/dt = (C_wv / (Ws b)) (Me - M),   Me = b RH_out
                  oxidation: oxygen is consumed at the intrinsic (Arrhenius) rate or the rate it gets in, whichever is
                             slower:  t_ox = max(t_intrinsic, (Q_crit - O2_headspace) / ingress)
  fresh_produce failure = anaerobic (O2 too low), CO2 injury, weight loss past the limit, or quality decay
                  steady-state gas balance with Michaelis-Menten respiration, Q10 on respiration and on quality decay,
                  transpiration limited by film + perforation (series resistances), condensation penalty.

Structural options (spec T3 part B). Every default reproduces v1 exactly; an option whose reference table has no sourced
row for a commodity / material / oil falls back to v1 for that row and says so in the `sim_model` output column:

  isotherm="gab"            B1  GAB sorption isotherm (reference/sorption_params.csv); time to the critical moisture by
                                 quadrature of dM / (RH - aw(M)) in log(RH - aw), stable when RH is close to the critical aw.
                                 Uses literature rows only (trust not starting "empirical_fit").
  isotherm="gab_empirical"  B1e the same physics with EMPIRICAL curve fits only (trust "empirical_fit..."), e.g. GAB fitted to
                                 (moisture, aw) pairs from storage trials. Only the curve is meaningful, so it is never used outside
                                 its valid_aw_min..valid_aw_max: a trial whose trajectory would leave that range falls back to v1
  oxidation="pv"            B2  peroxide value (PV) end point with a headspace-O2 mass balance and Michaelis-Menten in pO2
                                 (reference/oil_stability.csv via the commodity's oil_type)
  rh_dependent_otr=True     B3  per-layer OTR multiplier at the RH each layer sees (reference/rh_permeability.csv)
  temp_history="excursions" B4  per-trial temperature excursion events; every rate is time-averaged (Arrhenius-equivalent)
  produce_gas="transient"   B5  O2/CO2 from sealing to steady state; quality life consumed during the transient is counted

Units: gas conductances cm3/(day atm); vapour conductances g/(day per unit RH fraction); OTR cm3/(m2 day) at 23 C and
1 atm O2 difference; WVTR g/(m2 day) at 38 C / 90 % RH.

Every constant in CONSTANTS is an ASSUMPTION to be calibrated against literature pairings (see docs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

R_KJ = 8.314e-3
WVTR_DP_KPA = 0.90 * 0.61078 * np.exp(17.27 * 38.0 / (38.0 + 237.3))      # same constant as packr.units
MOLAR_VOLUME_CM3 = 22414.0                                                 # cm3(STP) per mol
LIFE_CAP_DAYS = 5000.0


def _structural(default):
    """A Constants field used only by a structural option (not part of the v1 model or its sensitivity analysis)."""
    return field(default=default, metadata={"structural": True})


@dataclass(frozen=True)
class Constants:
    ea_o2_kj: float = 40.0               # activation energy of film O2 permeability
    ea_wv_kj: float = 25.0               # activation energy of film water-vapour permeability
    d_o2_air: float = 0.20               # cm2/s, diffusion through holes
    d_wv_air: float = 0.25
    seal_channel_cm: float = 0.005       # length of a seal leak path (50 um)
    film_hole_cm: float = 0.0025         # length of a perforation (25 um film)
    leak_prob_by_seal: tuple = (0.30, 0.10, 0.04, 0.02, 0.01)      # seal rating 1..5
    leak_area_median_cm2: float = 1.0e-6
    leak_area_sigma_ln: float = 1.5
    damage_area_median_cm2: float = 1.0e-4
    damage_prob_normal: float = 0.03
    damage_prob_harsh: float = 0.12
    tol_sigma_ln: float = 0.15           # material / gauge tolerance on OTR and WVTR
    sigma_t_normal: float = 1.5          # temperature fluctuation, K
    sigma_t_harsh: float = 3.5
    harsh_t_offset: float = 1.5
    rh_sigma: float = 4.0
    o2_air: float = 0.209
    co2_o2_ratio_mean: float = 4.0       # film CO2 / O2 permeability ratio
    qcrit_ml_per_g_fat: float = 1.0
    headspace_cm3_per_g: float = 2.5

    # --- structural options (defaults = v1) ---------------------------------------------------------------------------
    isotherm: str = _structural("linear")              # "linear" | "gab" | "gab_empirical"
    oxidation: str = _structural("v1")                 # "v1" | "pv"
    rh_dependent_otr: bool = _structural(False)
    temp_history: str = _structural("constant")        # "constant" | "excursions"
    produce_gas: str = _structural("steady")           # "steady" | "transient"
    # B2: the PV end point is a measured quantity (mEq O2 / kg oil); 10 is a common rancidity limit for fried snacks
    pv_endpoint_meq_kg: float = _structural(10.0)
    # B4: ASSUMED excursion model (number of events per trial ~ Poisson, duration lognormal, amplitude normal)
    excursion_rate_normal: float = _structural(1.0)
    excursion_rate_harsh: float = _structural(3.0)
    excursion_days_median_normal: float = _structural(0.5)
    excursion_days_median_harsh: float = _structural(1.0)
    excursion_days_sigma_ln: float = _structural(0.7)
    excursion_amp_mean_normal: float = _structural(6.0)   # K
    excursion_amp_mean_harsh: float = _structural(10.0)
    excursion_amp_sd: float = _structural(3.0)
    excursion_max: int = _structural(8)
    # B5: ASSUMED free gas volume inside a produce pack
    produce_free_volume_cm3_per_kg: float = _structural(1000.0)
    transient_steps: int = _structural(160)

    def options(self) -> dict:
        return dict(isotherm=self.isotherm, oxidation=self.oxidation, rh_dependent_otr=self.rh_dependent_otr,
                    temp_history=self.temp_history, produce_gas=self.produce_gas)

    @property
    def is_v1(self) -> bool:
        return self.options() == Constants().options()


CONSTANTS = Constants()

COMMODITY_COLS = ["resp_ref_mgco2_kg_h", "resp_ref_temp_c", "q10_resp", "ideal_temp_c", "base_life_days", "chilling_sensitive",
                  "fragility", "o2_min_pct", "co2_max_pct", "crit_weight_loss_pct", "ma_gain", "transp_g_kg_d", "q10_decay",
                  "moisture_pct", "fat_pct", "crit_aw", "iso_b", "ox_ref_days", "ea_ox_kj", "light_sens"]
PACK_COLS = ["otr23", "wvtr38", "light_barrier", "seal", "strength", "mass_kg", "area_m2_per_kg", "perf_mm2_per_kg", "flush_o2_pct"]
SCEN_COLS = ["temp_c", "rh_pct", "harsh", "target_days"]
OUT_COLS = ["success_rate", "shelf_life_p10", "shelf_life_p50", "shelf_life_p90", "p_fail_moisture", "p_fail_oxidation",
            "p_fail_o2_low", "p_fail_co2_high", "p_fail_weight_loss", "p_fail_quality", "x_o2_pct_p50", "x_co2_pct_p50"]
# returned only with simulate(..., extra=True): per-mechanism lives and the quantities calibration compares with papers
EXTRA_COLS = ["life_moisture_p10", "life_moisture_p50", "life_moisture_p90", "life_oxidation_p10", "life_oxidation_p50",
              "life_oxidation_p90", "p_moisture_by_target", "p_oxidation_by_target", "aw0_p50", "pv_at_target_p50",
              "life_quality_p50", "life_weight_loss_p50", "moisture_db_at_target_p50", "aw_at_target_p50"]
# optional per-row input columns (calibration / tests): override the reference tables when present
OPTIONAL_COLS = ["gab_m0", "gab_C", "gab_K", "gab_crit_aw", "gab_aw_min", "gab_aw_max", "crit_moisture_db", "oil_pv_rate_25c", "oil_ea_kj", "oil_km_o2",
                 "otr_rh_factor", "oil_type", "layers", "gauge_scale"]


def psat_kpa(t_c):
    return 0.61078 * np.exp(17.27 * t_c / (t_c + 237.3))


def arrhenius(ea_kj, t_c, tref_c):
    """Rate multiplier at t_c relative to tref_c."""
    return np.exp(-ea_kj / R_KJ * (1.0 / (t_c + 273.15) - 1.0 / (tref_c + 273.15)))


def _rho_sat_g_cm3(t_c):
    return 2.1668 * psat_kpa(t_c) * 10.0 / (t_c + 273.15) * 1e-6     # g/cm3


def _col(df, name, n):
    return df[name].to_numpy(dtype=float)[:, None]


def _opt(df, name):
    """Optional per-row column as a float array (NaN where absent)."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), np.nan)


# ------------------------------------------------------------------------------------------------ GAB isotherm (B1)

def gab_moisture(aw, m0, c, k):
    """GAB: M = m0 C K aw / ((1 - K aw)(1 - K aw + C K aw)), M in g water / g solids."""
    return m0 * c * k * aw / ((1.0 - k * aw) * (1.0 - k * aw + c * k * aw))


def gab_dmoisture_daw(aw, m0, c, k):
    d = (1.0 - k * aw) * (1.0 - k * aw + c * k * aw)
    dd = -k * (1.0 - k * aw + c * k * aw) + (1.0 - k * aw) * (c - 1.0) * k
    return m0 * c * k * (d - aw * dd) / d ** 2


def gab_aw(m, m0, c, k):
    """Invert GAB: water activity at moisture m (dry basis). Solves -M(C-1)K^2 aw^2 + K(M(C-2) - m0 C) aw + M = 0."""
    m, m0, c, k = np.broadcast_arrays(np.asarray(m, float), m0, c, k)
    a = -m * (c - 1.0) * k ** 2
    b = k * (m * (c - 2.0) - m0 * c)
    cc = m
    disc = np.sqrt(np.clip(b * b - 4 * a * cc, 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = (-b - disc) / (2 * a)
        r2 = (-b + disc) / (2 * a)
        lin = -cc / b                                        # C == 1: a == 0
    ok1 = (r1 >= 0) & (r1 < 1.0 / k)
    root = np.where(np.abs(a) < 1e-15, lin, np.where(ok1, r1, r2))
    return np.where(m <= 0, 0.0, root)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)


def time_to_moisture_gab(aw0, awc, rh, k_eff, m0, c, k):
    """Days to go from aw0 to awc at outside RH (fraction), dM/dt = k_eff (RH - aw(M)), GAB isotherm.

    t = (1/k_eff) * integral_{aw0}^{awc} M'(aw) / (RH - aw) daw. With u = -ln(RH - aw) this is integral M'(aw(u)) du, a smooth
    integrand even when awc is close to RH (the stiff case), integrated by 24-point Gauss-Legendre.
    Infinite when RH <= awc; zero when aw0 >= awc.
    """
    aw0, awc, rh, k_eff, m0, c, k = np.broadcast_arrays(aw0, awc, rh, k_eff, m0, c, k)
    out = np.full(aw0.shape, np.inf)
    out = np.where(aw0 >= awc, 0.0, out)
    go = (aw0 < awc) & (rh > awc)
    if go.any():
        u0 = -np.log(rh[go] - aw0[go])
        u1 = -np.log(rh[go] - awc[go])
        half, mid = (u1 - u0) / 2.0, (u1 + u0) / 2.0
        u = mid[:, None] + half[:, None] * _GL_X[None, :]
        aw = rh[go][:, None] - np.exp(-u)
        integrand = gab_dmoisture_daw(aw, m0[go][:, None], c[go][:, None], k[go][:, None])
        out[go] = half * (integrand * _GL_W[None, :]).sum(1) / k_eff[go]
    return out


def aw_at_time_gab(aw0, rh, k_eff, t, m0, c, k, iters: int = 50):
    """Water activity after t days of uptake (aw0 < RH), GAB isotherm: bisection on aw for time_to_moisture_gab(aw0, aw) = t.
    NaN where aw0 >= RH (drying is not modelled)."""
    aw0, rh, k_eff, t, m0, c, k = np.broadcast_arrays(*(np.asarray(x, float) for x in (aw0, rh, k_eff, t, m0, c, k)))
    lo, hi = aw0.copy(), rh.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        slow = time_to_moisture_gab(aw0, mid, rh, k_eff, m0, c, k) < t       # reaching mid takes less than t: answer is higher
        lo, hi = np.where(slow, mid, lo), np.where(slow, hi, mid)
    return np.where(aw0 < rh, 0.5 * (lo + hi), np.nan)


# ------------------------------------------------------------------------------------------------ reference lookups

def _base_name(names: pd.Series) -> pd.Series:
    return names.astype(str).str.split("~").str[0]


@lru_cache(maxsize=4)
def _sorption_table():
    from ..paths import REFERENCE_DIR
    t = pd.read_csv(REFERENCE_DIR / "sorption_params.csv")
    t = t.dropna(subset=["m0", "C", "K"])
    t["empirical"] = t["trust"].fillna("").astype(str).str.startswith("empirical_fit")
    for col in ("valid_aw_min", "valid_aw_max"):
        t[col] = pd.to_numeric(t[col], errors="coerce") if col in t.columns else np.nan
    return t.set_index("commodity")


@lru_cache(maxsize=4)
def _oil_table():
    from ..paths import REFERENCE_DIR
    t = pd.read_csv(REFERENCE_DIR / "oil_stability.csv")
    rate = pd.to_numeric(t["pv_rate_25c_meq_kg_day"], errors="coerce")
    ind = pd.to_numeric(t["induction_days_25c"], errors="coerce")
    endpoint = pd.to_numeric(t["pv_endpoint_meq_kg"], errors="coerce").fillna(10.0)
    t["rate"] = rate.where(rate.notna(), endpoint / ind)       # an induction time is read as days to the PV end point in air
    return t.dropna(subset=["rate", "ea_kj", "km_o2_frac"]).set_index("oil")


@lru_cache(maxsize=4)
def _oil_type_by_commodity():
    from ..commodities import load_commodities
    m = load_commodities()
    return m.set_index("commodity")["oil_type"] if "oil_type" in m.columns else pd.Series(dtype=object)


def _gab_params(d: pd.DataFrame, empirical: bool = False):
    """(m0, C, K, crit_aw override, valid aw min, valid aw max) per row: row columns first, then reference/sorption_params.csv
    by commodity, literature rows or empirical-fit rows only (never mixed). NaN valid bounds mean no limit."""
    m0, c, k, caw, lo, hi = (_opt(d, x) for x in ("gab_m0", "gab_C", "gab_K", "gab_crit_aw", "gab_aw_min", "gab_aw_max"))
    tab = _sorption_table()
    tab = tab[tab["empirical"] == empirical]
    if len(tab) and "commodity" in d.columns:
        names = _base_name(d["commodity"])
        hit = names.isin(tab.index).to_numpy() & np.isnan(m0)
        if hit.any():
            sub = tab.loc[names[hit]]
            m0[hit], c[hit], k[hit] = sub["m0"].to_numpy(float), sub["C"].to_numpy(float), sub["K"].to_numpy(float)
            caw[hit] = pd.to_numeric(sub["crit_aw"], errors="coerce").to_numpy(float)
            lo[hit], hi[hit] = sub["valid_aw_min"].to_numpy(float), sub["valid_aw_max"].to_numpy(float)
    return m0, c, k, caw, lo, hi


def _oil_params(d: pd.DataFrame):
    rate, ea, km = (_opt(d, x) for x in ("oil_pv_rate_25c", "oil_ea_kj", "oil_km_o2"))
    tab = _oil_table()
    if len(tab):
        if "oil_type" in d.columns:
            oil = d["oil_type"]
        elif "commodity" in d.columns:
            oil = _base_name(d["commodity"]).map(_oil_type_by_commodity())
        else:
            oil = pd.Series([np.nan] * len(d))
        oil = oil.reset_index(drop=True)
        hit = (oil.isin(tab.index).to_numpy()) & np.isnan(rate)
        if hit.any():
            sub = tab.loc[oil[hit]]
            rate[hit], ea[hit], km[hit] = sub["rate"].to_numpy(float), sub["ea_kj"].to_numpy(float), sub["km_o2_frac"].to_numpy(float)
    return rate, ea, km


def _otr_rh_factor(d: pd.DataFrame, rh_in_pct: np.ndarray) -> tuple[np.ndarray, str]:
    """B3: OTR multiplier per row from its layers and the RH each layer sees. 1.0 where nothing is sourced."""
    f = _opt(d, "otr_rh_factor")
    if np.isfinite(f).all():
        return f, "row"
    from ..materials import load_materials, load_rh_factors, otr_rh_multiplier
    table = load_rh_factors()
    if table.empty or "layers" not in d.columns:
        return np.where(np.isfinite(f), f, 1.0), "no sourced RH curves (factor 1)"
    mats = load_materials()
    gs = _opt(d, "gauge_scale")
    gs = np.where(np.isfinite(gs), gs, 1.0)
    rh_out = d["rh_pct"].to_numpy(float)
    cache: dict = {}
    for i, lay in enumerate(d["layers"].astype(str)):
        if np.isfinite(f[i]):
            continue
        key = (lay, round(gs[i], 3), round(rh_in_pct[i]), round(rh_out[i]))
        if key not in cache:
            cache[key] = otr_rh_multiplier(lay.split("/"), rh_in_pct[i], rh_out[i], mats, table, gauge_scale=gs[i])
        f[i] = cache[key]
    return f, "sourced RH curves"


# ------------------------------------------------------------------------------------------------ temperature (B4)

class _Temperature:
    """v1: every rate is evaluated at the trial temperature t. B4: base t plus excursion events, and any rate F(T) is replaced by
    its time average (1 - W) F(t) + sum_j w_j F(t + a_j), w_j = duration_j / horizon: the Arrhenius-equivalent temperature."""

    def __init__(self, d, n, rng, c, t):
        self.t = t
        self.active = c.temp_history == "excursions"
        if not self.active:
            return
        R = len(d)
        harsh = _col(d, "harsh", n) > 0
        lam = np.where(harsh, c.excursion_rate_harsh, c.excursion_rate_normal) * np.ones((R, n))
        count = np.minimum(rng.poisson(lam), c.excursion_max)
        K = c.excursion_max
        med = np.where(harsh, c.excursion_days_median_harsh, c.excursion_days_median_normal)[..., None]
        dur = med * np.exp(rng.normal(0, c.excursion_days_sigma_ln, (R, n, K)))
        amp = np.where(harsh, c.excursion_amp_mean_harsh, c.excursion_amp_mean_normal)[..., None] + rng.normal(0, c.excursion_amp_sd, (R, n, K))
        horizon = np.maximum(_col(d, "target_days", n), 1.0)[..., None]
        w = np.where(np.arange(K) < count[..., None], dur / horizon, 0.0)
        tot = w.sum(-1, keepdims=True)
        w = w / np.maximum(tot, 1.0)                    # excursions longer than the horizon are rescaled to fill it
        self.w, self.amp, self.W = w, amp, np.minimum(tot[..., 0], 1.0)

    def avg(self, F):
        if not self.active:
            return F(self.t)
        out = (1.0 - self.W) * F(self.t)
        for j in range(self.w.shape[-1]):
            out = out + self.w[..., j] * F(self.t + self.amp[..., j])
        return out


# ------------------------------------------------------------------------------------------------ main entry

def simulate(df: pd.DataFrame, n_trials: int = 128, seed: int = 0, c: Constants = CONSTANTS, chunk: int = 2000,
             extra: bool = False) -> pd.DataFrame:
    """Simulate every row of `df` (needs SCEN_COLS + PACK_COLS + COMMODITY_COLS + `category`). Returns OUT_COLS, plus
    EXTRA_COLS when extra=True, plus a `sim_model` column whenever a structural option is switched on."""
    rng = np.random.default_rng(seed)
    cols = OUT_COLS + (EXTRA_COLS if extra else [])
    notes = not c.is_v1
    out = []
    for start in range(0, len(df), chunk):
        part = df.iloc[start:start + chunk]
        res = pd.DataFrame(index=part.index, columns=cols, dtype=float)
        model = pd.Series("", index=part.index, dtype=object)
        for cat, fn in (("dry_snack", _simulate_snack), ("fresh_produce", _simulate_produce)):
            m = (part["category"] == cat).to_numpy()
            if m.any():
                arr, txt = fn(part[m], n_trials, rng, c)
                res.loc[part.index[m]] = arr[:, :len(cols)] if extra else arr[:, :len(OUT_COLS)]
                model.loc[part.index[m]] = txt
        if notes:
            res["sim_model"] = model
        out.append(res)
    return pd.concat(out) if out else pd.DataFrame(columns=cols)


def _common_draws(d, n, rng, c):
    R = len(d)
    harsh = _col(d, "harsh", n)
    t = _col(d, "temp_c", n) + harsh * c.harsh_t_offset + rng.normal(0, 1, (R, n)) * np.where(harsh > 0, c.sigma_t_harsh, c.sigma_t_normal)
    rh = np.clip(_col(d, "rh_pct", n) + rng.normal(0, c.rh_sigma, (R, n)), 5, 99) / 100.0
    tol_o = np.exp(rng.normal(0, c.tol_sigma_ln, (R, n)))
    tol_w = np.exp(rng.normal(0, c.tol_sigma_ln, (R, n)))
    # leak / damage -> an extra pinhole of random area
    seal = np.clip(_col(d, "seal", n).astype(int), 1, 5)
    p_leak = np.array(c.leak_prob_by_seal)[seal - 1] * np.where(harsh > 0, 2.0, 1.0)
    leak = rng.random((R, n)) < p_leak
    a_leak = np.where(leak, np.exp(rng.normal(np.log(c.leak_area_median_cm2), c.leak_area_sigma_ln, (R, n))), 0.0)
    strength = _col(d, "strength", n)
    p_dam = np.where(harsh > 0, c.damage_prob_harsh, c.damage_prob_normal) * _col(d, "fragility", n) * np.clip((6 - strength) / 3.0, 0.2, 2.0)
    damage = rng.random((R, n)) < p_dam
    a_dam = np.where(damage, c.damage_area_median_cm2 * np.exp(rng.normal(0, 0.5, (R, n))), 0.0)
    return t, rh, tol_o, tol_w, a_leak + a_dam, damage


def _pct(x, q):
    return np.percentile(x, q, axis=1)


def _row_text(parts: list[tuple[str, np.ndarray]], R: int) -> np.ndarray:
    """Per-row 'key=value|...' description of the model actually used."""
    out = np.full(R, "", dtype=object)
    for key, vals in parts:
        out = out + np.where(out == "", "", "|") + key + "=" + vals.astype(str)
    return out


def _simulate_snack(d, n, rng, c):
    R = len(d)
    t, rh, tol_o, tol_w, a_pin, damage = _common_draws(d, n, rng, c)
    temp = _Temperature(d, n, rng, c, t)
    area = (_col(d, "area_m2_per_kg", n) * _col(d, "mass_kg", n))                       # m2
    mass_g = _col(d, "mass_kg", n) * 1000.0
    # --- moisture
    if not temp.active:
        ps = psat_kpa(t)
        wvtr = _col(d, "wvtr38", n) * tol_w * arrhenius(c.ea_wv_kj, t, 38.0)
        cond_kpa = wvtr / WVTR_DP_KPA                                                  # g/(m2 day kPa)
        g_wv_pin = c.d_wv_air * 86400.0 * a_pin / c.seal_channel_cm * _rho_sat_g_cm3(t)    # g/day per unit RH
        c_wv = cond_kpa * area * ps + g_wv_pin
    else:
        film = temp.avg(lambda T: arrhenius(c.ea_wv_kj, T, 38.0) * psat_kpa(T))
        c_wv = (_col(d, "wvtr38", n) * tol_w / WVTR_DP_KPA * area * film
                + c.d_wv_air * 86400.0 * a_pin / c.seal_channel_cm * temp.avg(_rho_sat_g_cm3))
    m0 = _col(d, "moisture_pct", n); m0 = m0 / (100.0 - m0)
    b = _col(d, "iso_b", n)
    mc = b * _col(d, "crit_aw", n)
    crit_m = _opt(d, "crit_moisture_db")[:, None]
    mc = np.where(np.isfinite(crit_m), crit_m, mc)
    ws = mass_g * (1.0 - _col(d, "moisture_pct", n) / 100.0)
    me = b * rh
    k_eff = c_wv / (ws * b)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_m = np.where(me <= mc, np.inf, np.log(np.clip((me - m0) / np.clip(me - mc, 1e-12, None), 1e-12, None)) / k_eff)
    t_m = np.where(m0 >= mc, 0.0, t_m)
    aw0 = m0 / b * np.ones((1, n))
    tgt = _col(d, "target_days", n)
    m_tgt = me - (me - m0) * np.exp(-k_eff * tgt)                                       # linear isotherm, closed form
    aw_tgt = m_tgt / b
    no_row = {"linear": "", "gab": "linear(fallback: no sourced GAB row)", "gab_empirical": "linear(fallback: no empirical GAB row)"}
    iso_used = np.full(R, "linear" if c.isotherm == "linear" else no_row[c.isotherm], dtype=object)
    if c.isotherm in ("gab", "gab_empirical"):
        g_m0, g_c, g_k, g_caw, g_lo, g_hi = _gab_params(d, empirical=c.isotherm == "gab_empirical")
        use = np.isfinite(g_m0) & np.isfinite(g_c) & np.isfinite(g_k)
        if use.any():
            gm0, gc, gk = g_m0[use][:, None], g_c[use][:, None], g_k[use][:, None]
            lo = np.where(np.isfinite(g_lo[use]), g_lo[use], 0.0)[:, None]
            hi = np.where(np.isfinite(g_hi[use]), g_hi[use], 1.0)[:, None]
            caw = np.where(np.isfinite(g_caw[use]), g_caw[use], d["crit_aw"].to_numpy(float)[use])[:, None]
            aw0_g = gab_aw(m0[use], gm0, gc, gk) * np.ones((1, n))
            awc = np.where(np.isfinite(crit_m[use]), gab_aw(crit_m[use], gm0, gc, gk), caw) * np.ones((1, n))
            k_g = c_wv[use] / ws[use]
            start_ok = (aw0_g >= lo) & (aw0_g <= hi)
            # time to the critical aw: the curve is needed from aw0 up to min(awc, RH) (drying never reaches awc)
            ok_t = start_ok & (np.minimum(awc, np.maximum(rh[use], aw0_g)) <= hi)
            t_m[use] = np.where(ok_t, time_to_moisture_gab(aw0_g, awc, rh[use], k_g, gm0, gc, gk), t_m[use])
            aw0[use] = np.where(start_ok, aw0_g, aw0[use])
            # state at the target day: the curve is needed from aw0 up to aw(target); drying (NaN) is not modelled
            aw_g = aw_at_time_gab(aw0_g, rh[use], k_g, tgt[use], gm0, gc, gk)
            ok_m = start_ok & np.isfinite(aw_g) & (np.nan_to_num(aw_g, nan=2.0) <= hi)
            aw_tgt[use] = np.where(ok_m, aw_g, aw_tgt[use])
            m_tgt[use] = np.where(ok_m, gab_moisture(np.where(ok_m, aw_g, 0.0), gm0, gc, gk), m_tgt[use])
            out_t, out_m = 1.0 - ok_t.mean(1), 1.0 - ok_m.mean(1)
            limited = np.isfinite(g_lo[use]) | np.isfinite(g_hi[use])
            rng_txt = np.array([f"outside aw {a:g}-{b:g} or drying" if lim else "when drying" for a, b, lim in zip(lo[:, 0], hi[:, 0], limited)],
                               dtype=object)
            txt = np.array([c.isotherm if (ot == 0 and om == 0) else
                            f"{c.isotherm}(v1 {r}: life {ot:.0%}, target-day state {om:.0%} of trials)"
                            for ot, om, r in zip(out_t, out_m, rng_txt)], dtype=object)
            iso_used[use] = txt
    # --- oxidation
    fat_g = mass_g * _col(d, "fat_pct", n) / 100.0
    q_crit = c.qcrit_ml_per_g_fat * fat_g
    o2_head = c.headspace_cm3_per_g * mass_g * _col(d, "flush_o2_pct", n) / 100.0
    g_pin_o2 = c.d_o2_air * 86400.0 * a_pin / c.seal_channel_cm
    otr_eff = _col(d, "otr23", n)
    rh_used = np.full(R, "off", dtype=object)
    if c.rh_dependent_otr:
        fac, how = _otr_rh_factor(d, (aw0[:, 0] * 100.0))
        otr_eff = otr_eff * fac[:, None]
        rh_used[:] = how
    if not temp.active:
        ingress = (otr_eff * tol_o * arrhenius(c.ea_o2_kj, t, 23.0) * area + g_pin_o2) * c.o2_air
    else:
        ingress = (otr_eff * tol_o * temp.avg(lambda T: arrhenius(c.ea_o2_kj, T, 23.0)) * area + g_pin_o2) * c.o2_air
    light_exp = rng.random((R, n))
    light_f = 1.0 + _col(d, "light_sens", n) * (1.0 - _col(d, "light_barrier", n)) * light_exp
    lot = np.exp(rng.normal(0, 0.25, (R, n)))
    ea_ox = _col(d, "ea_ox_kj", n)
    if not temp.active:
        t_int = (_col(d, "ox_ref_days", n) * np.exp(ea_ox / R_KJ * (1.0 / (t + 273.15) - 1.0 / 298.15)) * lot / light_f)
    else:
        t_int = _col(d, "ox_ref_days", n) / temp.avg(lambda T: arrhenius(ea_ox, T, 25.0)) * lot / light_f
    t_sup = (q_crit - o2_head) / np.clip(ingress, 1e-12, None)
    t_o = np.maximum(t_int, t_sup)
    target = _col(d, "target_days", n)
    pv_target = np.full((R, n), np.nan)
    ox_used = np.full(R, "v1" if c.oxidation == "v1" else "v1(fallback: no sourced oil stability row)", dtype=object)
    if c.oxidation == "pv":
        rate, ea_oil, km = _oil_params(d)
        use = np.isfinite(rate) & np.isfinite(ea_oil) & np.isfinite(km)
        if use.any():
            ox_used[use] = "pv"
            ea_full = np.where(use, ea_oil, 0.0)[:, None]
            kpv = rate[use][:, None] * temp.avg(lambda T: arrhenius(ea_full, T, 25.0))[use] * lot[use] * light_f[use]
            g_o2 = (ingress[use] / c.o2_air)                                     # cm3/(day atm)
            vhs = c.headspace_cm3_per_g * mass_g[use] * np.ones((1, n))
            p0 = _col(d, "flush_o2_pct", n)[use] / 100.0 * np.ones((1, n))
            oil_kg = fat_g[use] / 1000.0 * np.ones((1, n))
            t_o[use], pv_target[use] = _pv_life(kpv, km[use][:, None] * np.ones((1, n)), g_o2, vhs, p0, oil_kg,
                                                target[use] * np.ones((1, n)), c.pv_endpoint_meq_kg, c.o2_air)
    life = np.minimum(t_m, t_o) * np.where(damage, 0.9, 1.0)
    ok = life >= target
    life_f = np.minimum(life, LIFE_CAP_DAYS)
    res = np.full((R, len(OUT_COLS) + len(EXTRA_COLS)), np.nan)
    ix = {k: i for i, k in enumerate(OUT_COLS + EXTRA_COLS)}
    res[:, ix["success_rate"]] = ok.mean(1)
    for q, k in ((10, "shelf_life_p10"), (50, "shelf_life_p50"), (90, "shelf_life_p90")):
        res[:, ix[k]] = _pct(life_f, q)
    res[:, ix["p_fail_moisture"]] = ((t_m < target) & (t_m <= t_o)).mean(1)
    res[:, ix["p_fail_oxidation"]] = ((t_o < target) & (t_o < t_m)).mean(1)
    for k in ("p_fail_o2_low", "p_fail_co2_high", "p_fail_weight_loss", "p_fail_quality"):
        res[:, ix[k]] = 0.0
    tm_f, to_f = np.minimum(t_m, LIFE_CAP_DAYS), np.minimum(t_o, LIFE_CAP_DAYS)
    for q in (10, 50, 90):
        res[:, ix[f"life_moisture_p{q}"]] = _pct(tm_f, q)
        res[:, ix[f"life_oxidation_p{q}"]] = _pct(to_f, q)
    res[:, ix["p_moisture_by_target"]] = (t_m < target).mean(1)
    res[:, ix["p_oxidation_by_target"]] = (t_o < target).mean(1)
    res[:, ix["aw0_p50"]] = _pct(aw0, 50)
    res[:, ix["moisture_db_at_target_p50"]] = np.nanpercentile(m_tgt, 50, axis=1)
    res[:, ix["aw_at_target_p50"]] = np.nanpercentile(aw_tgt, 50, axis=1)
    res[:, ix["pv_at_target_p50"]] = np.where(np.isnan(pv_target).all(1), np.nan, np.nanpercentile(np.where(np.isnan(pv_target), 0, pv_target), 50, axis=1))
    text = _row_text([("iso", iso_used), ("ox", ox_used), ("otr_rh", rh_used), ("temp", np.full(R, c.temp_history, dtype=object)),
                      ("gas", np.full(R, "n/a", dtype=object))], R)
    return res, text


def _pv_life(kpv, km, g_o2, vhs, p0, oil_kg, target, endpoint, o2_air, steps: int = 400, t_max: float = LIFE_CAP_DAYS):
    """B2: integrate headspace O2 (fraction p) and peroxide value PV (mEq O2 / kg oil) from sealing.

        Vhs dp/dt = G (0.209 - p) - A f(p),   dPV/dt = kpv f(p) / f(0.209),   f(p) = p / (km + p)
        A = kpv * oil_kg * 0.5 mmol O2 per mEq * 22.414 cm3 per mmol / f(0.209)

    kpv is the PV rate in air at the trial temperature (mEq/kg/day). Semi-implicit (linearised Michaelis-Menten) steps on a
    geometric time grid: unconditionally stable. Returns (days to PV = endpoint, PV at the target day)."""
    f_air = o2_air / (km + o2_air)
    a = kpv * oil_kg * 0.5 * MOLAR_VOLUME_CM3 / 1000.0 / f_air
    grid = np.concatenate([[0.0], np.geomspace(0.01, t_max, steps)])
    p, pv = p0.copy(), np.zeros_like(p0)
    life = np.full(p0.shape, np.inf)
    pv_t = np.full(p0.shape, np.nan)
    for t0, t1 in zip(grid[:-1], grid[1:]):
        dt = t1 - t0
        p_new = (p + dt / vhs * g_o2 * o2_air) / (1.0 + dt / vhs * (g_o2 + a / (km + p)))
        pv_new = pv + dt * kpv * (p_new / (km + p_new)) / f_air
        cross = np.isinf(life) & (pv_new >= endpoint)
        life = np.where(cross, t0 + dt * (endpoint - pv) / np.maximum(pv_new - pv, 1e-30), life)
        at = np.isnan(pv_t) & (t1 >= target)
        pv_t = np.where(at, pv + (pv_new - pv) * (target - t0) / dt, pv_t)
        p, pv = p_new, pv_new
    return life, pv_t


def o2_transient(x0, x_ss_guess, g, r_air, km, norm, vol, horizon, steps):
    """B5 helper: in-pack O2 fraction from x0 under V dx/dt = G (0.209 - x) - R_air (x / (km + x)) / norm.
    Semi-implicit steps on a uniform grid of `steps` over `horizon` days (arrays broadcast). Returns (times, x(t))."""
    times = np.linspace(0.0, 1.0, steps + 1)[:, None, None] * horizon[None]
    x = x0 * np.ones_like(horizon)
    xs = [x]
    for i in range(steps):
        dt = times[i + 1] - times[i]
        x = (x + dt / vol * g * 0.209) / (1.0 + dt / vol * (g + r_air / norm / (km + x)))
        xs.append(x)
    return times, np.stack(xs)


def _simulate_produce(d, n, rng, c):
    R = len(d)
    t, rh, tol_o, tol_w, a_pin, damage = _common_draws(d, n, rng, c)
    temp = _Temperature(d, n, rng, c, t)
    mass = _col(d, "mass_kg", n)
    area = _col(d, "area_m2_per_kg", n) * mass
    perf_cm2 = _col(d, "perf_mm2_per_kg", n) * mass / 100.0
    a_tot = perf_cm2 + a_pin
    otr_eff = _col(d, "otr23", n)
    rh_used = np.full(R, "off", dtype=object)
    if c.rh_dependent_otr:
        fac, how = _otr_rh_factor(d, np.full(R, 99.0))
        otr_eff = otr_eff * fac[:, None]
        rh_used[:] = how
    # conductances (cm3/day/atm)
    if not temp.active:
        g_film_o2 = otr_eff * tol_o * arrhenius(c.ea_o2_kj, t, 23.0) * area
    else:
        g_film_o2 = otr_eff * tol_o * temp.avg(lambda T: arrhenius(c.ea_o2_kj, T, 23.0)) * area
    g_hole_o2 = c.d_o2_air * 86400.0 * (perf_cm2 / c.film_hole_cm + a_pin / c.seal_channel_cm)
    beta = np.clip(c.co2_o2_ratio_mean + rng.normal(0, 0.6, (R, n)), 2.0, 7.0)
    g_o2 = g_film_o2 + g_hole_o2
    g_co2 = beta * g_film_o2 + 0.8 * g_hole_o2
    # respiration
    q10 = np.clip(_col(d, "q10_resp", n) + rng.normal(0, 0.2, (R, n)), 1.5, 4.0)
    if not temp.active:
        r0 = (_col(d, "resp_ref_mgco2_kg_h", n) * np.exp(rng.normal(0, 0.25, (R, n)))
              * q10 ** ((t - _col(d, "resp_ref_temp_c", n)) / 10.0))
    else:
        tref = _col(d, "resp_ref_temp_c", n)
        r0 = _col(d, "resp_ref_mgco2_kg_h", n) * np.exp(rng.normal(0, 0.25, (R, n))) * temp.avg(lambda T: q10 ** ((T - tref) / 10.0))
    r_co2_air = r0 * mass * 13.1                                      # cm3 CO2 per day at full O2
    rq = np.clip(1.0 + rng.normal(0, 0.1, (R, n)), 0.7, 1.4)
    km = rng.uniform(0.01, 0.03, (R, n))
    norm = c.o2_air / (km + c.o2_air)
    r_o2_air = r_co2_air / rq
    lo, hi = np.zeros((R, n)), np.full((R, n), c.o2_air)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        h = g_o2 * (c.o2_air - mid) - r_o2_air * (mid / (km + mid)) / norm
        pos = h > 0
        lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
    x_o2 = 0.5 * (lo + hi)
    r_o2 = r_o2_air * (x_o2 / (km + x_o2)) / norm
    x_co2 = np.clip(r_o2 * rq / np.clip(g_co2, 1e-9, None) + 0.0004, 0.0, 0.5)
    viol_o2 = x_o2 * 100.0 < _col(d, "o2_min_pct", n) * rng.uniform(0.8, 1.2, (R, n))
    viol_co2 = x_co2 * 100.0 > _col(d, "co2_max_pct", n) * rng.uniform(0.8, 1.2, (R, n))
    # water balance (series: product transpiration, then pack)
    if not temp.active:
        wvtr = _col(d, "wvtr38", n) * tol_w * arrhenius(c.ea_wv_kj, t, 38.0)
        ps = psat_kpa(t)
        c_pack = (wvtr / WVTR_DP_KPA) * area * ps + c.d_wv_air * 86400.0 * _rho_sat_g_cm3(t) * (perf_cm2 / c.film_hole_cm + a_pin / c.seal_channel_cm)
        k_t = _col(d, "transp_g_kg_d", n) * mass * (ps / psat_kpa(20.0))
    else:
        c_pack = (_col(d, "wvtr38", n) * tol_w / WVTR_DP_KPA * area * temp.avg(lambda T: arrhenius(c.ea_wv_kj, T, 38.0) * psat_kpa(T))
                  + c.d_wv_air * 86400.0 * temp.avg(_rho_sat_g_cm3) * (perf_cm2 / c.film_hole_cm + a_pin / c.seal_channel_cm))
        k_t = _col(d, "transp_g_kg_d", n) * mass * (temp.avg(psat_kpa) / psat_kpa(20.0))
    rh_in = (k_t * 0.99 + c_pack * rh) / (k_t + c_pack)
    loss = c_pack * (rh_in - rh)                                       # g/day
    loss = np.clip(loss, 1e-9, None)
    t_wl = _col(d, "crit_weight_loss_pct", n) / 100.0 * mass * 1000.0 / loss
    # quality life
    ideal = _col(d, "ideal_temp_c", n)
    q10d = _col(d, "q10_decay", n)
    if not temp.active:
        life_q = _col(d, "base_life_days", n) / (q10d ** ((t - ideal) / 10.0))
    else:
        life_q = _col(d, "base_life_days", n) / temp.avg(lambda T: q10d ** ((T - ideal) / 10.0))
    chill = _col(d, "chilling_sensitive", n) > 0
    life_q = np.where(chill & (t < ideal - 2.0), life_q * np.clip(1.0 - 0.1 * (ideal - 2.0 - t), 0.3, 1.0), life_q)
    ma_gain = _col(d, "ma_gain", n)
    benefit = 0.7 * np.clip((c.o2_air - x_o2) / (c.o2_air - 0.03), 0, 1) + 0.3 * np.clip(x_co2 / 0.05, 0, 1)
    life_q = life_q * (1.0 + ma_gain * benefit) * np.exp(rng.normal(0, 0.15, (R, n)))
    cond_pen = np.clip(0.35 * np.clip((rh_in - 0.97) / 0.02, 0, 1) * (np.where(_col(d, "harsh", n) > 0, c.sigma_t_harsh, c.sigma_t_normal) / 2.0), 0, 0.6)
    life_q = life_q * (1.0 - cond_pen) * np.where(damage, 0.8, 1.0)
    gas_used = "steady"
    if c.produce_gas == "transient":
        gas_used = "transient"
        life_q = _transient_quality_life(life_q, benefit, ma_gain, x_o2, g_o2, g_co2, r_o2_air, km, norm, rq, mass,
                                         _col(d, "flush_o2_pct", n) / 100.0, c)
    life = np.minimum(life_q, t_wl)
    viol = viol_o2 | viol_co2
    life = np.where(viol, life * 0.3, life)
    target = _col(d, "target_days", n)
    ok = (life >= target) & ~viol
    life_f = np.minimum(life, LIFE_CAP_DAYS)
    res = np.full((R, len(OUT_COLS) + len(EXTRA_COLS)), np.nan)
    ix = {k: i for i, k in enumerate(OUT_COLS + EXTRA_COLS)}
    res[:, ix["success_rate"]] = ok.mean(1)
    for q, k in ((10, "shelf_life_p10"), (50, "shelf_life_p50"), (90, "shelf_life_p90")):
        res[:, ix[k]] = _pct(life_f, q)
    res[:, ix["p_fail_moisture"]] = 0.0
    res[:, ix["p_fail_oxidation"]] = 0.0
    res[:, ix["p_fail_o2_low"]] = viol_o2.mean(1)
    res[:, ix["p_fail_co2_high"]] = (viol_co2 & ~viol_o2).mean(1)
    res[:, ix["p_fail_weight_loss"]] = ((t_wl < target) & ~viol & (t_wl <= life_q)).mean(1)
    res[:, ix["p_fail_quality"]] = ((life_q < target) & ~viol & (life_q < t_wl)).mean(1)
    res[:, ix["x_o2_pct_p50"]] = _pct(x_o2 * 100.0, 50)
    res[:, ix["x_co2_pct_p50"]] = _pct(x_co2 * 100.0, 50)
    res[:, ix["life_quality_p50"]] = _pct(np.minimum(life_q, LIFE_CAP_DAYS), 50)
    res[:, ix["life_weight_loss_p50"]] = _pct(np.minimum(t_wl, LIFE_CAP_DAYS), 50)
    text = _row_text([("iso", np.full(R, "n/a", dtype=object)), ("ox", np.full(R, "n/a", dtype=object)), ("otr_rh", rh_used),
                      ("temp", np.full(R, c.temp_history, dtype=object)), ("gas", np.full(R, gas_used, dtype=object))], R)
    return res, text


def _transient_quality_life(life_ss, benefit_ss, ma_gain, x_ss, g_o2, g_co2, r_o2_air, km, norm, rq, mass, x0, c):
    """B5: quality life when the pack starts at the sealing gas (x0 O2, no CO2) instead of at steady state.

    Quality decays at rate 1 / (L0 (1 + ma_gain * benefit(x(t), y(t)))), L0 = life_ss / (1 + ma_gain * benefit_ss), so the
    steady-state rate is unchanged. O2: V dx/dt = G_O2 (0.209 - x) - R(x); CO2: V dy/dt = RQ R(x) - G_CO2 (y - 0.0004). Both are
    integrated semi-implicitly over 8 time constants V / min(G_O2, G_CO2); the life consumed there is added to the remainder at the
    steady-state rate. Violations are still judged at steady state (as v1)."""
    vol = c.produce_free_volume_cm3_per_kg * mass * np.ones_like(life_ss)
    tau = vol / np.maximum(np.minimum(g_o2, g_co2), 1e-9)
    horizon = 8.0 * tau
    steps = c.transient_steps
    l0 = life_ss / (1.0 + ma_gain * benefit_ss)
    x = x0 * np.ones_like(life_ss)
    y = np.zeros_like(life_ss)
    consumed = np.zeros_like(life_ss)
    life = np.full(life_ss.shape, np.nan)
    dt = horizon / steps

    def rate(xx, yy):
        ben = 0.7 * np.clip((c.o2_air - xx) / (c.o2_air - 0.03), 0, 1) + 0.3 * np.clip(yy / 0.05, 0, 1)
        return 1.0 / (l0 * (1.0 + ma_gain * ben))

    r_prev = rate(x, y)
    for i in range(steps):
        x = (x + dt / vol * g_o2 * c.o2_air) / (1.0 + dt / vol * (g_o2 + r_o2_air / norm / (km + x)))
        r_o2 = r_o2_air * (x / (km + x)) / norm
        y = (y + dt / vol * (rq * r_o2 + g_co2 * 0.0004)) / (1.0 + dt / vol * g_co2)
        r_now = rate(x, y)
        step = 0.5 * (r_prev + r_now) * dt
        cross = np.isnan(life) & (consumed + step >= 1.0)
        life = np.where(cross, i * dt + dt * (1.0 - consumed) / np.maximum(step, 1e-30), life)
        consumed = consumed + step
        r_prev = r_now
    tail = horizon + np.maximum(1.0 - consumed, 0.0) * life_ss
    return np.where(np.isnan(life), tail, life)
