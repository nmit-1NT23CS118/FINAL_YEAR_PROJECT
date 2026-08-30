"""
env_risk.py — Instant (single-day) Disease Pressure Index calculations
========================================================================
Ported from the Streamlit multi-disease dashboard (app/app.py in ev_project).
Unlike the LSTM forecaster, these formulas need ONLY today's four readings
(temperature, humidity, rainfall, soil_moisture) — no history required.
This is what powers the FR4 "combined diagnosis" today; the LSTM 5-day-ahead
forecast is a separate, optional feature that needs 7 days of stored history
(see cloud_store.py, added later once the DB is set up).

Now covers 8 diseases (previously 4). Added in this revision:
  - tomato_early_blight    (Alternaria solani)
  - potato_early_blight    (Alternaria solani — same pathogen as tomato)
  - pepper_bacterial_spot  (Xanthomonas spp., formerly X. campestris pv. vesicatoria)
  - potato_late_blight     (Phytophthora infestans — dedicated formula; previously
                            Potato___Late_blight was approximated via
                            "tomato_late_blight" in app.py's DISEASE_MAP since it's
                            the same pathogen. That approximation is still
                            scientifically reasonable, but a dedicated formula lets
                            us reflect potato-specific forecasting conventions
                            (Wallin/BLITECAST use a stricter RH>=90% threshold and a
                            cooler temperature optimum than typically cited for
                            tomato foliage).

Sources consulted for the new curves (temperature/humidity optima, not exact
model coefficients — the shapes/weights below are our own reasonable fit to
the qualitative consensus in these sources):
  - UMN Extension & UMaine Extension & APS: early blight optimum ~24-29C /
    82-86F, RH >=80-90%, free moisture from rain OR dew is sufficient (rain
    itself is "not a necessity" for onset).
  - NC State / Alabama / WVU Extension, and a Xanthomonas hot-pepper infection
    model (Lee et al., PMC4174847): bacterial spot favored 24-30C (75-86F),
    RH >85%, and critically requires wind-driven rain/splash for spread —
    rain is a much stronger driver here than for early blight.
  - Wallin severity values / BLITECAST, plus recent late-blight reviews:
    potato late blight infection accelerates sharply once RH stays >=90% for
    extended hours, with a cooler temperature optimum (~10-23C) than is often
    quoted for tomato foliage blight.

Tomato Early Blight, Potato Early Blight, and Pepper Bacterial Spot are now
covered. (Previously these 3 plus a dedicated Potato Late Blight formula were
listed as not-yet-covered — see DISEASE_MAP in app.py, which should now be
updated to point at these new keys instead of `None`.)
"""

import numpy as np

# ── Temperature (bell curves, disease-specific optima) ─────────────────────
def temp_score_tlb(T):
    """Tomato / Potato Late Blight (Phytophthora infestans): opt 15C, bounds 3-26C."""
    score = np.where(T < 15.0,
                      100.0 * np.exp(-0.5 * ((T - 15.0) / 5.0) ** 2),
                      100.0 * np.exp(-0.5 * ((T - 15.0) / 4.0) ** 2))
    return np.where((T < 3.0) | (T > 26.0), 0.0, np.clip(score, 0.0, 100.0))

def temp_score_gdm(T):
    """Grape Downy Mildew: opt 22C, bounds 10-30C."""
    score = np.where(T < 22.0,
                      100.0 * np.exp(-0.5 * ((T - 22.0) / 6.0) ** 2),
                      100.0 * np.exp(-0.5 * ((T - 22.0) / 4.0) ** 2))
    return np.where((T < 10.0) | (T > 30.0), 0.0, np.clip(score, 0.0, 100.0))

def temp_score_wlr(T):
    """Wheat Leaf Rust: opt 20C, bounds 5-30C."""
    score = np.where(T < 20.0,
                      100.0 * np.exp(-0.5 * ((T - 20.0) / 7.0) ** 2),
                      100.0 * np.exp(-0.5 * ((T - 20.0) / 5.0) ** 2))
    return np.where((T < 5.0) | (T > 30.0), 0.0, np.clip(score, 0.0, 100.0))

def temp_score_as(T):
    """Apple Scab: opt 17C, bounds 5-25C."""
    score = np.where(T < 17.0,
                      100.0 * np.exp(-0.5 * ((T - 17.0) / 6.0) ** 2),
                      100.0 * np.exp(-0.5 * ((T - 17.0) / 4.0) ** 2))
    return np.where((T < 5.0) | (T > 25.0), 0.0, np.clip(score, 0.0, 100.0))

def temp_score_eb(T):
    """Early Blight (Alternaria solani, tomato & potato): opt 27C, bounds 10-36C.
    Extension sources cite an 82-86F (28-30C) germination optimum and a wider
    24-29C infection-favorable band; conidia can still germinate (slowly) from
    around 15C up to the mid-30s."""
    score = np.where(T < 27.0,
                      100.0 * np.exp(-0.5 * ((T - 27.0) / 7.0) ** 2),
                      100.0 * np.exp(-0.5 * ((T - 27.0) / 5.0) ** 2))
    return np.where((T < 10.0) | (T > 36.0), 0.0, np.clip(score, 0.0, 100.0))

def temp_score_bs(T):
    """Pepper Bacterial Spot (Xanthomonas spp.): opt 29C, bounds 18-38C.
    Extension sources consistently cite 75-86F (24-30C) as favorable, with
    disease activity reported up to ~95F (35C)."""
    score = np.where(T < 29.0,
                      100.0 * np.exp(-0.5 * ((T - 29.0) / 8.0) ** 2),
                      100.0 * np.exp(-0.5 * ((T - 29.0) / 5.0) ** 2))
    return np.where((T < 18.0) | (T > 38.0), 0.0, np.clip(score, 0.0, 100.0))

def temp_score_plb(T):
    """Potato Late Blight (dedicated, Phytophthora infestans): opt 18C, bounds 7-24C.
    Wallin/BLITECAST-style models key off a ~7.2C (45F) minimum threshold and
    rapid spread in the 10-23C window; this is deliberately cooler/narrower
    than the shared tomato_late_blight curve above."""
    score = np.where(T < 18.0,
                      100.0 * np.exp(-0.5 * ((T - 18.0) / 6.0) ** 2),
                      100.0 * np.exp(-0.5 * ((T - 18.0) / 3.0) ** 2))
    return np.where((T < 7.0) | (T > 24.0), 0.0, np.clip(score, 0.0, 100.0))

# ── Humidity (logistic curves) ──────────────────────────────────────────────
def hum_score_tlb(RH):
    score = 100.0 / (1.0 + np.exp(-0.18 * (RH - 80.0)))
    return np.where(RH < 65.0, 0.0, np.clip(score, 0.0, 100.0))

def hum_score_gdm(RH):
    score = 100.0 / (1.0 + np.exp(-0.20 * (RH - 85.0)))
    return np.where(RH < 70.0, 0.0, np.clip(score, 0.0, 100.0))

def hum_score_wlr(RH):
    score = 100.0 / (1.0 + np.exp(-0.15 * (RH - 78.0)))
    return np.where(RH < 60.0, 0.0, np.clip(score, 0.0, 100.0))

def hum_score_as(RH):
    score = 100.0 / (1.0 + np.exp(-0.18 * (RH - 82.0)))
    return np.where(RH < 65.0, 0.0, np.clip(score, 0.0, 100.0))

def hum_score_eb(RH):
    """Early blight: extension sources cite RH >=80-90% as optimal, but dew
    alone (not captured by our rainfall input) can sustain infection, so the
    floor is set lower than for the other fungal diseases."""
    score = 100.0 / (1.0 + np.exp(-0.15 * (RH - 82.0)))
    return np.where(RH < 55.0, 0.0, np.clip(score, 0.0, 100.0))

def hum_score_bs(RH):
    """Bacterial spot: favored above ~85% RH; extended low-humidity spells
    (per WVU Extension) sharply curtail spread, hence the higher floor."""
    score = 100.0 / (1.0 + np.exp(-0.20 * (RH - 85.0)))
    return np.where(RH < 60.0, 0.0, np.clip(score, 0.0, 100.0))

def hum_score_plb(RH):
    """Potato late blight (dedicated): Wallin severity values only accrue once
    RH holds at/above ~90% for extended hours, so this curve is the steepest
    and has the highest floor of any disease modeled here."""
    score = 100.0 / (1.0 + np.exp(-0.35 * (RH - 90.0)))
    return np.where(RH < 75.0, 0.0, np.clip(score, 0.0, 100.0))

# ── Rainfall ─────────────────────────────────────────────────────────────────
def rain_score_saturating(R, scale=18.0):
    score = np.where(R == 0.0, 0.0, 90.0 * (1.0 - np.exp(-R / scale)))
    return np.clip(score, 0.0, 100.0)

def rain_score_wlr(R):
    score = np.where(R == 0.0, 0.0, np.where(R <= 10.0, 10.0 * R, 100.0 * np.exp(-0.04 * (R - 10.0))))
    return np.clip(score, 0.0, 100.0)

# ── Soil moisture (shared across diseases) ──────────────────────────────────
def soil_moisture_score(SM):
    return np.clip((SM - 10.0) / 50.0 * 100.0, 0.0, 100.0)

# ── Weights (must each sum to 1.0) ──────────────────────────────────────────
WEIGHTS = {
    "tomato_late_blight":   {"humidity": 0.35, "temperature": 0.30, "rainfall": 0.20, "soil_moisture": 0.15},
    "grape_downy_mildew":   {"humidity": 0.30, "temperature": 0.25, "rainfall": 0.30, "soil_moisture": 0.15},
    "wheat_leaf_rust":      {"humidity": 0.40, "temperature": 0.35, "rainfall": 0.10, "soil_moisture": 0.15},
    "apple_scab":           {"humidity": 0.35, "temperature": 0.25, "rainfall": 0.25, "soil_moisture": 0.15},
    # New in this revision:
    # Early blight spreads fine on dew alone, so rainfall carries less weight
    # than humidity/temperature; identical weighting for tomato and potato
    # since both are caused by the same pathogen (Alternaria solani).
    "tomato_early_blight":  {"humidity": 0.35, "temperature": 0.35, "rainfall": 0.15, "soil_moisture": 0.15},
    "potato_early_blight":  {"humidity": 0.35, "temperature": 0.35, "rainfall": 0.15, "soil_moisture": 0.15},
    # Bacterial spot needs wind-driven rain/splash to disperse, so rainfall is
    # weighted more heavily here than for any other disease in this file.
    "pepper_bacterial_spot": {"humidity": 0.30, "temperature": 0.25, "rainfall": 0.30, "soil_moisture": 0.15},
    # Dedicated potato late blight: Wallin/BLITECAST are fundamentally
    # RH-hour-accumulation models, so humidity dominates even more than in
    # the shared tomato_late_blight curve.
    "potato_late_blight":   {"humidity": 0.45, "temperature": 0.25, "rainfall": 0.15, "soil_moisture": 0.15},
}

# ── Disease registry ─────────────────────────────────────────────────────────
# Each entry wires a disease key to its temperature/humidity/rainfall score
# functions. compute_current_dpis() loops over this instead of hand-repeating
# a weighted-sum block per disease, so adding disease #9+ later is a matter of
# adding one entry here (plus a matching WEIGHTS row above).
DISEASE_MODELS = {
    "tomato_late_blight":    {"temp_fn": temp_score_tlb, "hum_fn": hum_score_tlb, "rain_fn": lambda R: rain_score_saturating(R, 18.0)},
    "grape_downy_mildew":    {"temp_fn": temp_score_gdm, "hum_fn": hum_score_gdm, "rain_fn": lambda R: rain_score_saturating(R, 15.0)},
    "wheat_leaf_rust":       {"temp_fn": temp_score_wlr, "hum_fn": hum_score_wlr, "rain_fn": rain_score_wlr},
    "apple_scab":            {"temp_fn": temp_score_as,  "hum_fn": hum_score_as,  "rain_fn": lambda R: rain_score_saturating(R, 12.0)},
    "tomato_early_blight":   {"temp_fn": temp_score_eb,  "hum_fn": hum_score_eb,  "rain_fn": lambda R: rain_score_saturating(R, 20.0)},
    "potato_early_blight":   {"temp_fn": temp_score_eb,  "hum_fn": hum_score_eb,  "rain_fn": lambda R: rain_score_saturating(R, 20.0)},
    "pepper_bacterial_spot": {"temp_fn": temp_score_bs,  "hum_fn": hum_score_bs,  "rain_fn": lambda R: rain_score_saturating(R, 10.0)},
    "potato_late_blight":    {"temp_fn": temp_score_plb, "hum_fn": hum_score_plb, "rain_fn": lambda R: rain_score_saturating(R, 16.0)},
}


def get_risk_label(dpi: float) -> str:
    """Low: 0-30, Medium: 31-60, High: 61-100."""
    if dpi <= 30.0:
        return "Low"
    elif dpi <= 60.0:
        return "Medium"
    return "High"


def compute_current_dpis(temperature: float, humidity: float, rainfall: float, soil_moisture: float) -> dict:
    """
    Computes today's DPI for all currently-covered diseases from a single
    day's reading. Returns {disease_key: {"dpi": float, "risk": "Low"/"Medium"/"High"}}.
    """
    T  = np.array([temperature], dtype=float)
    RH = np.array([humidity], dtype=float)
    R  = np.array([rainfall], dtype=float)
    SM = np.array([soil_moisture], dtype=float)

    sm_score = soil_moisture_score(SM)

    results = {}
    for key, model in DISEASE_MODELS.items():
        w = WEIGHTS[key]
        raw = (w["humidity"]      * model["hum_fn"](RH) +
               w["temperature"]   * model["temp_fn"](T) +
               w["rainfall"]      * model["rain_fn"](R) +
               w["soil_moisture"] * sm_score)
        val = round(float(np.clip(raw[0], 0.0, 100.0)), 2)
        results[key] = {"dpi": val, "risk": get_risk_label(val)}

    return results