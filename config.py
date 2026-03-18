"""
Central configuration for the HTHP comparison study.

Collects all tuneable parameters (fluid lists, economic assumptions,
sensitivity ranges and output paths) in one place so that downstream
modules never hard-code magic numbers.
"""

import json
import os

import numpy as np
from CoolProp.CoolProp import PropsSI

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
SIM_CACHE_DIR = os.path.join(CACHE_DIR, "sims")
SIM_INDEX_FILE = os.path.join(CACHE_DIR, "sim_index.json")
HEATER_CACHE_FILE = os.path.join(CACHE_DIR, "heater.json")
GAS_HEATER_CACHE_FILE = os.path.join(CACHE_DIR, "gas_heater.json")
ANALYSIS_JSON_FILE = os.path.join(CACHE_DIR, "analysis.json")

# ── Fluid combinations ──────────────────────────────────────────────────────
FLUIDS_C1 = ["R290", "R1270", "R717"]
FLUIDS_C2 = ["R600a", "R600", "R717"]

# ── Base-case economic parameters ────────────────────────────────────────────
BASE_FULL_LOAD_HOURS = 7500       # h/a (fixed, no sensitivity on this)
BASE_E1_C = 18.0                  # ct/kWh

# ── Alternative scenario parameters ─────────────────────────────────────────
ALT_E1_C = 35.0                   # ct/kWh  (high electricity price)

# ── Gas heater reference parameters ──────────────────────────────────────────
BASE_GAS_C = 3.5                  # ct/kWh  (= 35 EUR/MWh)

# ── Sensitivity ranges ───────────────────────────────────────────────────────
E1_C_RANGE = np.arange(10, 41, 2.5)  # ct/kWh

# ── Steam temperature (fixed boundary condition) ────────────────────────────
P_WATER = 2  # bar
T_STEAM = PropsSI("T", "P", P_WATER * 1e5, "Q", 0, "water") - 273.15  # ≈ 120.2 °C

# ── Lift share sensitivity ─────────────────────────────────────────────────
#    Fraction of the total temperature lift (T_steam - T_air_in) handled by
#    the lower cycle.  T34 = T_air_in + lift_share * (T_steam - T_air_in).
LIFT_SHARE_RANGE = [0.30, 0.40, 0.50, 0.60, 0.70]  # lower / upper
LIFT_SHARE_DEFAULT = 0.50                             # base case (50/50)

# ── T_source_in sensitivity ────────────────────────────────────────────────
T_SOURCE_IN_RANGE = list(range(20, 61, 10))  # °C (source water inlet temperature)
T_SOURCE_IN_DEFAULT = 20                     # °C

# ── Source water constraints ──────────────────────────────────────────────
#    Two modes for defining source water boundary conditions:
#    - "fixed_delta_T": fix T_in and T_out = T_in - SOURCE_DELTA_T (m free)
#    - "fixed_mass_flow": fix T_in and m = SOURCE_MASS_FLOW (T_out free)
SOURCE_DELTA_T = 10                          # K (fixed for all T_source_in cases)
SOURCE_MASS_FLOW = 40.0                      # kg/s (base case for fixed_mass_flow mode)
SOURCE_MASS_FLOW_RANGE = [20, 30, 40, 50]   # kg/s (sensitivity range)


# ── Helpers ──────────────────────────────────────────────────────────────────

def lift_share_to_T34(lift_share, T_source_in):
    """
    Compute the cycle-2 evaporation temperature T34 from lift share.

    Parameters
    ----------
    lift_share : float
        Fraction of total lift handled by the lower cycle (0–1).
    T_source_in : float
        Source water inlet temperature [deg C].

    Returns
    -------
    float
        Evaporation temperature of cycle 2 (T34) [deg C].
    """
    return T_source_in + lift_share * (T_STEAM - T_source_in)


def ls_label(lift_share):
    """
    Return a human-readable lift share label, e.g. ``"40/60"``.

    Parameters
    ----------
    lift_share : float
        Lower cycle fraction (0–1).

    Returns
    -------
    str
        ``"<lower>/<upper>"`` percentage string.
    """
    lower = int(round(lift_share * 100))
    upper = 100 - lower
    return f"{lower}/{upper}"


# ── JSON serialisation helpers ────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy types to native Python."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def sim_cache_folder(f1, f2, ls, T_src, source_mode="fixed_delta_T"):
    """Return the cache directory for a specific simulation combo."""
    ls_pct = int(round(ls * 100))
    base = os.path.join(SIM_CACHE_DIR, f"{f1}_{f2}", f"LS_{ls_pct}_Tsrc_{int(T_src)}")
    if source_mode != "fixed_delta_T":
        base += f"_{source_mode}"
    return base


def sim_key_to_str(key):
    """Encode a (f1, f2, ls, T_src) tuple as a JSON-safe string."""
    f1, f2, ls, T_src = key
    return f"{f1}|{f2}|{ls}|{T_src}"


def str_to_sim_key(s):
    """Decode a string key back to a (f1, f2, ls, T_src) tuple."""
    parts = s.split("|")
    return (parts[0], parts[1], float(parts[2]), float(parts[3]))
