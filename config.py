"""
Central configuration for the HTHP comparison study.

Collects all tuneable parameters (fluid lists, economic assumptions,
sensitivity ranges and output paths) in one place so that downstream
modules never hard-code magic numbers.
"""

import os

import numpy as np
from CoolProp.CoolProp import PropsSI

# ── Exergy-fuel definition (source-water outlet, stream 12) ─────────────────
# Two system-boundary conventions for the source water are studied:
#   "outlet_loss" — stream 11 (source in) is always FUEL, stream 12 (source
#                   out) is always a LOSS:  E_F = E_e1 + E_11,  E_L = E_12.
#   "outlet_fuel" — stream 12 is a FUEL OUTPUT while it leaves at or above
#                   ambient (net water fuel E_11 − E_12); only when it leaves
#                   BELOW ambient is its exergy counted as a LOSS.
# The choice only moves E_12 between the fuel and loss ledgers: E_P, E_D and
# every cost quantity (c_P, Z, PEC) are identical under both conventions —
# only E_F, E_L, epsilon and the component y-factor (E_D / E_F,tot) change.
# Each convention writes to its own results tree, results/ef_<definition>/,
# so both datasets coexist; plot_compare_EF_definitions.py reads both trees
# and writes comparison figures to results/<EF_COMPARE_DIR_NAME>/.
EF_DEFINITIONS = ("outlet_loss", "outlet_fuel")
EF_DEFINITION = "outlet_fuel"
EF_COMPARE_DIR_NAME = "ef_definition_compare"

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_BASE_DIR = os.path.join(BASE_DIR, "results")
RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, f"ef_{EF_DEFINITION}")


def ef_results_dir(ef_definition=None):
    """Results tree for one EF definition (defaults to the active one)."""
    return os.path.join(RESULTS_BASE_DIR, f"ef_{ef_definition or EF_DEFINITION}")


def ef_compare_dir():
    """Output directory for the cross-definition comparison plots."""
    return os.path.join(RESULTS_BASE_DIR, EF_COMPARE_DIR_NAME)

# ── Fluid combinations ──────────────────────────────────────────────────────
# Lower-cycle / upper-cycle working fluids screened in this study.
FLUIDS_C1 = ["R290", "R1270", "R717"]
FLUIDS_C2 = ["R600a", "R600"]

# ── Base-case economic parameters ────────────────────────────────────────────
BASE_FULL_LOAD_HOURS = 7500       # h/a (fixed, no sensitivity on this)

# Industrial electricity, German medium-industry full price (BDEW, 2026).
BASE_E1_C = 159.0                 # EUR/MWh

# ── Gas heater reference parameters ──────────────────────────────────────────
# Natural gas, German THE Day-Ahead wholesale (BDEW, 2026).
BASE_GAS_C = 47                  # EUR/MWh

# ── CO2 emission price (gas-heater reference only) ────────────────────────────
# Carbon charge on the gas fuel; midpoint of the EU ETS / BEHG 55-65 EUR/tCO2
# corridor (2026). Set to 0 to drop the carbon charge.
BASE_CO2_PRICE = 60.0            # EUR/tCO2

# ── Economic levelization parameters ─────────────────────────────────────────
# Discount rate and plant lifetime for the TRR / CELF computation.
I_EFF       = 0.10              # 1/a  effective discount rate
N_YEARS     = 20                # years, plant lifetime

# ── Annual nominal cost escalation rates (TRR / CELF) ────────────────────────
# Nominal escalation rates for the CELF levelization (BDEW pre-2022 trend;
# CO2 from the EU-ETS / BEHG forward midpoint).
R_N_OM      = 0.02              # 1/a   O&M escalation
R_N_EL      = 0.02              # 1/a   electricity escalation
R_N_GAS     = 0.03              # 1/a   natural-gas escalation
R_N_CO2     = 0.05              # 1/a   CO2 emission-price escalation

# ── Sensitivity ranges ───────────────────────────────────────────────────────
E1_C_RANGE = np.arange(100, 201, 10.0)  # EUR/MWh — brackets BASE_E1_C = 159

# ± 50 % electricity/gas price sweep (10 % steps) for the 2-D price sensitivity.
PRICE_SENS_FRAC_RANGE = np.arange(-0.5, 0.51, 0.1)  # -50 % … +50 %

# ── Steam temperature (sink boundary) ───────────────────────────────────────
# Fallback sink steam temperature when no override is passed.
T_STEAM_DEFAULT = 100.0  # °C

# ── Case study: native scale ─────────────────────────────────────────────────
# Sink steam mass flow. Sets Q_H (= m_steam x Dh_vap of water at T_steam),
# about 1.1 MWth across the three steam temperatures.
M_STEAM = 0.5                  # kg/s  sink-side steam mass flow (native scale)

T_STEAM_CASE_DEFAULT = 110.0      # °C  default for the single-design fast path

# Run main.py over each of these temperatures when --t-steam=all (default).
# Output lands under results/case_steam_<int(T)>/ for each one.
T_STEAMS_TO_RUN = [100.0, 110.0, 120.0]
# Folder where the cross-T_steam comparison plots are written.
T_STEAM_COMPARE_DIR_NAME = "case_steam_compare"


def m_steam_label():
    """Display string for the native scale, e.g. ``'m_steam = 0.5 kg/s'``."""
    return f"m_steam = {M_STEAM:g} kg/s"


def p_water_for_T_steam(T_steam_degC):
    """Saturation pressure of water at the given steam temperature [bar]."""
    return PropsSI("P", "T", T_steam_degC + 273.15, "Q", 0, "water") / 1e5


def case_results_dir(T_steam):
    """Return the per-T_steam case-study output directory."""
    return os.path.join(RESULTS_DIR, f"case_steam_{int(T_steam)}")


def case_data_dir(T_steam):
    """``case_steam_<T>/data/`` — economics-pipeline CSV inputs."""
    return os.path.join(case_results_dir(T_steam), "data")


def case_plots_dir(T_steam):
    """``case_steam_<T>/plots/`` — every PNG generated by the plotters."""
    return os.path.join(case_results_dir(T_steam), "plots")


def case_gas_heater_dir(T_steam):
    """``case_steam_<T>/gas_heater/`` — gas-reference exergoeconomic dump."""
    return os.path.join(case_results_dir(T_steam), "gas_heater")


def t_steam_compare_dir():
    """Return the cross-T_steam comparison output directory."""
    return os.path.join(RESULTS_DIR, T_STEAM_COMPARE_DIR_NAME)

# ── Lift share sensitivity ─────────────────────────────────────────────────
# Fraction of the total lift (T_steam - T_src,in) handled by the lower cycle:
# T34 = T_src,in + lift_share * (T_steam - T_src,in).
LIFT_SHARE_RANGE = [0.30, 0.40, 0.50, 0.60, 0.70]  # lower / upper
LIFT_SHARE_DEFAULT = 0.50                            # base case (50/50)

# ── T_source_in sensitivity ────────────────────────────────────────────────
T_SOURCE_IN_RANGE = list(range(20, 61, 10))  # °C (source water inlet temperature)
T_SOURCE_IN_DEFAULT = 20                     # °C

# ── Source water constraints ──────────────────────────────────────────────
#    Two modes for defining source water boundary conditions:
#    - "fixed_delta_T": fix T_in and T_out = T_in - SOURCE_DELTA_T (m free)
#    - "fixed_mass_flow": fix T_in and m = SOURCE_MASS_FLOW (T_out free)
SOURCE_DELTA_T = 10                          # K (used only in fixed_delta_T mode)
SOURCE_MASS_FLOW = 15.0                      # kg/s (default for fixed_mass_flow mode;
                                              # T_in fixed, T_src,out free)


# ── Helpers ──────────────────────────────────────────────────────────────────

def lift_share_to_T34(lift_share, T_source_in, T_steam=None):
    """
    Compute the cycle-2 evaporation temperature T34 from lift share.

    Parameters
    ----------
    lift_share : float
        Fraction of total lift handled by the lower cycle (0-1).
    T_source_in : float
        Source water inlet temperature [deg C].
    T_steam : float, optional
        Sink steam temperature [deg C]. If ``None``, falls back to
        ``T_STEAM_DEFAULT``.

    Returns
    -------
    float
        Evaporation temperature of cycle 2 (T34) [deg C].
    """
    T_steam_val = T_STEAM_DEFAULT if T_steam is None else T_steam
    return T_source_in + lift_share * (T_steam_val - T_source_in)
