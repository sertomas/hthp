"""
Central configuration for the HTHP comparison study.

Collects all tuneable parameters (fluid lists, economic assumptions,
sensitivity ranges and output paths) in one place so that downstream
modules never hard-code magic numbers.
"""

import os

import numpy as np
from CoolProp.CoolProp import PropsSI

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── Fluid combinations ──────────────────────────────────────────────────────
# Cycle-2 fluids are restricted to hydrocarbons (R600a, R600). R717 was
# considered for cycle 2 but excluded after analysis:
#   1. R717 in cycle 2 condenses at T_steam + pinch ≈ 115 °C → p_high ≈ 71 bar,
#      which sits above the Ommen 2015 R717-HP envelope (50 bar) and at the
#      edge of modern commercial compressors (Vilter VSSH 76 bar). Cost data
#      for R717-HP at this pressure is NDA in Ommen Table 4; the project uses
#      a 1.5× engineering multiplier on R717-LP as a placeholder, introducing
#      uncertainty into any economic ranking.
#   2. Industrial demonstrators that pair R717 with a high-temperature top
#      cycle universally use R717+R718 (steam) — Aneo Industry, AGO Energie,
#      GEA Ammonia steam generator (IEA HPT Project 68 Table 2-1). There is
#      no commercial precedent for R717+butane cascade.
#   3. R717/R717 specifically is not a real cascade — same fluid in two loops
#      offers no thermodynamic advantage over single-stage two-compression
#      with intermediate-pressure vessel (economizer / flash tank). TESPy
#      converges but the configuration would never be built.
# R717 in cycle 1 (R717/R600, R717/R600a) is retained: cycle-1 R717 condenses
# at the intermediate temperature T34, which for LS ≤ 0.50 stays below 28 bar
# (R717-LP envelope, properly costed in Ommen Table 4).
#
# R744 (CO2) is intentionally NOT included. Two reasons:
#  1. Compressor scope: Ommen 2015 Table 3 lists R744 capacity at 6-25 m³/h
#     (an order of magnitude smaller than HC compressors at 5-280 m³/h).
#     The cost correlation is calibrated for small machines, well below
#     the suction volumetric flow required by an industrial steam-generating
#     HTHP.
#  2. Heat-sink mismatch: R744 transcritical wins on NPV in Ommen 2015 only
#     for HIGH sink-temperature glide (≥ 40 K, e.g. district heating). Our
#     sink is saturated steam at constant T — an isothermal sink — which
#     wastes R744's gliding gas-cooler advantage. The transcritical
#     plumbing in models.py (_is_transcritical, _R744_P_HIGH_BAR) is kept
#     in place as a starting point for a possible single-stage R744
#     follow-up study, but R744 is not in any active fluid list.
FLUIDS_C1 = ["R290", "R1270", "R717"]
FLUIDS_C2 = ["R600a", "R600"]

# ── Base-case economic parameters ────────────────────────────────────────────
BASE_FULL_LOAD_HOURS = 7500       # h/a (fixed, no sensitivity on this)

# Industrial electricity, medium consumer (20-70 GWh/a).
# Full end-customer price including taxes, surcharges and grid fees.
# Source: BDEW Strompreisanalyse (Bundesverband der Energie- und
# Wasserwirtschaft), 2025 release. The 144 EUR/MWh row in the same table
# applies to large industry (70-150 GWh/a); a steam-generating HTHP at
# Q_H ≈ 2.2 MW and 7500 h/a, COP ≈ 3, draws ~5.5 GWh_el/a, which sits inside
# the medium-industry band → 159 EUR/MWh is the appropriate full-cost reference.
BASE_E1_C = 159.0                 # EUR/MWh

# ── Gas heater reference parameters ──────────────────────────────────────────
# Natural gas wholesale, Day-Ahead spot at the German THE virtual hub.
# End-of-April 2026 reading. Source: BDEW gas market monitoring.
# Industrial end-customer prices are typically wholesale + ~10-20 EUR/MWh
# of grid/distribution fees, but this study mirrors Ommen (2015) which uses
# bare market prices for both fuels (no industrial gas margin added).
BASE_GAS_C = 47                  # EUR/MWh

# ── CO2 emission price (gas-heater reference only) ────────────────────────────
# Carbon cost added on top of the gas-fuel cost in run_economics_gas_heater.
# Ommen et al. (2015) priced the gas burner on fuel only; we extend the
# comparison with a carbon charge.
# Source: Destatis, EU ETS / BEHG corridor 55-65 EUR/tCO2 valid from
# 1 January 2026 (start of BEHG market-based phase, transitioning out of
# the fixed-price regime). Midpoint adopted as the base case.
# Set to 0 to reproduce the Ommen baseline.
BASE_CO2_PRICE = 60.0            # EUR/tCO2

# ── Economic levelization parameters ─────────────────────────────────────────
# Effective discount rate and plant lifetime for the TRR / CELF computation.
# Held constant in this study; passed into exerpy.EconomicAnalysis and to the
# fuel-cost levelization helper _celf() in economics.py.
I_EFF       = 0.10              # 1/a  effective discount rate
N_YEARS     = 20                # years, plant lifetime

# ── Annual nominal cost escalation rates (TRR / CELF) ────────────────────────
# Applied via the constant-escalation levelization factor CELF(r_n).
# OMC: 2 %/a, general inflation.
# Electricity and gas: 2 %/a and 3 %/a — derived from the BDEW
# Strompreisanalyse 04/2026 for German medium-industry consumers
# (160k-20M kWh/a), pre-2022 trend (2015-2020). The 2022-2023 energy-crisis
# spike is treated as a one-off event and excluded from the long-term trend.
# CO2: 5 %/a — forward EU-ETS / BEHG trajectory, conservative midpoint
# between the soft (~3 %/a) and tight (~8 %/a) policy scenarios.
R_N_OM      = 0.02              # 1/a   O&M escalation
R_N_EL      = 0.02              # 1/a   electricity escalation
R_N_GAS     = 0.03              # 1/a   natural-gas escalation
R_N_CO2     = 0.05              # 1/a   CO2 emission-price escalation

# ── Sensitivity ranges ───────────────────────────────────────────────────────
E1_C_RANGE = np.arange(100, 201, 10.0)  # EUR/MWh — brackets BASE_E1_C = 159

# ── ±50 % price sensitivity (Ommen 2015, Fig. 3) ─────────────────────────────
# 2-D sweep of electricity and gas prices around their base values, in ±50 %
# steps of 10 %. Used by the price_sensitivity_2d analysis & plot.
PRICE_SENS_FRAC_RANGE = np.arange(-0.5, 0.51, 0.1)  # -50 % … +50 %

# ── Steam temperature (sink boundary) ───────────────────────────────────────
# T_STEAM_DEFAULT is the fallback used by simulate_hthp / simulate_gas_heater
# when no T_steam_override is passed. The pipeline always passes overrides,
# so this default is exercised only by ad-hoc programmatic use.
T_STEAM_DEFAULT = 100.0  # °C

# ── Case study: native scale ─────────────────────────────────────────────────
# All designs are simulated at the same native scale, set by ``M_STEAM`` on
# the sink side. With M_STEAM = 0.5 kg/s the case study delivers Q_H ≈
# 1.10–1.13 MWth depending on T_steam (just Δh_vap of water at the saturation
# pressure). All cost and efficiency metrics are reported at this native scale
# — no virtual rescaling. Three industrial steam temperatures are studied:
# 100 °C (≈ atmospheric), 110 °C (≈ 1.4 bar), 120 °C (≈ 2 bar).
M_STEAM = 0.5                  # kg/s  sink-side steam mass flow (native scale)

T_STEAM_CASE_DEFAULT = 110.0      # °C  default for the single-design fast path in
                              #     main.py and for stage main() helpers when
                              #     no T_steam argument is passed

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
    """Return the per-T_steam case-study output directory.

    All six case-study stages share this naming scheme so that
    `results/case_steam_100/`, `results/case_steam_110/` and
    `results/case_steam_120/` sit side-by-side without collision.
    """
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
#    Fraction of the total temperature lift (T_steam - T_air_in) handled by
#    the lower cycle.  T34 = T_air_in + lift_share * (T_steam - T_air_in).
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
                                              # ≈ mean ṁ_src observed under the previous
                                              # ΔT=10 K mode, so designs span similar
                                              # operating points but with T_src,out free)


# ── Helpers ──────────────────────────────────────────────────────────────────

def lift_share_to_T34(lift_share, T_source_in, T_steam=None):
    """
    Compute the cycle-2 evaporation temperature T34 from lift share.

    Parameters
    ----------
    lift_share : float
        Fraction of total lift handled by the lower cycle (0–1).
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
