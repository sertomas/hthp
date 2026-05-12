"""
Economic and exergoeconomic analysis for the cascaded HTHP.

Provides:

* **PEC cost correlations** (``pec_compressor``, ``pec_motor``,
  ``pec_plate_hx``) — power-law scaling with CI adjustment.
* ``run_economics`` — full exergoeconomic analysis for an HTHP simulation.
* ``run_economics_gas_heater`` — fuel-only cost balance for the gas heater
  reference (existing-burner retrofit assumption per Ommen et al. 2015).
"""

import math

from exerpy import EconomicAnalysis, ExergoeconomicAnalysis

# Cost index ratios (reference year → 2024)
_CI_RATIO = {
    2013: 1.4102,   # 2013 → 2024
    2020: 1.3418,   # 2020 → 2024
}

# Cost correlations reference year
COST_REF_YEAR = 2013

# Installation cost factor: TCI = F_INSTALL * PEC
# Accounts for installation, piping, instrumentation, engineering, contingencies.
# Value 4.16 originates from Bejan, Tsatsaronis & Moran (1995),
# "Thermal Design and Optimization" (sum of direct + indirect installation
# factors for industrial chemical plants); subsequently adopted by Ommen et
# al. (2015), Table 1, for industrial heat-pump / refrigeration costing.
F_INSTALL = 4.16

# Gas heater reference case parameters
# Natural gas burner thermal efficiency: Ommen et al. (2015),
# "Technical and economic working domains of industrial heat pumps: Part 1",
# Int. J. Refrigeration 55, 168-182, Table 1.
# Note: this constant is currently informational; the actual eta_gas applied
# in simulate_gas_heater() is set via its eta_gas argument (default 0.90).
GAS_HEATER_EFFICIENCY = 0.90  # thermal efficiency

# PEC Cost Correlations (power-law scaling)
# PEC_y = PEC_W * (X_y / X_W) ^ alpha
# All pec_*() functions return costs in the reference year;
# CI adjustment is applied in run_economics().

# Refrigerant → cost-type mapping (Ommen 2015, Tables 3-4).
# R1270 (propylene) and R600 (n-butane) are not in Ommen's data but use the
# same Type-2 reciprocating piston compressor as R290 / R600a, so they're
# proxied onto the R290_R600a cost row.
# R717 has two cost rows: LP (28 bar machine) and HP (50 bar machine). Which
# one applies is decided at runtime by _r717_class_for() based on the
# simulated discharge pressure.
_FLUID_COST_TYPE = {
    "R290":  "R290_R600a",
    "R600a": "R290_R600a",
    "R1270": "R290_R600a",
    "R600":  "R290_R600a",
    "R717":  "R717_LP",   # default; overridden to R717_HP if p_disch > 28 bar
    "R744":  "R744",
}

# Switch threshold between R717-LP and R717-HP cost rows [bar]. Equal to
# the LP compressor pressure limit in Ommen Table 3.
_R717_LP_HP_THRESHOLD_BAR = 28.0

# Engineering multiplier for R717-HP cost rows where Ommen Table 4 lists
# NDA. Public industry data on HP ammonia screw / piston compressors
# typically shows ~30-60 % higher cost than LP machines of equivalent
# swept volume; we adopt the midpoint of that range as a placeholder.
# This is an estimate, not a measured correlation.
_R717_HP_OVER_LP_FACTOR = 1.5


def _r717_class_for(p_high_bar):
    """Return the Ommen cost class for an R717 stage given its discharge
    pressure: 'R717_LP' (≤ 28 bar) or 'R717_HP' (28 < p ≤ 50 bar)."""
    if p_high_bar <= _R717_LP_HP_THRESHOLD_BAR:
        return "R717_LP"
    return "R717_HP"


# --- Compressor: sized by suction volumetric flow rate [m³/h] ---
# R717_HP coefficients are NDA in Ommen Table 4; we keep the LP shape and
# scale the intercept by _R717_HP_OVER_LP_FACTOR (engineering estimate).
_COMP_COST = {
    #                PEC_W [€]                                X_W [m³/h]  alpha
    "R290_R600a":   (19_850,                                  279.8,      0.73),
    "R717_LP":      (11_914,                                  178.4,      0.66),
    "R717_HP":      (11_914 * _R717_HP_OVER_LP_FACTOR,        178.4,      0.66),
    "R744":         (12_109,                                  25.6,       0.42),
}

# --- Electrical motor: sized by shaft power [kW] ---
# For R290 / R600a / R1270 / R600 / R744 motor cost = 0 (included in
# compressor cost per Ommen Table 4). Both R717-LP and R717-HP price the
# motor separately on the same row.
_MOTOR_COST = {
    #                PEC_W [€]   X_W [kW]    alpha
    "R290_R600a":   (0,          0,          0),
    "R717_LP":      (10_710,     250,        0.65),
    "R717_HP":      (10_710,     250,        0.65),
    "R744":         (0,          0,          0),
}

_USD_TO_EUR = 0.85  # approximate 2020 average exchange rate

# --- Centrifugal pump: sized by shaft power [kW] ---
# Source: Shamoushaki, Niknam, Talluri, Manfrida & Fiaschi (2021),
#   "Development of Cost Correlations for the Economic Assessment of Power
#    Plant Equipment", Energies 14(9), 2665. doi:10.3390/en14092665
#   Eq. (5) and Table 4 (centrifugal pump, carbon steel):
#       C = log(W_P) + a·W_P² + b·W_P + c            (cost in USD, Q1-2020)
#       a = -0.03195,  b = 467.2,  c = 2.048e4,  R² = 0.97
#   Calibration range: 20-3500 kW (140 data points from QUE$TOR Q1-2020 db).
# CAVEAT: pumps in this study operate at W_P < 1 kW, well below the 20 kW
# lower bound of the calibration range. The correlation is therefore
# extrapolated; in this regime the linear+quadratic terms become negligible
# and the cost is essentially the constant floor (≈ 17.4 kEUR per pump after
# USD→EUR), making the pump contribution effectively a small fixed cost
# rather than a duty-driven one.
_PUMP_COEFF = {"a": -0.03195, "b": 467.2, "c": 2.048e4}
PUMP_REF_YEAR = 2020

# --- Plate heat exchanger: sized by area [m²] ---
# R717-HP and R744 plate-HX coefficients are NDA in Ommen Table 4. Plate-HX
# cost is dominated by area and material (stainless steel either way) and
# only weakly by working fluid, so we apply the same shape as the HC group
# for both NDA rows.
_PHX_COST = {
    #                PEC_W [€]   X_W [m²]    alpha
    "R290_R600a":   (15_526,     42,         0.8),
    "R717_LP":      (15_526,     42,         0.8),
    "R717_HP":      (15_526,     42,         0.8),
    "R744":         (15_526,     42,         0.8),
}


def _resolve_cost_type(fluid, p_high_bar=None):
    """Look up the Ommen cost class for a fluid, with R717 routed to LP or
    HP based on ``p_high_bar`` (None ⇒ assume LP)."""
    cost_type = _FLUID_COST_TYPE[fluid]
    if fluid == "R717" and p_high_bar is not None:
        cost_type = _r717_class_for(p_high_bar)
    return cost_type


def pec_compressor(V_dot_m3h, fluid, p_high_bar=None, eta_vol=1.0):
    """
    Purchased-equipment cost of a compressor (reference-year EUR).

    Parameters
    ----------
    V_dot_m3h : float
        Suction volumetric flow rate [m^3/h].
    fluid : str
        Refrigerant name (mapped to a cost-type internally).
    p_high_bar : float, optional
        Cycle high-side (discharge) pressure [bar]. Used to route R717 to
        the LP or HP cost row. Ignored for non-R717 fluids.
    eta_vol : float, optional
        Compressor volumetric efficiency [-]. The correlation is keyed on
        the *displaced* (swept) volume rather than the actual suction
        flow, so the sizing parameter passed into the power law is
        ``V_dot / eta_vol``. Default 1.0 reproduces the pre-Dincer
        behaviour (treat suction flow as displaced volume).

    Returns
    -------
    float
        PEC in reference-year EUR (before CI adjustment).
    """
    cost_type = _resolve_cost_type(fluid, p_high_bar)
    PEC_W, X_W, alpha = _COMP_COST[cost_type]
    return PEC_W * (V_dot_m3h / (eta_vol * X_W)) ** alpha


def pec_motor(W_kW, fluid, p_high_bar=None):
    """
    Purchased-equipment cost of an electrical motor (reference-year EUR).

    Returns 0 for fluid types whose motor cost is already included in the
    compressor correlation (Ommen Table 4: R290 / R600a / R1270 / R600 /
    R744). For R717 the motor is priced separately for both LP and HP
    classes; ``p_high_bar`` selects the row.

    Parameters
    ----------
    W_kW : float
        Shaft power [kW].
    fluid : str
        Refrigerant name.
    p_high_bar : float, optional
        Cycle high-side (discharge) pressure [bar]. R717-only.

    Returns
    -------
    float
        PEC in reference-year EUR (before CI adjustment).
    """
    cost_type = _resolve_cost_type(fluid, p_high_bar)
    PEC_W, X_W, alpha = _MOTOR_COST[cost_type]
    if PEC_W == 0:
        return 0.0
    return PEC_W * (W_kW / X_W) ** alpha


def pec_plate_hx(A_m2, fluid, p_high_bar=None):
    """
    Purchased-equipment cost of a plate heat exchanger (reference-year EUR).

    Parameters
    ----------
    A_m2 : float
        Heat-transfer area [m^2].
    fluid : str
        Refrigerant name.
    p_high_bar : float, optional
        Cycle high-side pressure [bar]. R717-only (routes between LP and HP
        cost rows; both rows are currently identical, so this has no effect
        on the result, but the argument is accepted for API symmetry).

    Returns
    -------
    float
        PEC in reference-year EUR (before CI adjustment).
    """
    cost_type = _resolve_cost_type(fluid, p_high_bar)
    PEC_W, X_W, alpha = _PHX_COST[cost_type]
    return PEC_W * (A_m2 / X_W) ** alpha


def pec_pump(W_kW):
    """
    Purchased-equipment cost of a centrifugal pump (2020 EUR).

    C = log10(W_P) + a*W_P^2 + b*W_P + c  (USD, converted to EUR via _USD_TO_EUR)

    Parameters
    ----------
    W_kW : float
        Shaft power [kW].

    Returns
    -------
    float
        PEC in 2020 EUR (before CI adjustment).
    """
    a = _PUMP_COEFF["a"]
    b = _PUMP_COEFF["b"]
    c = _PUMP_COEFF["c"]
    cost_usd = math.log10(W_kW) + a * W_kW ** 2 + b * W_kW + c
    return cost_usd * _USD_TO_EUR


def run_economics(sim, full_load_hours, e1_c_ct_kwh):
    """
    Run a full exergoeconomic analysis on a pre-computed HTHP simulation.

    Computes PEC for every component (with CI adjustment), converts to
    hourly cost rates via ``EconomicAnalysis``, and solves the
    ``ExergoeconomicAnalysis`` cost balance.

    Parameters
    ----------
    sim : dict
        Output of ``simulate_hthp`` (must not be ``None``).
    full_load_hours : float
        Annual full-load operating hours [h/a].
    e1_c_ct_kwh : float
        Electricity price [ct/kWh].

    Returns
    -------
    dict or None
        ``None`` on failure.  Otherwise a dict with keys:

        - **c_P** — specific product cost [EUR/GJ].
        - **Z_sum** — total hourly cost rate [EUR/h].
        - **exergoeco** — ``ExergoeconomicAnalysis`` object.
    """
    try:
        e1_c = e1_c_ct_kwh / 0.36  # convert ct/kWh → EUR/GJ

        ean = sim["ean"]
        sz = sim["sizing"]
        f1 = sim["fluid_cycle1"]
        f2 = sim["fluid_cycle2"]

        # Cycle high-side pressures (used to route R717 between LP and HP
        # cost classes). Pulled from the saved cycle_states points.
        c1_pts = sim["cycle_states"]["cycle1"]["points"]
        c2_pts = sim["cycle_states"]["cycle2"]["points"]
        p_high_c1 = max(p["p"] for p in c1_pts)
        p_high_c2 = max(p["p"] for p in c2_pts)

        # --- Compute PEC for each component (ref-year cost, ref-year tag) ---
        # Each entry is (cost_in_ref_year, ref_year).
        # Pumps are NOT included: brownfield-retrofit assumption — both the
        # source-loop circulator and the sink-side feedwater pump are
        # pre-existing site infrastructure (the latter formerly served the
        # displaced gas-fired boiler) and are not new equipment in scope.
        PEC_ref = {
            "COMP1":     (pec_compressor(sz["V_dot_comp1"], f1, p_high_c1,
                                        eta_vol=sz.get("eta_vol_comp1", 1.0)), COST_REF_YEAR),
            "COMP2":     (pec_compressor(sz["V_dot_comp2"], f2, p_high_c2,
                                        eta_vol=sz.get("eta_vol_comp2", 1.0)), COST_REF_YEAR),
            "SRC_HX":    (pec_plate_hx(sz["A_src_hx"], f1, p_high_c1), COST_REF_YEAR),
            "IHX":       (pec_plate_hx(sz["A_ihx"], f1, p_high_c1), COST_REF_YEAR),
            "SNK_HX":    (pec_plate_hx(sz["A_snk_hx"], f2, p_high_c2), COST_REF_YEAR),
            "MOT1":      (pec_motor(sz["W_comp1"], f1, p_high_c1), COST_REF_YEAR),
            "MOT2":      (pec_motor(sz["W_comp2"], f2, p_high_c2), COST_REF_YEAR),
            "VAL1":      (0.0, COST_REF_YEAR),
            "VAL2":      (0.0, COST_REF_YEAR),
        }
        PEC = {k: F_INSTALL * cost * _CI_RATIO[ref_year]
               for k, (cost, ref_year) in PEC_ref.items()}

        # --- PEC → Z [EUR/h] via EconomicAnalysis ---
        # f_tci = 1.0 disables exerpy's internal 6.32× module factor: the
        # values we pass are already TCI (= F_INSTALL × bare_PEC × CI), so
        # the only PEC→TCI multiplier in the chain is Ommen's 4.16, applied
        # exactly once at PEC_ref → PEC time. Without this, exerpy would
        # multiply by another 6.32 → 26.3× double-count of installation.
        econ = EconomicAnalysis({
            "tau": full_load_hours,
            "i_eff": 0.10,
            "n": 20,
            "r_n": 0.02,
            "f_tci": 1.0,
        })
        comp_names = list(PEC.keys())
        PEC_list = list(PEC.values())
        OMC_relative = [0.03] * len(PEC_list)
        _, _, Z_total = econ.compute_component_costs(PEC_list, OMC_relative)

        # --- Build cost dict for exergoeconomic analysis ---
        cost_dict = {}
        for name, z in zip(comp_names, Z_total):
            cost_dict[f"{name}_Z"] = z
        cost_dict["e1_c"] = e1_c
        cost_dict["11_c"] = 0.0   # source water inlet — free reservoir
        cost_dict["13_c"] = 0.0   # source water outlet — free disposal (loss with c_L = 0)
        cost_dict["41_c"] = 0.0

        exergoeco = ExergoeconomicAnalysis(ean)
        exergoeco.run(cost_dict)

        E_P_kW = ean.E_P / 1000
        c_P = exergoeco.system_costs["C_P"] / E_P_kW * 1e6 / 3600
        Z_sum = exergoeco.system_costs["Z"]
        return {"c_P": c_P, "Z_sum": Z_sum, "exergoeco": exergoeco}
    except Exception as e:
        print(f"    Economics failed: {e}")
        return None
    

def run_economics_gas_heater(sim, full_load_hours, gas_c_ct_kwh,
                              co2_price_eur_per_t=0.0):
    """
    Compute the specific product cost for the gas heater reference.

    Follows the retrofit assumption of Ommen et al. (2015), Section 2.4:
    "The investment cost of already installed natural gas burners were
    neglected. This is the case if the heat pump replaces an existing
    installation." → TCI_gas = 0, OMC_gas = 0.

    Cost balance reduces to fuel cost plus optional CO2 emission cost:
    ``C_P = c_gas · Q_gas + c_CO2 · ṁ_CO2``. Ommen et al. omitted the CO2
    term entirely; this implementation adds it because EU ETS carbon
    pricing is now a non-trivial fraction of the total gas-fired heat
    cost. ``co2_price_eur_per_t = 0`` reproduces Ommen's original
    formulation exactly.

    Parameters
    ----------
    sim : dict
        Output of ``simulate_gas_heater``.
    full_load_hours : float
        Annual full-load operating hours [h/a]. Unused under the retrofit
        assumption (kept in signature for API consistency with the other
        ``run_economics_*`` functions).
    gas_c_ct_kwh : float
        Gas price [ct/kWh].
    co2_price_eur_per_t : float, optional
        CO2 emission price [EUR / tonne CO2]. Multiplied by the CO2 mass
        flow rate from ``sim["m_dot_CO2"]`` (computed from the TESPy
        combustion-chamber CH4 input via stoichiometry). The function-
        level default is ``0.0`` (Ommen 2015 baseline, fuel-only), but
        ``case_steam_economics.py`` and ``plot_case_steam_economics.py``
        always call this function with ``co2_price_eur_per_t = BASE_CO2_PRICE``
        (60 EUR/tCO2 by default), so the gas-reference c_P shown in the
        pipeline outputs includes the EU ETS / BEHG charge.

    Returns
    -------
    dict
        Keys: **c_P** [EUR/GJ], **Z_sum** [EUR/h] (always 0 under retrofit).
    """
    gas_c = gas_c_ct_kwh / 0.36  # convert ct/kWh → EUR/GJ_LHV
    del full_load_hours  # unused under retrofit assumption; kept for API parity

    # ── Hourly cost rates that the exergoeco balance needs ──────────────
    Q_gas_GJ_per_h = sim["Q_gas"] * 3600 / 1e9   # GJ_LHV/h
    E_F_GJ_per_h   = sim["E_F"]   * 3600 / 1e9   # GJ_chemical-exergy/h
    E_P_GJ_per_h   = sim["E_P"]   * 3600 / 1e9   # GJ_exergy/h

    # Fuel cost rate: gas at LHV price + CO2 charge.
    C_fuel_only = gas_c * Q_gas_GJ_per_h                # EUR/h
    m_dot_CO2 = sim.get("m_dot_CO2", 0.0)               # kg/s
    m_CO2_t_per_h = m_dot_CO2 * 3600.0 / 1000.0         # tCO2/h
    C_CO2_hourly = co2_price_eur_per_t * m_CO2_t_per_h  # EUR/h
    C_F_hourly = C_fuel_only + C_CO2_hourly             # EUR/h

    # Convert to a per-GJ-exergy price on stream g2 so the exergoeconomic
    # chain can propagate it. Polluter-pays-via-the-fuel formulation: the
    # CO2 surcharge is folded into c_g2 rather than carried as a Z on the
    # combustion chamber, which keeps Z = 0 under the retrofit assumption
    # while the CO2 cost still flows through C_F → C_P. The ratio
    # Q_gas / E_F ≈ 0.97 corrects between LHV input and chemical exergy.
    c_g2_GJ_ex = C_F_hourly / E_F_GJ_per_h              # EUR/GJ_ex

    # ── Backwards-compat fallback ───────────────────────────────────────
    # Older cached sim dicts (pre-2026 update) carried no `ean` object;
    # in that case we revert to the manual scalar balance and emit no
    # exergoeconomic decomposition.
    ean = sim.get("ean")
    if ean is None:
        c_P = C_F_hourly / E_P_GJ_per_h
        return {"c_P": c_P, "Z_sum": 0.0}

    # ── Real exergoeconomic analysis ────────────────────────────────────
    # Components: CC (combustion chamber) + GAS_HX (heat exchanger).
    # Material inputs that need a price set: g1 (air, ambient → 0),
    # g2 (CH4 fuel → c_g2_GJ_ex), w1 (water, ambient → 0).
    cost_dict = {
        "CC_Z":     0.0,        # retrofit ⇒ no CAPEX
        "GAS_HX_Z": 0.0,        # retrofit ⇒ no CAPEX
        "g1_c":     0.0,        # ambient air, zero exergy cost
        "g2_c":     c_g2_GJ_ex, # CH4 fuel (incl. CO2 surcharge)
        "w1_c":     0.0,        # ambient water, zero exergy cost
    }

    exergoeco = ExergoeconomicAnalysis(ean)
    exergoeco.run(cost_dict)

    E_P_kW = ean.E_P / 1000
    c_P = exergoeco.system_costs["C_P"] / E_P_kW * 1e6 / 3600
    Z_sum = exergoeco.system_costs["Z"]
    return {"c_P": c_P, "Z_sum": Z_sum, "exergoeco": exergoeco}