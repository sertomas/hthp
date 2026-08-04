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

from config import I_EFF, N_YEARS, R_N_OM, R_N_EL, R_N_GAS, R_N_CO2


def _celf(r_n: float, i_eff: float = I_EFF, n: int = N_YEARS) -> float:
    """End-of-year constant-escalation levelization factor (CELF(0) = 1).

    Multiplying a first-year cost rate by CELF gives the constant-equivalent
    annual rate over ``n`` years at discount rate ``i_eff``. exerpy uses the
    begin-of-year convention, so OMC is rescaled in ``run_economics`` to match.
    """
    k = (1.0 + r_n) / (1.0 + i_eff)
    crf = i_eff * (1.0 + i_eff) ** n / ((1.0 + i_eff) ** n - 1.0)
    return (1.0 - k ** n) / ((1.0 + i_eff) * (1.0 - k)) * crf

# Cost index ratios (reference year → 2024) from CEPCI data.
_CI_RATIO = {
    2013: 1.4102,   # 2013 → 2024
    2020: 1.3418,   # 2020 → 2024
}

# Cost correlations reference year
COST_REF_YEAR = 2013

# Analysis year. CEPCI scales PEC to 2024; the 2024 -> 2026 step uses the
# general inflation rate R_N_OM, matching the year-1 fuel prices in config.py.
ANALYSIS_REF_YEAR = 2026
_CEPCI_TO_REF_YEAR = (1.0 + R_N_OM) ** (ANALYSIS_REF_YEAR - 2024)   # = 1.0404

# Installation factor TCI = F_INSTALL * PEC (Bejan, Tsatsaronis & Moran 1995;
# adopted by Ommen 2015, Table 1).
F_INSTALL = 4.16

# Gas burner thermal efficiency (Ommen 2015); the applied value is the
# eta_gas argument of simulate_gas_heater (default 0.90).
GAS_HEATER_EFFICIENCY = 0.90  # thermal efficiency

# PEC Cost Correlations (power-law scaling)
# PEC_y = PEC_W * (X_y / X_W) ^ alpha
# All pec_*() functions return costs in the reference year;
# CI adjustment is applied in run_economics().

# Refrigerant -> cost-type mapping (Ommen 2015, Tables 3-4). R1270 and R600
# are proxied onto the R290/R600a Type-2 piston row. R717 has LP and HP rows
# selected at runtime by _r717_class_for() from the discharge pressure.
_FLUID_COST_TYPE = {
    "R290":  "R290_R600a",
    "R600a": "R290_R600a",
    "R1270": "R290_R600a",
    "R600":  "R290_R600a",
    "R717":  "R717_LP",   # default; overridden to R717_HP if p_disch > 28 bar
}

# Switch threshold between R717-LP and R717-HP cost rows [bar]. Equal to
# the LP compressor pressure limit in Ommen Table 3.
_R717_LP_HP_THRESHOLD_BAR = 28.0

# Engineering multiplier for R717-HP rows (NDA in Ommen Table 4): LP shape
# scaled by 1.5, midpoint of the ~30-60 % HP-over-LP industry range.
_R717_HP_OVER_LP_FACTOR = 1.5


def _r717_class_for(p_high_bar):
    """Return the Ommen cost class for an R717 stage given its discharge
    pressure: 'R717_LP' (≤ 28 bar) or 'R717_HP' (28 < p ≤ 50 bar)."""
    if p_high_bar <= _R717_LP_HP_THRESHOLD_BAR:
        return "R717_LP"
    return "R717_HP"


# --- Compressor: sized by suction volumetric flow rate [m^3/h] ---
_COMP_COST = {
    #                PEC_W [€]                                X_W [m³/h]  alpha
    "R290_R600a":   (19_850,                                  279.8,      0.73),
    "R717_LP":      (11_914,                                  178.4,      0.66),
    "R717_HP":      (11_914 * _R717_HP_OVER_LP_FACTOR,        178.4,      0.66),
}

# --- Electrical motor: sized by shaft power [kW] ---
# Zero for the HC group (rolled into the compressor PEC); priced separately
# for R717.
_MOTOR_COST = {
    #                PEC_W [€]   X_W [kW]    alpha
    "R290_R600a":   (0,          0,          0),
    "R717_LP":      (10_710,     250,        0.65),
    "R717_HP":      (10_710,     250,        0.65),
}

_USD_TO_EUR = 0.85  # approximate 2020 average exchange rate

# --- Centrifugal pump: sized by shaft power [kW] ---
# Shamoushaki et al. (2021), Energies 14(9), 2665, Eq. (5), Table 4 (USD, 2020).
_PUMP_COEFF = {"a": -0.03195, "b": 467.2, "c": 2.048e4}
PUMP_REF_YEAR = 2020

# --- Plate heat exchanger: sized by area [m^2] ---
# Area-dominated and only weakly fluid-dependent, so all rows share one shape.
_PHX_COST = {
    #                PEC_W [€]   X_W [m²]    alpha
    "R290_R600a":   (15_526,     42,         0.8),
    "R717_LP":      (15_526,     42,         0.8),
    "R717_HP":      (15_526,     42,         0.8),
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
        ``V_dot / eta_vol``. Default 1.0 treats the suction flow as the
        displaced volume.

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
    compressor correlation (Ommen Table 4: R290 / R600a / R1270 / R600).
    For R717 the motor is priced separately for both LP and HP classes;
    ``p_high_bar`` selects the row.

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
        e1_c = e1_c_ct_kwh / 0.36  # convert ct/kWh → EUR/GJ_ex (electricity = pure exergy, first-year)
        # Levelize electricity to a constant-equivalent annual cost (end-of-year).
        e1_c_lev = e1_c * _celf(R_N_EL)

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

        # --- Compute PEC for each component; entries are (cost, ref_year) ---
        # Pumps excluded (brownfield retrofit: existing site infrastructure).
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
        PEC = {k: F_INSTALL * cost * _CI_RATIO[ref_year] * _CEPCI_TO_REF_YEAR
               for k, (cost, ref_year) in PEC_ref.items()}

        # --- PEC → Z [EUR/h] via EconomicAnalysis ---
        # f_tci = 1.0 disables exerpy's internal module factor; the PEC values
        # passed are already TCI (F_INSTALL applied once above).
        econ = EconomicAnalysis({
            "tau": full_load_hours,
            "i_eff": I_EFF,
            "n": N_YEARS,
            "r_n": R_N_OM,
            "f_tci": 1.0,
        })
        comp_names = list(PEC.keys())
        PEC_list = list(PEC.values())
        OMC_relative = [0.03] * len(PEC_list)
        Z_CC_list, Z_OM_begin, _ = econ.compute_component_costs(PEC_list, OMC_relative)
        # Rescale exerpy's begin-of-year OMC to end-of-year so O&M and fuel
        # share the _celf convention; the capital part Z_CC is unaffected.
        Z_OM_list = [z / (1.0 + I_EFF) for z in Z_OM_begin]
        Z_total = [zcc + zom for zcc, zom in zip(Z_CC_list, Z_OM_list)]

        # --- Build cost dict for exergoeconomic analysis ---
        cost_dict = {}
        for name, z in zip(comp_names, Z_total):
            cost_dict[f"{name}_Z"] = z
        cost_dict["e1_c"] = e1_c_lev
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

    Retrofit assumption (Ommen et al. 2015): the existing burner has zero
    CAPEX and O&M, so the balance reduces to fuel plus an optional CO2 charge,
    ``C_P = c_gas * Q_gas + c_CO2 * m_CO2``. ``co2_price_eur_per_t = 0`` gives
    the fuel-only baseline.

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
        CO2 price [EUR/tCO2], applied to the CO2 mass flow from
        ``sim["m_dot_CO2"]``. Default 0.0 (fuel-only baseline).

    Returns
    -------
    dict
        Keys: **c_P** [EUR/GJ], **Z_sum** [EUR/h] (always 0 under retrofit).
    """
    gas_c = gas_c_ct_kwh / 0.36  # convert ct/kWh → EUR/GJ_LHV (first-year)
    del full_load_hours  # unused under retrofit assumption; kept for API parity

    # ── Hourly cost rates that the exergoeco balance needs ──────────────
    Q_gas_GJ_per_h = sim["Q_gas"] * 3600 / 1e9   # GJ_LHV/h
    E_F_GJ_per_h   = sim["E_F"]   * 3600 / 1e9   # GJ_chemical-exergy/h
    E_P_GJ_per_h   = sim["E_P"]   * 3600 / 1e9   # GJ_exergy/h

    # Levelize gas and CO2 prices independently — the two streams escalate at
    # different nominal rates over the plant lifetime (see config.R_N_GAS,
    # config.R_N_CO2). End-of-year (BTM) convention, matching `_celf`.
    celf_gas = _celf(R_N_GAS)
    celf_co2 = _celf(R_N_CO2)

    # Fuel cost rate: gas at LHV price + CO2 charge, both levelized.
    C_fuel_only = gas_c * Q_gas_GJ_per_h * celf_gas               # EUR/h
    m_dot_CO2 = sim.get("m_dot_CO2", 0.0)                         # kg/s
    m_CO2_t_per_h = m_dot_CO2 * 3600.0 / 1000.0                   # tCO2/h
    C_CO2_hourly = co2_price_eur_per_t * m_CO2_t_per_h * celf_co2 # EUR/h
    C_F_hourly = C_fuel_only + C_CO2_hourly                       # EUR/h

    # Per-GJ-exergy price on stream g2. The CO2 surcharge is folded into the
    # fuel price (not a Z on the combustion chamber), keeping Z = 0 while the
    # CO2 cost still propagates through C_F -> C_P.
    c_g2_GJ_ex = C_F_hourly / E_F_GJ_per_h              # EUR/GJ_ex

    # Fallback when sim carries no `ean`: manual scalar balance, no decomposition.
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