"""
Economic and exergoeconomic analysis for the cascaded HTHP.

Provides:

* **PEC cost correlations** (``pec_compressor``, ``pec_motor``,
  ``pec_plate_hx``) — power-law scaling with CI adjustment.
* ``run_economics`` — full exergoeconomic analysis for an HTHP simulation.
* ``run_economics_heater`` — simplified cost balance for the heater reference.
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
F_INSTALL = 6.32

# Heater reference case parameters
HEATER_COST_PER_KW = 150  # EUR/kW installed (industrial resistance heater)
GAS_HEATER_COST_PER_KW = 110  # EUR/kW installed (industrial gas heater)
GAS_HEATER_EFFICIENCY = 0.95  # thermal efficiency

# PEC Cost Correlations (power-law scaling)
# PEC_y = PEC_W * (X_y / X_W) ^ alpha
# All pec_*() functions return costs in the reference year;
# CI adjustment is applied in run_economics().

# Refrigerant → cost-type mapping
# (R1270 treated as R290; R600 treated as R600a; R717 treated as R717-LP)
_FLUID_COST_TYPE = {
    "R290": "R290_R600a",
    "R600a": "R290_R600a",
    "R1270": "R290_R600a",
    "R600": "R290_R600a",
    "R717": "R717_LP",
}

# --- Compressor: sized by suction volumetric flow rate [m³/h] ---
_COMP_COST = {
    #                PEC_W [€]   X_W [m³/h]  alpha
    "R290_R600a":   (19_850,     279.8,      0.73),
    "R717_LP":      (11_914,     178.4,      0.66),
}

# --- Electrical motor: sized by shaft power [kW] ---
# For R290/R600a/R1270/R600 motor cost = 0 (included in compressor)
_MOTOR_COST = {
    #                PEC_W [€]   X_W [kW]    alpha
    "R290_R600a":   (0,          0,          0),
    "R717_LP":      (10_710,     250,        0.65),
}

# --- Air cooler: sized by area [m²] ---
# C = log10(A) + a·A² + b·A + c  (cost in USD, ref year 2020)
_AIR_COOLER_COEFF = {"a": 0.01764, "b": 617.4, "c": 3.31e4}
AIR_COOLER_REF_YEAR = 2020
_USD_TO_EUR = 0.85  # approximate 2020 average exchange rate

# --- Centrifugal pump: sized by shaft power [kW] ---
# C = log10(W_P) + a·W_P² + b·W_P + c  (cost in USD, ref year 2020)
_PUMP_COEFF = {"a": -0.03195, "b": 467.2, "c": 2.048e4}
PUMP_REF_YEAR = 2020

# --- Plate heat exchanger: sized by area [m²] ---
_PHX_COST = {
    #                PEC_W [€]   X_W [m²]    alpha
    "R290_R600a":   (15_526,     42,         0.8),
    "R717_LP":      (15_526,     42,         0.8),  # same as R290/R600a row
}


def pec_compressor(V_dot_m3h, fluid):
    """
    Purchased-equipment cost of a compressor (reference-year EUR).

    Parameters
    ----------
    V_dot_m3h : float
        Suction volumetric flow rate [m^3/h].
    fluid : str
        Refrigerant name (mapped to a cost-type internally).

    Returns
    -------
    float
        PEC in reference-year EUR (before CI adjustment).
    """
    cost_type = _FLUID_COST_TYPE[fluid]
    PEC_W, X_W, alpha = _COMP_COST[cost_type]
    return PEC_W * (V_dot_m3h / X_W) ** alpha


def pec_motor(W_kW, fluid):
    """
    Purchased-equipment cost of an electrical motor (reference-year EUR).

    Returns 0 for fluid types whose motor cost is already included in the
    compressor correlation.

    Parameters
    ----------
    W_kW : float
        Shaft power [kW].
    fluid : str
        Refrigerant name.

    Returns
    -------
    float
        PEC in reference-year EUR (before CI adjustment).
    """
    cost_type = _FLUID_COST_TYPE[fluid]
    PEC_W, X_W, alpha = _MOTOR_COST[cost_type]
    if PEC_W == 0:
        return 0.0
    return PEC_W * (W_kW / X_W) ** alpha


def pec_plate_hx(A_m2, fluid):
    """
    Purchased-equipment cost of a plate heat exchanger (reference-year EUR).

    Parameters
    ----------
    A_m2 : float
        Heat-transfer area [m^2].
    fluid : str
        Refrigerant name.

    Returns
    -------
    float
        PEC in reference-year EUR (before CI adjustment).
    """
    cost_type = _FLUID_COST_TYPE[fluid]
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


def pec_air_cooler(A_m2):
    """
    Purchased-equipment cost of an air cooler (2020 EUR).

    C = log10(A) + a*A^2 + b*A + c  (USD, converted to EUR via _USD_TO_EUR)

    The correlation is valid up to 1000 m². For larger areas, the cost at
    1000 m² is extrapolated using the six-tenths rule: C(A) = C(1000) * (A/1000)^0.6.

    Parameters
    ----------
    A_m2 : float
        Heat-transfer area [m^2].

    Returns
    -------
    float
        PEC in 2020 EUR (before CI adjustment).
    """
    a = _AIR_COOLER_COEFF["a"]
    b = _AIR_COOLER_COEFF["b"]
    c = _AIR_COOLER_COEFF["c"]
    A_max = 1000.0
    if A_m2 <= A_max:
        cost_usd = math.log10(A_m2) + a * A_m2 ** 2 + b * A_m2 + c
    else:
        cost_at_max = math.log10(A_max) + a * A_max ** 2 + b * A_max + c
        cost_usd = cost_at_max * (A_m2 / A_max) ** 0.6
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

        # --- Compute PEC for each component (ref-year cost, ref-year tag) ---
        # Each entry is (cost_in_ref_year, ref_year).
        # Existing correlations → 2013; air cooler → 2020.
        PEC_ref = {
            "COMP1":     (pec_compressor(sz["V_dot_comp1"], f1), COST_REF_YEAR),
            "COMP2":     (pec_compressor(sz["V_dot_comp2"], f2), COST_REF_YEAR),
            "SRC_PUMP":  (pec_pump(sz["W_src_pump"]), PUMP_REF_YEAR),
            "SNK_PUMP":  (pec_pump(sz["W_snk_pump"]), PUMP_REF_YEAR),
            "SRC_HX":    (pec_plate_hx(sz["A_src_hx"], f1), COST_REF_YEAR),
            "IHX":       (pec_plate_hx(sz["A_ihx"], f1), COST_REF_YEAR),
            "SNK_HX":    (pec_plate_hx(sz["A_snk_hx"], f2), COST_REF_YEAR),
            "MOT1":      (pec_motor(sz["W_comp1"], f1), COST_REF_YEAR),
            "MOT2":      (pec_motor(sz["W_comp2"], f2), COST_REF_YEAR),
            "MOT3":      (0.0, COST_REF_YEAR),  # SRC_PUMP motor (included in pump)
            "MOT4":      (0.0, COST_REF_YEAR),  # SNK_PUMP motor (included in pump)
            "VAL1":      (0.0, COST_REF_YEAR),
            "VAL2":      (0.0, COST_REF_YEAR),
        }
        PEC = {k: F_INSTALL * cost * _CI_RATIO[ref_year]
               for k, (cost, ref_year) in PEC_ref.items()}

        # --- PEC → Z [EUR/h] via EconomicAnalysis ---
        econ = EconomicAnalysis({
            "tau": full_load_hours,
            "i_eff": 0.10,
            "n": 20,
            "r_n": 0.02,
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
        cost_dict["11_c"] = 0.0
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
    

def run_economics_heater(sim, full_load_hours, e1_c_ct_kwh):
    """
    Compute the specific product cost for the electrical heater reference.

    Uses a simple cost balance: ``c_P = (c_el * E_F + Z) / E_P``, with the
    same annualisation parameters as the HTHP.

    Parameters
    ----------
    sim : dict
        Output of ``simulate_heater``.
    full_load_hours : float
        Annual full-load operating hours [h/a].
    e1_c_ct_kwh : float
        Electricity price [ct/kWh].

    Returns
    -------
    dict
        Keys: **c_P** [EUR/GJ], **Z_sum** [EUR/h].
    """
    e1_c = e1_c_ct_kwh / 0.36  # convert ct/kWh → EUR/GJ for cost balance

    Q_H_kW = sim["Q_H"] / 1000
    equipment_cost = F_INSTALL * HEATER_COST_PER_KW * Q_H_kW  # EUR

    # Annualize (same parameters as HTHP)
    i = 0.10
    n = 20
    CRF = i * (1 + i) ** n / ((1 + i) ** n - 1)
    annual_cost = equipment_cost * (CRF + 0.03)  # capital + maintenance
    Z_hourly = annual_cost / full_load_hours  # EUR/h

    # Cost balance: C_P = C_F + Z
    E_F_GJ_per_h = sim["E_F"] * 3600 / 1e9  # W → GJ/h
    E_P_GJ_per_h = sim["E_P"] * 3600 / 1e9  # W → GJ/h

    C_F_hourly = e1_c * E_F_GJ_per_h  # EUR/h
    C_P_hourly = C_F_hourly + Z_hourly  # EUR/h
    c_P = C_P_hourly / E_P_GJ_per_h  # EUR/GJ

    return {"c_P": c_P, "Z_sum": Z_hourly}


def run_economics_gas_heater(sim, full_load_hours, gas_c_ct_kwh):
    """
    Compute the specific product cost for the gas heater reference.

    Uses a simple cost balance: ``c_P = (c_gas * Q_gas + Z) / E_P``.

    Parameters
    ----------
    sim : dict
        Output of ``simulate_gas_heater``.
    full_load_hours : float
        Annual full-load operating hours [h/a].
    gas_c_ct_kwh : float
        Gas price [ct/kWh].

    Returns
    -------
    dict
        Keys: **c_P** [EUR/GJ], **Z_sum** [EUR/h].
    """
    gas_c = gas_c_ct_kwh / 0.36  # convert ct/kWh → EUR/GJ

    Q_H_kW = sim["Q_H"] / 1000
    equipment_cost = F_INSTALL * GAS_HEATER_COST_PER_KW * Q_H_kW  # EUR

    # Annualize (same parameters as HTHP)
    i = 0.10
    n = 20
    CRF = i * (1 + i) ** n / ((1 + i) ** n - 1)
    annual_cost = equipment_cost * (CRF + 0.03)  # capital + maintenance
    Z_hourly = annual_cost / full_load_hours  # EUR/h

    # Cost balance: C_P = C_F + Z
    # Fuel cost based on LHV energy input (gas priced per energy, not exergy)
    Q_gas_GJ_per_h = sim["Q_gas"] * 3600 / 1e9  # W → GJ/h
    E_P_GJ_per_h = sim["E_P"] * 3600 / 1e9      # W → GJ/h

    C_F_hourly = gas_c * Q_gas_GJ_per_h  # EUR/h
    C_P_hourly = C_F_hourly + Z_hourly  # EUR/h
    c_P = C_P_hourly / E_P_GJ_per_h  # EUR/GJ

    return {"c_P": c_P, "Z_sum": Z_hourly}