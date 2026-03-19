"""
Thermodynamic simulation models for the cascaded HTHP and the heater reference.

Provides two public functions:

* ``simulate_hthp`` — builds and solves a two-stage heat-pump TESPy network,
  runs an exergy analysis, and returns all data required by the economics
  and plotting stages.
* ``simulate_heater`` — analytically computes the same metrics for an
  electrical resistance heater used as a reference case.
"""

import json
import logging

import numpy as np

from CoolProp.CoolProp import PropsSI
from tespy.components import Compressor, CycleCloser, Motor, PowerBus, PowerSource, Pump, Sink, Source, Valve
from tespy.components import MovingBoundaryHeatExchanger as HeatExchanger
from tespy.connections import Connection, PowerConnection, Ref
from tespy.networks import Network

from exerpy import ExergyAnalysis
from exerpy.parser.from_tespy.tespy_parser import to_exerpy

from config import NumpyEncoder, SOURCE_DELTA_T, SOURCE_MASS_FLOW

# Shared parameters
Tamb = 293.15  # K
pamb = 101325  # Pa
T_source_in = Tamb - 273.15  # °C (default: 20 °C)
p_source = 3  # bar (source water circulation pressure)
p_water = 2  # bar (sink water / steam pressure)
T_evap_c2 = 60  # °C
pinch = 5  # K
T_water_sat = PropsSI("T", "P", p_water * 1e5, "Q", 0, "water") - 273.15

# Absolute pressure drops [bar] (average from literature ranges)
_DP = {
    "SRC_HX": (0.15,  0.175),   # (source water side, refrigerant side)
    "IHX":    (0.300, 0.150),    # (C1 condensing, C2 evaporating)
    "SNK_HX": (0.250, 0.175),   # (C2 condensing, sink water side)
}

# U-Value Lookup Table [W/(m2K)]
_U_VALUES = {
    "R717":  {"R717": 2200, "R290": 2000, "R1270": 2000, "R600a": 1600, "R600": 1400, "Water": 2000, "Air": 35},
    "R290":  {"R717": 2000, "R290": 1800, "R1270": 1800, "R600a": 1400, "R600": 1200, "Water": 1700, "Air": 35},
    "R1270": {"R717": 2000, "R290": 1800, "R1270": 1800, "R600a": 1400, "R600": 1200, "Water": 1700, "Air": 35},
    "R600a": {"R717": 1600, "R290": 1400, "R1270": 1400, "R600a": 1100, "R600": 1000, "Water": 1300, "Air": 30},
    "R600":  {"R717": 1400, "R290": 1200, "R1270": 1200, "R600a": 1000, "R600": 900,  "Water": 1100, "Air": 30},
    "Water": {"R717": 1800, "R290": 1500, "R1270": 1500, "R600a": 1100, "R600": 1000, "Water": 1500, "Air": 35},
    "Air":   {"R717": 35,   "R290": 35,   "R1270": 35,   "R600a": 30,   "R600": 30,   "Water": 35,   "Air": 15},
}


def get_U(fluid_a, fluid_b):
    """
    Look up the overall heat-transfer coefficient for a fluid pair.

    Parameters
    ----------
    fluid_a : str
        Hot-side fluid name (e.g. ``"R717"``, ``"Air"``, ``"Water"``).
    fluid_b : str
        Cold-side fluid name.

    Returns
    -------
    float
        Overall heat-transfer coefficient U [W/(m^2 K)].
    """
    return _U_VALUES[fluid_a][fluid_b]


def get_source_delta_T(T_src_in):
    """
    Return the fixed source water temperature drop [K].

    Parameters
    ----------
    T_src_in : float
        Source water inlet temperature [°C] (unused, kept for API compat).

    Returns
    -------
    float
        Temperature drop [K].
    """
    return float(SOURCE_DELTA_T)


def _patch_dissipative_hx(ean):
    """
    Monkey-patch dissipative HX components (Case 6) in an ExergyAnalysis.

    When a heat exchanger has hot side > Tamb and cold side < Tamb (Case 6),
    exerpy's ``aux_eqs`` returns None, crashing the exergoeconomic matrix.
    This patch replaces ``aux_eqs`` on those components with a version that
    uses the F-rule on the hot stream (thermal cost equality on the fuel side).
    """
    T0 = ean.Tamb

    for comp in ean.components.values():
        if comp.type != "heat_exchanger":
            continue

        inl0_T = comp.inl[0]["T"]
        inl1_T = comp.inl[1]["T"]
        outl0_T = comp.outl[0]["T"]
        outl1_T = comp.outl[1]["T"]

        # Case 6: hot always above T0, cold always below T0
        is_case6 = (inl0_T > T0 and outl0_T > T0
                     and inl1_T <= T0 and outl1_T <= T0)
        if not is_case6:
            continue

        logging.info(f"Patching dissipative HX '{comp.name}' (Case 6)")

        def _patched_aux_eqs(A, b, counter, T0, equations,
                             chemical_exergy_enabled, _comp=comp):
            # F-rule on hot stream (thermal)
            inl0 = _comp.inl[0]
            outl0 = _comp.outl[0]
            if inl0["e_T"] != 0 and outl0["e_T"] != 0:
                A[counter, inl0["CostVar_index"]["T"]] = 1 / inl0["E_T"]
                A[counter, outl0["CostVar_index"]["T"]] = -1 / outl0["E_T"]
            elif inl0["e_T"] == 0 and outl0["e_T"] != 0:
                A[counter, inl0["CostVar_index"]["T"]] = 1
            elif inl0["e_T"] != 0 and outl0["e_T"] == 0:
                A[counter, outl0["CostVar_index"]["T"]] = 1
            else:
                A[counter, inl0["CostVar_index"]["T"]] = 1
                A[counter, outl0["CostVar_index"]["T"]] = -1

            equations[counter] = {
                "kind": "aux_f_rule_hot",
                "objects": [_comp.name, inl0["name"], outl0["name"]],
                "property": "c_T",
            }

            # Mechanical equality (both streams)
            for idx, (i, o) in enumerate([
                (_comp.inl[0], _comp.outl[0]),
                (_comp.inl[1], _comp.outl[1]),
            ]):
                row = counter + 1 + idx
                if i["e_M"] != 0 and o["e_M"] != 0:
                    A[row, i["CostVar_index"]["M"]] = 1 / i["E_M"]
                    A[row, o["CostVar_index"]["M"]] = -1 / o["E_M"]
                elif i["e_M"] == 0 and o["e_M"] != 0:
                    A[row, i["CostVar_index"]["M"]] = 1
                elif i["e_M"] != 0 and o["e_M"] == 0:
                    A[row, o["CostVar_index"]["M"]] = 1
                else:
                    A[row, i["CostVar_index"]["M"]] = 1
                    A[row, o["CostVar_index"]["M"]] = -1
                equations[row] = {
                    "kind": "aux_equality",
                    "objects": [_comp.name, i["name"], o["name"]],
                    "property": "c_M",
                }

            num_aux = 3
            if chemical_exergy_enabled:
                for idx, (i, o) in enumerate([
                    (_comp.inl[0], _comp.outl[0]),
                    (_comp.inl[1], _comp.outl[1]),
                ]):
                    row = counter + 3 + idx
                    if i["e_CH"] != 0 and o["e_CH"] != 0:
                        A[row, i["CostVar_index"]["CH"]] = 1 / i["E_CH"]
                        A[row, o["CostVar_index"]["CH"]] = -1 / o["E_CH"]
                    elif i["e_CH"] == 0 and o["e_CH"] != 0:
                        A[row, i["CostVar_index"]["CH"]] = 1
                    elif i["e_CH"] != 0 and o["e_CH"] == 0:
                        A[row, o["CostVar_index"]["CH"]] = 1
                    else:
                        A[row, i["CostVar_index"]["CH"]] = 1
                        A[row, o["CostVar_index"]["CH"]] = -1
                    equations[row] = {
                        "kind": "aux_equality",
                        "objects": [_comp.name, i["name"], o["name"]],
                        "property": "c_CH",
                    }
                num_aux = 5

            for i in range(num_aux):
                b[counter + i] = 0

            return A, b, counter + num_aux, equations

        comp.aux_eqs = _patched_aux_eqs


def _extract_qt_sections(hx_dict):
    """
    Pre-compute Q-T section data from TESPy HX objects.

    Returns a dict mapping HX name to lists of Q, T_hot, T_cold values
    that can be serialised to JSON.
    """
    qt = {}
    for name, hx in hx_dict.items():
        Q_sec, T_hot, T_cold, _, _ = hx.calc_sections()
        qt[name] = {
            "Q": (Q_sec / 1000).tolist(),       # W → kW
            "T_hot": (T_hot - 273.15).tolist(),  # K → °C
            "T_cold": (T_cold - 273.15).tolist(),
        }
    return qt


def reconstruct_ean(exerpy_json_path, T_source_in_val):
    """
    Reconstruct an ExergyAnalysis from a saved exerpy JSON file.

    Loads the JSON, creates an ExergyAnalysis, classifies fuel/product/loss
    streams, runs the analysis, and patches dissipative HX components.

    Parameters
    ----------
    exerpy_json_path : str
        Path to the exerpy_data.json file.
    T_source_in_val : float
        Source water inlet temperature [deg C] (needed for stream classification).

    Returns
    -------
    ExergyAnalysis
        Fully analysed ExergyAnalysis object.
    """
    ean = ExergyAnalysis.from_json(exerpy_json_path)

    # Classify streams (same logic as in simulate_hthp)
    Tamb_degC = Tamb - 273.15
    product = {"inputs": ["43"], "outputs": ["41"]}

    fuel_inputs = ["e1"]
    fuel_outputs = []
    loss_inputs = []
    loss_outputs = []

    if T_source_in_val > Tamb_degC:
        fuel_inputs.append("11")
        fuel_outputs.append("13")
    else:
        loss_outputs.append("11")
        loss_inputs.append("13")

    fuel = {"inputs": fuel_inputs, "outputs": fuel_outputs}
    if loss_inputs or loss_outputs:
        loss = {"inputs": loss_inputs, "outputs": loss_outputs}
        ean.analyse(E_F=fuel, E_P=product, E_L=loss)
    else:
        ean.analyse(E_F=fuel, E_P=product)

    _patch_dissipative_hx(ean)
    return ean


def simulate_hthp(fluid_cycle1, fluid_cycle2, T_evap_c2_override=None,
                   T_source_in_override=None, source_mode="fixed_delta_T",
                   m_source=None):
    """
    Build and solve the cascaded HTHP network for a given fluid combination.

    The model consists of two vapour-compression cycles coupled through an
    internal heat exchanger (IHX).  An exergy analysis is performed after
    convergence, and component sizing data (volumetric flows, shaft powers,
    HX areas) is computed for the cost correlations in ``economics.py``.

    Parameters
    ----------
    fluid_cycle1 : str
        Refrigerant for the lower (source-side) cycle (e.g. ``"R290"``).
    fluid_cycle2 : str
        Refrigerant for the upper (sink-side) cycle (e.g. ``"R600a"``).
    T_evap_c2_override : float, optional
        Evaporation temperature of the upper cycle [deg C].  Defaults to the
        module-level ``T_evap_c2`` (60 deg C).
    T_source_in_override : float, optional
        Source water inlet temperature [deg C].  Defaults to the module-level
        ``T_source_in`` (20 deg C).
    source_mode : str, optional
        Source water constraint mode:
        - ``"fixed_delta_T"``: fix T_in and T_out = T_in - SOURCE_DELTA_T (m free).
        - ``"fixed_mass_flow"``: fix T_in and m (T_out free).
    m_source : float, optional
        Source water mass flow [kg/s] for ``"fixed_mass_flow"`` mode.
        Defaults to ``SOURCE_MASS_FLOW`` from config.

    Returns
    -------
    dict or None
        ``None`` if the combination is thermodynamically infeasible.
        Otherwise a dict with keys:

        - **ean** — ``ExergyAnalysis`` object (needed by ``run_economics``).
        - **COP**, **epsilon**, **E_F**, **E_P**, **E_D** — performance.
        - **fluid_cycle1**, **fluid_cycle2** — fluid names.
        - **sizing** — dict of component sizing values.
        - **U_values** — dict of heat-transfer coefficients [W/(m^2 K)].
        - **heat_exchangers** — TESPy HX objects (for Q-T diagrams).
        - **cycle_states** — state-point data (for log(p)-h diagrams).
    """
    T_evap_c2_val = T_evap_c2_override if T_evap_c2_override is not None else T_evap_c2
    T_src_in_val = T_source_in_override if T_source_in_override is not None else T_source_in
    delta_T_src = get_source_delta_T(T_src_in_val)
    T_src_out_val = T_src_in_val - delta_T_src
    T_cond_c1_est = T_evap_c2_val + pinch
    T_cond_c2_est = T_water_sat + pinch
    T_evap_c1_est = T_src_out_val - pinch

    T_crit_c1 = PropsSI("Tcrit", fluid_cycle1) - 273.15
    T_crit_c2 = PropsSI("Tcrit", fluid_cycle2) - 273.15

    if T_cond_c1_est >= T_crit_c1:
        print(f"  SKIP: {fluid_cycle1} critical temp ({T_crit_c1:.1f} °C) < condensing temp ({T_cond_c1_est:.1f} °C)")
        return None
    if T_cond_c2_est >= T_crit_c2:
        print(f"  SKIP: {fluid_cycle2} critical temp ({T_crit_c2:.1f} °C) < condensing temp ({T_cond_c2_est:.1f} °C)")
        return None

    try:

        nw = Network(iterinfo=False)
        nw.units.set_defaults(
            temperature="°C",
            pressure="bar",
            enthalpy="kJ/kg",
            mass_flow="kg/s",
            power="kW",
            heat="kW",
            volumetric_flow="m3/h"
        )

        # Source water loop
        src_in = Source("source inlet")
        src_out = Sink("source outlet")
        src_pump = Pump("SRC_PUMP")

        # Sink water loop
        snk_in = Source("sink inlet")
        snk_out = Sink("sink outlet")
        snk_pump = Pump("SNK_PUMP")

        # Heat exchangers & cycle components
        src_hx = HeatExchanger("SRC_HX")
        comp1 = Compressor("COMP1")
        valve1 = Valve("VAL1")
        ihx = HeatExchanger("IHX")
        cc1 = CycleCloser("cc1")

        snk_hx = HeatExchanger("SNK_HX")
        comp2 = Compressor("COMP2")
        valve2 = Valve("VAL2")
        cc2 = CycleCloser("cc2")

        # Source water connections
        c11 = Connection(src_in, "out1", src_pump, "in1", label="11")
        c12 = Connection(src_pump, "out1", src_hx, "in1", label="12")
        c13 = Connection(src_hx, "out1", src_out, "in1", label="13")

        # Cycle 1 (lower)
        c21 = Connection(src_hx, "out2", comp1, "in1", label="21")
        c22 = Connection(comp1, "out1", ihx, "in1", label="22")
        c22c = Connection(ihx, "out1", cc1, "in1", label="22c")
        c23 = Connection(cc1, "out1", valve1, "in1", label="23")
        c24 = Connection(valve1, "out1", src_hx, "in2", label="24")

        # Cycle 2 (upper)
        c31 = Connection(ihx, "out2", comp2, "in1", label="31")
        c32 = Connection(comp2, "out1", snk_hx, "in1", label="32")
        c32c = Connection(snk_hx, "out1", cc2, "in1", label="32c")
        c33 = Connection(cc2, "out1", valve2, "in1", label="33")
        c34 = Connection(valve2, "out1", ihx, "in2", label="34")

        # Sink water connections
        c41 = Connection(snk_in, "out1", snk_pump, "in1", label="41")
        c42 = Connection(snk_pump, "out1", snk_hx, "in2", label="42")
        c43 = Connection(snk_hx, "out2", snk_out, "in1", label="43")

        nw.add_conns(c21, c22, c22c, c23, c24)
        nw.add_conns(c11, c12, c13)
        nw.add_conns(c31, c32, c32c, c33, c34)
        nw.add_conns(c41, c42, c43)

        # Electrical power network
        power_input = PowerSource("grid")
        distribution = PowerBus("electricity distribution", num_in=1, num_out=4)
        motor1 = Motor("MOT1")
        motor2 = Motor("MOT2")
        motor3 = Motor("MOT3")  # drives SRC_PUMP
        motor4 = Motor("MOT4")  # drives SNK_PUMP

        e1 = PowerConnection(power_input, "power", distribution, "power_in1", label="e1")
        e2 = PowerConnection(distribution, "power_out1", motor1, "power_in", label="e2")
        e3 = PowerConnection(motor1, "power_out", comp1, "power", label="e3")
        e4 = PowerConnection(distribution, "power_out2", motor2, "power_in", label="e4")
        e5 = PowerConnection(motor2, "power_out", comp2, "power", label="e5")
        e6 = PowerConnection(distribution, "power_out3", motor3, "power_in", label="e6")
        e7 = PowerConnection(motor3, "power_out", src_pump, "power", label="e7")
        e8 = PowerConnection(distribution, "power_out4", motor4, "power_in", label="e8")
        e9 = PowerConnection(motor4, "power_out", snk_pump, "power", label="e9")
        nw.add_conns(e1, e2, e3, e4, e5, e6, e7, e8, e9)

        # Source water boundary conditions
        if source_mode == "fixed_mass_flow":
            m_val = m_source if m_source is not None else SOURCE_MASS_FLOW
            c11.set_attr(fluid={"water": 1}, T=T_src_in_val, p=p_source,
                         m=m_val)
            c13.set_attr(p=Ref(c11, 1, 0))
        else:  # fixed_delta_T
            c11.set_attr(fluid={"water": 1}, T=T_src_in_val, p=p_source)
            c13.set_attr(T=T_src_out_val, p=Ref(c11, 1, 0))

        # Cycle 1 boundary conditions
        c21.set_attr(fluid={fluid_cycle1: 1}, td_dew=pinch)
        c22.set_attr(T_dew=T_cond_c1_est)
        c23.set_attr(td_bubble=pinch)
        c24.set_attr(T_dew=T_evap_c1_est)

        # Cycle 2 boundary conditions
        c31.set_attr(fluid={fluid_cycle2: 1}, td_dew=pinch)
        c32.set_attr(T_dew=T_cond_c2_est)
        c33.set_attr(x=0)
        c34.set_attr(T=T_evap_c2_val)

        # Sink water boundary conditions
        c41.set_attr(fluid={"water": 1}, p=p_water, x=0, m=1)
        c43.set_attr(x=1, p=Ref(c41, 1, 0))

        # Component parameters
        comp1.set_attr(eta_s=0.74)
        comp2.set_attr(eta_s=0.74)
        src_pump.set_attr(eta_s=0.8)
        snk_pump.set_attr(eta_s=0.8)
        src_hx.set_attr(dp1=_DP["SRC_HX"][0], dp2=_DP["SRC_HX"][1])
        ihx.set_attr(dp1=_DP["IHX"][0], dp2=_DP["IHX"][1])
        snk_hx.set_attr(dp1=_DP["SNK_HX"][0], dp2=_DP["SNK_HX"][1])
        motor1.set_attr(eta=0.985)
        motor2.set_attr(eta=0.985)
        motor3.set_attr(eta=0.985)
        motor4.set_attr(eta=0.985)

        nw.solve("design")

        # Second solve: relax pressures and use pinch constraints
        c22.set_attr(T_dew=None)
        c24.set_attr(T_dew=None)
        c32.set_attr(T_dew=None)
        src_hx.set_attr(ttd_l=5)
        ihx.set_attr(td_pinch=5)
        snk_hx.set_attr(ttd_l=5)

        nw.solve("design")

        # Reject if cycle-1 pressure reaches 95 % of its critical pressure.
        # Cycle 2 is not checked — its condensing conditions are fixed by the
        # steam temperature and already validated by the T_crit pre-check.
        p_crit_c1 = PropsSI("Pcrit", fluid_cycle1) / 1e5  # bar
        for conn in [c21, c22, c22c, c23, c24]:
            if conn.p.val >= 0.95 * p_crit_c1:
                print(f"  SKIP: {fluid_cycle1} at {conn.label} reaches "
                      f"{conn.p.val:.1f} bar >= 95% of p_crit "
                      f"({p_crit_c1:.1f} bar)")
                return None

        Q_H = c41.m.val * (c43.h.val - c41.h.val)  # kW

        # COP from energy balance (independent of exergy definitions)
        W_shaft_total = (abs(comp1.P.val) + abs(comp2.P.val)
                         + abs(src_pump.P.val) + abs(snk_pump.P.val))  # kW
        eta_motor = 0.985
        W_el = W_shaft_total / eta_motor
        COP = Q_H / W_el

        # Exergy analysis — classify source water streams based on temperature vs Tamb
        ean = ExergyAnalysis.from_tespy(nw, Tamb=Tamb, pamb=pamb)
        product = {"inputs": ["43"], "outputs": ["41"]}
        Tamb_degC = Tamb - 273.15

        fuel_inputs = ["e1"]
        fuel_outputs = []
        loss_inputs = []
        loss_outputs = []

        # Source water streams must stay paired in the same category.
        # When T_source > Tamb: net exergy from water is fuel → both in fuel.
        # When T_source ≈ Tamb: cooling below ambient is a loss → both in loss.
        if T_src_in_val > Tamb_degC:
            fuel_inputs.append("11")
            fuel_outputs.append("13")
        else:
            loss_outputs.append("11")
            loss_inputs.append("13")

        fuel = {"inputs": fuel_inputs, "outputs": fuel_outputs}
        if loss_inputs or loss_outputs:
            loss = {"inputs": loss_inputs, "outputs": loss_outputs}
            ean.analyse(E_F=fuel, E_P=product, E_L=loss)
        else:
            ean.analyse(E_F=fuel, E_P=product)

        # Patch dissipative HX before economics can run
        _patch_dissipative_hx(ean)

        # Capture exerpy-format dict for JSON serialisation
        exerpy_data = to_exerpy(nw, Tamb=Tamb, pamb=pamb)

        # Pre-compute Q-T section data (before TESPy objects go out of scope)
        hx_dict = {"SRC_HX": src_hx, "IHX": ihx, "SNK_HX": snk_hx}
        qt_sections = _extract_qt_sections(hx_dict)

        # --- Sizing data for cost correlations ---
        # Compressor inlet volumetric flow [m³/h]: V_dot = m / rho * 3600
        V_dot_comp1 = c21.v.val # m³/h
        V_dot_comp2 = c31.v.val # m³/h

        # Shaft powers [kW] (TESPy P.val is in kW)
        W_comp1 = abs(comp1.P.val)
        W_comp2 = abs(comp2.P.val)
        W_src_pump = abs(src_pump.P.val)
        W_snk_pump = abs(snk_pump.P.val)

        # HX areas [m²]: A = kA / U
        U_vals = {
            "SRC_HX": get_U("Water", fluid_cycle1),
            "IHX": get_U(fluid_cycle1, fluid_cycle2),
            "SNK_HX": get_U(fluid_cycle2, "Water"),
        }
        A_src_hx = src_hx.kA.val / U_vals["SRC_HX"]
        A_ihx = ihx.kA.val / U_vals["IHX"]
        A_snk_hx = snk_hx.kA.val / U_vals["SNK_HX"]

        return {
            "ean": ean,
            "COP": COP,
            "epsilon": ean.epsilon,
            "E_F": ean.E_F,
            "E_P": ean.E_P,
            "E_D": ean.E_D,
            "fluid_cycle1": fluid_cycle1,
            "fluid_cycle2": fluid_cycle2,
            "T_source_in": T_src_in_val,
            "T_source_out": c13.T.val,
            "m_source": c11.m.val,
            "source_mode": source_mode,
            "sizing": {
                "V_dot_comp1": V_dot_comp1,  # m³/h
                "V_dot_comp2": V_dot_comp2,  # m³/h
                "W_comp1": W_comp1,           # kW
                "W_comp2": W_comp2,           # kW
                "W_src_pump": W_src_pump,     # kW
                "W_snk_pump": W_snk_pump,     # kW
                "A_src_hx": A_src_hx,         # m²
                "A_ihx": A_ihx,               # m²
                "A_snk_hx": A_snk_hx,         # m²
            },
            "U_values": U_vals,
            "exerpy_data": exerpy_data,
            "qt_sections": qt_sections,
            "cycle_states": {
                "cycle1": {
                    "fluid": fluid_cycle1,
                    "points": [
                        {"label": "21", "p": c21.p.val, "h": c21.h.val},
                        {"label": "22", "p": c22.p.val, "h": c22.h.val},
                        {"label": "23", "p": c23.p.val, "h": c23.h.val},
                        {"label": "24", "p": c24.p.val, "h": c24.h.val},
                    ],
                },
                "cycle2": {
                    "fluid": fluid_cycle2,
                    "points": [
                        {"label": "31", "p": c31.p.val, "h": c31.h.val},
                        {"label": "32", "p": c32.p.val, "h": c32.h.val},
                        {"label": "33", "p": c33.p.val, "h": c33.h.val},
                        {"label": "34", "p": c34.p.val, "h": c34.h.val},
                    ],
                },
            },
        }

    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def simulate_heater():
    """
    Compute thermodynamic and exergy results for an electrical resistance heater.

    Uses the same water-side boundary conditions as the HTHP
    (p = 2 bar, x = 0 -> x = 1, m = 1 kg/s).  Because electricity is
    pure exergy the fuel exergy equals the heating duty: E_F = Q_H.

    Returns
    -------
    dict
        Keys: ``COP`` (always 1.0), ``epsilon``, ``E_F`` [W], ``E_P`` [W],
        ``E_D`` [W], ``Q_H`` [W].
    """
    p = p_water * 2e5  # Pa
    m = 1.0  # kg/s

    h_in = PropsSI("H", "P", p, "Q", 0, "water")  # J/kg
    h_out = PropsSI("H", "P", p, "Q", 1, "water")  # J/kg
    s_in = PropsSI("S", "P", p, "Q", 0, "water")  # J/(kgK)
    s_out = PropsSI("S", "P", p, "Q", 1, "water")  # J/(kgK)

    h0 = PropsSI("H", "T", Tamb, "P", pamb, "water")  # J/kg
    s0 = PropsSI("S", "T", Tamb, "P", pamb, "water")  # J/(kgK)

    Q_H = m * (h_out - h_in)  # W

    e_in = (h_in - h0) - Tamb * (s_in - s0)  # J/kg
    e_out = (h_out - h0) - Tamb * (s_out - s0)  # J/kg

    E_F = Q_H  # W (electricity = pure exergy)
    E_P = m * (e_out - e_in)  # W
    E_D = E_F - E_P  # W

    return {
        "COP": 1.0,
        "epsilon": E_P / E_F,
        "E_F": E_F,
        "E_P": E_P,
        "E_D": E_D,
        "Q_H": Q_H,
    }


def simulate_gas_heater(eta_gas=0.95):
    """
    Build and solve a TESPy gas heater: CombustionChamber + HeatExchanger.

    Uses the Ahrendts chemical exergy database for proper exergy accounting
    of the natural gas (CH4) fuel stream.  The water side uses the same
    boundary conditions as the electric heater (p = 2 bar, x = 0 → 1, m = 1 kg/s).

    Parameters
    ----------
    eta_gas : float, optional
        Thermal efficiency of the gas heater (default 0.95).

    Returns
    -------
    dict
        Keys: ``COP`` (= eta_gas), ``epsilon``, ``E_F`` [W], ``E_P`` [W],
        ``E_D`` [W], ``Q_H`` [W], ``Q_gas`` [W].
    """
    from tespy.components import CombustionChamber
    from tespy.components import HeatExchanger

    # Pre-compute water-side duty to set CC thermal input
    h_in_w = PropsSI("H", "P", p_water * 1e5, "Q", 0, "water")  # J/kg
    h_out_w = PropsSI("H", "P", p_water * 1e5, "Q", 1, "water")  # J/kg
    Q_H_W = 1.0 * (h_out_w - h_in_w)  # W (m = 1 kg/s)
    ti_W = Q_H_W / eta_gas  # CC thermal input [W]

    nw = Network(iterinfo=False)
    nw = Network(iterinfo=False)
    nw.units.set_defaults(
        temperature="°C",
        pressure="bar",
        enthalpy="kJ/kg",
        mass_flow="kg/s",
    )

    # Components
    air_src = Source("air inlet")
    fuel_src = Source("fuel inlet")
    cc = CombustionChamber("CC")
    hx = HeatExchanger("GAS_HX")
    exh_snk = Sink("exhaust outlet")
    water_src = Source("water inlet")
    steam_snk = Sink("steam outlet")

    # Gas-side connections
    g1 = Connection(air_src, "out1", cc, "in1", label="g1")
    g2 = Connection(fuel_src, "out1", cc, "in2", label="g2")
    g3 = Connection(cc, "out1", hx, "in1", label="g3")
    g4 = Connection(hx, "out1", exh_snk, "in1", label="g4")

    # Water-side connections
    w1 = Connection(water_src, "out1", hx, "in2", label="w1")
    w2 = Connection(hx, "out2", steam_snk, "in1", label="w2")

    nw.add_conns(g1, g2, g3, g4, w1, w2)

    # Air and fuel at ambient conditions (only set p on one gas-side connection)
    g1.set_attr(fluid={"CH4": 0, "O2": 0.2314, "N2": 0.7553, "CO2": 0.0004, "H2O": 0, "Ar": 0.0129}, T=Tamb - 273.15, p=pamb / 1e5)
    g2.set_attr(fluid={"CH4": 1}, T=Tamb - 273.15)

    # Water: saturated liquid in → saturated vapour out at p_water
    w1.set_attr(fluid={"H2O": 1}, p=p_water, x=0, m=1)
    w2.set_attr(x=1)

    # CC: excess air ratio λ = 1.2, thermal input in W
    cc.set_attr(lamb=1.2, ti=ti_W)

    # HX: no pressure drops
    hx.set_attr(pr1=1, pr2=1)

    nw.solve("design")

    # Exergy analysis with Ahrendts chemical exergy
    ean = ExergyAnalysis.from_tespy(
        nw, Tamb=Tamb, pamb=pamb, chemExLib="Ahrendts",
    )
    ean.analyse(
        E_F={"inputs": ["g2"]},
        E_P={"inputs": ["w2"], "outputs": ["w1"]},
        E_L={"inputs": ["g4"], "outputs": ["g1"]},
    )

    return {
        "COP": eta_gas,
        "epsilon": ean.epsilon,
        "E_F": ean.E_F,      # chemical exergy of fuel [W]
        "E_P": ean.E_P,      # exergy gained by water [W]
        "E_D": ean.E_D,      # exergy destruction [W]
        "Q_H": Q_H_W,        # heating duty [W]
        "Q_gas": ti_W,       # LHV-based gas energy input [W]
    }
