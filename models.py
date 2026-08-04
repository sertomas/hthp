"""
Thermodynamic simulation models for the cascaded HTHP and the gas-heater reference.

Provides two public functions:

* ``simulate_hthp`` — builds and solves a two-stage heat-pump TESPy network,
  runs an exergy analysis, and returns all data required by the economics
  and plotting stages.
* ``simulate_gas_heater`` — TESPy combustion-chamber + heat-exchanger model
  for the natural gas heater used as a reference case.
"""

import logging

import numpy as np

from CoolProp.CoolProp import PropsSI
from tespy.components import Compressor, CycleCloser, Motor, PowerBus, PowerSource, Sink, Source, Valve
from tespy.components import MovingBoundaryHeatExchanger as HeatExchanger
from tespy.connections import Connection, PowerConnection, Ref
from tespy.networks import Network

from exerpy import ExergyAnalysis
from exerpy.parser.from_tespy.tespy_parser import to_exerpy

from calculate_heatexchanger_area import get_hex_area
from config import (
    EF_DEFINITION,
    M_STEAM,
    SOURCE_DELTA_T,
    SOURCE_MASS_FLOW,
    T_STEAM_DEFAULT,
    p_water_for_T_steam,
)

# Shared parameters
Tamb = 293.15  # K
pamb = 101325  # Pa
T_source_in = Tamb - 273.15  # °C (default: 20 °C)
p_source = 3  # bar (source water circulation pressure)
T_evap_c2 = 60  # °C
pinch = 5  # K

# Default sink-steam conditions (used when no T_steam_override is passed).
p_water = p_water_for_T_steam(T_STEAM_DEFAULT)  # bar
T_water_sat = T_STEAM_DEFAULT  # °C

# Pressure drops are neglected on every heat-exchanger side (pr = 1.0),
# both for the refrigerant streams and for the source/sink water streams.
_PR = {
    "SRC_HX": (1.0, 1.0),   # (source water side, refrigerant side)
    "IHX":    (1.0, 1.0),   # (C1 condensing, C2 evaporating)
    "SNK_HX": (1.0, 1.0),   # (C2 condensing, sink water side)
}

# Compressor high-side pressure limits (Ommen 2015, Table 3). Designs above
# (limit x _OMMEN_P_TOL) are excluded by simulate_hthp. R600/R1270 are proxied
# onto the Type-2 28 bar row; R717 uses the 50 bar HP envelope.
_OMMEN_P_MAX_BAR = {
    "R290":  28.0,
    "R1270": 28.0,
    "R600a": 28.0,
    "R600":  28.0,
    "R717":  50.0,
}

# 10 % buffer on the Ommen pressure limits to keep marginal cases in scope.
_OMMEN_P_TOL = 1.10


def _check_ommen_p_limit(fluid, p_high, cycle_label):
    """
    Return True iff the cycle's high-side pressure is within
    (Ommen 2015 limit) × _OMMEN_P_TOL for the given fluid; otherwise log
    and return False.
    """
    p_max_eff = _OMMEN_P_MAX_BAR[fluid] * _OMMEN_P_TOL
    if p_high > p_max_eff:
        print(f"  SKIP: {fluid} ({cycle_label}) high-side p={p_high:.1f} bar "
              f"> {_OMMEN_P_MAX_BAR[fluid]:.0f} bar × {_OMMEN_P_TOL:.2f} "
              f"= {p_max_eff:.1f} bar (Ommen 2015 limit + tol)")
        return False
    return True


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


def simulate_hthp(fluid_cycle1, fluid_cycle2, T_evap_c2_override=None,
                   T_source_in_override=None, source_mode="fixed_delta_T",
                   m_source=None, T_steam_override=None,
                   skip_ommen_check=False):
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
        ``"fixed_delta_T"`` fixes T_in and T_out = T_in - SOURCE_DELTA_T (m free);
        ``"fixed_mass_flow"`` fixes T_in and m (T_out free). The pipeline uses
        ``"fixed_mass_flow"``.
    m_source : float, optional
        Source water mass flow [kg/s] for ``"fixed_mass_flow"`` mode.
        Defaults to ``SOURCE_MASS_FLOW`` from config.
    T_steam_override : float, optional
        Sink-steam saturation temperature [deg C]. Defaults to
        ``T_STEAM_DEFAULT`` (100 °C).
    skip_ommen_check : bool, optional
        If ``True``, skip the Ommen pressure feasibility gate so the screener
        can keep out-of-envelope designs visible. Default ``False``.

    Returns
    -------
    dict or None
        ``None`` if the combination is thermodynamically infeasible.
        Otherwise a dict with keys:

        - **ean** — ``ExergyAnalysis`` object (needed by ``run_economics``).
        - **COP**, **epsilon**, **E_F**, **E_P**, **E_D** — performance.
        - **fluid_cycle1**, **fluid_cycle2** — fluid names.
        - **sizing** — component sizing values (volumetric flows, shaft
          powers, HX areas) for the cost correlations.
        - **qt_sections** — Q-T section data (kW, °C) for plotting.
        - **cycle_states** — state-point data for log(p)-h diagrams.
    """
    T_evap_c2_val = T_evap_c2_override if T_evap_c2_override is not None else T_evap_c2
    T_src_in_val = T_source_in_override if T_source_in_override is not None else T_source_in
    T_steam_val = T_steam_override if T_steam_override is not None else T_water_sat
    p_water_val = p_water_for_T_steam(T_steam_val)
    delta_T_src = get_source_delta_T(T_src_in_val)
    T_src_out_val = T_src_in_val - delta_T_src
    T_cond_c1_est = T_evap_c2_val + pinch
    T_cond_c2_est = T_steam_val + pinch
    T_evap_c1_est = T_src_out_val - pinch

    T_crit_c1 = PropsSI("Tcrit", fluid_cycle1) - 273.15
    T_crit_c2 = PropsSI("Tcrit", fluid_cycle2) - 273.15

    # Critical-temperature pre-checks: reject fluids whose condensing
    # temperature reaches or exceeds the critical temperature.
    if T_cond_c1_est >= T_crit_c1:
        print(f"  SKIP: {fluid_cycle1} critical temp ({T_crit_c1:.1f} C) < condensing temp ({T_cond_c1_est:.1f} C)")
        return None
    if T_cond_c2_est >= T_crit_c2:
        print(f"  SKIP: {fluid_cycle2} critical temp ({T_crit_c2:.1f} C) < condensing temp ({T_cond_c2_est:.1f} C)")
        return None

    try:
        p_evap_c1 = PropsSI("P", "T", T_evap_c1_est + 273.15, "Q", 1, fluid_cycle1) / 1e5
        p_cond_c1 = PropsSI("P", "T", T_cond_c1_est + 273.15, "Q", 1, fluid_cycle1) / 1e5
        p_cond_c2 = PropsSI("P", "T", T_cond_c2_est + 273.15, "Q", 1, fluid_cycle2) / 1e5

        nw = Network(T_unit="C", p_unit="bar", h_unit="kJ / kg", m_unit="kg / s", iterinfo=False)

        # Source water loop (no pump — existing site infrastructure)
        src_in = Source("source inlet")
        src_out = Sink("source outlet")

        # Sink water loop (no pump — pre-existing boiler feedwater pump)
        snk_in = Source("sink inlet")
        snk_out = Sink("sink outlet")

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

        # Source water connections (direct, no pump)
        c11 = Connection(src_in, "out1", src_hx, "in1", label="11")
        c12 = Connection(src_hx, "out1", src_out, "in1", label="12")

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

        # Sink water connections (direct, no pump)
        c41 = Connection(snk_in, "out1", snk_hx, "in2", label="41")
        c42 = Connection(snk_hx, "out2", snk_out, "in1", label="42")

        nw.add_conns(c21, c22, c22c, c23, c24)
        nw.add_conns(c11, c12)
        nw.add_conns(c31, c32, c32c, c33, c34)
        nw.add_conns(c41, c42)

        # Electrical power network: grid feeding the two compressor motors
        power_input = PowerSource("grid")
        distribution = PowerBus("electricity distribution", num_in=1, num_out=2)
        motor1 = Motor("MOT1")
        motor2 = Motor("MOT2")

        e1 = PowerConnection(power_input, "power", distribution, "power_in1", label="e1")
        e2 = PowerConnection(distribution, "power_out1", motor1, "power_in", label="e2")
        e3 = PowerConnection(motor1, "power_out", comp1, "power", label="e3")
        e4 = PowerConnection(distribution, "power_out2", motor2, "power_in", label="e4")
        e5 = PowerConnection(motor2, "power_out", comp2, "power", label="e5")
        nw.add_conns(e1, e2, e3, e4, e5)

        # Source water boundary conditions (c12.p follows c11.p via SRC_HX pr1=1).
        if source_mode == "fixed_mass_flow":
            m_val = m_source if m_source is not None else SOURCE_MASS_FLOW
            c11.set_attr(fluid={"water": 1}, T=T_src_in_val, p=p_source,
                         m=m_val)
        else:  # fixed_delta_T
            c11.set_attr(fluid={"water": 1}, T=T_src_in_val, p=p_source)
            c12.set_attr(T=T_src_out_val)

        # Cycle 1 boundary conditions
        c21.set_attr(fluid={fluid_cycle1: 1}, td_dew=pinch)
        c22.set_attr(p=p_cond_c1)
        c23.set_attr(td_bubble=pinch)
        c24.set_attr(p=p_evap_c1)

        # Cycle 2 boundary conditions
        c31.set_attr(fluid={fluid_cycle2: 1}, td_dew=pinch)
        c32.set_attr(p=p_cond_c2)
        c33.set_attr(x=0)
        c34.set_attr(T=T_evap_c2_val)

        # Sink water boundary conditions (c42.p follows c41.p via SNK_HX pr2=1).
        c41.set_attr(fluid={"water": 1}, p=p_water_val, x=0, m=M_STEAM)
        c42.set_attr(x=1)

        # Component parameters (eta_s, eta_motor from Ommen 2015, Table 1).
        comp1.set_attr(eta_s=0.80)   # Ommen 2015, Table 1
        comp2.set_attr(eta_s=0.80)   # Ommen 2015, Table 1
        src_hx.set_attr(pr1=_PR["SRC_HX"][0], pr2=_PR["SRC_HX"][1])
        ihx.set_attr(pr1=_PR["IHX"][0], pr2=_PR["IHX"][1])
        snk_hx.set_attr(pr1=_PR["SNK_HX"][0], pr2=_PR["SNK_HX"][1])
        motor1.set_attr(eta=0.95)    # Ommen 2015, Table 1
        motor2.set_attr(eta=0.95)    # Ommen 2015, Table 1

        nw.solve("design")

        # Second solve: relax the high-side and evaporator pressures and let
        # the pinch constraints (HX terminal temperature differences and the
        # IHX internal pinch) drive the operating point.
        c24.set_attr(p=None)
        c22.set_attr(p=None)
        c32.set_attr(p=None)
        src_hx.set_attr(ttd_l=5)
        ihx.set_attr(td_pinch=5)
        snk_hx.set_attr(ttd_l=5)

        nw.solve("design")

        # Reject if the cycle-1 pressure reaches 95 % of critical.
        p_crit_c1 = PropsSI("Pcrit", fluid_cycle1) / 1e5  # bar
        for conn in [c21, c22, c22c, c23, c24]:
            if conn.p.val >= 0.95 * p_crit_c1:
                print(f"  SKIP: {fluid_cycle1} at {conn.label} reaches "
                      f"{conn.p.val:.1f} bar >= 95% of p_crit "
                      f"({p_crit_c1:.1f} bar)")
                return None

        # Reject if either cycle exceeds the Ommen compressor envelope
        # (bypassed when skip_ommen_check=True for the feasibility screener).
        p_high_c1 = max(c22.p.val, c22c.p.val, c23.p.val)
        p_high_c2 = max(c32.p.val, c32c.p.val, c33.p.val)
        if not skip_ommen_check:
            if not _check_ommen_p_limit(fluid_cycle1, p_high_c1, "cycle 1"):
                return None
            if not _check_ommen_p_limit(fluid_cycle2, p_high_c2, "cycle 2"):
                return None

        Q_H = c41.m.val * (c42.h.val - c41.h.val)  # kW

        # COP from energy balance (independent of exergy definitions)
        W_shaft_total = abs(comp1.P.val) + abs(comp2.P.val)  # W
        eta_motor = 0.95   # Ommen 2015, Table 1
        W_el = W_shaft_total / eta_motor
        COP = Q_H * 1000 / W_el  # Q_H [kW] → [W]

        # Exergy analysis — classify the source water streams (11 in, 12 out).
        ean = ExergyAnalysis.from_tespy(nw, Tamb=Tamb, pamb=pamb)
        product = {"inputs": ["42"], "outputs": ["41"]}

        # exerpy convention: "inputs" ADD to a category, "outputs" SUBTRACT.
        #
        # Source water inlet 11 is always a FUEL input (water arrives carrying
        # exergy from the free reservoir, priced at c = 0 downstream).
        #
        # Source water outlet 12 depends on config.EF_DEFINITION:
        #   "outlet_loss" — 12 is always a LOSS (E_F = E_e1 + E_11).
        #   "outlet_fuel" — in fixed-mass-flow mode the outlet temperature
        #     floats. While the water leaves AT OR ABOVE ambient it still
        #     carries usable thermal exergy, so 12 is a FUEL OUTPUT — the net
        #     water fuel E_11 − E_12 is then the exergy actually drawn, and
        #     its leaving exergy is not charged to the system. Only when it
        #     leaves BELOW ambient is that exergy dumped unused → LOSS.
        Tamb_degC = Tamb - 273.15
        if EF_DEFINITION == "outlet_fuel" and c12.T.val >= Tamb_degC:
            fuel = {"inputs": ["e1", "11"], "outputs": ["12"]}
            loss = {}
        else:
            fuel = {"inputs": ["e1", "11"], "outputs": []}
            loss = {"inputs": ["12"], "outputs": []}
        ean.analyse(E_F=fuel, E_P=product, E_L=loss)

        # Patch dissipative HX before economics can run
        _patch_dissipative_hx(ean)

        # Capture exerpy-format dict for JSON serialisation
        exerpy_data = to_exerpy(nw, Tamb=Tamb, pamb=pamb)

        # Pre-compute Q-T section data (before TESPy objects go out of scope)
        hx_dict = {"SRC_HX": src_hx, "IHX": ihx, "SNK_HX": snk_hx}
        qt_sections = _extract_qt_sections(hx_dict)

        # --- Sizing data for cost correlations ---
        # Compressor inlet volumetric flow [m³/h]: V_dot = m / rho * 3600
        rho_21 = PropsSI("D", "H", c21.h.val * 1000, "P", c21.p.val * 1e5, fluid_cycle1)
        rho_22 = PropsSI("D", "H", c22.h.val * 1000, "P", c22.p.val * 1e5, fluid_cycle1)
        rho_31 = PropsSI("D", "H", c31.h.val * 1000, "P", c31.p.val * 1e5, fluid_cycle2)
        rho_32 = PropsSI("D", "H", c32.h.val * 1000, "P", c32.p.val * 1e5, fluid_cycle2)
        V_dot_comp1 = c21.m.val / rho_21 * 3600  # m³/h
        V_dot_comp2 = c31.m.val / rho_31 * 3600  # m³/h

        # Volumetric efficiency, Dincer (2010): eta_vol = 1 - R*(rho_disch/rho_suc - 1),
        # clearance ratio R = 0.06 (Hoang et al. 2025). Enters the compressor
        # PEC as the displaced volume V_dot / eta_vol.
        CLEARANCE_RATIO = 0.06
        eta_vol_comp1 = 1.0 - CLEARANCE_RATIO * ((rho_22 / rho_21) - 1.0)
        eta_vol_comp2 = 1.0 - CLEARANCE_RATIO * ((rho_32 / rho_31) - 1.0)

        # Shaft powers [kW] (TESPy P.val is in W)
        W_comp1 = abs(comp1.P.val) / 1000
        W_comp2 = abs(comp2.P.val) / 1000

        # HX areas [m^2] from get_hex_area (section-wise A = Q / (LMTD * U)).
        A_src_hx = get_hex_area(src_hx)
        A_ihx = get_hex_area(ihx)
        A_snk_hx = get_hex_area(snk_hx)

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
            "T_source_out": c12.T.val,
            "m_source": c11.m.val,
            "source_mode": source_mode,
            "sizing": {
                "V_dot_comp1": V_dot_comp1,  # m³/h
                "V_dot_comp2": V_dot_comp2,  # m³/h
                "eta_vol_comp1": eta_vol_comp1,  # [-]
                "eta_vol_comp2": eta_vol_comp2,  # [-]
                "W_comp1": W_comp1,           # kW
                "W_comp2": W_comp2,           # kW
                "A_src_hx": A_src_hx,         # m²
                "A_ihx": A_ihx,               # m²
                "A_snk_hx": A_snk_hx,         # m²
            },
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


def simulate_gas_heater(eta_gas=0.90, T_steam_override=None):
    """
    Build and solve a TESPy gas heater: CombustionChamber + HeatExchanger.

    Uses the Ahrendts chemical exergy database for proper exergy accounting
    of the natural gas (CH4) fuel stream.  The water side uses the same
    boundary conditions as the HTHP sink (saturated water → saturated steam,
    m = ``config.M_STEAM``, the same sink steam flow as the HTHP).

    Parameters
    ----------
    eta_gas : float, optional
        Gas-heater thermal efficiency. Default 0.90 (Ommen 2015, Table 1).
    T_steam_override : float, optional
        Sink-steam saturation temperature [deg C]. Defaults to
        ``T_STEAM_DEFAULT`` (100 °C). The water-side pressure is derived
        from this temperature.

    Returns
    -------
    dict
        Keys: ``COP`` (= eta_gas), ``epsilon``, ``E_F`` [W], ``E_P`` [W],
        ``E_D`` [W], ``Q_H`` [W], ``Q_gas`` [W], ``m_dot_CO2`` [kg/s],
        ``exerpy_data`` (full TESPy + exerpy network dump: connections,
        components, parameters; same shape as ``simulate_hthp``),
        ``ean`` (the ``ExergyAnalysis`` object — needed by
        ``run_economics_gas_heater`` to build the exergoeconomic balance).
    """
    from tespy.components import CombustionChamber
    from tespy.components import HeatExchanger

    T_steam_val = T_steam_override if T_steam_override is not None else T_water_sat
    p_water_val = p_water_for_T_steam(T_steam_val)

    # Pre-compute water-side duty to set CC thermal input
    h_in_w = PropsSI("H", "P", p_water_val * 1e5, "Q", 0, "water")  # J/kg
    h_out_w = PropsSI("H", "P", p_water_val * 1e5, "Q", 1, "water")  # J/kg
    Q_H_W = M_STEAM * (h_out_w - h_in_w)  # W (same sink steam flow as the HTHP)
    ti_W = Q_H_W / eta_gas  # CC thermal input [W]

    nw = Network(
        T_unit="C", p_unit="bar", h_unit="kJ / kg", m_unit="kg / s",
        iterinfo=False,
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

    # Water: saturated liquid in → saturated vapour out at p_water_val
    w1.set_attr(fluid={"H2O": 1}, p=p_water_val, x=0, m=M_STEAM)
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
        E_F={"inputs": ["g2", "g1"]},
        E_P={"inputs": ["w2"], "outputs": ["w1"]},
        E_L={"inputs": ["g4"]},
    )

    # CO2 emission rate from CH4 combustion. Stoichiometry CH4 + 2 O2 →
    # CO2 + 2 H2O gives 1 mol CO2 per mol CH4; m_CO2 = m_CH4 · M_CO2/M_CH4.
    # Fuel stream g2 is pure methane (set above), so g2.m.val is m_CH4.
    M_CH4 = 16.043   # kg/kmol
    M_CO2 = 44.010   # kg/kmol
    m_dot_CO2 = g2.m.val * (M_CO2 / M_CH4)   # kg/s

    # Full exerpy-format network dump (same shape as the HTHP path).
    exerpy_data = to_exerpy(nw, Tamb=Tamb, pamb=pamb)

    return {
        "COP": eta_gas,
        "epsilon": ean.epsilon,
        "E_F": ean.E_F,      # chemical exergy of fuel [W]
        "E_P": ean.E_P,      # exergy gained by water [W]
        "E_D": ean.E_D,      # exergy destruction [W]
        "Q_H": Q_H_W,        # heating duty [W]
        "Q_gas": ti_W,       # LHV-based gas energy input [W]
        "m_dot_CO2": m_dot_CO2,      # CO2 emitted by combustion [kg/s]
        "exerpy_data": exerpy_data,  # full state-point + component dump
        "ean": ean,                  # for run_economics_gas_heater
    }
