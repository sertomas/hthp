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

from config import (
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

# Default sink-steam conditions (used when T_steam_override is not provided
# to simulate_hthp / simulate_gas_heater). These are derived from
# T_STEAM_DEFAULT so that changing the default in config.py automatically
# propagates here.
p_water = p_water_for_T_steam(T_STEAM_DEFAULT)  # bar
T_water_sat = T_STEAM_DEFAULT  # °C — by definition of T_STEAM_DEFAULT

# Absolute pressure drops [bar].
# Brownfield-retrofit assumption: the source-water and sink-water (feedwater)
# loops are pre-existing site infrastructure (the existing source-loop
# circulator and the displaced gas-boiler feedwater pump remain in place),
# so the heat-pump scope incurs *no* new pressure drop on the water sides.
# Refrigerant-side Δp values are kept at typical plate-HX simulation defaults
# (Jensen 2015, Mateu-Royo 2019; calibration reference pending).
_DP = {
    "SRC_HX": (0.0,   0.175),   # (source water side, refrigerant side)
    "IHX":    (0.300, 0.150),   # (C1 condensing, C2 evaporating)
    "SNK_HX": (0.250, 0.0),     # (C2 condensing, sink water side)
}

# Compressor high-side pressure limits per Ommen et al. (2015), Table 3.
# Cases with cycle high-side pressure exceeding (limit × _OMMEN_P_TOL) fall
# outside every commercially-available compressor surveyed in the paper and
# are excluded by simulate_hthp (returns None → cached as "failed").
# - R290 / R600a: Type 2, 28 bar.
# - R600 / R1270: not in Ommen Table 3; treated as Type 2 here, mirroring
#   the proxy mapping used in economics._FLUID_COST_TYPE.
# - R717: HP envelope (Type 4, 50 bar). LP envelope (Type 3, 28 bar) is
#   much tighter; switching between LP and HP cost classes happens later
#   in run_economics based on the simulated discharge pressure.
# - R744: transcritical Type 5, 140 bar. Cost row exists in Ommen Table 4.
#   Transcritical handling IS implemented (see _TRANSCRITICAL_FLUIDS,
#   _R744_P_HIGH_BAR, and the is_c?_transcritical branches in simulate_hthp)
#   but R744 is not in any active fluid list (FLUIDS_C1 / FLUIDS_C2) — see
#   the comment block at the top of config.py for the exclusion rationale.
_OMMEN_P_MAX_BAR = {
    "R290":  28.0,
    "R1270": 28.0,
    "R600a": 28.0,
    "R600":  28.0,
    "R717":  50.0,
    "R744": 140.0,
}

# Engineering tolerance applied on top of the Ommen pressure limits.
# Real-world catalogue pressure ratings are not exactly the rounded values
# in Ommen Table 3 and machine families do exist a few bar above the listed
# ceilings. 10 % is a reasonable buffer to keep marginal cases in scope.
_OMMEN_P_TOL = 1.10

# Transcritical fluids — high-side pressure cannot be set from saturation
# at the cooling temperature (no phase change above T_crit). Currently only
# R744 (CO2). Cycle-1 R744 uses a fixed transcritical pressure; the IHX
# acts as a gas cooler instead of a condenser.
_TRANSCRITICAL_FLUIDS = {"R744"}

# Default transcritical high-side pressure [bar]. 100 bar is in the
# typically-optimal 90-120 bar range for CO2 heat-pumping duty
# (Neksa et al. 1998, "CO2-heat pump water heater"). Not optimised here;
# could be a design variable in a follow-up study.
_R744_P_HIGH_BAR = 100.0


def _is_transcritical(fluid):
    return fluid in _TRANSCRITICAL_FLUIDS


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


# U-Value Lookup Table [W/(m2K)]
# R744 (CO2) values for transcritical gas cooling are typically a touch
# higher than HC condensing duty thanks to its thermophysical properties —
# 1800-2200 W/(m²·K) is a representative range for plate gas coolers.
_U_VALUES = {
    "R717":  {"R717": 2200, "R290": 2000, "R1270": 2000, "R600a": 1600, "R600": 1400, "R744": 1800, "Water": 2000},
    "R290":  {"R717": 2000, "R290": 1800, "R1270": 1800, "R600a": 1400, "R600": 1200, "R744": 1500, "Water": 1700},
    "R1270": {"R717": 2000, "R290": 1800, "R1270": 1800, "R600a": 1400, "R600": 1200, "R744": 1500, "Water": 1700},
    "R600a": {"R717": 1600, "R290": 1400, "R1270": 1400, "R600a": 1100, "R600": 1000, "R744": 1300, "Water": 1300},
    "R600":  {"R717": 1400, "R290": 1200, "R1270": 1200, "R600a": 1000, "R600": 900,  "R744": 1100, "Water": 1100},
    "R744":  {"R717": 1800, "R290": 1500, "R1270": 1500, "R600a": 1300, "R600": 1100, "R744": 1500, "Water": 2000},
    "Water": {"R717": 1800, "R290": 1500, "R1270": 1500, "R600a": 1100, "R600": 1000, "R744": 2000, "Water": 1500},
}


def get_U(fluid_a, fluid_b):
    """
    Look up the overall heat-transfer coefficient for a fluid pair.

    Parameters
    ----------
    fluid_a : str
        Hot-side fluid name (e.g. ``"R717"``, ``"Water"``).
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
        Source water constraint mode:
        - ``"fixed_delta_T"``: fix T_in and T_out = T_in - SOURCE_DELTA_T (m free).
        - ``"fixed_mass_flow"``: fix T_in and m (T_out free).
        The function-level default is ``"fixed_delta_T"``, but every call
        site in the pipeline (``main.py``, ``case_steam.py``,
        ``case_steam_economics.py``, ``export_design_details.py``)
        passes ``"fixed_mass_flow"`` explicitly. The ΔT mode is retained
        for ad-hoc programmatic use only.
    m_source : float, optional
        Source water mass flow [kg/s] for ``"fixed_mass_flow"`` mode.
        Defaults to ``SOURCE_MASS_FLOW`` from config.
    T_steam_override : float, optional
        Sink-steam saturation temperature [deg C]. Defaults to
        ``T_STEAM_DEFAULT`` (100 °C). Lower values relax the cycle-2
        condensing pressure and bring fluids like R600a back into the
        Ommen 2015 compressor envelope.
    skip_ommen_check : bool, optional
        If ``True``, skip the Ommen 2015 high-side pressure feasibility
        gate inside the solver and return the converged state regardless
        of envelope. Used by the screening stage (``case_steam.py``)
        to keep out-of-envelope designs visible in the 4-state grid
        instead of reporting them as ``None``. Default ``False``.

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
        - **qt_sections** — pre-computed Q-T section data (kW, °C) ready
          for plotting; see ``_extract_qt_sections``.
        - **cycle_states** — state-point data (for log(p)-h diagrams).
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

    is_c1_transcritical = _is_transcritical(fluid_cycle1)
    is_c2_transcritical = _is_transcritical(fluid_cycle2)

    # Critical-temperature pre-checks. For transcritical fluids the cycle
    # high-side runs above T_crit by design — skip the rejection there.
    if not is_c1_transcritical and T_cond_c1_est >= T_crit_c1:
        print(f"  SKIP: {fluid_cycle1} critical temp ({T_crit_c1:.1f} °C) < condensing temp ({T_cond_c1_est:.1f} °C)")
        return None
    if not is_c2_transcritical and T_cond_c2_est >= T_crit_c2:
        print(f"  SKIP: {fluid_cycle2} critical temp ({T_crit_c2:.1f} °C) < condensing temp ({T_cond_c2_est:.1f} °C)")
        return None
    # R744 cycle-1 evaporator is still subcritical (T_evap < 31 °C); reject
    # if T_evap_c1_est is at or above T_crit (would require an unusual
    # transcritical evaporation that this model doesn't support).
    if is_c1_transcritical and T_evap_c1_est >= T_crit_c1:
        print(f"  SKIP: {fluid_cycle1} evaporator T ({T_evap_c1_est:.1f} °C) >= T_crit "
              f"({T_crit_c1:.1f} °C) — transcritical evap not supported")
        return None

    try:
        p_evap_c1 = PropsSI("P", "T", T_evap_c1_est + 273.15, "Q", 1, fluid_cycle1) / 1e5
        if is_c1_transcritical:
            p_cond_c1 = _R744_P_HIGH_BAR  # fixed transcritical high-side pressure
        else:
            p_cond_c1 = PropsSI("P", "T", T_cond_c1_est + 273.15, "Q", 1, fluid_cycle1) / 1e5
        if is_c2_transcritical:
            p_cond_c2 = _R744_P_HIGH_BAR
        else:
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

        # Sink water connections (direct, no pump)
        c41 = Connection(snk_in, "out1", snk_hx, "in2", label="41")
        c43 = Connection(snk_hx, "out2", snk_out, "in1", label="43")

        nw.add_conns(c21, c22, c22c, c23, c24)
        nw.add_conns(c11, c13)
        nw.add_conns(c31, c32, c32c, c33, c34)
        nw.add_conns(c41, c43)

        # Electrical power network — only the two compressor motors remain
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

        # Source water boundary conditions
        # No Ref constraint on c13.p needed: with SRC_HX dp1=0 (water side),
        # c13.p is automatically equal to c11.p, and adding the Ref would
        # over-determine the system.
        if source_mode == "fixed_mass_flow":
            m_val = m_source if m_source is not None else SOURCE_MASS_FLOW
            c11.set_attr(fluid={"water": 1}, T=T_src_in_val, p=p_source,
                         m=m_val)
        else:  # fixed_delta_T
            c11.set_attr(fluid={"water": 1}, T=T_src_in_val, p=p_source)
            c13.set_attr(T=T_src_out_val)

        # Cycle 1 boundary conditions
        c21.set_attr(fluid={fluid_cycle1: 1}, td_dew=pinch)
        c22.set_attr(p=p_cond_c1)
        if is_c1_transcritical:
            # Gas cooler outlet: no saturation. Set hot-end exit T directly
            # (= T_evap_c2 + pinch — same target as the subcritical condenser).
            c23.set_attr(T=T_cond_c1_est)
        else:
            c23.set_attr(td_bubble=pinch)
        c24.set_attr(p=p_evap_c1)

        # Cycle 2 boundary conditions
        c31.set_attr(fluid={fluid_cycle2: 1}, td_dew=pinch)
        c32.set_attr(p=p_cond_c2)
        if is_c2_transcritical:
            c33.set_attr(T=T_cond_c2_est)
        else:
            c33.set_attr(x=0)
        c34.set_attr(T=T_evap_c2_val)

        # Sink water boundary conditions.
        # No Ref constraint on c43.p: with SNK_HX dp2=0 (water/steam side),
        # c43.p is automatically equal to c41.p (both at the steam saturation
        # pressure), and adding the Ref would over-determine the system.
        c41.set_attr(fluid={"water": 1}, p=p_water_val, x=0, m=M_STEAM)
        c43.set_attr(x=1)

        # Component parameters
        # Compressor isentropic efficiency and motor electrical efficiency taken
        # from Ommen et al. (2015), "Technical and economic working domains of
        # industrial heat pumps: Part 1 - Single stage vapour compression heat
        # pumps", Int. J. Refrigeration 55, 168-182 — Table 1.
        comp1.set_attr(eta_s=0.80)   # Ommen 2015, Table 1
        comp2.set_attr(eta_s=0.80)   # Ommen 2015, Table 1
        src_hx.set_attr(dp1=_DP["SRC_HX"][0], dp2=_DP["SRC_HX"][1])
        ihx.set_attr(dp1=_DP["IHX"][0], dp2=_DP["IHX"][1])
        snk_hx.set_attr(dp1=_DP["SNK_HX"][0], dp2=_DP["SNK_HX"][1])
        motor1.set_attr(eta=0.95)    # Ommen 2015, Table 1
        motor2.set_attr(eta=0.95)    # Ommen 2015, Table 1

        nw.solve("design")

        # Second solve: relax pressures and use pinch constraints. For
        # transcritical cycles the high-side pressure is a design variable
        # we hold fixed (not driven by saturation), and the gas-cooler
        # outlet temperature is already pinned by c23.T (cycle 1) or c33.T
        # (cycle 2), so adding td_pinch on the IHX would over-constrain.
        c24.set_attr(p=None)
        if not is_c1_transcritical:
            c22.set_attr(p=None)
        if not is_c2_transcritical:
            c32.set_attr(p=None)
        src_hx.set_attr(ttd_l=5)
        if not (is_c1_transcritical or is_c2_transcritical):
            ihx.set_attr(td_pinch=5)
        snk_hx.set_attr(ttd_l=5)

        nw.solve("design")

        # Reject if subcritical cycle-1 pressure reaches 95 % of critical.
        # Skipped for transcritical cycle 1 — operation above p_crit is the
        # whole point of the cycle there.
        if not is_c1_transcritical:
            p_crit_c1 = PropsSI("Pcrit", fluid_cycle1) / 1e5  # bar
            for conn in [c21, c22, c22c, c23, c24]:
                if conn.p.val >= 0.95 * p_crit_c1:
                    print(f"  SKIP: {fluid_cycle1} at {conn.label} reaches "
                          f"{conn.p.val:.1f} bar >= 95% of p_crit "
                          f"({p_crit_c1:.1f} bar)")
                    return None

        # Reject if either cycle's high-side pressure exceeds the Ommen 2015
        # compressor envelope (no machine in the paper covers that operating
        # point and the cost correlation cannot be evaluated honestly).
        # Bypassed when ``skip_ommen_check=True`` so the feasibility screener
        # can post-process the full data and report which constraint failed.
        p_high_c1 = max(c22.p.val, c22c.p.val, c23.p.val)
        p_high_c2 = max(c32.p.val, c32c.p.val, c33.p.val)
        if not skip_ommen_check:
            if not _check_ommen_p_limit(fluid_cycle1, p_high_c1, "cycle 1"):
                return None
            if not _check_ommen_p_limit(fluid_cycle2, p_high_c2, "cycle 2"):
                return None

        Q_H = c41.m.val * (c43.h.val - c41.h.val)  # kW

        # COP from energy balance (independent of exergy definitions)
        W_shaft_total = abs(comp1.P.val) + abs(comp2.P.val)  # W
        eta_motor = 0.95   # Ommen 2015, Table 1
        W_el = W_shaft_total / eta_motor
        COP = Q_H * 1000 / W_el  # Q_H [kW] → [W]

        # Exergy analysis — classify source water streams based on temperature vs Tamb
        ean = ExergyAnalysis.from_tespy(nw, Tamb=Tamb, pamb=pamb)
        product = {"inputs": ["43"], "outputs": ["41"]}
        Tamb_degC = Tamb - 273.15

        # Source water inlet (11) → system FUEL: water arrives carrying
        # useful exergy and is priced at c = 0 (free reservoir).
        # Source water outlet (13) → system LOSS: whatever exergy remains
        # after the SRC_HX leaves the system unrecovered.
        # Split (rather than the previous paired fuel/loss with T_src vs
        # Tamb conditional) gives a clean, continuous boundary definition
        # that does not flip at T_src = Tamb. Tamb_degC is kept available
        # for downstream patches but no longer drives the classification.
        del Tamb_degC
        # exerpy convention: "inputs" ADD to the category, "outputs" SUBTRACT.
        # 11 in fuel.inputs  → +E_11 to system fuel (water arriving with exergy)
        # 13 in loss.inputs  → +E_13 to system loss (water leaving with exergy)
        fuel = {"inputs": ["e1", "11"], "outputs": []}
        loss = {"inputs": ["13"], "outputs": []}
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
        rho_31 = PropsSI("D", "H", c31.h.val * 1000, "P", c31.p.val * 1e5, fluid_cycle2)
        V_dot_comp1 = c21.m.val / rho_21 * 3600  # m³/h
        V_dot_comp2 = c31.m.val / rho_31 * 3600  # m³/h

        # Shaft powers [kW] (TESPy P.val is in W)
        W_comp1 = abs(comp1.P.val) / 1000
        W_comp2 = abs(comp2.P.val) / 1000

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


def simulate_gas_heater(eta_gas=0.90, T_steam_override=None):
    """
    Build and solve a TESPy gas heater: CombustionChamber + HeatExchanger.

    Uses the Ahrendts chemical exergy database for proper exergy accounting
    of the natural gas (CH4) fuel stream.  The water side uses the same
    boundary conditions as the HTHP sink (saturated water → saturated steam,
    m = 1 kg/s).

    Parameters
    ----------
    eta_gas : float, optional
        Thermal efficiency of the gas heater. Default 0.90 from Ommen et al.
        (2015), "Technical and economic working domains of industrial heat
        pumps: Part 1", Int. J. Refrigeration 55, 168-182, Table 1
        ("Natural gas burner efficiency = 0.9").
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
    Q_H_W = 1.0 * (h_out_w - h_in_w)  # W (m = 1 kg/s)
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
    w1.set_attr(fluid={"H2O": 1}, p=p_water_val, x=0, m=1)
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

    # Full exerpy-format network dump (state points + components +
    # exergy decomposition per stream). Same shape as the HTHP path so
    # downstream serialisation can reuse the same JSON / CSV layout.
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
