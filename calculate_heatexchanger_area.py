"""
Phase-aware heat-exchanger area calculation.

For each section delimited by TESPy's MovingBoundaryHeatExchanger
phase-change breakpoints, the appropriate overall heat-transfer coefficient
U is selected from a phase-pair lookup (rows: hot-side phase; columns:
cold-side phase) and the area contribution is ``Q / (LMTD * U)``.

Single-phase / two-phase U values come from Ommen 2015 (single-stage
heat-pump study, Table A.3). The remaining combinations (two-phase /
two-phase and two-phase / gas in both directions) are not reported by
Ommen and are engineering assumptions for the cascaded topology, where
the IHX has two-phase fluid on both sides.

Section heat duties and LMTDs come from TESPy's ``hex.calc_sections()``.

References
----------
Ommen, T.; Markussen, W.B.; Elmegaard, B. Technical and economic working
domains of industrial heat pumps: Part 1 - Single stage vapour
compression heat pumps. Int. J. Refrigeration 2015, 55, 168-182.
"""

import numpy as np

from tespy.tools.fluid_properties import phase_mix_ph, h_mix_pQ


# Overall heat-transfer coefficients U [W/(m^2 K)] by phase pair.
# Phase indices: 0 = liquid, 1 = two-phase, 2 = gas.
# Ommen 2015 contributes the five entries that involve water (one of the
# two sides is therefore always liquid):
#   (0,0) condenser subcooling      U_sub  = 1494
#   (0,1) evaporator two-phase zone U_evap = 1483
#   (0,2) evaporator superheat zone U_sh,e =  380
#   (1,0) condenser two-phase zone  U_cond = 3696
#   (2,0) condenser desuperheat     U_sh,c =  466
# The other four entries (1,1), (1,2), (2,1), (2,2) are not given by
# Ommen and are engineering assumptions not reported in Ommen.
U_VALUES = {
    (0, 0): 1494,
    (0, 1): 1483,
    (0, 2):  380,
    (1, 0): 3696,
    (1, 1): 2500,  # assumed (not in Ommen)
    (1, 2): 1000,  # assumed (not in Ommen)
    (2, 0):  466,
    (2, 1):  750,  # assumed (not in Ommen)
    (2, 2):   35,  # assumed (not in Ommen)
}

def _sat_enthalpies(conn):
    """Saturation liquid (x=0) and vapor (x=1) specific enthalpies [J/kg]
    at the connection's pressure (constant along each HX side, pr=1).
    Returns ``(None, None)`` when the side is supercritical and no
    saturation enthalpies exist.
    """
    p = conn.p.val_SI
    try:
        return (h_mix_pQ(p, 0.0, conn.fluid_data, conn.mixing_rule),
                h_mix_pQ(p, 1.0, conn.fluid_data, conn.mixing_rule))
    except Exception:
        return None, None


def _phase_of(h, h_L, h_V):
    """Phase index (0=liquid, 1=two-phase, 2=gas) from the enthalpy
    relative to the side's saturation enthalpies. Supercritical
    (``h_L is None``) is treated as gas. Deterministic: classification
    depends only on the saturation positions, never on a CoolProp phase
    query at a section boundary (which lands on the saturation line and
    fails for R717 near saturation).
    """
    if h_L is None:
        return 2
    if h <= h_L:
        return 0
    if h >= h_V:
        return 2
    return 1


def get_hex_area(hex_component):
    """Heat-exchanger area [m^2] using the phase-pair U lookup.

    Uses TESPy's ``calc_sections()`` to obtain heat duty and LMTD per
    section (breakpoints inserted at every saturation crossing on either
    side), classifies the phase pair on each section, and integrates
    ``area += Q_i / (LMTD_i * U_phase_pair_i)``.
    """
    Q_sections, _, _, Q_per_section, td_log_per_section = hex_component.calc_sections()

    hot_in = hex_component.inl[0]
    hot_out = hex_component.outl[0]
    cold_in = hex_component.inl[1]

    # Same expressions TESPy uses internally in _get_T_at_steps to map
    # cumulative Q back onto enthalpies on each side.
    h_steps_hot = hot_out.h.val_SI + Q_sections / hot_in.m.val_SI
    h_steps_cold = cold_in.h.val_SI + Q_sections / cold_in.m.val_SI

    # calc_sections() inserts a breakpoint at every saturation crossing on
    # either side, so each section is single-phase on each side. Classify
    # each section by its midpoint enthalpy against that side's saturation
    # enthalpies — deterministic and free of boundary/CoolProp ambiguity.
    hL_hot, hV_hot = _sat_enthalpies(hot_in)
    hL_cold, hV_cold = _sat_enthalpies(cold_in)

    area = 0.0
    for i, (Q, td_log) in enumerate(zip(Q_per_section, td_log_per_section)):
        h_mid_hot = 0.5 * (h_steps_hot[i] + h_steps_hot[i + 1])
        h_mid_cold = 0.5 * (h_steps_cold[i] + h_steps_cold[i + 1])
        ph = _phase_of(h_mid_hot, hL_hot, hV_hot)
        pc = _phase_of(h_mid_cold, hL_cold, hV_cold)
        area += Q / (td_log * U_VALUES[(ph, pc)])
    return area
