"""
Phase-aware heat-exchanger area calculation.

For each section delimited by TESPy's MovingBoundaryHeatExchanger
phase-change breakpoints, the appropriate overall heat-transfer coefficient
U is selected from a phase-pair lookup (rows: hot-side phase; columns:
cold-side phase) and the area contribution is ``Q / (LMTD * U)``.

Single-phase / two-phase U values come from Ommen 2015 (single-stage
heat-pump study, Table A.3). The remaining combinations (two-phase /
two-phase and two-phase / gas in both directions) are not reported by
Ommen and are engineering assumptions retained for compatibility with
the cascaded topology, where the IHX has two-phase fluid on both sides.

Ported from the hthp_optimization repository
(calculate_heatexchanger_area.py), adapted to the TESPy 0.9.x public API
(``hex.calc_sections()`` replaces the older ``_assign_steps`` /
``_get_moving_steps`` helpers).

References
----------
Ommen, T.; Markussen, W.B.; Elmegaard, B. Technical and economic working
domains of industrial heat pumps: Part 1 - Single stage vapour
compression heat pumps. Int. J. Refrigeration 2015, 55, 168-182.
"""

import numpy as np

from tespy.tools.fluid_properties import phase_mix_ph


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
# Ommen and are taken over verbatim from the hthp_optimization repo.
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

_PHASE_TO_INT = {"l": 0, "tp": 1, "g": 2}


def _phase_at_step(conn, h_step):
    """Phase index (0=liquid, 1=two-phase, 2=gas) at enthalpy ``h_step``
    for the stream described by ``conn`` (pressure / fluid taken from the
    connection; pr=1 in every HX so pressure is constant along the side).

    Falls back to two-phase (1) when CoolProp returns ``"state not
    recognised"``, which happens for some refrigerants (notably R717 near
    saturation at high reduced pressure) when the enthalpy step lands
    exactly on the saturation boundary. The two-phase fallback is
    consistent with ``_phase_per_section`` snapping straddled sections to
    the bordering single-phase value, which keeps the sizing conservative.
    """
    label = phase_mix_ph(conn.p.val_SI, h_step, conn.fluid_data, conn.mixing_rule)
    return _PHASE_TO_INT.get(label, 1)


def _phase_per_section(phase_steps):
    """Collapse a per-step phase array (length N) into a per-section
    phase array (length N-1).

    Sections that lie cleanly inside one phase keep their phase. Sections
    that straddle a saturation boundary (0<->1 or 1<->2) are snapped to
    the bordering single-phase value (0 or 2 respectively) — same
    convention as the hthp_optimization implementation. This is the
    conservative choice for sizing: single-phase U values are smaller,
    so the section contributes a larger area.
    """
    t_from = phase_steps[:-1].copy()
    t_to = phase_steps[1:]

    mask_01 = ((t_from == 0) & (t_to == 1)) | ((t_from == 1) & (t_to == 0))
    t_from[mask_01] = 0

    mask_12 = ((t_from == 1) & (t_to == 2)) | ((t_from == 2) & (t_to == 1))
    t_from[mask_12] = 2

    return t_from


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

    phase_hot = np.array([_phase_at_step(hot_in, h) for h in h_steps_hot])
    phase_cold = np.array([_phase_at_step(cold_in, h) for h in h_steps_cold])

    sec_hot = _phase_per_section(phase_hot)
    sec_cold = _phase_per_section(phase_cold)

    area = 0.0
    for Q, td_log, ph, pc in zip(Q_per_section, td_log_per_section, sec_hot, sec_cold):
        U = U_VALUES[(int(ph), int(pc))]
        area += Q / (td_log * U)
    return area
