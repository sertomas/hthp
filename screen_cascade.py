"""
Ommen 2015 Table 3 envelope helpers.

Compressor envelopes (p, V̇, T_disch limits), envelope-class routing for
R717 (LP vs HP), state-point conversion, and a cheap pre-feasibility check
that mirrors the early-exit conditions inside ``simulate_hthp``.

Imported by ``case_steam.py`` (during the 150-case TESPy screen) and by
``reclassify_modern.py`` (when relabelling a screened design under modern
compressor envelopes).
"""

from __future__ import annotations

from CoolProp.CoolProp import PropsSI

from config import lift_share_to_T34


# ──────────────────────────────────────────────────────────────────────────────
# Compressor envelope per Ommen 2015 Table 3.
# Tuple: (p_max [bar], V_min [m³/h], V_max [m³/h], T_disch_max [°C])
# R600 / R1270 are not in Ommen Table 3; treated as Type 2 (matches the
# proxy mapping in economics._FLUID_COST_TYPE).
# ──────────────────────────────────────────────────────────────────────────────
COMPRESSOR_SPEC = {
    # label   : (p_max, V_min, V_max, T_disch_max)
    "R134a":   (28,    5,     280,   180),
    "R290":    (28,    5,     280,   180),
    "R600a":   (28,    5,     280,   180),
    "R600":    (28,    5,     280,   180),
    "R1270":   (28,    5,     280,   180),
    "R717-LP": (28,    5,     180,   180),
    "R717-HP": (50,    90,    200,   180),
    "R744":    (140,   6,     25,    180),
}

OMMEN_P_TOL = 1.10
T_DISCH_MAX = 180.0  # °C, Ommen 2015 Table 3 oil-degradation limit


def _r717_class_for(p_high_bar: float) -> str:
    """Route an R717 stage to the LP envelope (≤ 28 bar) or HP (≤ 50 bar)."""
    if p_high_bar <= 28.0:
        return "R717-LP"
    return "R717-HP"


def _envelope_label(fluid: str, p_high_bar: float) -> str:
    """Resolve which Ommen Table 3 envelope to apply to this stage."""
    return _r717_class_for(p_high_bar) if fluid == "R717" else fluid


def _T_from_p_h(p_bar: float, h_kJkg: float, fluid: str) -> float:
    return PropsSI("T", "P", p_bar * 1e5, "H", h_kJkg * 1e3, fluid) - 273.15


_PINCH = 5.0  # K — must match the value used inside simulate_hthp


def _pre_classify_failure(f1, f2, ls, T_src, T_steam):
    """Run the cheap thermodynamic pre-checks that ``simulate_hthp`` does
    internally, but return a *named* failure reason. Returns ``None`` if
    all pre-checks pass (i.e. we should attempt the TESPy solve)."""
    T34 = lift_share_to_T34(ls, T_src, T_steam=T_steam)
    T_cond_c1_est = T34 + _PINCH
    T_cond_c2_est = T_steam + _PINCH

    T_crit_c1 = PropsSI("Tcrit", f1) - 273.15
    T_crit_c2 = PropsSI("Tcrit", f2) - 273.15
    p_crit_c1 = PropsSI("Pcrit", f1) / 1e5

    if T_cond_c2_est >= T_crit_c2:
        return ("T_crit_c2",
                f"f2={f2}: T_cond_c2={T_cond_c2_est:.1f} °C ≥ T_crit={T_crit_c2:.1f} °C")
    if T_cond_c1_est >= T_crit_c1:
        return ("T_crit_c1",
                f"f1={f1}: T_cond_c1={T_cond_c1_est:.1f} °C ≥ T_crit={T_crit_c1:.1f} °C")

    p_cond_c1 = PropsSI("P", "T", T_cond_c1_est + 273.15, "Q", 1, f1) / 1e5
    if p_cond_c1 >= 0.95 * p_crit_c1:
        return ("Pcrit_c1",
                f"f1={f1}: p_cond_c1={p_cond_c1:.1f} bar ≥ 0.95·p_crit={0.95 * p_crit_c1:.1f} bar")

    return None
