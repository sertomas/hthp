"""
screen_cascade.py — Technical-feasibility screening of the CASCADED HTHP.

For every (f1, f2, LS, T_source_in, T_steam) combination:

  1. Run ``simulate_hthp`` with ``skip_ommen_check=True`` so we get the
     thermodynamic state even when one or both compressors fall outside
     Ommen 2015 Table 3 envelopes.
  2. Read suction volumetric flow on cycles 1 and 2.
  3. Scale the whole machine by the *binding* compressor (the one whose
     V_max is hit first):
         k = min(V_max,C1 / V̇_C1 ,  V_max,C2 / V̇_C2)
     Mass flow, V̇'s, Q_H, W_el all scale linearly by k. Pressures,
     T_disch and COP are unchanged by the scaling.
  4. Apply Ommen 2015 Table 3 constraints on each stage (with the same
     +10 % pressure tolerance as in the cost work):
         p_high  ≤ p_max × 1.10
         T_disch ≤ 180 °C
         V_dot   ∈ [V_min, V_max]    (one side automatically at V_max
                                       after scaling; the other must be
                                       ≥ its V_min)
  5. R717 stages are routed to LP cost class if p_high ≤ 28 bar, otherwise
     HP (≤ 50 bar). The LP/HP envelopes are checked accordingly.

No exergy / cost analysis at this stage — pure technical feasibility plus
COP from energy balance and steam mass-flow per machine.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

import pandas as pd
from CoolProp.CoolProp import PropsSI

from config import (
    FLUIDS_C1,
    FLUIDS_C2,
    LIFT_SHARE_RANGE,
    lift_share_to_T34,
)
from models import simulate_hthp


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


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


# ──────────────────────────────────────────────────────────────────────────────
# Operating-condition sweep
# ──────────────────────────────────────────────────────────────────────────────
T_SOURCE_IN_RANGE = [20, 30, 40, 50]    # °C
T_STEAM_RANGE = [100, 110, 120]          # °C — saturated industrial steam

OUTPUT_DIR = os.path.join("results", "screen_cascade")


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


def screen_one(
    f1: str, f2: str, ls: float, T_src: float, T_steam: float
) -> dict:
    """Screen a single cascade configuration. Always returns a dict (the
    ``status`` field carries the failure reason where relevant)."""

    common = {
        "f1": f1, "f2": f2, "ls": ls,
        "T_src": T_src, "T_steam": T_steam,
    }

    # ── Pre-classify thermodynamic infeasibilities before the TESPy call ──
    pre = _pre_classify_failure(f1, f2, ls, T_src, T_steam)
    if pre is not None:
        code, detail = pre
        return {**common, "feasible": False,
                "reason": "no_solve",
                "no_solve_class": code,
                "status": detail}

    T34 = lift_share_to_T34(ls, T_src, T_steam=T_steam)
    sim = simulate_hthp(
        f1, f2,
        T_evap_c2_override=T34,
        T_source_in_override=T_src,
        T_steam_override=T_steam,
        skip_ommen_check=True,
    )
    if sim is None:
        # Pre-checks all passed but TESPy still failed → genuine convergence
        # issue (Newton-iteration didn't settle, bad starting point, etc.).
        return {**common, "feasible": False,
                "reason": "no_solve",
                "no_solve_class": "convergence",
                "status": "TESPy did not converge"}

    sz = sim["sizing"]
    cs1 = sim["cycle_states"]["cycle1"]["points"]
    cs2 = sim["cycle_states"]["cycle2"]["points"]

    # ── State extraction ─────────────────────────────────────────────────
    # Cycle-1 high side: discharge "22" and condenser exit "23"
    p_high_c1 = max(p["p"] for p in cs1)
    p_low_c1 = min(p["p"] for p in cs1)
    h_disch_c1 = next(p["h"] for p in cs1 if p["label"] == "22")
    p_disch_c1 = next(p["p"] for p in cs1 if p["label"] == "22")
    T_disch_c1 = _T_from_p_h(p_disch_c1, h_disch_c1, f1)

    p_high_c2 = max(p["p"] for p in cs2)
    p_low_c2 = min(p["p"] for p in cs2)
    h_disch_c2 = next(p["h"] for p in cs2 if p["label"] == "32")
    p_disch_c2 = next(p["p"] for p in cs2 if p["label"] == "32")
    T_disch_c2 = _T_from_p_h(p_disch_c2, h_disch_c2, f2)

    V_dot_c1 = sz["V_dot_comp1"]   # m³/h, computed at m_steam=1 kg/s
    V_dot_c2 = sz["V_dot_comp2"]
    W_comp1 = sz["W_comp1"]        # kW
    W_comp2 = sz["W_comp2"]

    # ── Resolve Ommen envelope per stage (handles R717 LP/HP routing) ────
    label_c1 = _envelope_label(f1, p_high_c1)
    label_c2 = _envelope_label(f2, p_high_c2)
    p_max_c1, V_min_c1, V_max_c1, _ = COMPRESSOR_SPEC[label_c1]
    p_max_c2, V_min_c2, V_max_c2, _ = COMPRESSOR_SPEC[label_c2]

    # ── Binding-compressor sizing ────────────────────────────────────────
    k = min(V_max_c1 / V_dot_c1, V_max_c2 / V_dot_c2)
    V_dot_c1_scaled = V_dot_c1 * k
    V_dot_c2_scaled = V_dot_c2 * k
    m_steam_scaled = 1.0 * k  # original simulation runs at m_steam = 1 kg/s

    # Q_H ∝ m_steam (full latent heat at saturated steam) and W_comp ∝ m
    Q_H_full_kW = (
        # at m_steam=1 kg/s, Q_H = 1 × Δh_steam(p_water)
        # we recompute it from the saturation enthalpy difference
        (PropsSI("H", "P", PropsSI("P", "T", T_steam + 273.15, "Q", 0, "water"),
                 "Q", 1, "water")
         - PropsSI("H", "P", PropsSI("P", "T", T_steam + 273.15, "Q", 0, "water"),
                   "Q", 0, "water")) / 1e3  # → kW per kg/s of steam
    )
    Q_H_kW = Q_H_full_kW * m_steam_scaled
    W_el_kW = (W_comp1 + W_comp2) * k / 0.95   # divide by motor eta
    COP = Q_H_kW / W_el_kW

    # ── Feasibility flags ───────────────────────────────────────────────
    p_OK_c1 = p_high_c1 <= p_max_c1 * OMMEN_P_TOL
    p_OK_c2 = p_high_c2 <= p_max_c2 * OMMEN_P_TOL
    T_OK_c1 = T_disch_c1 <= T_DISCH_MAX
    T_OK_c2 = T_disch_c2 <= T_DISCH_MAX
    V_OK_c1 = (V_min_c1 * 0.999) <= V_dot_c1_scaled <= (V_max_c1 * 1.001)
    V_OK_c2 = (V_min_c2 * 0.999) <= V_dot_c2_scaled <= (V_max_c2 * 1.001)
    feasible = all([p_OK_c1, p_OK_c2, T_OK_c1, T_OK_c2, V_OK_c1, V_OK_c2])

    fail_reasons = [name for name, ok in [
        ("p_c1", p_OK_c1), ("p_c2", p_OK_c2),
        ("T_c1", T_OK_c1), ("T_c2", T_OK_c2),
        ("V_c1", V_OK_c1), ("V_c2", V_OK_c2),
    ] if not ok]

    return {
        **common,
        "no_solve_class": "",
        "label_c1": label_c1,
        "label_c2": label_c2,
        "V_dot_c1_target [m3/h]": round(V_dot_c1_scaled, 1),
        "V_dot_c2_target [m3/h]": round(V_dot_c2_scaled, 1),
        "binding": "C1" if (V_max_c1 / V_dot_c1) <= (V_max_c2 / V_dot_c2) else "C2",
        "scale_k": round(k, 4),
        "m_steam [kg/s]": round(m_steam_scaled, 4),
        "m_steam [kg/h]": round(m_steam_scaled * 3600, 1),
        "Q_H [kW]": round(Q_H_kW, 1),
        "W_el [kW]": round(W_el_kW, 2),
        "COP": round(COP, 3),
        "p_low_c1 [bar]": round(p_low_c1, 2),
        "p_high_c1 [bar]": round(p_high_c1, 2),
        "T_disch_c1 [°C]": round(T_disch_c1, 1),
        "p_low_c2 [bar]": round(p_low_c2, 2),
        "p_high_c2 [bar]": round(p_high_c2, 2),
        "T_disch_c2 [°C]": round(T_disch_c2, 1),
        "p_OK_c1": p_OK_c1, "p_OK_c2": p_OK_c2,
        "T_OK_c1": T_OK_c1, "T_OK_c2": T_OK_c2,
        "V_OK_c1": V_OK_c1, "V_OK_c2": V_OK_c2,
        "feasible": feasible,
        "reason": ", ".join(fail_reasons) if not feasible else "",
        "status": "ok",
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    n_total = (len(FLUIDS_C1) * len(FLUIDS_C2) * len(LIFT_SHARE_RANGE)
               * len(T_SOURCE_IN_RANGE) * len(T_STEAM_RANGE))
    n_done = 0

    for f1 in FLUIDS_C1:
        for f2 in FLUIDS_C2:
            for ls in LIFT_SHARE_RANGE:
                for T_src in T_SOURCE_IN_RANGE:
                    for T_steam in T_STEAM_RANGE:
                        n_done += 1
                        tag = (f"[{n_done:>3d}/{n_total}] "
                               f"{f1}/{f2:<6s} LS={ls:.2f} "
                               f"T_src={T_src} T_steam={T_steam}")
                        res = screen_one(f1, f2, ls, T_src, T_steam)
                        rows.append(res)
                        if res.get("status") == "ok":
                            mark = "OK" if res["feasible"] else "--"
                            print(f"{tag}  {mark}  COP={res['COP']:.2f}  "
                                  f"m_steam={res['m_steam [kg/h]']:.0f} kg/h  "
                                  f"reason={res['reason'] or '-'}")
                        else:
                            print(f"{tag}  --  {res['status']}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "cascade_screen.csv"), index=False)
    df[df["feasible"] == True].to_csv(
        os.path.join(OUTPUT_DIR, "cascade_feasible_only.csv"), index=False
    )

    n_feas = int((df["feasible"] == True).sum())
    print()
    print("=" * 90)
    print(f"Total scenarios: {len(df)}")
    print(f"Feasible:        {n_feas} ({100 * n_feas / len(df):.1f} %)")
    print("=" * 90)
    print(f"\nCSV written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
