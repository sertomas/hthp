"""
case_steam.py — Cascade HTHP feasibility screen at fixed Q_H.

For a given saturated-steam sink temperature (default 110 °C, ≈ 1.43 bar;
the same pipeline runs for 100 / 110 / 120 °C from main.py), sweep the
design space:

    fluid pair  ∈  FLUIDS_C1 × FLUIDS_C2   (3 × 2 = 6 combinations)
    T_source_in ∈  T_SOURCE_IN_RANGE        ([20, 30, 40, 50, 60] °C)
    lift share  ∈  LIFT_SHARE_RANGE         ([0.30, 0.40, 0.50, 0.60, 0.70])

Total: 6 × 5 × 5 = 150 cases per T_steam.

Every design runs at native scale (``config.M_STEAM`` kg/s on the sink
side) in ``models.simulate_hthp``, giving Q_H = M_STEAM · Δh_vap of water
at T_steam. Pressures, T_disch, COP, η_Lorenz are scale-invariant.
No virtual rescaling — all reported values are at the actual TESPy-converged
scale.

Out-of-Ommen-envelope designs are flagged, not rejected. Four-state
classification (in plot_common.classify_status):

    OK      — within Ommen 2015 Table 3 envelope (p, T, V̇).
    V_ONLY  — V̇ outside Ommen but p, T compliant. Operable per IEA HPT
              Annex 58 supplier evidence (e.g. Mayekawa, Hybrid Energy at
              MWth scale routinely exceed Ommen V_max). Cost extrapolation
              is the only risk and is cross-checked in stage 5.
    HARD    — pressure or discharge-temperature violation. Genuinely
              problematic (compressor pressure rating, lubricant stability).
    NOSOLVE — thermodynamic infeasibility (T_crit, P_crit) or TESPy
              non-convergence.

Outputs:
    results/case_steam_<int(T_steam)>/case_steam_<int(T_steam)>.csv
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
    M_STEAM,
    T_SOURCE_IN_RANGE,
    T_STEAM_CASE_DEFAULT,
    lift_share_to_T34,
    m_steam_label,
    p_water_for_T_steam,
)
from models import simulate_hthp
from plot_common import classify_status
from screen_cascade import (
    COMPRESSOR_SPEC,
    T_DISCH_MAX,
    envelope_flags,
    _T_from_p_h,
    _pre_classify_failure,
)


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


def _paths_for(T_steam):
    """Return (output_dir, csv_path) for a given case-study T_steam."""
    from config import case_results_dir
    out_dir = case_results_dir(T_steam)
    csv = os.path.join(out_dir, f"case_steam_{int(T_steam)}.csv")
    return out_dir, csv


# ── Steam latent heat at the given T_steam (kJ per kg of steam) ──────────────
def _dh_steam_kJ_per_kg(T_steam_degC: float) -> float:
    p_pa = p_water_for_T_steam(T_steam_degC) * 1e5
    h_v = PropsSI("H", "P", p_pa, "Q", 1, "water")
    h_l = PropsSI("H", "P", p_pa, "Q", 0, "water")
    return (h_v - h_l) / 1e3  # J/kg → kJ/kg


def _row_common(f1, f2, ls, T_src, T_steam):
    return {"f1": f1, "f2": f2, "ls": ls,
            "T_src": T_src, "T_steam": T_steam}


def _nosolve_row(common, code, detail):
    return {**common,
            "feasible": False,
            "reason": "no_solve",
            "no_solve_class": code,
            "status": detail,
            "status_4state": "NOSOLVE"}


def screen_one_case(
    f1: str, f2: str, ls: float, T_src: float,
    T_steam: float = T_STEAM_CASE_DEFAULT,
    dh_steam_kJ_per_kg: float | None = None,
) -> dict:
    """Screen one cascade design at native scale (config.M_STEAM kg/s).
    Q_H is determined by the cycle physics: Q_H = M_STEAM × Δh_vap(T_steam)."""

    common = _row_common(f1, f2, ls, T_src, T_steam)

    # ── Pre-classify thermodynamic infeasibilities ──────────────────────
    pre = _pre_classify_failure(f1, f2, ls, T_src, T_steam)
    if pre is not None:
        code, detail = pre
        return _nosolve_row(common, code, detail)

    T34 = lift_share_to_T34(ls, T_src, T_steam=T_steam)
    sim = simulate_hthp(
        f1, f2,
        T_evap_c2_override=T34,
        T_source_in_override=T_src,
        T_steam_override=T_steam,
        source_mode="fixed_mass_flow",
        skip_ommen_check=True,
    )
    if sim is None:
        return _nosolve_row(common, "convergence", "TESPy did not converge")

    sz = sim["sizing"]
    cs1 = sim["cycle_states"]["cycle1"]["points"]
    cs2 = sim["cycle_states"]["cycle2"]["points"]

    # ── Cycle high-side state extraction ────────────────────────────────
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

    # Native sizing — simulation runs at config.M_STEAM kg/s. No post-scaling:
    # the case study reports values at the actual TESPy-converged scale.
    V_dot_c1 = sz["V_dot_comp1"]   # m³/h
    V_dot_c2 = sz["V_dot_comp2"]
    W_comp1 = sz["W_comp1"]        # kW
    W_comp2 = sz["W_comp2"]

    if dh_steam_kJ_per_kg is None:
        dh_steam_kJ_per_kg = _dh_steam_kJ_per_kg(T_steam)
    m_steam = M_STEAM
    Q_H_native_kW = dh_steam_kJ_per_kg * m_steam

    eta_motor = 0.95
    W_el_kW = (W_comp1 + W_comp2) / eta_motor

    # COP is scale-invariant — use the simulation value directly (it
    # already includes pump powers via models.simulate_hthp).
    COP = sim["COP"]

    # ── Lorenz benchmark ────────────────────────────────────────────────
    T_src_in_K = T_src + 273.15
    T_src_out_K = sim["T_source_out"] + 273.15
    T_bar_source = 0.5 * (T_src_in_K + T_src_out_K)
    # Sink: water enters as saturated liquid (x=0) at T_steam, exits as
    # saturated vapor (x=1) at T_steam → isothermal sink at T_steam.
    T_bar_sink = T_steam + 273.15
    if T_bar_sink <= T_bar_source:
        eta_Lorenz = float("nan")  # would yield negative COP_Lorenz
    else:
        COP_Lorenz = T_bar_sink / (T_bar_sink - T_bar_source)
        eta_Lorenz = COP / COP_Lorenz

    # ── Ommen envelope flags (hard Table-3 limits, no tolerance) ─────────
    flags, reason, feasible, label_c1, label_c2 = envelope_flags(
        f1, f2, p_high_c1, p_high_c2, T_disch_c1, T_disch_c2,
        V_dot_c1, V_dot_c2, COMPRESSOR_SPEC, T_DISCH_MAX)

    row = {
        **common,
        "no_solve_class": "",
        "label_c1": label_c1,
        "label_c2": label_c2,
        "Q_H_kW": round(Q_H_native_kW, 2),
        "m_steam [kg/s]": round(m_steam, 4),
        "m_steam [kg/h]": round(m_steam * 3600, 1),
        "V_dot_c1_target [m3/h]": round(V_dot_c1, 1),
        "V_dot_c2_target [m3/h]": round(V_dot_c2, 1),
        "W_comp1 [kW]": round(W_comp1, 2),
        "W_comp2 [kW]": round(W_comp2, 2),
        "W_el [kW]": round(W_el_kW, 2),
        "scale_k": 1.0,   # native scale; field kept for backward compatibility
        "COP": round(COP, 3),
        "eta_Lorenz": round(eta_Lorenz, 3) if eta_Lorenz == eta_Lorenz else "",  # NaN→""
        "p_low_c1 [bar]": round(p_low_c1, 2),
        "p_high_c1 [bar]": round(p_high_c1, 2),
        "T_disch_c1 [°C]": round(T_disch_c1, 1),
        "p_low_c2 [bar]": round(p_low_c2, 2),
        "p_high_c2 [bar]": round(p_high_c2, 2),
        "T_disch_c2 [°C]": round(T_disch_c2, 1),
        **flags,
        "feasible": feasible,
        "reason": reason,
        "status": "ok",
    }
    row["status_4state"] = classify_status(row)

    # TODO Stage 4 — exergy sidecar JSON dump for OK/V_ONLY rows would go here:
    #   results/case_steam_<int(T_steam)>/exergy/<f1>_<f2>_LS<ls_pct>_Tsrc<T_src>.json
    # TODO Stage 5 — compute_pec(sim, k) gated behind --with-pec; columns
    #   PEC_comp1_kEUR, PEC_comp2_kEUR, ..., PEC_total_kEUR appended here.

    return row


# ── Self-test ───────────────────────────────────────────────────────────────

def _self_test():
    """Cheap sanity tests run by ``--check`` before the full sweep."""
    print("Running self-test ...")

    # 1. Saturation pressure sanity
    p_bar = p_water_for_T_steam(110.0)
    print(f"  p_water(110 °C) = {p_bar:.4f} bar  (expected ≈ 1.4327 bar)")
    assert abs(p_bar - 1.4327) / 1.4327 < 0.01, \
        f"p_water_for_T_steam(110) deviates >1% from 1.4327 bar (got {p_bar})"

    # 2. Latent heat at 110 °C
    dh = _dh_steam_kJ_per_kg(110.0)
    print(f"  Δh_steam(110 °C) = {dh:.1f} kJ/kg  (expected ≈ 2230 kJ/kg)")
    assert 2200 < dh < 2260, f"Δh_steam at 110 °C out of band (got {dh})"

    # 3. classify_status precedence
    base = dict(p_OK_c1=True, p_OK_c2=True, T_OK_c1=True, T_OK_c2=True,
                V_OK_c1=True, V_OK_c2=True, feasible=True, reason="",
                no_solve_class="")
    assert classify_status(base) == "OK"

    v_only = dict(base, V_OK_c1=False, feasible=False,
                  reason="V_c1")
    assert classify_status(v_only) == "V_ONLY", \
        f"V_ONLY misclassified as {classify_status(v_only)}"

    hard = dict(base, p_OK_c1=False, V_OK_c1=False, feasible=False,
                reason="p_c1, V_c1")
    assert classify_status(hard) == "HARD", \
        f"HARD misclassified as {classify_status(hard)} (must beat V_ONLY)"

    nosolve = dict(feasible=False, reason="no_solve",
                   no_solve_class="T_crit_c2",
                   p_OK_c1="", p_OK_c2="", T_OK_c1="", T_OK_c2="",
                   V_OK_c1="", V_OK_c2="")
    assert classify_status(nosolve) == "NOSOLVE", \
        f"NOSOLVE misclassified as {classify_status(nosolve)}"

    print("  ✓ classify_status precedence: OK / V_ONLY / HARD / NOSOLVE")
    print("Self-test passed.")


# ── Main sweep ──────────────────────────────────────────────────────────────

def main(T_steam=None):
    """Run the 150-case screen at the given steam temperature.

    Parameters
    ----------
    T_steam : float, optional
        Sink steam temperature [°C]. Defaults to ``T_STEAM_CASE_DEFAULT``
        (110 °C). Pass 100 / 120 to write under
        ``results/case_steam_100/`` or ``results/case_steam_120/``.
    """
    if T_steam is None:
        T_steam = T_STEAM_CASE_DEFAULT
    out_dir, csv_path = _paths_for(T_steam)
    os.makedirs(out_dir, exist_ok=True)

    dh = _dh_steam_kJ_per_kg(T_steam)
    Q_H_native = dh * M_STEAM
    print(f"Case study: T_steam = {T_steam:.0f} °C, "
          f"p = {p_water_for_T_steam(T_steam):.3f} bar, "
          f"Δh = {dh:.1f} kJ/kg, Q_H_native = {Q_H_native:.0f} kW "
          f"({m_steam_label()})")
    print(f"Pairs: {len(FLUIDS_C1)} × {len(FLUIDS_C2)} = "
          f"{len(FLUIDS_C1) * len(FLUIDS_C2)}; "
          f"sweep over {len(T_SOURCE_IN_RANGE)} T_src × "
          f"{len(LIFT_SHARE_RANGE)} LS")

    rows = []
    n_total = (len(FLUIDS_C1) * len(FLUIDS_C2)
               * len(LIFT_SHARE_RANGE) * len(T_SOURCE_IN_RANGE))
    n_done = 0

    for f1 in FLUIDS_C1:
        for f2 in FLUIDS_C2:
            for ls in LIFT_SHARE_RANGE:
                for T_src in T_SOURCE_IN_RANGE:
                    n_done += 1
                    tag = (f"[{n_done:>3d}/{n_total}] "
                           f"{f1}/{f2:<6s} LS={ls:.2f} T_src={T_src}")
                    res = screen_one_case(
                        f1, f2, ls, T_src,
                        T_steam=T_steam,
                        dh_steam_kJ_per_kg=dh,
                    )
                    rows.append(res)
                    state = res.get("status_4state", "?")
                    cop = res.get("COP", "—")
                    if state == "NOSOLVE":
                        print(f"{tag}  NOSOLVE  ({res.get('no_solve_class', '')})")
                    else:
                        print(f"{tag}  {state:7s}  COP={cop}  "
                              f"reason={res.get('reason', '') or '-'}")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # Summary
    print()
    print("=" * 90)
    counts = df["status_4state"].value_counts().to_dict()
    for state in ("OK", "V_ONLY", "HARD", "NOSOLVE"):
        n = counts.get(state, 0)
        pct = 100.0 * n / len(df) if len(df) else 0.0
        print(f"  {state:8s} : {n:>3d}  ({pct:5.1f} %)")
    print("=" * 90)

    # Q_H consistency assertion: every solved design must produce the same
    # native Q_H (within numerical tolerance), since m_steam = config.M_STEAM
    # is fixed and Q_H = m_steam × Δh_vap depends only on T_steam.
    non_nosolve = df[df["status_4state"] != "NOSOLVE"]
    if not non_nosolve.empty:
        max_dev = (non_nosolve["Q_H_kW"] - Q_H_native).abs().max()
        print(f"  Q_H consistency: max |Q_H − {Q_H_native:.0f}| = "
              f"{max_dev:.4f} kW (target < 0.1 kW)")
        if max_dev > 0.1:
            print("  ! WARNING: Q_H consistency check failed")

    print(f"\nCSV written to {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        _self_test()
    else:
        main()
