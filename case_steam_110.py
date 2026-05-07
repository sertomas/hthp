"""
case_steam_110.py — Cascade HTHP screening: steam at 110 °C, fixed Q_H.

For one specific industrial use case (saturated steam delivered at 110 °C,
≈ 1.43 bar) sweep the design space:

    fluid pair  ∈  FLUIDS_C1 × FLUIDS_C2   (3 × 3 = 9 combinations)
    T_source_in ∈  T_SOURCE_IN_RANGE        ([20, 30, 40, 50, 60] °C)
    lift share  ∈  LIFT_SHARE_RANGE         ([0.30, 0.40, 0.50, 0.60, 0.70])

Every design is sized to deliver ``Q_H_NOMINAL_KW`` (1 MWth by default).
The simulation in ``models.simulate_hthp`` runs natively at m_steam = 1 kg/s;
linear scaling by ``k = m_steam_target / 1`` is exact for V̇, m, W_comp,
Q_H. Pressures, T_disch, COP, η_Lorenz are scale-invariant.

Out-of-Ommen-envelope designs are flagged, not rejected. Four-state
classification (in plot_common.classify_status):

    OK      — within Ommen 2015 Table 3 envelope (p, T, V̇).
    V_ONLY  — V̇ outside Ommen but p, T compliant. Operable per IEA HPT
              Annex 58 supplier evidence (e.g. Mayekawa, Hybrid Energy at
              MWth scale routinely exceed Ommen V_max). Cost extrapolation
              is the only risk and is cross-checked in step 5.
    HARD    — pressure or discharge-temperature violation. Genuinely
              problematic (compressor pressure rating, lubricant stability).
    NOSOLVE — thermodynamic infeasibility (T_crit, P_crit) or TESPy
              non-convergence.

Forward-compatibility hooks (deferred):
    - exergy sidecar JSON (per OK / V_ONLY case) for step 4 sensitivities.
    - compute_pec(sim, k) PEC stub gated behind --with-pec.
    - data/annex58_pec_bands.csv reader for step 5. Expected schema:
          compressor_type, T_source_min, T_source_max,
          T_sink_min, T_sink_max,
          EUR_per_kW_low, EUR_per_kW_high, source

Outputs:
    results/case_steam_110/case_steam_110.csv
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
    Q_H_NOMINAL_KW,
    T_SOURCE_IN_RANGE,
    T_STEAM_CASE_110,
    lift_share_to_T34,
    p_water_for_T_steam,
)
from models import simulate_hthp
from plot_common import classify_status
from screen_cascade import (
    COMPRESSOR_SPEC,
    OMMEN_P_TOL,
    T_DISCH_MAX,
    _T_from_p_h,
    _envelope_label,
    _pre_classify_failure,
)


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


# Default output paths for the legacy 110 °C case. These remain importable
# so existing entry points still work, but main(T_steam=...) now derives
# its own paths via case_results_dir() — supporting 100 / 110 / 120 °C.
OUTPUT_DIR = os.path.join("results", "case_steam_110")
CSV_PATH = os.path.join(OUTPUT_DIR, "case_steam_110.csv")


def _paths_for(T_steam):
    """Return (output_dir, csv_path) for a given case-study T_steam."""
    from config import case_results_dir
    out_dir = case_results_dir(T_steam)
    csv = os.path.join(out_dir, f"case_steam_{int(T_steam)}.csv")
    return out_dir, csv


# ── Steam latent-heat constant for 110 °C (kJ per kg of steam) ──────────────
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


def screen_one_case_steam_110(
    f1: str, f2: str, ls: float, T_src: float,
    T_steam: float = T_STEAM_CASE_110,
    Q_H_target_kW: float = Q_H_NOMINAL_KW,
    dh_steam_kJ_per_kg: float | None = None,
) -> dict:
    """Screen one cascade design at fixed Q_H. Always returns a dict."""

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

    # Per-(kg/s steam) sizing — simulation runs at m_steam = 1 kg/s
    V_dot_c1_unit = sz["V_dot_comp1"]   # m³/h
    V_dot_c2_unit = sz["V_dot_comp2"]
    W_comp1_unit = sz["W_comp1"]        # kW
    W_comp2_unit = sz["W_comp2"]
    W_src_pump_unit = sz["W_src_pump"]
    W_snk_pump_unit = sz["W_snk_pump"]

    # ── Fixed-Q_H scaling ───────────────────────────────────────────────
    if dh_steam_kJ_per_kg is None:
        dh_steam_kJ_per_kg = _dh_steam_kJ_per_kg(T_steam)
    Q_H_per_unit_kW = dh_steam_kJ_per_kg  # Q_H = m_steam · Δh; m_steam=1 kg/s
    k = Q_H_target_kW / Q_H_per_unit_kW   # = m_steam_target [kg/s]

    V_dot_c1 = V_dot_c1_unit * k
    V_dot_c2 = V_dot_c2_unit * k
    W_comp1 = W_comp1_unit * k
    W_comp2 = W_comp2_unit * k
    W_src_pump = W_src_pump_unit * k
    W_snk_pump = W_snk_pump_unit * k
    m_steam = 1.0 * k

    eta_motor = 0.95
    W_el_kW = (W_comp1 + W_comp2 + W_src_pump + W_snk_pump) / eta_motor

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

    # ── Ommen envelope flags ────────────────────────────────────────────
    label_c1 = _envelope_label(f1, p_high_c1)
    label_c2 = _envelope_label(f2, p_high_c2)
    p_max_c1, V_min_c1, V_max_c1, _ = COMPRESSOR_SPEC[label_c1]
    p_max_c2, V_min_c2, V_max_c2, _ = COMPRESSOR_SPEC[label_c2]

    p_OK_c1 = p_high_c1 <= p_max_c1 * OMMEN_P_TOL
    p_OK_c2 = p_high_c2 <= p_max_c2 * OMMEN_P_TOL
    T_OK_c1 = T_disch_c1 <= T_DISCH_MAX
    T_OK_c2 = T_disch_c2 <= T_DISCH_MAX
    V_OK_c1 = V_min_c1 * 0.999 <= V_dot_c1 <= V_max_c1 * 1.001
    V_OK_c2 = V_min_c2 * 0.999 <= V_dot_c2 <= V_max_c2 * 1.001
    feasible = all([p_OK_c1, p_OK_c2, T_OK_c1, T_OK_c2, V_OK_c1, V_OK_c2])

    fail_reasons = [name for name, ok in [
        ("p_c1", p_OK_c1), ("p_c2", p_OK_c2),
        ("T_c1", T_OK_c1), ("T_c2", T_OK_c2),
        ("V_c1", V_OK_c1), ("V_c2", V_OK_c2),
    ] if not ok]

    row = {
        **common,
        "no_solve_class": "",
        "label_c1": label_c1,
        "label_c2": label_c2,
        "Q_H_kW": round(Q_H_per_unit_kW * k, 2),
        "m_steam [kg/s]": round(m_steam, 4),
        "m_steam [kg/h]": round(m_steam * 3600, 1),
        "V_dot_c1_target [m3/h]": round(V_dot_c1, 1),
        "V_dot_c2_target [m3/h]": round(V_dot_c2, 1),
        "W_comp1 [kW]": round(W_comp1, 2),
        "W_comp2 [kW]": round(W_comp2, 2),
        "W_el [kW]": round(W_el_kW, 2),
        "scale_k": round(k, 4),
        "COP": round(COP, 3),
        "eta_Lorenz": round(eta_Lorenz, 3) if eta_Lorenz == eta_Lorenz else "",  # NaN→""
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
    row["status_4state"] = classify_status(row)

    # TODO step 4 — exergy sidecar JSON dump for OK/V_ONLY rows would go here:
    #   results/case_steam_110/exergy/<f1>_<f2>_LS<ls_pct>_Tsrc<T_src>.json
    # TODO step 5 — compute_pec(sim, k) gated behind --with-pec; columns
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
        T_steam = T_STEAM_CASE_110
    out_dir, csv_path = _paths_for(T_steam)
    os.makedirs(out_dir, exist_ok=True)

    dh = _dh_steam_kJ_per_kg(T_steam)
    print(f"Case study: T_steam = {T_steam:.0f} °C, "
          f"p = {p_water_for_T_steam(T_steam):.3f} bar, "
          f"Δh = {dh:.1f} kJ/kg, Q_H = {Q_H_NOMINAL_KW:.0f} kW")
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
                    res = screen_one_case_steam_110(
                        f1, f2, ls, T_src,
                        T_steam=T_steam,
                        Q_H_target_kW=Q_H_NOMINAL_KW,
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

    # Q_H consistency assertion
    non_nosolve = df[df["status_4state"] != "NOSOLVE"]
    if not non_nosolve.empty:
        max_dev = (non_nosolve["Q_H_kW"] - Q_H_NOMINAL_KW).abs().max()
        print(f"  Q_H consistency: max |Q_H − {Q_H_NOMINAL_KW:.0f}| = "
              f"{max_dev:.4f} kW (target < 0.1 kW)")
        if max_dev > 0.1:
            print("  ! WARNING: Q_H consistency check failed")

    print(f"\nCSV written to {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        _self_test()
    else:
        main()
