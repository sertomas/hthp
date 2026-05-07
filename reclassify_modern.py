"""
reclassify_modern.py — Post-process case_steam_110.csv with two envelopes
                       (Ommen 2015 strict + Project 68 / Annex 58 modern)
                       and write enriched, sorted CSV views.

The original screen in case_steam_110.py applies Ommen 2015 Table 3 limits
(p_max 28 / 50 bar, V̇ 5–280 m³/h Type-2, T_disch ≤ 180 °C). Project 68
(IEA HPT, Nov. 2025) and Annex 58 (2023) document multiple commercial
compressors operating beyond those limits. This script:

  1. Reads results/case_steam_110/case_steam_110.csv (already produced).
  2. Enriches it with T_evap_c1, T_cond_c1, T_evap_c2, T_cond_c2 derived
     from the saturation properties at the recorded p_low / p_high.
  3. Reclassifies each row under MODERN_COMPRESSOR_SPEC (justified per
     parameter from Project 68 supplier data). NOSOLVE rows (T_crit /
     P_crit / convergence failures) stay NOSOLVE — those are real
     thermodynamic infeasibilities and are not affected by the envelope.
  4. Writes:
       - case_steam_110_enriched.csv     (full data + new columns)
       - case_steam_110_sorted.csv        (sorted by status_modern, COP)
       - case_steam_110_per_pair.csv      (sorted f1, f2, ls, T_src)
       - case_steam_110_modern_OK.csv     (rows that become OK under modern)

Run:
    python reclassify_modern.py

Then re-plot with:
    python plot_case_steam_110.py --modern
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from CoolProp.CoolProp import PropsSI

# Reuse the helpers from screen_cascade and plot_common
from screen_cascade import OMMEN_P_TOL, T_DISCH_MAX, _envelope_label
from plot_common import classify_status


CSV_IN = os.path.join("results", "case_steam_110", "case_steam_110.csv")
OUT_DIR = os.path.join("results", "case_steam_110")


def _paths_for(T_steam):
    """Return (csv_in, out_dir) for a given case-study T_steam."""
    from config import case_results_dir
    out_dir = case_results_dir(T_steam)
    csv_in = os.path.join(out_dir, f"case_steam_{int(T_steam)}.csv")
    return csv_in, out_dir


# ──────────────────────────────────────────────────────────────────────────────
# Modern compressor envelope — IEA HPT Project 68 (2025) / Annex 58 (2023)
# evidence. Per-parameter justification:
#
# T_disch_max = 200 °C
#   Project 68 Table 1-2: Heaten HC piston rated to 200 °C; SPH Sustainable
#   Process Heat HC/HFO piston to 165–180 °C; AGO Energie & Anlagen R717/R718
#   piston/screw to 160 °C; Aneo Industry R717/R718 piston/screw to 190 °C;
#   Mayekawa Europe FC R601 screw to 145 °C. Modern POE / PAG / PFPE
#   lubricants are stable to 200–220 °C continuous; older mineral oils
#   degraded above 150 °C, which set Ommen's 180 °C limit.
#
# p_max:
#   HC Type-2  : 28 → 35 bar  (R290/R1270 piston compressors are routinely
#                              rated to 30–40 bar; Heaten and SPH machines
#                              cluster at 30–35 bar high-side for 1 MWth class)
#   R717-LP    : 28 bar       (kept — LP class is conservative)
#   R717-HP    : 50 → 76 bar  (Star Vilter VSSH single-screw, special steel
#                              cast design; GEA Grasso twin-screw 63 bar;
#                              Sabroe HeatPAC HPX piston 60 bar — Annex 58
#                              p. 11; AGO/Aneo/GEA R717-cycle MW-scale plants
#                              all run above 50 bar)
#   R744       : 140 bar      (kept — already high)
#
# V̇ envelope:
#   HC Type-2  : 5–280 → 5–1500 m³/h
#                Mayekawa Europe HS R600 piston (750 kW) ~600 m³/h;
#                Heaten HC piston cascade up to 6 MW → ~3000–5000 m³/h.
#                1500 m³/h is conservative against Heaten / SPH MW-scale.
#   R717-LP    : 5–180 → 5–1500 m³/h  (industrial NH3 plants run > 1000 m³/h)
#   R717-HP    : 90–200 → 5–1500 m³/h
#   R744       : 6–25  (kept — tight Ommen-only data)
# ──────────────────────────────────────────────────────────────────────────────

MODERN_COMPRESSOR_SPEC = {
    # label   : (p_max, V_min, V_max, T_disch_max)
    "R134a":   (35,    5,    1500, 200),
    "R290":    (35,    5,    1500, 200),
    "R600a":   (35,    5,    1500, 200),
    "R600":    (35,    5,    1500, 200),
    "R1270":   (35,    5,    1500, 200),
    "R717-LP": (28,    5,    1500, 200),
    "R717-HP": (76,    5,    1500, 200),
    "R744":    (140,   6,    25,   200),
}

MODERN_T_DISCH_MAX = 200.0


# ──────────────────────────────────────────────────────────────────────────────
# Enrichment: derive T_evap / T_cond from saturation pressure
# ──────────────────────────────────────────────────────────────────────────────

def _T_sat_C(p_bar: float, fluid: str) -> float | None:
    """Saturation temperature [°C] for the given pressure [bar].
    Returns None if above the critical pressure."""
    try:
        p_crit = PropsSI("Pcrit", fluid) / 1e5
        if p_bar >= 0.999 * p_crit:
            return None
        T = PropsSI("T", "P", p_bar * 1e5, "Q", 0, fluid) - 273.15
        return T
    except Exception:
        return None


def _enrich_row(row):
    """Add saturation T_evap, T_cond per cycle. NOSOLVE rows just get NaN."""
    if row.get("status_4state") == "NOSOLVE":
        return None, None, None, None
    f1 = row["f1"]
    f2 = row["f2"]
    p_low_c1 = row.get("p_low_c1 [bar]")
    p_high_c1 = row.get("p_high_c1 [bar]")
    p_low_c2 = row.get("p_low_c2 [bar]")
    p_high_c2 = row.get("p_high_c2 [bar]")
    T_evap_c1 = _T_sat_C(p_low_c1, f1) if pd.notna(p_low_c1) else None
    T_cond_c1 = _T_sat_C(p_high_c1, f1) if pd.notna(p_high_c1) else None
    T_evap_c2 = _T_sat_C(p_low_c2, f2) if pd.notna(p_low_c2) else None
    T_cond_c2 = _T_sat_C(p_high_c2, f2) if pd.notna(p_high_c2) else None
    return T_evap_c1, T_cond_c1, T_evap_c2, T_cond_c2


# ──────────────────────────────────────────────────────────────────────────────
# Modern-envelope reclassification
# ──────────────────────────────────────────────────────────────────────────────

def _reclassify(row):
    """Re-evaluate the six envelope flags under the modern spec.
    Returns (p_OK_c1, p_OK_c2, T_OK_c1, T_OK_c2, V_OK_c1, V_OK_c2,
             reason_modern, status_modern)."""
    if row.get("status_4state") == "NOSOLVE":
        # Genuine thermodynamic infeasibility — envelope doesn't apply
        return ("", "", "", "", "", "", "no_solve", "NOSOLVE")

    f1 = row["f1"]
    f2 = row["f2"]
    p_high_c1 = row["p_high_c1 [bar]"]
    p_high_c2 = row["p_high_c2 [bar]"]
    T_disch_c1 = row["T_disch_c1 [°C]"]
    T_disch_c2 = row["T_disch_c2 [°C]"]
    V_dot_c1 = row["V_dot_c1_target [m3/h]"]
    V_dot_c2 = row["V_dot_c2_target [m3/h]"]

    label_c1 = _envelope_label(f1, p_high_c1)
    label_c2 = _envelope_label(f2, p_high_c2)
    p_max_c1, V_min_c1, V_max_c1, _ = MODERN_COMPRESSOR_SPEC[label_c1]
    p_max_c2, V_min_c2, V_max_c2, _ = MODERN_COMPRESSOR_SPEC[label_c2]

    p_OK_c1 = p_high_c1 <= p_max_c1 * OMMEN_P_TOL
    p_OK_c2 = p_high_c2 <= p_max_c2 * OMMEN_P_TOL
    T_OK_c1 = T_disch_c1 <= MODERN_T_DISCH_MAX
    T_OK_c2 = T_disch_c2 <= MODERN_T_DISCH_MAX
    V_OK_c1 = V_min_c1 * 0.999 <= V_dot_c1 <= V_max_c1 * 1.001
    V_OK_c2 = V_min_c2 * 0.999 <= V_dot_c2 <= V_max_c2 * 1.001
    feasible = all([p_OK_c1, p_OK_c2, T_OK_c1, T_OK_c2, V_OK_c1, V_OK_c2])

    fail_reasons = [name for name, ok in [
        ("p_c1", p_OK_c1), ("p_c2", p_OK_c2),
        ("T_c1", T_OK_c1), ("T_c2", T_OK_c2),
        ("V_c1", V_OK_c1), ("V_c2", V_OK_c2),
    ] if not ok]
    reason_modern = ", ".join(fail_reasons) if not feasible else ""

    # Apply 4-state classification using the modern flags
    proxy = {
        "feasible": feasible,
        "reason": reason_modern,
        "no_solve_class": "",
        "p_OK_c1": p_OK_c1, "p_OK_c2": p_OK_c2,
        "T_OK_c1": T_OK_c1, "T_OK_c2": T_OK_c2,
        "V_OK_c1": V_OK_c1, "V_OK_c2": V_OK_c2,
    }
    status_modern = classify_status(proxy)

    return (p_OK_c1, p_OK_c2, T_OK_c1, T_OK_c2, V_OK_c1, V_OK_c2,
            reason_modern, status_modern)


def main(T_steam=None):
    """Reclassify the screen CSV under the modern envelope.

    Parameters
    ----------
    T_steam : float, optional
        Steam temperature to look up the per-T_steam screen output.
        Defaults to 110 °C (legacy folder).
    """
    if T_steam is None:
        T_steam = 110.0
    csv_in, out_dir = _paths_for(T_steam)
    if not os.path.exists(csv_in):
        print(f"Could not find {csv_in}. Run case_steam_110.py(T_steam={T_steam}) first.")
        sys.exit(1)

    df = pd.read_csv(csv_in)
    df["pair"] = df["f1"] + "/" + df["f2"]

    # ── Enrichment: T_evap, T_cond per cycle ─────────────────────────────
    print("Enriching CSV with saturation T_evap / T_cond ...")
    enrich = df.apply(_enrich_row, axis=1, result_type="expand")
    enrich.columns = ["T_evap_c1 [°C]", "T_cond_c1 [°C]",
                      "T_evap_c2 [°C]", "T_cond_c2 [°C]"]
    df = pd.concat([df, enrich.round(1)], axis=1)

    # ── Reclassify under modern envelope ─────────────────────────────────
    print("Re-applying modern envelope ...")
    rec = df.apply(_reclassify, axis=1, result_type="expand")
    rec.columns = ["p_OK_c1_modern", "p_OK_c2_modern",
                   "T_OK_c1_modern", "T_OK_c2_modern",
                   "V_OK_c1_modern", "V_OK_c2_modern",
                   "reason_modern", "status_modern"]
    df = pd.concat([df, rec], axis=1)

    # ── Compare classification distributions ─────────────────────────────
    print()
    print("=" * 80)
    print("Status distribution: Ommen 2015 strict  vs  Project 68 modern")
    print("=" * 80)
    a = df["status_4state"].value_counts().to_dict()
    b = df["status_modern"].value_counts().to_dict()
    for state in ("OK", "V_ONLY", "HARD", "NOSOLVE"):
        na = a.get(state, 0); nb = b.get(state, 0)
        print(f"  {state:8s} : Ommen {na:>3d}  →  Modern {nb:>3d}  (Δ {nb-na:+d})")

    # ── Transition matrix ────────────────────────────────────────────────
    print()
    print("Transition matrix (Ommen → Modern):")
    pivot = pd.crosstab(df["status_4state"], df["status_modern"],
                        rownames=["Ommen"], colnames=["Modern"])
    print(pivot)

    # ── Reordered column list for readability ────────────────────────────
    front_cols = [
        "f1", "f2", "pair", "ls", "T_src", "T_steam",
        "status_4state", "status_modern",
        "reason", "reason_modern", "no_solve_class",
        "COP", "eta_Lorenz", "Q_H_kW", "m_steam [kg/h]",
        "T_evap_c1 [°C]", "T_cond_c1 [°C]", "T_disch_c1 [°C]",
        "T_evap_c2 [°C]", "T_cond_c2 [°C]", "T_disch_c2 [°C]",
        "p_low_c1 [bar]", "p_high_c1 [bar]",
        "p_low_c2 [bar]", "p_high_c2 [bar]",
        "V_dot_c1_target [m3/h]", "V_dot_c2_target [m3/h]",
        "W_comp1 [kW]", "W_comp2 [kW]", "W_el [kW]",
        "label_c1", "label_c2", "scale_k",
        # Ommen flags
        "p_OK_c1", "p_OK_c2", "T_OK_c1", "T_OK_c2", "V_OK_c1", "V_OK_c2",
        # Modern flags
        "p_OK_c1_modern", "p_OK_c2_modern",
        "T_OK_c1_modern", "T_OK_c2_modern",
        "V_OK_c1_modern", "V_OK_c2_modern",
        "feasible", "status",
    ]
    other = [c for c in df.columns if c not in front_cols]
    df = df[[c for c in front_cols if c in df.columns] + other]

    # ── Write the enriched, unsorted master CSV (overwrites input) ───────
    tag = f"case_steam_{int(T_steam)}"
    enriched_path = os.path.join(out_dir, f"{tag}_enriched.csv")
    df.to_csv(enriched_path, index=False)
    print(f"\nWrote {enriched_path}")

    # ── Sorted view 1: by modern status (best-first), then COP descending ─
    state_rank = {"OK": 0, "V_ONLY": 1, "HARD": 2, "NOSOLVE": 3}
    df_sorted = df.copy()
    df_sorted["_rank"] = df_sorted["status_modern"].map(state_rank).fillna(9)
    df_sorted = df_sorted.sort_values(
        by=["_rank", "COP"],
        ascending=[True, False],
    ).drop(columns=["_rank"])
    sorted_path = os.path.join(out_dir, f"{tag}_sorted.csv")
    df_sorted.to_csv(sorted_path, index=False)
    print(f"Wrote {sorted_path}  (sorted by modern status, COP desc)")

    # ── Sorted view 2: per-pair grouped (alphabetical f1/f2, then ls, T_src) ─
    df_pp = df.sort_values(by=["f1", "f2", "ls", "T_src"])
    per_pair_path = os.path.join(out_dir, f"{tag}_per_pair.csv")
    df_pp.to_csv(per_pair_path, index=False)
    print(f"Wrote {per_pair_path}  (sorted by f1, f2, ls, T_src)")

    # ── Sorted view 3: rows that flip from HARD/V_ONLY to OK under modern ─
    flips = df[(df["status_4state"] != "OK") & (df["status_modern"] == "OK")]
    flips_path = os.path.join(out_dir, f"{tag}_modern_OK.csv")
    flips_sorted = flips.sort_values(
        by=["status_4state", "COP"], ascending=[True, False],
    )
    flips_sorted.to_csv(flips_path, index=False)
    print(f"Wrote {flips_path}  ({len(flips)} rows now OK under modern)")

    # ── Quick sample of newly-OK rows for the user ─────────────────────────
    print()
    print("Sample of designs newly OK under modern envelope (top 8 by COP):")
    cols = ["f1", "f2", "ls", "T_src", "status_4state", "reason",
            "T_disch_c1 [°C]", "T_disch_c2 [°C]",
            "p_high_c1 [bar]", "p_high_c2 [bar]",
            "V_dot_c1_target [m3/h]", "V_dot_c2_target [m3/h]",
            "COP", "eta_Lorenz"]
    print(flips_sorted.head(8)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
