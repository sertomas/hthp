"""
enrich_screen.py — Stage 2: enrich the single Ommen screen with saturation
temperatures and write the sorted / per-pair CSV views.

There is ONE screening (``case_steam.py``, Ommen 2015 envelope). A design is
EXCLUDED only on thermodynamic grounds — no TESPy convergence, or a cycle
at/over its critical point. Those are the ``NOSOLVE`` rows. The Ommen
pressure / discharge-temperature / volume-flow limits are recorded for
information only (which limit each design exceeds and by how much, visualised
by the ``feasibility_ommen`` heatmap); they do NOT exclude a design. Every
non-NOSOLVE design is carried into the economics.

Run via main.py (Stage 2) or directly:
    python enrich_screen.py            # uses T_STEAM_CASE_DEFAULT
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from CoolProp.CoolProp import PropsSI


def _paths_for(T_steam):
    """Return (csv_in, out_dir) for a given case-study T_steam."""
    from config import case_results_dir
    out_dir = case_results_dir(T_steam)
    csv_in = os.path.join(out_dir, f"case_steam_{int(T_steam)}.csv")
    return csv_in, out_dir


def _T_sat_C(p_bar, fluid):
    """Saturation temperature [°C] at the given pressure [bar].
    Returns None at/above the critical pressure."""
    try:
        p_crit = PropsSI("Pcrit", fluid) / 1e5
        if p_bar >= 0.999 * p_crit:
            return None
        return PropsSI("T", "P", p_bar * 1e5, "Q", 0, fluid) - 273.15
    except Exception:
        return None


def _enrich_row(row):
    """Per-cycle saturation T_evap / T_cond. NOSOLVE rows get NaN."""
    if row.get("status_4state") == "NOSOLVE":
        return None, None, None, None
    f1, f2 = row["f1"], row["f2"]

    def sat(col, fluid):
        v = row.get(col)
        return _T_sat_C(v, fluid) if pd.notna(v) else None

    return (sat("p_low_c1 [bar]", f1), sat("p_high_c1 [bar]", f1),
            sat("p_low_c2 [bar]", f2), sat("p_high_c2 [bar]", f2))


def main(T_steam=None):
    """Enrich the Stage-1 screen CSV and write the sorted / per-pair views."""
    if T_steam is None:
        from config import T_STEAM_CASE_DEFAULT
        T_steam = T_STEAM_CASE_DEFAULT
    csv_in, out_dir = _paths_for(T_steam)
    if not os.path.exists(csv_in):
        print(f"Could not find {csv_in}. "
              f"Run `python main.py --t-steam {int(T_steam)}` "
              f"(or case_steam.main(T_steam={T_steam})) first.")
        sys.exit(1)

    df = pd.read_csv(csv_in)
    df["pair"] = df["f1"] + "/" + df["f2"]

    print("Enriching with saturation T_evap / T_cond ...")
    enrich = df.apply(_enrich_row, axis=1, result_type="expand")
    enrich.columns = ["T_evap_c1 [°C]", "T_cond_c1 [°C]",
                      "T_evap_c2 [°C]", "T_cond_c2 [°C]"]
    df = pd.concat([df, enrich.round(1)], axis=1)

    # ── Status distribution (single Ommen screen) ────────────────────────
    print()
    print("Status distribution (Ommen 2015, informational envelope):")
    for state in ("OK", "V_ONLY", "HARD", "NOSOLVE"):
        n = int((df["status_4state"] == state).sum())
        print(f"  {state:8s} : {n:>3d}")
    n_considered = int((df["status_4state"] != "NOSOLVE").sum())
    print(f"  -> {n_considered} designs carried into economics "
          f"(every design except NOSOLVE)")

    tag = f"case_steam_{int(T_steam)}"
    enriched_path = os.path.join(out_dir, f"{tag}_enriched.csv")
    df.to_csv(enriched_path, index=False)
    print(f"\nWrote {enriched_path}")

    # ── Sorted view: by status (best-first), then COP descending ─────────
    state_rank = {"OK": 0, "V_ONLY": 1, "HARD": 2, "NOSOLVE": 3}
    df_sorted = df.copy()
    df_sorted["_rank"] = df_sorted["status_4state"].map(state_rank).fillna(9)
    df_sorted = df_sorted.sort_values(
        by=["_rank", "COP"], ascending=[True, False]).drop(columns=["_rank"])
    df_sorted.to_csv(os.path.join(out_dir, f"{tag}_sorted.csv"), index=False)

    # ── Per-pair view: alphabetical f1/f2, then ls, T_src ────────────────
    df.sort_values(by=["f1", "f2", "ls", "T_src"]).to_csv(
        os.path.join(out_dir, f"{tag}_per_pair.csv"), index=False)
    print(f"Wrote {tag}_sorted.csv and {tag}_per_pair.csv")


if __name__ == "__main__":
    main()
