"""
plot_common.py — Helpers shared by the cascade-screening plotters.

Originally extracted from ``plot_screen_cascade.py`` to support the
``case_steam_110.py`` study, which adds a 4-state classification
(OK / V_ONLY / HARD / NOSOLVE) so V̇-only Ommen-envelope violations can
be rendered as "operable per IEA HPT Annex 58 evidence" rather than
lumped together with hard pressure / discharge-temperature failures.

The existing ``plot_screen_cascade.py`` is not rewritten; it keeps its
own copies of ``_is_true`` and ``_slice_grid`` to remain isolated from
this case-study extension.
"""

from __future__ import annotations


# ── 4-state classification ──────────────────────────────────────────────────
# OK      — all Ommen flags pass (p_OK_*, T_OK_*, V_OK_* all True)
# V_ONLY  — p_OK_* and T_OK_* all True; V_OK_c1 or V_OK_c2 False.
#           Operable per Annex 58 supplier evidence: real machines exist
#           above Ommen 2015's 5–280 m³/h Type-2 V_max (e.g. Mayekawa, Hybrid
#           Energy, Heaten at 1–5 MWth). Cost extrapolation is the only
#           risk, mitigated by Annex 58 €/kW band cross-checks (step 5).
# HARD    — p_OK_c1, p_OK_c2, T_OK_c1 or T_OK_c2 False. Pressure rating or
#           oil-degradation issue — genuinely problematic.
# NOSOLVE — pre-classified thermodynamic infeasibility or TESPy non-convergence.
STATUS_COLORS = {
    "OK":      "#66bb6a",
    "V_ONLY":  "#ffb74d",
    "HARD":    "#e57373",
    "NOSOLVE": "#cccccc",
}

STATUS_ORDER = ["OK", "V_ONLY", "HARD", "NOSOLVE"]


def _is_true(v) -> bool:
    """Robust truthy parser for booleans round-tripped through CSV."""
    return v is True or v == "True" or v == "true" or v == 1 or v == 1.0


def classify_status(row) -> str:
    """
    Map a screening-CSV row to one of OK / V_ONLY / HARD / NOSOLVE.

    Precedence: NOSOLVE > HARD > V_ONLY > OK.
    """
    feasible_field = row.get("feasible")
    if not _is_true(feasible_field):
        # `feasible == False` could mean either no_solve or constraint failure.
        # Distinguish via the explicit `reason` field used by screen_cascade.
        reason = row.get("reason", "")
        if reason == "no_solve" or row.get("no_solve_class"):
            return "NOSOLVE"
        # Otherwise fall through into the per-flag logic below.

    p_OK = _is_true(row.get("p_OK_c1")) and _is_true(row.get("p_OK_c2"))
    T_OK = _is_true(row.get("T_OK_c1")) and _is_true(row.get("T_OK_c2"))
    V_OK = _is_true(row.get("V_OK_c1")) and _is_true(row.get("V_OK_c2"))

    if not (p_OK and T_OK):
        return "HARD"
    if not V_OK:
        return "V_ONLY"
    return "OK"


def slice_grid(df, ls, value_col, *, status_col="status_4state",
               accept_states=("OK", "V_ONLY"),
               pair_order=None, T_src_order=None):
    """
    Pivot ``value_col`` into a (fluid-pair × T_src) grid for one lift share.

    Cells whose ``status_4state`` is not in ``accept_states`` come back NaN
    so the caller can mask them in heatmaps.
    """
    import numpy as np  # local import keeps this module light

    sub = df[df["ls"] == ls]
    sub = sub[sub[status_col].isin(accept_states)]

    if pair_order is None:
        pair_order = sorted(df["pair"].unique())
    if T_src_order is None:
        T_src_order = sorted(df["T_src"].unique())

    grid = np.full((len(pair_order), len(T_src_order)), np.nan)
    for i, pair in enumerate(pair_order):
        for j, T_src in enumerate(T_src_order):
            row = sub[(sub["pair"] == pair) & (sub["T_src"] == T_src)]
            if not row.empty:
                grid[i, j] = row.iloc[0][value_col]
    return grid, pair_order, T_src_order


def reason_short(reason: str) -> str:
    """Compress a comma-separated failed-flag list into subscripted symbols
    for cell annotations (e.g. 'p_c1, V_c2' → 'p₁, V₂')."""
    return (str(reason)
            .replace("p_c1", "p₁").replace("p_c2", "p₂")
            .replace("T_c1", "T₁").replace("T_c2", "T₂")
            .replace("V_c1", "V₁").replace("V_c2", "V₂"))


def nosolve_short(cls: str) -> str:
    """Compact label for the NOSOLVE annotation cell."""
    return {
        "T_crit_c1":   "T_crit\nc1",
        "T_crit_c2":   "T_crit\nc2",
        "Pcrit_c1":    "P_crit\nc1",
        "convergence": "TESPy\nconv.",
    }.get(str(cls), "no\nsolve")
