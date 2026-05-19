"""
plot_common.py — Helpers shared by the cascade-screening plotters.

Provides the 4-state classification (OK / V_ONLY / HARD / NOSOLVE) used
by ``case_steam.py``, ``reclassify_modern.py`` and the Stage 3 grid
plotters. V̇-only Ommen-envelope violations are kept distinct from hard
pressure / discharge-temperature failures so they can be rendered as
"operable per IEA HPT Annex 58 evidence" rather than lumped together.
"""

from __future__ import annotations

import os


# ── 4-state classification ──────────────────────────────────────────────────
# OK      — all Ommen flags pass (p_OK_*, T_OK_*, V_OK_* all True)
# V_ONLY  — p_OK_* and T_OK_* all True; V_OK_c1 or V_OK_c2 False.
#           Operable per Annex 58 supplier evidence: real machines exist
#           above Ommen 2015's 5–280 m³/h Type-2 V_max (e.g. Mayekawa, Hybrid
#           Energy, Heaten at 1–5 MWth). Cost extrapolation is the only
#           risk, mitigated by Annex 58 €/kW band cross-checks (Stage 5).
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


# ── Component palette ─────────────────────────────────────────────────────
# Single source of truth for every component-level plot in the project so
# the visual language stays consistent. Conventions:
#   * COMP and MOT of the same cycle are bundled into a single "COMP+MOT"
#     group (one bar segment, one legend entry). This matches the Ommen
#     Tab. 4 cost-row granularity where the motor PEC is rolled into the
#     compressor for HC fluids.
#   * Every other component is shown separately: VAL1 ≠ VAL2, SRC_HX (cycle
#     1 evaporator), IHX (cascade), SNK_HX (cycle 2 condenser).
#   * Cycle-1 components inherit the original "aggregated" hues from
#     z_breakdown_aggregated.png (blue, green, pink). The corresponding
#     cycle-2 component uses a lighter pastel tone of the same hue so the
#     two cycles read as a saturation pair at a glance. SRC_HX and SNK_HX
#     have no opposite-cycle sibling, so each keeps its own distinctive
#     colour (green / red).
#
# Stacking order: cycle-1 components at the bottom, cascade IHX in the
# middle, cycle-2 components on top — each bar reads
# "low-pressure side at the bottom, high-pressure side on top".

# Per-group plotting info. ``members`` maps the group to the raw component
# names found in exergoeco_components.csv that should be summed into that
# group. ``color`` is the fill colour; ``label`` is the legend / annotation
# label.
GROUP_STYLE = {
    "COMP1+MOT1": {"members": ["COMP1", "MOT1"], "color": "#4C72B0",
                   "label": "COMP1+MOT1", "cycle": 1},
    "SRC_HX":     {"members": ["SRC_HX"],         "color": "#55A868",
                   "label": "SRC_HX",     "cycle": 1},
    "VAL1":       {"members": ["VAL1"],           "color": "#E8A0BF",
                   "label": "VAL1",       "cycle": 1},
    "IHX":        {"members": ["IHX"],            "color": "#D4A039",
                   "label": "IHX",        "cycle": 0},
    "COMP2+MOT2": {"members": ["COMP2", "MOT2"], "color": "#8FAFD3",
                   "label": "COMP2+MOT2", "cycle": 2},
    "SNK_HX":     {"members": ["SNK_HX"],         "color": "#C44E52",
                   "label": "SNK_HX",     "cycle": 2},
    "VAL2":       {"members": ["VAL2"],           "color": "#F4CFDD",
                   "label": "VAL2",       "cycle": 2},
}

# Stacking order: cycle 1 bottom → IHX cascade → cycle 2 top
GROUP_ORDER = ["COMP1+MOT1", "SRC_HX", "VAL1",
               "IHX",
               "COMP2+MOT2", "SNK_HX", "VAL2"]

# Convenience views derived from GROUP_STYLE, kept so existing call sites
# can pick the dict they want without rebuilding it on each plot.
GROUP_COLORS  = {g: GROUP_STYLE[g]["color"]   for g in GROUP_ORDER}
GROUP_LABELS  = {g: GROUP_STYLE[g]["label"]   for g in GROUP_ORDER}
GROUP_MEMBERS = {g: GROUP_STYLE[g]["members"] for g in GROUP_ORDER}


# ── Alternative grouping: motors shown separately ──────────────────────────
# Used by the E_D components breakdown. The motor losses (electrical-to-
# shaft) are physically distinct from the compressor losses (shaft-to-
# refrigerant) and become visible as their own stack segments only when
# unbundled. MOT colours follow the same dark/light cycle-1/cycle-2
# pattern as the rest of the palette: saturated violet for MOT1, pastel
# violet for MOT2 — so each "drive package" (COMP*+MOT*) reads as a
# blue+violet pair on cycle 1, light blue+light violet on cycle 2.
GROUP_STYLE_SPLIT_MOT = {
    "COMP1":  {"members": ["COMP1"], "color": "#4C72B0",
                "label": "COMP1",  "cycle": 1},
    "MOT1":   {"members": ["MOT1"],  "color": "#5B3F8B",
                "label": "MOT1",   "cycle": 1},
    "SRC_HX": {"members": ["SRC_HX"], "color": "#55A868",
                "label": "SRC_HX", "cycle": 1},
    "VAL1":   {"members": ["VAL1"],  "color": "#E8A0BF",
                "label": "VAL1",   "cycle": 1},
    "IHX":    {"members": ["IHX"],   "color": "#D4A039",
                "label": "IHX",    "cycle": 0},
    "COMP2":  {"members": ["COMP2"], "color": "#8FAFD3",
                "label": "COMP2",  "cycle": 2},
    "MOT2":   {"members": ["MOT2"],  "color": "#A89AC9",
                "label": "MOT2",   "cycle": 2},
    "SNK_HX": {"members": ["SNK_HX"], "color": "#C44E52",
                "label": "SNK_HX", "cycle": 2},
    "VAL2":   {"members": ["VAL2"],  "color": "#F4CFDD",
                "label": "VAL2",   "cycle": 2},
}

# Same stacking convention as GROUP_ORDER: cycle 1 (bottom) → cascade IHX
# → cycle 2 (top). Motors come right after their compressor.
GROUP_ORDER_SPLIT_MOT = ["COMP1", "MOT1", "SRC_HX", "VAL1",
                         "IHX",
                         "COMP2", "MOT2", "SNK_HX", "VAL2"]
GROUP_COLORS_SPLIT_MOT  = {g: GROUP_STYLE_SPLIT_MOT[g]["color"]
                            for g in GROUP_ORDER_SPLIT_MOT}
GROUP_LABELS_SPLIT_MOT  = {g: GROUP_STYLE_SPLIT_MOT[g]["label"]
                            for g in GROUP_ORDER_SPLIT_MOT}
GROUP_MEMBERS_SPLIT_MOT = {g: GROUP_STYLE_SPLIT_MOT[g]["members"]
                            for g in GROUP_ORDER_SPLIT_MOT}


# ── Publication-figure style ────────────────────────────────────────────────
# Elsevier journal column widths (Applied Energy, Energy Conversion and
# Management, Energy, Int. J. of Refrigeration). Figures are rendered at
# the PHYSICAL widths used in the manuscript — no figure-size scaling, no
# font-size scaling — so the PDF imports into LaTeX at 1:1 with the body
# text font sizes intact. The only "more pixels" knob is
# ``savefig.dpi``: it raises the resolution of *raster* elements embedded
# in the PDF (imshow heatmap fills, hatched bars, etc.) without touching
# the vector text or axes.
COL_SINGLE_IN = 3.54   # 90 mm  — single column
COL_DOUBLE_IN = 7.48   # 190 mm — full-page / double column


_STYLE_APPLIED = False


def apply_publication_style():
    """Set matplotlib rcParams to journal-figure defaults.

    Sans-serif (Arial / DejaVu Sans fallback), 9 pt body, 0.75 pt axes,
    ``pdf.fonttype = 42`` so PDF text remains real text (selectable /
    editable, required by some journal vector pipelines), and a high
    ``savefig.dpi`` so embedded rasters (imshow fills, hatches) stay
    sharp when LaTeX upsamples them. Idempotent.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "figure.titlesize":  10,
        "axes.linewidth":    0.75,
        "lines.linewidth":   1.0,
        "patch.linewidth":   0.5,
        "pdf.fonttype":      42,   # keep text as text, not outlines
        "ps.fonttype":       42,
        "savefig.bbox":      "tight",
        # Raster resolution embedded inside the PDF — vector text/lines
        # are unaffected; only imshow / hatch fills get this many pixels.
        # 200 dpi at 7.48" wide ≈ 1496 px wide → enough for journal
        # zoom while staying memory-safe on Windows numpy builds.
        "savefig.dpi":       200,
        "figure.dpi":        100,
    })
    _STYLE_APPLIED = True


def fs(pt):
    """Identity helper — kept for source-level compatibility with the
    short-lived ``RENDER_SCALE`` experiment so all per-call
    ``fontsize=fs(N)`` literals continue to work unchanged."""
    return pt


# Apply on import so every plotter that imports anything from this module
# inherits the journal style without an explicit call.
apply_publication_style()


# ── Title / no-title PDF variant helper ─────────────────────────────────────
def save_titled_and_paper(fig, plots_dir, base_name, **savefig_kwargs):
    """Save a figure twice as PDF: titled (``<name>_titled.pdf``) and
    paper-clean (``<name>.pdf``, no suptitle).

    PDF is the vector format used by LaTeX ``\\includegraphics``.
    Per-axis ``ax.set_title`` panel labels are preserved in both variants
    — only ``fig.suptitle`` is hidden in the paper version. Closes the
    figure when done.
    """
    import matplotlib.pyplot as plt
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f"{base_name}_titled.pdf"),
                **savefig_kwargs)
    sup = getattr(fig, "_suptitle", None)
    if sup is not None:
        sup.set_visible(False)
    fig.savefig(os.path.join(plots_dir, f"{base_name}.pdf"),
                **savefig_kwargs)
    plt.close(fig)
