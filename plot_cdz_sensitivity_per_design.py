"""
plot_cdz_sensitivity_per_design.py — Per-design Z+C_D sensitivity plots.

For each fluid pair, generate two figures inside the pair's own
``designs/<pair>/`` folder:

1. ``cdz_sensitivity.png`` — two side-by-side stacked-bar panels:
       LEFT  : T_src sweep at LS = 50 %
       RIGHT : LS sweep   at T_src = 60 °C
   Same colors as ``cost_breakdown_per_design.png``.

2. ``cP_heatmap.png`` — heatmap of specific product cost c_P [EUR/GJ_ex] on
   the full (T_src × LS) grid for the pair. Cells annotated with the
   numerical EUR/GJ value.

Run with::

    python plot_cdz_sensitivity_per_design.py            # all T_steam cases
    python plot_cdz_sensitivity_per_design.py 110        # single case
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from matplotlib.colors import ListedColormap

from config import (
    BASE_E1_C, BASE_FULL_LOAD_HOURS, BASE_GAS_C, T_STEAMS_TO_RUN,
    case_data_dir, case_plots_dir, case_results_dir, m_steam_label,
)
from economics import F_INSTALL
from plot_common import (
    COL_DOUBLE_IN, COL_SINGLE_IN,
    GROUP_COLORS, GROUP_ORDER, GROUP_MEMBERS,
    STATUS_COLORS, classify_status, save_titled_and_paper,
    fs,
)

# Component grouping pulled from the central plot_common dict. COMP+MOT of
# each cycle are bundled into one segment; everything else (VAL1, VAL2,
# SRC_HX, IHX, SNK_HX) is plotted individually.
COMP_GROUPS = dict(GROUP_MEMBERS)

LS_FIXED = 0.50      # 50 % — used in the T_src-sweep panel
T_SRC_FIXED = 60.0   # °C  — used in the LS-sweep panel


def _op_string():
    return (f"FLH = {BASE_FULL_LOAD_HOURS:.0f} h/a, "
            f"e1 = {BASE_E1_C:.0f} EUR/MWh, "
            f"gas = {BASE_GAS_C:.0f} EUR/MWh")


def _pair_dir(designs_dir, pair):
    return os.path.join(designs_dir, pair.replace("/", "_"))


def _design_dir(designs_dir, pair, ls, T_src):
    ls_pct = int(round(ls * 100))
    return os.path.join(_pair_dir(designs_dir, pair),
                        f"LS{ls_pct}_Tsrc{int(T_src)}")


def _load_cdz_row(designs_dir, pair, ls, T_src,
                  value_col="C_D+Z [EUR/h]"):
    """Return {group: value} for one design, or None if file is missing."""
    path = os.path.join(_design_dir(designs_dir, pair, ls, T_src),
                        "exergoeco_components.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["Component"] != "TOT"].copy()
    cdz = dict(zip(df["Component"], df[value_col].astype(float)))
    row = {g: sum(cdz.get(m, 0.0) for m in members)
           for g, members in COMP_GROUPS.items()}
    row["TOTAL"] = sum(row[g] for g in COMP_GROUPS)
    return row


def _discover_grid(designs_dir, pair):
    """Discover the LS (%) and T_src (°C) values that exist on disk.

    The parametric study restricts the lift share to {0.30, 0.40, 0.50};
    any extra LS values created upstream by the per-pair screening (LS =
    0.60, 0.70) are dropped here so they do not appear as empty rows in
    the per-design heatmaps.
    """
    pdir = _pair_dir(designs_dir, pair)
    if not os.path.isdir(pdir):
        return [], []
    ls_set, tsrc_set = set(), set()
    for sub in os.listdir(pdir):
        # Folder names look like ``LS50_Tsrc60``
        try:
            ls_part, t_part = sub.split("_")
            ls_set.add(int(ls_part.replace("LS", "")))
            tsrc_set.add(int(t_part.replace("Tsrc", "")))
        except ValueError:
            continue
    ls_set = {ls for ls in ls_set if ls in (30, 40, 50)}
    return sorted(ls_set), sorted(tsrc_set)


# ── Bar-plot panels ───────────────────────────────────────────────────────
def _draw_stacked_panel(ax, data, x_labels, y_max, xlabel):
    """Draw one stacked-bar panel; ``data`` is a list of group→value dicts."""
    x = np.arange(len(data))
    bottom = np.zeros(len(data))
    for group in COMP_GROUPS:
        vals = np.array([d[group] for d in data])
        ax.bar(x, vals, bottom=bottom,
               color=GROUP_COLORS[group],
               edgecolor="black", linewidth=0.3)
        bottom += vals
    for k, d in enumerate(data):
        ax.text(k, d["TOTAL"] + y_max * 0.01, f"{d['TOTAL']:.0f}",
                ha="center", va="bottom", fontsize=fs(8), fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=fs(9))
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", alpha=0.25)


def _collect_tsrc_sweep(designs_dir, pair, ls_pct, t_src_vals):
    """Return (data list, labels) for the T_src sweep at a single fixed LS
    (in percent)."""
    data, labels = [], []
    for T in t_src_vals:
        row = _load_cdz_row(designs_dir, pair, ls_pct / 100.0, T)
        if row is None:
            continue
        data.append(row)
        labels.append(f"{T:g}")
    return data, labels


def _collect_ls_sweep(designs_dir, pair, T_src, ls_vals):
    """Return (data list, labels) for the LS sweep at a single fixed
    T_src (°C)."""
    data, labels = [], []
    for ls_pct in ls_vals:
        row = _load_cdz_row(designs_dir, pair, ls_pct / 100.0, T_src)
        if row is None:
            continue
        data.append(row)
        labels.append(f"{ls_pct}%")
    return data, labels


def _plot_bars(pair, designs_dir, plots_dir, base_name, T_steam,
                ls_vals, t_src_vals):
    """Full stacked-bar grid: row 1 = T_src sweeps (one per LS),
    row 2 = LS sweeps (one per T_src). All subplots share a common y axis
    so heights are directly comparable across the grid. The two reference
    cases (LS = ``LS_FIXED``, T_src = ``T_SRC_FIXED``) are flagged in the
    panel titles."""
    # ── Row 1: T_src sweep, one panel per LS ──────────────────────────────
    top_panels = []  # list of (ls_pct, data, labels)
    for ls_pct in ls_vals:
        d, lab = _collect_tsrc_sweep(designs_dir, pair, ls_pct, t_src_vals)
        if d:
            top_panels.append((ls_pct, d, lab))

    # ── Row 2: LS sweep, one panel per T_src ──────────────────────────────
    bot_panels = []  # list of (T_src, data, labels)
    for T in t_src_vals:
        d, lab = _collect_ls_sweep(designs_dir, pair, T, ls_vals)
        if d:
            bot_panels.append((T, d, lab))

    if not top_panels and not bot_panels:
        return False

    # Shared y limit across the whole grid for visual comparability.
    all_totals = [r["TOTAL"]
                  for _, d, _ in top_panels + bot_panels
                  for r in d]
    y_max = max(all_totals) * 1.20

    n_top = len(top_panels)
    n_bot = len(bot_panels)
    ncols = max(n_top, n_bot, 1)
    # Per-pair grid lives in designs/<pair>/ — full-page (double-column)
    # double-row sensitivity layout.
    fig, axes = plt.subplots(
        2, ncols,
        figsize=(COL_DOUBLE_IN, 6.5),
        sharey=True, squeeze=False,
    )

    # Row 1 — T_src sweeps
    for j, (ls_pct, d, lab) in enumerate(top_panels):
        ax = axes[0][j]
        _draw_stacked_panel(ax, d, lab, y_max,
                            xlabel=r"$T_\mathrm{src,in}$  [°C]")
        ref = (ls_pct == int(LS_FIXED * 100))
        ax.set_title(
            rf"$T_\mathrm{{src,in}}$ sweep @ $\mathit{{LS}} = {ls_pct}$ %"
            + ("  ★" if ref else ""),
            fontsize=fs(8),
            fontweight=("bold" if ref else "normal"),
            color=("#a02020" if ref else "black"),
        )
    for j in range(n_top, ncols):
        axes[0][j].set_visible(False)

    # Row 2 — LS sweeps
    for j, (T, d, lab) in enumerate(bot_panels):
        ax = axes[1][j]
        _draw_stacked_panel(ax, d, lab, y_max,
                            xlabel=r"Lift share $\mathit{LS}$  [-]")
        ref = (int(T) == int(T_SRC_FIXED))
        ax.set_title(
            rf"$\mathit{{LS}}$ sweep @ $T_\mathrm{{src,in}} = {T:g}$ °C"
            + ("  ★" if ref else ""),
            fontsize=fs(8),
            fontweight=("bold" if ref else "normal"),
            color=("#a02020" if ref else "black"),
        )
    for j in range(n_bot, ncols):
        axes[1][j].set_visible(False)

    axes[0][0].set_ylabel(r"$\dot{C}_D + \dot{Z}$  [EUR/h]")
    axes[1][0].set_ylabel(r"$\dot{C}_D + \dot{Z}$  [EUR/h]")

    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g)
               for g in COMP_GROUPS]
    handles.append(mpatches.Patch(
        facecolor="white", edgecolor="white",
        label=(rf"★ reference slices ($\mathit{{LS}} = {int(LS_FIXED*100)}$ %, "
               rf"$T_\mathrm{{src,in}} = {int(T_SRC_FIXED)}$ °C)")))
    fig.legend(handles=handles, loc="lower center",
               ncol=min(4, len(handles)),
               fontsize=fs(7), frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        rf"$\dot{{C}}_D + \dot{{Z}}$ — {pair}  "
        rf"($T_\mathrm{{steam}}$ = {T_steam:.0f} °C)",
        fontsize=fs(10), fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    save_titled_and_paper(fig, plots_dir, base_name)
    return True


# ── Heatmap loaders ──────────────────────────────────────────────────────
def _load_base_grid(data_dir, pair, ls_vals, t_src_vals, column,
                     transform=None):
    """Pull a column from ``data/economics_base.csv`` into a (LS × T_src) grid.

    ``transform`` may be a callable applied to the raw column value (e.g.
    converting fraction → percent for ε).
    """
    base_csv = os.path.join(data_dir, "economics_base.csv")
    if not os.path.exists(base_csv):
        return None
    df = pd.read_csv(base_csv)
    df = df[df["pair"] == pair]
    grid = np.full((len(ls_vals), len(t_src_vals)), np.nan)
    for i, ls_pct in enumerate(ls_vals):
        for j, T in enumerate(t_src_vals):
            row = df[(np.isclose(df["ls"], ls_pct / 100.0))
                     & (np.isclose(df["T_src"], T))]
            if not row.empty:
                v = float(row.iloc[0][column])
                grid[i, j] = transform(v) if transform else v
    return grid


def _load_enriched_grid(case_dir, T_steam, pair, ls_vals, t_src_vals,
                         row_to_value, accept_states=("OK", "V_ONLY")):
    """Pull a derived value from ``case_steam_<T>_enriched.csv``.

    ``row_to_value`` is a callable ``row -> float | nan`` so non-trivial
    quantities (pressure ratios, temperature lift, …) can be computed
    inline from one or more columns.

    By default cells whose ``status_modern`` is not OK or V_ONLY come back
    NaN — keeps cost / pressure / temperature heatmaps confined to the
    feasible operating envelope. Pass ``accept_states=None`` to see every
    converged cell regardless of status (used by the feasibility plots
    themselves).
    """
    enriched_csv = os.path.join(case_dir,
                                f"case_steam_{int(T_steam)}_enriched.csv")
    if not os.path.exists(enriched_csv):
        return None
    df = pd.read_csv(enriched_csv)
    df = df[df["pair"] == pair]
    if accept_states is not None and "status_modern" in df.columns:
        df = df[df["status_modern"].isin(accept_states)]
    grid = np.full((len(ls_vals), len(t_src_vals)), np.nan)
    for i, ls_pct in enumerate(ls_vals):
        for j, T in enumerate(t_src_vals):
            row = df[(np.isclose(df["ls"], ls_pct / 100.0))
                     & (np.isclose(df["T_src"], T))]
            if row.empty:
                continue
            try:
                v = row_to_value(row.iloc[0])
            except (KeyError, ValueError, TypeError):
                v = np.nan
            grid[i, j] = float(v) if v is not None else np.nan
    return grid


def _load_LCOH_grid(data_dir, pair, ls_vals, t_src_vals):
    """Levelised cost of heat [EUR/MWh_th] on the (LS × T_src) grid.

    LCOH is the energy-basis counterpart of the SPECO ``c_P`` produced by
    ExerPy: it normalises the all-in cost rate by the *thermal* output
    rather than by the steam-exergy product, so the resulting number is
    directly comparable to retail steam or boiler prices.

        LCOH [EUR/MWh_th] = (1000 · Ż_sum / Q_H) + c_el / COP
    """
    grid = np.full((len(ls_vals), len(t_src_vals)), np.nan)
    base_csv = os.path.join(data_dir, "economics_base.csv")
    if not os.path.exists(base_csv):
        return grid
    df = pd.read_csv(base_csv)
    sub = df[df["pair"] == pair]
    for i, ls_pct in enumerate(ls_vals):
        for j, T in enumerate(t_src_vals):
            row = sub[(np.isclose(sub["ls"], ls_pct / 100.0))
                      & (np.isclose(sub["T_src"], T))]
            if row.empty:
                continue
            r = row.iloc[0]
            Q_H = float(r["Q_H_kW"])
            COP = float(r["COP"])
            Z_sum = float(r["Z_sum [EUR/h]"])
            if Q_H <= 0 or COP <= 0:
                continue
            grid[i, j] = 1000.0 * Z_sum / Q_H + BASE_E1_C / COP
    return grid


def _draw_metric_heatmap(ax, grid, ls_vals, t_src_vals, vmin, vmax,
                         cmap, fmt="{:.1f}", fontsize=fs(10)):
    """Render a single (LS × T_src) heatmap into ``ax``. Returns the imshow
    handle. ``fmt`` controls the cell annotation format string.

    Cell-text colour is picked from the actual luminance of the colormap
    sample at that cell (Rec. 601 coefficients), so the rule works for
    every colormap direction — including ``viridis`` / ``inferno`` /
    ``cividis`` where the low end is dark.
    """
    masked = np.ma.masked_invalid(grid)
    # ``aspect="equal"`` forces square cells (one data unit = one display
    # unit in both axes); ``aspect="auto"`` would stretch them to fill the
    # axes box and the LS / T_src cells end up as rectangles.
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="equal", origin="upper")
    for i in range(len(ls_vals)):
        for j in range(len(t_src_vals)):
            if np.isnan(grid[i, j]):
                continue
            norm = (grid[i, j] - vmin) / max(vmax - vmin, 1e-6)
            r, g, b, _ = cmap(norm)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            txt_color = "white" if luminance < 0.5 else "black"
            ax.text(j, i, fmt.format(grid[i, j]),
                    ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=txt_color)
    ax.set_xticks(range(len(t_src_vals)))
    ax.set_xticklabels([f"{T:g}" for T in t_src_vals])
    ax.set_yticks(range(len(ls_vals)))
    ax.set_yticklabels([f"{ls}%" for ls in ls_vals])
    return im


def _plot_heatmap(pair, data_dir, plots_dir, base_name, T_steam,
                   ls_vals, t_src_vals):
    """Per-design c_P heatmap over the full (T_src × LS) grid."""
    if not ls_vals or not t_src_vals:
        return False
    grid = _load_base_grid(data_dir, pair, ls_vals, t_src_vals,
                            "c_P [EUR/GJ]")
    if grid is None or np.all(np.isnan(grid)):
        return False

    # Single-column figure (one panel for one fluid pair, LS × T_src).
    fig, ax = plt.subplots(figsize=(COL_SINGLE_IN, 2.8))
    cmap = plt.get_cmap("plasma_r").copy()
    cmap.set_bad(color="#cccccc")
    vmin = float(np.nanmin(grid))
    vmax = float(np.nanmax(grid))
    im = _draw_metric_heatmap(ax, grid, ls_vals, t_src_vals,
                               vmin, vmax, cmap, fmt="{:.1f}", fontsize=fs(7))
    ax.set_xlabel(r"$T_\mathrm{src,in}$  [°C]")
    ax.set_ylabel(r"Lift share $\mathit{LS}$  [-]")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$c_P$  [EUR/GJ$_{ex}$]")
    fig.suptitle(rf"$c_P$ — {pair}  ($T_\mathrm{{steam}}$ = {T_steam:.0f} °C)",
                  fontsize=fs(9), fontweight="bold")
    fig.tight_layout()
    save_titled_and_paper(fig, plots_dir, base_name)
    return True


def _plot_metric_grid(designs_dir, plots_dir, pairs, T_steam,
                      loader, cbar_label, suptitle, base_name,
                      fmt="{:.1f}", cmap_name="plasma_r"):
    """Combined 2×3 figure of one heatmap per fluid pair (shared colorbar).

    ``loader(pair, ls_vals, t_src_vals)`` produces a per-pair (LS × T_src)
    numpy grid. The result is written twice via ``save_titled_and_paper``:
    ``plots/<base_name>.png`` (no suptitle, paper variant) and
    ``plots/<base_name>_titled.png`` (with suptitle).

    Defaults to ``plasma_r`` (lower-is-better, cost-style); pass
    ``cmap_name="viridis"`` for higher-is-better performance metrics
    (COP, ε, T_lift).
    """
    # Pre-compute the UNION of LS and T_src values across every pair so
    # the x/y axes line up between panels. Pairs that don't have a design
    # at some cell (e.g. R1270 has no T_src = 60 °C) get NaN there and
    # render as grey, instead of having the column dropped from the panel.
    ls_union, tsrc_union = set(), set()
    for pair in pairs:
        ls_vals, t_src_vals = _discover_grid(designs_dir, pair)
        ls_union.update(ls_vals)
        tsrc_union.update(t_src_vals)
    ls_axis = sorted(ls_union)
    tsrc_axis = sorted(tsrc_union)
    if not ls_axis or not tsrc_axis:
        return None

    # Gather the per-pair grids on the COMMON axis.
    grids = {}
    for pair in pairs:
        g = loader(pair, ls_axis, tsrc_axis)
        if g is None:
            continue
        grids[pair] = g
    if not grids or all(np.all(np.isnan(g)) for g in grids.values()):
        return None

    all_vals = np.concatenate([g[~np.isnan(g)].ravel()
                               for g in grids.values()
                               if not np.all(np.isnan(g))])
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#cccccc")

    n = len(pairs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    # Double-column Elsevier figure (190 mm wide). Each panel is ~2.5"
    # — annotation font drops to 7 pt to keep numbers from running into
    # each other at journal scale.
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(COL_DOUBLE_IN, 3.6),
                              squeeze=False)

    last_im = None
    for idx, pair in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        if pair not in grids:
            ax.set_title(pair, fontsize=fs(9), fontweight="bold")
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#888888", fontsize=fs(8))
            ax.set_xticks([]); ax.set_yticks([])
            continue
        g = grids[pair]
        last_im = _draw_metric_heatmap(ax, g, ls_axis, tsrc_axis,
                                        vmin, vmax, cmap, fmt=fmt,
                                        fontsize=fs(7))
        ax.set_title(pair, fontsize=fs(9), fontweight="bold")
        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"$T_\mathrm{src,in}$  [°C]")
        if idx % ncols == 0:
            ax.set_ylabel(r"Lift share $\mathit{LS}$  [-]")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    if last_im is not None:
        fig.tight_layout(rect=[0, 0, 0.93, 0.93])
        cbar_ax = fig.add_axes([0.945, 0.10, 0.012, 0.80])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label(cbar_label)

    fig.suptitle(rf"{suptitle}  ($T_\mathrm{{steam}}$ = {T_steam:.0f} °C)",
                  fontsize=fs(10), fontweight="bold")
    save_titled_and_paper(fig, plots_dir, base_name)
    return os.path.join(plots_dir, f"{base_name}.pdf")


# ── Categorical (feasibility) heatmaps ────────────────────────────────────
# Two heatmap variants (one per row): Ommen envelope status (OK / V_ONLY /
# HARD) and thermodynamic-solver status (none / T_crit_c1 / T_crit_c2 /
# Pcrit_c1 / convergence). Both rely on integer-coded grids paired with an
# index→(color, label) map.

OMMEN_STATUS_PALETTE = {
    0: (STATUS_COLORS["OK"],      "OK (within Ommen envelope)"),
    1: (STATUS_COLORS["V_ONLY"],  "V_ONLY (operable per Annex 58 evidence)"),
    2: (STATUS_COLORS["HARD"],    "HARD (Ommen p / T_disch violation)"),
    3: (STATUS_COLORS["NOSOLVE"], "NOSOLVE (no thermo solution)"),
}

# Thermodynamic-only palette. Ordered by failure mode severity.
THERMO_STATUS_PALETTE = {
    0: ("#66bb6a", "feasible (thermo OK)"),
    1: ("#ef6c00", "T_crit_c1 (cycle 1 over-critical)"),
    2: ("#d84315", "T_crit_c2 (cycle 2 over-critical)"),
    3: ("#c62828", "Pcrit_c1 (cycle 1 over critical pressure)"),
    4: ("#6d4c41", "convergence (TESPy failed)"),
}

THERMO_LABEL_TO_CODE = {
    "":             0,
    "T_crit_c1":    1,
    "T_crit_c2":    2,
    "Pcrit_c1":     3,
    "convergence": 4,
}


def _draw_categorical_heatmap(ax, grid, ls_vals, t_src_vals, cmap,
                               cell_labels=None, fontsize=fs(9)):
    """Render a discrete-status (LS × T_src) heatmap into ``ax``.

    ``grid`` is integer-coded (NaN for missing); ``cmap`` is a
    ``ListedColormap`` with one entry per category. Cell annotation is
    drawn from the optional ``cell_labels`` 2-D list (e.g. short reason
    strings); when ``None`` no text is drawn.
    """
    masked = np.ma.masked_invalid(grid)
    n_codes = cmap.N
    im = ax.imshow(masked, cmap=cmap, vmin=-0.5, vmax=n_codes - 0.5,
                   aspect="equal", origin="upper")
    for i in range(len(ls_vals)):
        for j in range(len(t_src_vals)):
            if cell_labels is None:
                continue
            txt = cell_labels[i][j]
            if not txt:
                continue
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color="black")
    ax.set_xticks(range(len(t_src_vals)))
    ax.set_xticklabels([f"{T:g}" for T in t_src_vals])
    ax.set_yticks(range(len(ls_vals)))
    ax.set_yticklabels([f"{ls}%" for ls in ls_vals])
    return im


def _load_ommen_status_grid(case_dir, T_steam, pair, ls_vals, t_src_vals):
    """Build integer-coded (LS × T_src) Ommen-status grid.

    Codes match ``OMMEN_STATUS_PALETTE``: 0 OK, 1 V_ONLY, 2 HARD, 3 NOSOLVE.
    """
    enriched_csv = os.path.join(case_dir,
                                f"case_steam_{int(T_steam)}_enriched.csv")
    if not os.path.exists(enriched_csv):
        return None
    df = pd.read_csv(enriched_csv)
    df = df[df["pair"] == pair]
    status_to_code = {"OK": 0, "V_ONLY": 1, "HARD": 2, "NOSOLVE": 3}
    grid = np.full((len(ls_vals), len(t_src_vals)), np.nan)
    for i, ls_pct in enumerate(ls_vals):
        for j, T in enumerate(t_src_vals):
            row = df[(np.isclose(df["ls"], ls_pct / 100.0))
                     & (np.isclose(df["T_src"], T))]
            if row.empty:
                continue
            status = str(row.iloc[0].get("status_modern")
                          or classify_status(row.iloc[0]))
            grid[i, j] = status_to_code.get(status, np.nan)
    return grid


def _load_thermo_status_grid(case_dir, T_steam, pair, ls_vals, t_src_vals):
    """Build integer-coded (LS × T_src) thermodynamic-solver status grid.

    Reads ``no_solve_class`` directly (empty string = feasible).
    """
    enriched_csv = os.path.join(case_dir,
                                f"case_steam_{int(T_steam)}_enriched.csv")
    if not os.path.exists(enriched_csv):
        return None
    df = pd.read_csv(enriched_csv)
    df = df[df["pair"] == pair]
    grid = np.full((len(ls_vals), len(t_src_vals)), np.nan)
    for i, ls_pct in enumerate(ls_vals):
        for j, T in enumerate(t_src_vals):
            row = df[(np.isclose(df["ls"], ls_pct / 100.0))
                     & (np.isclose(df["T_src"], T))]
            if row.empty:
                continue
            cls = str(row.iloc[0].get("no_solve_class") or "")
            if cls == "nan":
                cls = ""
            grid[i, j] = THERMO_LABEL_TO_CODE.get(cls, 0)
    return grid


def _plot_categorical_grid(designs_dir, plots_dir, pairs, T_steam,
                            loader, palette, suptitle, base_name):
    """2×3 categorical heatmap (one panel per fluid pair, legend instead of cbar).

    Uses the union of (LS, T_src) axes across all pairs so every panel
    has the same x/y grid; pairs that don't have a design for a given
    cell render that cell as NaN (grey).
    """
    ls_union, tsrc_union = set(), set()
    for pair in pairs:
        ls_vals, t_src_vals = _discover_grid(designs_dir, pair)
        ls_union.update(ls_vals)
        tsrc_union.update(t_src_vals)
    ls_axis = sorted(ls_union)
    tsrc_axis = sorted(tsrc_union)
    if not ls_axis or not tsrc_axis:
        return None

    grids = {}
    for pair in pairs:
        g = loader(pair, ls_axis, tsrc_axis)
        if g is None:
            continue
        grids[pair] = g
    if not grids:
        return None

    codes = sorted(palette.keys())
    colors = [palette[c][0] for c in codes]
    cmap = ListedColormap(colors)

    n = len(pairs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    # Extra height reserves room for the legend at the bottom so it does
    # not collide with the x-axis labels.
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(COL_DOUBLE_IN, 4.2),
                              squeeze=False)

    for idx, pair in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        if pair not in grids:
            ax.set_title(pair, fontsize=fs(9), fontweight="bold")
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#888888", fontsize=fs(8))
            ax.set_xticks([]); ax.set_yticks([])
            continue
        g = grids[pair]
        _draw_categorical_heatmap(ax, g, ls_axis, tsrc_axis, cmap,
                                   fontsize=fs(7))
        ax.set_title(pair, fontsize=fs(9), fontweight="bold")
        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"$T_\mathrm{src,in}$  [°C]")
        if idx % ncols == 0:
            ax.set_ylabel(r"Lift share $\mathit{LS}$  [-]")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    legend_handles = [mpatches.Patch(color=palette[c][0], label=palette[c][1])
                      for c in codes]
    # Reserve ~18 % of the figure height at the bottom for the legend so
    # it sits clear of the x-axis labels; suptitle takes ~7 % at the top.
    fig.tight_layout(rect=[0, 0.18, 1, 0.93])
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=min(2, len(legend_handles)),
               fontsize=fs(7), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(rf"{suptitle}  ($T_\mathrm{{steam}}$ = {T_steam:.0f} °C)",
                  fontsize=fs(10), fontweight="bold")
    save_titled_and_paper(fig, plots_dir, base_name)
    return os.path.join(plots_dir, f"{base_name}.pdf")


# ── Per-design agent ──────────────────────────────────────────────────────
def run_agent_for_design(pair, T_steam):
    """Build both Z+C_D figures for one fluid pair × T_steam case.

    This is the "agent" entry point: invoking it for each (pair, T_steam) is
    enough to reproduce the full set of plots. ``run_all`` is just a thin
    loop that drives this agent across every available design.

    Outputs land in ``results/case_steam_<T>/designs/<pair>/``:
        cdz_sensitivity.png   (two stacked-bar panels of C_D + Z)
        cP_heatmap.png        (c_P [EUR/GJ_ex] over the T_src × LS grid)
    """
    case_dir = case_results_dir(T_steam)
    designs_dir = os.path.join(case_dir, "designs")
    data_dir = case_data_dir(T_steam)
    if not os.path.isdir(designs_dir):
        print(f"  [skip] no designs dir for T_steam={T_steam:g} C")
        return False

    pdir = _pair_dir(designs_dir, pair)
    if not os.path.isdir(pdir):
        print(f"  [skip] T_steam={T_steam:g} C  {pair}: pair folder missing")
        return False

    ls_vals, t_src_vals = _discover_grid(designs_dir, pair)

    # Drop legacy PNG / heatmap files from earlier runs so the design
    # folder stays clean.
    for legacy_name in ("cdz_heatmap.png", "cdz_sensitivity.png",
                         "cP_heatmap.png", "cdz_heatmap.pdf"):
        legacy = os.path.join(pdir, legacy_name)
        if os.path.exists(legacy):
            os.remove(legacy)

    ok_bar = _plot_bars(pair, designs_dir, pdir, "cdz_sensitivity",
                        T_steam, ls_vals, t_src_vals)
    ok_hm = _plot_heatmap(pair, data_dir, pdir, "cP_heatmap",
                          T_steam, ls_vals, t_src_vals)

    print(f"  [{'ok' if ok_bar else 'skip'}/"
          f"{'ok' if ok_hm else 'skip'}] "
          f"T_steam={T_steam:g} C  {pair:<14s} -> {pdir}")
    return ok_bar or ok_hm


def _discover_pairs(designs_dir):
    """Return sorted list of fluid pairs (e.g. ``R290/R600``) present on disk."""
    if not os.path.isdir(designs_dir):
        return []
    pairs = []
    for name in sorted(os.listdir(designs_dir)):
        full = os.path.join(designs_dir, name)
        if not os.path.isdir(full) or "_" not in name:
            continue
        f1, f2 = name.split("_", 1)
        pairs.append(f"{f1}/{f2}")
    return pairs


# ── Metric registry ───────────────────────────────────────────────────────
# Single source of truth for the 13 continuous + 2 categorical per-design
# heatmaps. Each entry is consumed by ``run_all`` and produces a pair of
# PNGs (``<base_name>.png`` paper variant + ``<base_name>_titled.png``).

def _build_continuous_metrics(data_dir, designs_dir, case_dir, T_steam):
    """Return the 13 continuous-metric specs as (label, kwargs) tuples for
    ``_plot_metric_grid``. Closes ``data_dir`` / ``case_dir`` / ``T_steam``
    into the loader callables so the run_all driver only needs the spec."""

    def base_loader(col, transform=None):
        return lambda p, ls, t: _load_base_grid(
            data_dir, p, ls, t, col, transform=transform)

    def enriched_loader(row_to_value):
        return lambda p, ls, t: _load_enriched_grid(
            case_dir, T_steam, p, ls, t, row_to_value)

    return [
        ("Total PEC",  dict(
            # CSV column "PEC [EUR/kW]" is actually TCI/kW (F_INSTALL × PEC).
            # Divide it out to recover the component-only PEC/kW.
            loader=base_loader("PEC [EUR/kW]",
                                transform=lambda v: v / F_INSTALL),
            cbar_label=r"PEC / $\dot{Q}_\mathrm{H}$  [EUR/kW]",
            suptitle=r"Total PEC per kW heat",
            base_name="PEC_per_kW_heatmap",
            fmt="{:.0f}", cmap_name="plasma_r")),
        ("Total TCI",  dict(
            loader=base_loader("PEC [EUR/kW]"),
            cbar_label=r"TCI / $\dot{Q}_\mathrm{H}$  [EUR/kW]",
            suptitle=r"Total TCI per kW heat",
            base_name="TCI_per_kW_heatmap",
            fmt="{:.0f}", cmap_name="plasma_r")),
        ("c_P",        dict(
            loader=base_loader("c_P [EUR/GJ]"),
            cbar_label=r"$c_P$  [EUR/GJ$_{ex}$]",
            suptitle=r"$c_P$ heatmap per design",
            base_name="cP_heatmap",
            fmt="{:.1f}", cmap_name="plasma_r")),
        ("COP",        dict(
            loader=base_loader("COP"),
            cbar_label=r"$\mathrm{COP}$  [-]",
            suptitle=r"Coefficient of performance $\mathrm{COP}$ per design",
            base_name="COP_heatmap",
            fmt="{:.2f}", cmap_name="viridis")),
        ("epsilon",    dict(
            loader=base_loader("epsilon", transform=lambda v: v * 100.0),
            cbar_label=r"$\varepsilon_\mathrm{tot}$  [%]",
            suptitle=r"Exergetic efficiency $\varepsilon_\mathrm{tot}$ per design",
            base_name="epsilon_heatmap",
            fmt="{:.1f}", cmap_name="viridis")),
        ("LCOH",       dict(
            loader=lambda p, ls, t: _load_LCOH_grid(data_dir, p, ls, t),
            cbar_label=r"LCOH  [EUR/MWh$_\mathrm{th}$]",
            suptitle="LCOH heatmap per design (heat basis)",
            base_name="LCOH_heatmap",
            fmt="{:.0f}", cmap_name="plasma_r")),
        ("p_max COMP1", dict(
            loader=enriched_loader(lambda r: float(r["p_high_c1 [bar]"])),
            cbar_label=r"$p_\mathrm{max,COMP1}$  [bar]",
            suptitle=r"Maximum pressure in COMP1 ($p_\mathrm{max,COMP1}$)",
            base_name="p_max_COMP1_heatmap",
            fmt="{:.1f}", cmap_name="magma_r")),
        ("p_max COMP2", dict(
            loader=enriched_loader(lambda r: float(r["p_high_c2 [bar]"])),
            cbar_label=r"$p_\mathrm{max,COMP2}$  [bar]",
            suptitle=r"Maximum pressure in COMP2 ($p_\mathrm{max,COMP2}$)",
            base_name="p_max_COMP2_heatmap",
            fmt="{:.1f}", cmap_name="magma_r")),
        ("pr cycle 1", dict(
            loader=enriched_loader(
                lambda r: float(r["p_high_c1 [bar]"])
                            / float(r["p_low_c1 [bar]"])),
            cbar_label=r"$\Pi$ cycle 1  [-]",
            suptitle="Pressure ratio in lower cycle (cycle 1)",
            base_name="pr_lower_heatmap",
            fmt="{:.2f}", cmap_name="magma_r")),
        ("pr cycle 2", dict(
            loader=enriched_loader(
                lambda r: float(r["p_high_c2 [bar]"])
                            / float(r["p_low_c2 [bar]"])),
            cbar_label=r"$\Pi$ cycle 2  [-]",
            suptitle="Pressure ratio in upper cycle (cycle 2)",
            base_name="pr_upper_heatmap",
            fmt="{:.2f}", cmap_name="magma_r")),
        ("T_max upper", dict(
            loader=enriched_loader(lambda r: float(r["T_disch_c2 [°C]"])),
            cbar_label=r"$T_\mathrm{cond,c2}$  [°C]",
            suptitle=r"Maximum temperature in upper cycle ($T_\mathrm{cond,c2}$)",
            base_name="T_max_upper_heatmap",
            fmt="{:.1f}", cmap_name="inferno")),
        ("T_min lower", dict(
            loader=enriched_loader(lambda r: float(r["T_evap_c1 [°C]"])),
            cbar_label=r"$T_\mathrm{evap,c1}$  [°C]",
            suptitle=r"Minimum temperature in lower cycle ($T_\mathrm{evap,c1}$)",
            base_name="T_min_lower_heatmap",
            fmt="{:.1f}", cmap_name="cividis")),
        ("T_lift",     dict(
            loader=enriched_loader(
                lambda r: float(r["T_cond_c2 [°C]"])
                            - float(r["T_evap_c1 [°C]"])),
            cbar_label=r"$\Delta T_\mathrm{lift}$  [K]",
            suptitle=r"Temperature lift ($T_\mathrm{cond,c2} - T_\mathrm{evap,c1}$)",
            base_name="T_lift_heatmap",
            fmt="{:.1f}", cmap_name="viridis")),
    ]


def run_all(T_steams=None):
    """Run the per-design agent across every fluid pair × T_steam case."""
    if T_steams is None:
        T_steams = T_STEAMS_TO_RUN
    for T_steam in T_steams:
        case_dir = case_results_dir(T_steam)
        designs_dir = os.path.join(case_dir, "designs")
        data_dir = case_data_dir(T_steam)
        plots_dir = case_plots_dir(T_steam)
        os.makedirs(plots_dir, exist_ok=True)
        pairs = _discover_pairs(designs_dir)
        if not pairs:
            print(f"[T_steam={T_steam:g} C] no designs found, skipping")
            continue
        print(f"[T_steam={T_steam:g} C] {len(pairs)} pairs: {pairs}")
        for pair in pairs:
            run_agent_for_design(pair, T_steam)

        # 13 continuous-metric per-design heatmaps (paper + titled variants).
        for label, kwargs in _build_continuous_metrics(
                data_dir, designs_dir, case_dir, T_steam):
            out = _plot_metric_grid(designs_dir, plots_dir, pairs, T_steam,
                                     **kwargs)
            if out:
                print(f"  [ok] T_steam={T_steam:g} C  {label:<12s} heatmap "
                      f"-> {out}")

        # 2 categorical feasibility heatmaps (Ommen envelope + thermo).
        for label, palette, loader, suptitle, base_name in (
            ("feasibility_ommen", OMMEN_STATUS_PALETTE,
             lambda p, ls, t: _load_ommen_status_grid(
                 case_dir, T_steam, p, ls, t),
             "Ommen feasibility envelope per design",
             "feasibility_ommen_heatmap"),
            ("feasibility_thermo", THERMO_STATUS_PALETTE,
             lambda p, ls, t: _load_thermo_status_grid(
                 case_dir, T_steam, p, ls, t),
             "Thermodynamic feasibility per design",
             "feasibility_thermo_heatmap"),
        ):
            out = _plot_categorical_grid(designs_dir, plots_dir, pairs,
                                           T_steam, loader=loader,
                                           palette=palette,
                                           suptitle=suptitle,
                                           base_name=base_name)
            if out:
                print(f"  [ok] T_steam={T_steam:g} C  {label:<20s} -> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        T_list = [float(x) for x in sys.argv[1:]]
    else:
        T_list = None
    run_all(T_list)
