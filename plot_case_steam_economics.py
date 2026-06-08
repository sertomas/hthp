"""
plot_case_steam_economics.py — Visualise the exergoeconomic results
for one case-study T_steam (modern envelope, LS ∈ {0.30, 0.40, 0.50}).

main.py invokes this module once per ``T_steam`` ∈ T_STEAMS_TO_RUN; paths
are repointed at runtime via ``_set_paths_for_T_steam``. The single-T
default is ``T_STEAM_CASE_DEFAULT`` (110 °C).

Reads (from results/case_steam_<int(T_steam)>/economics/):
    economics_base.csv
    economics_pec_breakdown.csv
    economics_sensitivity_e1.csv
    economics_sensitivity_FLH.csv
    gas_heater/gas_heater.json     (for the gas-reference c_P with carbon)

Writes (to the same folder):
    cP_grid.png                 — c_P heatmap (panels per LS, fluid pair × T_src)
    PEC_per_kW_grid.png         — PEC/kW (TCI) (native scale, config.M_STEAM kg/s) with Annex 58 band
    PEC_components_only.png     — components-only PEC = TCI / F_INSTALL with band
    PEC_breakdown.png           — stacked-bar component breakdown for top designs
    cP_vs_e1c.png               — c_P sensitivity to electricity price
    cP_vs_FLH.png               — c_P sensitivity to full-load hours
    cP_vs_gas.png               — c_P at base electricity vs gas reference
    cP_sorted_by_T_src.png      — designs sorted by c_P, faceted by T_src
    cP_vs_ED_EL_by_Tsrc.png     — c_P against E_D + E_L, faceted by T_src
    cost_breakdown_per_design.png   — C_D + Z per design, by component group
    z_breakdown_per_design.png      — Z per design, by component group
    cost_breakdown_aggregated.png   — same, aggregated per fluid pair
    z_breakdown_aggregated.png      — same, aggregated per fluid pair
    tsatsaronis_quadrant.png        — improvement-priority quadrant per pair
    economic_vs_exergoeconomic_ranking.png
    best_designs_per_T_src.png      — best-design dashboard
    lift_share_vs_T_src.png         — LS trends vs T_src per fluid pair
    price_sensitivity_2d.png        — joint e1 / gas price sweep at base FLH
    price_sensitivity_2d_FLH<NNNN>.png — same, one per non-base FLH value
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from config import (
    BASE_CO2_PRICE, BASE_E1_C, BASE_FULL_LOAD_HOURS, BASE_GAS_C,
    PRICE_SENS_FRAC_RANGE, T_SOURCE_IN_RANGE, T_STEAM_CASE_DEFAULT,
    m_steam_label,
)
from economics import F_INSTALL
from plot_common import (
    COL_DOUBLE_IN, COL_SINGLE_IN,
    GROUP_STYLE, GROUP_ORDER, GROUP_COLORS, GROUP_LABELS, GROUP_MEMBERS,
    GROUP_ORDER_SPLIT_MOT, GROUP_COLORS_SPLIT_MOT,
    GROUP_MEMBERS_SPLIT_MOT,
    save_titled_and_paper,
    fs,
)


def _op_string():
    """Operating-assumption tag used in every economics plot title."""
    return (f"FLH = {BASE_FULL_LOAD_HOURS:.0f} h/a, "
            f"e1 = {BASE_E1_C:.0f} EUR/MWh, "
            f"gas = {BASE_GAS_C:.0f} EUR/MWh")


# Gas reference: computed once via the proper exergy-based gas heater model
# so the c_P axis stays apples-to-apples with HTHP. The previous shortcut
# `(BASE_GAS_C / 0.36) / 0.90` was buggy on two counts: it used the
# ct/kWh → EUR/GJ factor (0.36) on a value in EUR/MWh, and it produced a
# heat-basis c_P (per Q_steam) instead of exergy-basis (per E_P_steam),
# which is what HTHP c_P uses.

_gas_reference_cache = {}


def _gas_reference_cP_at(c_gas_eur_mwh, full_load_hours=None):
    """c_P_gas [EUR/GJ_exergy] from simulate_gas_heater + run_economics_gas_heater.

    The cost balance c_P = slope · c_gas + co2_offset is linear in c_gas
    (retrofit ⇒ Z_gas = 0) with a constant CO2 surcharge baked in at
    BASE_CO2_PRICE, so we cache the slope and offset and reconstruct any
    point along the gas-price axis without re-simulating.
    """
    if full_load_hours is None:
        full_load_hours = BASE_FULL_LOAD_HOURS
    if "slope" not in _gas_reference_cache:
        # Local imports keep TESPy-heavy modules out of the cold plot path
        from models import simulate_gas_heater
        from economics import run_economics_gas_heater
        gas_sim = simulate_gas_heater(eta_gas=0.90,
                                       T_steam_override=_T_STEAM_CURRENT)
        # Pure gas-fuel cost (no CO2) → gives the gas-price slope
        eco_no_co2 = run_economics_gas_heater(gas_sim, full_load_hours,
                                               BASE_GAS_C / 10.0,
                                               co2_price_eur_per_t=0.0)
        # CO2 component is independent of gas price, so just take its delta
        eco_with_co2 = run_economics_gas_heater(gas_sim, full_load_hours,
                                                 BASE_GAS_C / 10.0,
                                                 co2_price_eur_per_t=BASE_CO2_PRICE)
        _gas_reference_cache["slope"] = eco_no_co2["c_P"] / BASE_GAS_C
        _gas_reference_cache["co2_offset"] = eco_with_co2["c_P"] - eco_no_co2["c_P"]
    return (_gas_reference_cache["slope"] * c_gas_eur_mwh
            + _gas_reference_cache["co2_offset"])


def _gas_reference_cP():
    """c_P_gas at base prices (most-used helper for plots)."""
    return _gas_reference_cP_at(BASE_GAS_C)


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# Module-level path globals are rewritten by ``_set_paths_for_T_steam`` at
# the start of ``main`` for each case-study T_steam; the placeholders below
# only matter if the helpers below are called before that override.
DATA_DIR = ""
PLOTS_DIR = ""
BASE_CSV = ""
BREAKDOWN_CSV = ""
SENS_CSV = ""
SENS_FLH_CSV = ""
DESIGNS_DIR = ""
_T_STEAM_CURRENT = 0.0   # set at the same time, used by suptitle strings


def _set_paths_for_T_steam(T_steam):
    """Rewrite the module-level path globals for a given T_steam.

    Also invalidates the gas-reference c_P cache (it depends on T_steam
    because steam exergy E_P is T-dependent).
    """
    global DATA_DIR, PLOTS_DIR, BASE_CSV, BREAKDOWN_CSV, SENS_CSV, SENS_FLH_CSV
    global DESIGNS_DIR, _T_STEAM_CURRENT, _gas_reference_cache
    from config import case_results_dir, case_data_dir, case_plots_dir
    case_dir = case_results_dir(T_steam)
    DATA_DIR = case_data_dir(T_steam)
    PLOTS_DIR = case_plots_dir(T_steam)
    BASE_CSV = os.path.join(DATA_DIR, "economics_base.csv")
    BREAKDOWN_CSV = os.path.join(DATA_DIR, "economics_pec_breakdown.csv")
    SENS_CSV = os.path.join(DATA_DIR, "economics_sensitivity_e1.csv")
    SENS_FLH_CSV = os.path.join(DATA_DIR, "economics_sensitivity_FLH.csv")
    DESIGNS_DIR = os.path.join(case_dir, "designs")
    _T_STEAM_CURRENT = T_steam
    # Force the gas-reference cache to recompute at the new T_steam
    _gas_reference_cache = {}

# Component grouping for every breakdown plot. COMP+MOT of each cycle are
# bundled into one bar segment (matches Ommen Tab. 4 cost-row granularity);
# everything else (VAL1, VAL2, SRC_HX, IHX, SNK_HX) is shown separately.
# Colours / labels / member list come from ``plot_common.GROUP_STYLE``.
COMP_GROUPS = dict(GROUP_MEMBERS)

# Project 68 (2025) "per unit, no integration" supplier band for 0.5–3 MWth,
# 110–150 °C — see Figure 1-6 of HPT-PR68-2.
ANNEX58_BAND_LOW = 400.0   # EUR/kW
ANNEX58_BAND_HIGH = 700.0  # EUR/kW


def _load():
    base = pd.read_csv(BASE_CSV)
    bk = pd.read_csv(BREAKDOWN_CSV)
    sens = pd.read_csv(SENS_CSV)
    sens_flh = (pd.read_csv(SENS_FLH_CSV)
                if os.path.exists(SENS_FLH_CSV) else None)

    # Add components-only PEC (purchase equipment cost, no installation)
    # for fair comparison to Annex 58 / Project 68 supplier €/kW which
    # excludes integration.
    base["PEC_components_only [EUR/kW]"] = (
        base["PEC [EUR/kW]"] / F_INSTALL
    ).round(1)
    return base, bk, sens, sens_flh


# NOTE: the old `plot_cP_grid`, `plot_pec_per_kW_TCI` and
# `plot_pec_per_kW_components` produced ``cP_grid.png`` /
# ``PEC_per_kW_grid.png`` / ``PEC_components_only.png`` — the same data as
# the 2×3 per-design heatmaps in ``plot_cdz_sensitivity_per_design.py``
# (``cP_heatmap.png`` / ``TCI_per_kW_heatmap.png`` /
# ``PEC_per_kW_heatmap.png``), just with a different layout. They were
# removed to drop the duplicates; the Annex 58 band overlay that they
# carried has been ported into the new TCI / PEC heatmaps.


# ── Plot 4: PEC component breakdown for the cheapest 12 designs by c_P ──────

def plot_pec_breakdown(base, bk):
    top = base.nsmallest(12, "c_P [EUR/GJ]").copy()
    top["label"] = (top["pair"] + "\nLS=" + (top["ls"]*100).astype(int).astype(str)
                    + "%, T_src=" + top["T_src"].astype(int).astype(str) + "°C\n"
                    + "c_P=" + top["c_P [EUR/GJ]"].round(0).astype(int).astype(str)
                    + r" EUR/GJ$_{ex}$")

    # PEC group list — VAL1, VAL2 have zero PEC by design so they would
    # render as empty stacks; skip them in this plot.
    groups = [g for g in GROUP_ORDER if g not in ("VAL1", "VAL2")]

    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, 4.2))
    bottom = np.zeros(len(top))
    for group in groups:
        members = COMP_GROUPS[group]
        vals = []
        for _, r in top.iterrows():
            sub = bk[(bk["pair"] == r["pair"])
                     & (bk["ls"] == r["ls"])
                     & (bk["T_src"] == r["T_src"])
                     & (bk["component"].isin(members))]
            v = float(sub["PEC [EUR/kW]"].sum()) if not sub.empty else 0.0
            vals.append(v)
        ax.bar(range(len(top)), vals, bottom=bottom,
               color=GROUP_COLORS[group],
               label=GROUP_LABELS[group],
               edgecolor="white", linewidth=0.3)
        bottom += np.array(vals)

    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top["label"], fontsize=fs(8), rotation=30, ha="right")
    ax.set_ylabel("PEC per kW heating  [EUR/kW]")
    ax.axhspan(ANNEX58_BAND_LOW * F_INSTALL, ANNEX58_BAND_HIGH * F_INSTALL,
               color="#2e7d32", alpha=0.10, zorder=0,
               label=f"Annex 58 band × F_INSTALL ({F_INSTALL:.1f})")
    ax.legend(loc="upper right", ncol=2, fontsize=fs(7), framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(rf"PEC breakdown — 12 cheapest designs  "
                  rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
                  fontsize=fs(10), fontweight="bold")
    fig.tight_layout()
    save_titled_and_paper(fig, PLOTS_DIR, "PEC_breakdown")
    plt.close(fig)


# ── Plot 5: c_P sensitivity to electricity price ───────────────────────────

def plot_cP_vs_e1(base, sens):
    # Pick the 12 best designs (by base c_P) for clarity
    best12 = base.nsmallest(12, "c_P [EUR/GJ]")
    keys = list(zip(best12["pair"], best12["ls"], best12["T_src"]))

    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, 4.0))
    # Tab10 + tab10b extension for 12 distinguishable colors
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[:12]
    for k, (pair, ls, T_src) in enumerate(keys):
        sub = sens[(sens["pair"] == pair)
                   & (sens["ls"] == ls)
                   & (sens["T_src"] == T_src)].sort_values("e1_c [EUR/MWh]")
        if sub.empty:
            continue
        label = rf"{pair} $\mathit{{LS}}={ls*100:.0f}$ % $T_\mathrm{{src,in}}={T_src:.0f}$ °C"
        ax.plot(sub["e1_c [EUR/MWh]"], sub["c_P [EUR/GJ]"],
                marker="o", markersize=4, linewidth=1.4,
                color=colors[k], label=label)

    # Gas heater horizontal reference: exergy-based c_P at base gas price
    gas_c_eur_GJ = _gas_reference_cP()
    ax.axhline(gas_c_eur_GJ, color="black", linestyle="--", linewidth=1.5,
               label=rf"Gas heater @ {BASE_GAS_C:.0f} EUR/MWh, η=0.90"
                     rf"  ($c_P$ = {gas_c_eur_GJ:.0f} EUR/GJ$_{{ex}}$)")
    ax.axvline(BASE_E1_C, color="grey", linestyle=":", alpha=0.6,
               label=rf"Base $c_\mathrm{{el,0}} = {BASE_E1_C:.0f}$ EUR/MWh")

    ax.set_xlabel(r"Electricity price $c_\mathrm{el,0}$  [EUR/MWh]")
    ax.set_ylabel(r"$c_P$  [EUR/GJ$_{ex}$]")
    fig.suptitle(
        rf"$c_P$ vs electricity price — 12 cheapest designs  "
        rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fontsize=fs(10), fontweight="bold",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=fs(6), ncol=1, framealpha=0.90)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_titled_and_paper(fig, PLOTS_DIR, "cP_vs_e1c")
    plt.close(fig)


# ── Plot: c_P sensitivity to full-load hours ───────────────────────────────

def plot_cP_vs_FLH(base, sens_flh):
    if sens_flh is None or sens_flh.empty:
        print("  ! no FLH sensitivity data; skipping cP_vs_FLH")
        return
    best12 = base.nsmallest(12, "c_P [EUR/GJ]")
    keys = list(zip(best12["pair"], best12["ls"], best12["T_src"]))

    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, 4.0))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[:12]
    for k, (pair, ls, T_src) in enumerate(keys):
        sub = sens_flh[(sens_flh["pair"] == pair)
                        & (sens_flh["ls"] == ls)
                        & (sens_flh["T_src"] == T_src)
                        ].sort_values("full_load_hours [h/a]")
        if sub.empty:
            continue
        label = rf"{pair} $\mathit{{LS}}={ls*100:.0f}$ % $T_\mathrm{{src,in}}={T_src:.0f}$ °C"
        ax.plot(sub["full_load_hours [h/a]"], sub["c_P [EUR/GJ]"],
                marker="o", markersize=4, linewidth=1.4,
                color=colors[k], label=label)

    # Gas reference (retrofit: Z_gas = 0, so c_P_gas independent of FLH)
    gas_c_eur_GJ = _gas_reference_cP()
    ax.axhline(gas_c_eur_GJ, color="black", linestyle="--", linewidth=1.5,
               label=rf"Gas heater (retrofit) {gas_c_eur_GJ:.0f} EUR/GJ$_{{ex}}$")
    # Vertical at base FLH
    from config import BASE_FULL_LOAD_HOURS
    ax.axvline(BASE_FULL_LOAD_HOURS, color="grey", linestyle=":", alpha=0.6,
               label=rf"Base $\tau = {BASE_FULL_LOAD_HOURS:.0f}$ h/a")

    ax.set_xlabel(r"Full-load operating hours $\tau$  [h/a]")
    ax.set_ylabel(r"$c_P$  [EUR/GJ$_{ex}$]")
    fig.suptitle(
        rf"$c_P$ vs full-load hours — 12 cheapest designs  "
        rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fontsize=fs(10), fontweight="bold",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=fs(6), ncol=1, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_titled_and_paper(fig, PLOTS_DIR, "cP_vs_FLH")
    plt.close(fig)


# ── Plot 6: c_P at base price vs gas reference ─────────────────────────────

def plot_cP_vs_gas(base):
    # Sort by c_P ascending; color by fluid pair
    df = base.sort_values("c_P [EUR/GJ]").reset_index(drop=True)
    pair_colors = {p: c for p, c in zip(sorted(df["pair"].unique()),
                                          plt.cm.tab10(np.linspace(0, 1, 10)))}

    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, 3.5))
    for k, (_, r) in enumerate(df.iterrows()):
        ax.bar(k, r["c_P [EUR/GJ]"], color=pair_colors[r["pair"]],
               edgecolor="black", linewidth=0.2)

    gas_c_eur_GJ = _gas_reference_cP()
    ax.axhline(gas_c_eur_GJ, color="black", linestyle="--", linewidth=1.5,
               label=rf"Gas heater (retrofit) at {BASE_GAS_C:.0f} EUR/MWh, "
                     rf"η=0.90: $c_P$ = {gas_c_eur_GJ:.0f} EUR/GJ$_{{ex}}$")

    ax.set_xlabel(r"Designs (sorted by $c_P$, lowest left)")
    ax.set_ylabel(r"$c_P$  [EUR/GJ$_{ex}$]")
    fig.suptitle(rf"$c_P$ — all OK designs sorted  "
                  rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
                  fontsize=fs(10), fontweight="bold")
    handles = [mpatches.Patch(color=c, label=p) for p, c in pair_colors.items()]
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                              label=rf"Gas reference $c_P$ = {gas_c_eur_GJ:.0f} EUR/GJ$_{{ex}}$"))
    ax.legend(handles=handles, ncol=5, fontsize=fs(8), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_titled_and_paper(fig, PLOTS_DIR, "cP_vs_gas")
    plt.close(fig)


# ── Plot 7: c_P sorted within each T_source case study ─────────────────────

def plot_cP_sorted_by_T_src(base):
    """One panel per T_src; designs sorted by c_P inside each panel.

    The full-pool ranking (cP_vs_gas.png) collapses 5 case studies into one
    chart. This view keeps each T_src as its own ranking so the reader can
    pick the best fluid pair / lift share for a given source-water inlet.
    """
    T_src_vals = sorted(base["T_src"].unique())
    pair_colors = {p: c for p, c in zip(sorted(base["pair"].unique()),
                                          plt.cm.tab10(np.linspace(0, 1, 10)))}
    gas_c_eur_GJ = _gas_reference_cP()
    # Extra headroom for the vertical value labels above each bar.
    y_max = float(base["c_P [EUR/GJ]"].max()) * 1.22
    y_max = max(y_max, gas_c_eur_GJ * 1.25)

    # Vertical layout: 1 column × N rows so each panel gets the full
    # double-column width — at single-row 5-panel layout the 18 bars in
    # the T_src = 20 °C case would overlap unreadably at journal scale.
    n = len(T_src_vals)
    fig, axes = plt.subplots(n, 1, figsize=(COL_DOUBLE_IN, 1.5 * n + 0.8),
                             sharex=False, squeeze=False)
    for j, T_src in enumerate(T_src_vals):
        ax = axes[j][0]
        sub = (base[base["T_src"] == T_src]
               .sort_values("c_P [EUR/GJ]")
               .reset_index(drop=True))
        if sub.empty:
            ax.set_title(rf"$T_\mathrm{{src,in}} = {T_src:.0f}$ °C  (no OK designs)",
                         fontsize=fs(9))
            ax.axis("off")
            continue

        labels = [rf"{r['pair']}" + "\n" + rf"$\mathit{{LS}}={r['ls']*100:.0f}$ %"
                  for _, r in sub.iterrows()]
        colors = [pair_colors[r["pair"]] for _, r in sub.iterrows()]
        x = np.arange(len(sub))
        ax.bar(x, sub["c_P [EUR/GJ]"], color=colors,
               edgecolor="black", linewidth=0.3)
        ax.axhline(gas_c_eur_GJ, color="black", linestyle="--", linewidth=1.0)

        # value labels above each bar — at full row width the bars get
        # enough horizontal space that horizontal labels fit.
        for k, v in enumerate(sub["c_P [EUR/GJ]"]):
            ax.text(k, v + y_max * 0.01, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=fs(6))

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=fs(6))
        ax.set_title(rf"$T_\mathrm{{src,in}} = {T_src:.0f}$ °C  (n={len(sub)})",
                     fontsize=fs(8), fontweight="bold", loc="left")
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel(r"$c_P$  [EUR/GJ$_{ex}$]")

    handles = [mpatches.Patch(color=c, label=p)
               for p, c in pair_colors.items()
               if p in base["pair"].unique()]
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                              label=rf"Gas heater  $c_P$ = {gas_c_eur_GJ:.0f} EUR/GJ$_{{ex}}$"))
    fig.legend(handles=handles, ncol=min(len(handles), 4),
               loc="lower center", fontsize=fs(7), frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        rf"$c_P$ ranked per $T_\mathrm{{src,in}}$  "
        rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fontsize=fs(10), fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_titled_and_paper(fig, PLOTS_DIR, "cP_sorted_by_T_src")
    plt.close(fig)


# ── Plot 8 & 9: C_D + Z component breakdown per design ────────────────────

def _design_dir(pair, ls, T_src):
    f1, f2 = pair.split("/")
    ls_pct = int(round(ls * 100))
    return os.path.join(DESIGNS_DIR, f"{f1}_{f2}",
                         f"LS{ls_pct}_Tsrc{int(T_src)}")


def _load_cdz_per_design(base, value_col="C_D+Z [EUR/h]", groups=None):
    """Per-design cost-rate metric [EUR/h] grouped by ``groups``.

    Returns a DataFrame with columns: pair, ls, T_src, <group>... (one column
    per group in ``groups``) and a TOTAL column. One row per OK design.

    ``groups`` defaults to the bundled ``COMP_GROUPS`` (COMP+MOT per
    cycle); pass ``GROUP_MEMBERS_SPLIT_MOT`` to keep COMP and MOT as
    separate stack segments (used by the E_D breakdown).
    """
    if groups is None:
        groups = COMP_GROUPS
    rows = []
    for _, r in base.iterrows():
        pair, ls, T_src = r["pair"], float(r["ls"]), float(r["T_src"])
        path = os.path.join(_design_dir(pair, ls, T_src),
                            "exergoeco_components.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df[df["Component"] != "TOT"].copy()
        cdz = dict(zip(df["Component"], df[value_col].astype(float)))

        row = {"pair": pair, "ls": ls, "T_src": T_src}
        for group, members in groups.items():
            row[group] = sum(cdz.get(m, 0.0) for m in members)
        row["TOTAL"] = sum(row[g] for g in groups)
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_breakdown_per_design(cdz, ylabel, suptitle, fname):
    """One panel per fluid pair; bars per (T_src × LS) within the panel.

    The (T_src × LS) grid is the canonical 5×3 parametric envelope
    (T_src ∈ [20, 30, 40, 50, 60] °C × LS ∈ {0.30, 0.40, 0.50}) so every
    panel keeps the same 15 x-positions. Cells where the design did not
    converge are drawn as a greyed-out "infeasible" placeholder so the
    reader can see *where* the parametric study fails."""
    pairs = sorted(cdz["pair"].unique())
    n = len(pairs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    # Canonical 15-cell grid (T_src × LS) in row-major order.
    T_src_axis = list(T_SOURCE_IN_RANGE)
    ls_axis = [0.30, 0.40, 0.50]
    cells = [(T, ls) for T in T_src_axis for ls in ls_axis]

    # 2 × double-column width so each of the 2×3 fluid-pair panels gets
    # ~5 in of horizontal room. 15 bars per panel leaves >0.3 in per bar
    # — plenty for the rotated 6 pt "T20 LS30" labels.
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(COL_DOUBLE_IN * 2.0, 3.5 * nrows),
                              squeeze=False)

    y_max = float(cdz["TOTAL"].max()) * 1.18

    for idx, pair in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        sub = (cdz[cdz["pair"] == pair]
               .set_index(["T_src", "ls"]))
        labels = [f"T{int(T)} LS{int(ls*100)}" for T, ls in cells]
        x = np.arange(len(cells))
        bottom = np.zeros(len(cells))
        for group in COMP_GROUPS:
            vals = np.array([
                float(sub.loc[(T, ls), group]) if (T, ls) in sub.index else 0.0
                for T, ls in cells
            ])
            ax.bar(x, vals, bottom=bottom,
                   color=GROUP_COLORS[group],
                   edgecolor="black", linewidth=0.3,
                   label=group if idx == 0 else None)
            bottom += vals
        for k, (T, ls) in enumerate(cells):
            if (T, ls) in sub.index:
                v = float(sub.loc[(T, ls), "TOTAL"])
                ax.text(k, v + y_max * 0.01, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=fs(6),
                        fontweight="bold")
            else:
                ax.add_patch(plt.Rectangle(
                    (k - 0.4, 0), 0.8, y_max,
                    fill=True, color="#eeeeee", zorder=0,
                ))
                ax.text(k, y_max * 0.04, "infeasible",
                        ha="center", va="bottom", fontsize=fs(6),
                        rotation=90, color="#888")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=fs(6), rotation=90)
        ax.set_ylim(0, y_max)
        ax.set_title(pair, fontweight="bold", fontsize=fs(9))
        ax.grid(axis="y", alpha=0.25)
        if idx % ncols == 0:
            ax.set_ylabel(ylabel)

    # Hide any unused panels
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g)
               for g in COMP_GROUPS]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)),
               fontsize=fs(9), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(suptitle, fontsize=fs(12), fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_titled_and_paper(fig, PLOTS_DIR, fname.removesuffix(".pdf"))
    plt.close(fig)


def plot_cost_breakdown_per_design(base, cdz):
    _plot_breakdown_per_design(
        cdz,
        ylabel=r"$\dot{C}_D + \dot{Z}$  [EUR/h]",
        suptitle=rf"$\dot{{C}}_D + \dot{{Z}}$ per design  "
                 rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fname="cost_breakdown_per_design.pdf",
    )


def plot_z_breakdown_per_design(base, z):
    _plot_breakdown_per_design(
        z,
        ylabel=r"$\dot{Z}$  [EUR/h]",
        suptitle=rf"$\dot{{Z}}$ per design  "
                 rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fname="z_breakdown_per_design.pdf",
    )


# ── Components breakdown — best-LS-per-T_src view ──────────────────────────
# A pruned variant of ``_plot_breakdown_per_design``: for each (pair, T_src)
# combination we keep only the LS that gives the lowest c_P. This collapses
# the 11-18 bars per panel down to ≤ 5 (one per T_src), so the figure fits
# comfortably at double-column width with no rotated labels.

def _filter_best_ls(per_design_df, base=None, criterion="c_P"):
    """Keep rows whose (pair, ls, T_src) is the best LS per (pair, T_src) cell.

    ``criterion`` picks what "best" means:
      - ``"c_P"``     : LS that minimizes ``c_P [EUR/GJ]`` (looked up in
                        ``base``). Exergoeconomic optimum — used by the C_D+Z
                        plot.
      - ``"epsilon"`` : LS that maximizes ``epsilon`` (looked up in ``base``).
                        Thermodynamic optimum — used by E_D / exergy-balance
                        plots.
      - ``"Z_sum"``   : LS that minimizes the ``TOTAL`` column of
                        ``per_design_df`` itself. Capital-cost optimum — used
                        by the Z plot. ``base`` is unused in this branch.
    """
    if criterion == "Z_sum":
        idx = per_design_df.groupby(["pair", "T_src"])["TOTAL"].idxmin()
        best = per_design_df.loc[idx, ["pair", "ls", "T_src"]]
    elif criterion == "epsilon":
        idx = base.groupby(["pair", "T_src"])["epsilon"].idxmax()
        best = base.loc[idx, ["pair", "ls", "T_src"]]
    else:  # "c_P"
        idx = base.groupby(["pair", "T_src"])["c_P [EUR/GJ]"].idxmin()
        best = base.loc[idx, ["pair", "ls", "T_src"]]
    keep = set(zip(best["pair"], best["ls"].round(2), best["T_src"]))
    mask = per_design_df.apply(
        lambda r: (r["pair"], round(float(r["ls"]), 2),
                    float(r["T_src"])) in keep,
        axis=1,
    )
    return per_design_df[mask].copy()


def _plot_components_breakdown(per_design_df, ylabel, suptitle, fname,
                                groups=None, colors=None, labels=None,
                                total_fmt="{:.0f}"):
    """One panel per fluid pair; ≤ 5 bars per panel (one per T_src, best LS).

    Mirrors ``_plot_breakdown_per_design`` but with the (T_src × LS) grid
    collapsed to a single bar per T_src — the LS that gives the lowest
    c_P for that (pair, T_src). The chosen LS is shown as the x-tick
    annotation so the reader sees which lift share is "best" at each
    source temperature.

    ``groups`` / ``colors`` default to the bundled ``COMP_GROUPS`` /
    ``GROUP_COLORS``; pass the split-MOT views to render compressor and
    motor as separate stack segments (used by the E_D plot).
    """
    if groups is None:
        groups = COMP_GROUPS
    if colors is None:
        colors = GROUP_COLORS
    pairs = sorted(per_design_df["pair"].unique())
    n = len(pairs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    # Canonical T_src axis so every panel keeps the same x-axis even when a
    # fluid pair has no converged design at one of the source temperatures
    # (rendered as a greyed-out "infeasible" placeholder).
    T_src_axis = list(T_SOURCE_IN_RANGE)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(COL_DOUBLE_IN, 3.5 * nrows),
                              squeeze=False)
    y_max = float(per_design_df["TOTAL"].max()) * 1.18

    for idx, pair in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        sub = (per_design_df[per_design_df["pair"] == pair]
               .set_index("T_src"))
        xticklabels = [
            (f"T{int(T)}\nLS{int(sub.loc[T, 'ls']*100)}" if T in sub.index
             else f"T{int(T)}\n—")
            for T in T_src_axis
        ]
        x = np.arange(len(T_src_axis))
        bottom = np.zeros(len(T_src_axis))
        for group in groups:
            vals = np.array([
                float(sub.loc[T, group]) if T in sub.index else 0.0
                for T in T_src_axis
            ])
            ax.bar(x, vals, bottom=bottom,
                   color=colors[group],
                   edgecolor="black", linewidth=0.3,
                   label=group if idx == 0 else None)
            bottom += vals
        for k, T in enumerate(T_src_axis):
            if T in sub.index:
                v = float(sub.loc[T, "TOTAL"])
                ax.text(k, v + y_max * 0.01, total_fmt.format(v),
                        ha="center", va="bottom", fontsize=fs(7),
                        fontweight="bold")
            else:
                ax.add_patch(plt.Rectangle(
                    (k - 0.4, 0), 0.8, y_max,
                    fill=True, color="#eeeeee", zorder=0,
                ))
                ax.text(k, y_max * 0.04, "infeasible",
                        ha="center", va="bottom", fontsize=fs(7),
                        rotation=90, color="#888")
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, fontsize=fs(7))
        ax.set_ylim(0, y_max)
        ax.set_title(pair, fontweight="bold", fontsize=fs(9))
        ax.grid(axis="y", alpha=0.25)
        if idx % ncols == 0:
            ax.set_ylabel(ylabel)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [mpatches.Patch(color=colors[g],
                               label=(labels[g] if labels else g))
               for g in groups]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(5, len(handles)),
               fontsize=fs(8), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(suptitle, fontsize=fs(10), fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_titled_and_paper(fig, PLOTS_DIR, fname.removesuffix(".pdf"))
    plt.close(fig)


def plot_CDZ_components_breakdown(base, cdz):
    sub = _filter_best_ls(cdz, base, criterion="c_P")
    _plot_components_breakdown(
        sub,
        ylabel=r"$\dot{C}_D + \dot{Z}$  [EUR/h]",
        suptitle=rf"$\dot{{C}}_D + \dot{{Z}}$ — best $\mathit{{LS}}$ per "
                 rf"$T_\mathrm{{src,in}}$  "
                 rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fname="CDZ_components_breakdown_per_design.pdf",
    )


def plot_z_components_breakdown(base, z):
    sub = _filter_best_ls(z, criterion="Z_sum")
    _plot_components_breakdown(
        sub,
        ylabel=r"$\dot{Z}$  [EUR/h]",
        total_fmt="{:.1f}",
        suptitle=rf"$\dot{{Z}}$ — best $\mathit{{LS}}$ per "
                 rf"$T_\mathrm{{src,in}}$  "
                 rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fname="z_components_breakdown_per_design.pdf",
    )


def plot_ED_components_breakdown(base, ed):
    sub = _filter_best_ls(ed, base, criterion="epsilon")
    _plot_components_breakdown(
        sub,
        groups={g: GROUP_MEMBERS_SPLIT_MOT[g] for g in GROUP_ORDER_SPLIT_MOT},
        colors=GROUP_COLORS_SPLIT_MOT,
        ylabel=r"$\dot{E}_D$  [kW]",
        suptitle=rf"$\dot{{E}}_D$ — best $\mathit{{LS}}$ per "
                 rf"$T_\mathrm{{src,in}}$  "
                 rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fname="ED_components_breakdown_per_design.pdf",
    )


# ── System exergy balance breakdown (E_F = E_P + E_L + E_D) ─────────────────
# Same layout as the components breakdown but the stack is the three
# system-level exergy terms instead of the per-component decomposition.
# Pulled from the ``TOT`` row of each design's ``exergoeco_components.csv``.

EXERGY_BALANCE_GROUPS = ["E_P", "E_L", "E_D"]
EXERGY_BALANCE_COLORS = {
    "E_P": "#2ca02c",   # product — green
    "E_L": "#ff7f0e",   # loss    — orange
    "E_D": "#d62728",   # destruction — red
}
EXERGY_BALANCE_LABELS = {
    "E_P": r"$\dot{E}_P$ (product)",
    "E_L": r"$\dot{E}_L$ (loss)",
    "E_D": r"$\dot{E}_D$ (destruction)",
}


def _load_exergy_balance_per_design(base):
    """Per-design system exergy balance from the TOT row of each
    ``exergoeco_components.csv``.

    Returns a DataFrame with columns pair, ls, T_src, E_P, E_L, E_D, TOTAL
    (= E_F). One row per OK design.
    """
    rows = []
    for _, r in base.iterrows():
        pair, ls, T_src = r["pair"], float(r["ls"]), float(r["T_src"])
        path = os.path.join(_design_dir(pair, ls, T_src),
                            "exergoeco_components.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        tot = df[df["Component"] == "TOT"]
        if tot.empty:
            continue
        t = tot.iloc[0]
        rows.append({
            "pair":   pair,
            "ls":     ls,
            "T_src":  T_src,
            "E_P":    float(t["E_P [kW]"]),
            "E_L":    float(t["E_L [kW]"]),
            "E_D":    float(t["E_D [kW]"]),
            "TOTAL":  float(t["E_F [kW]"]),
        })
    return pd.DataFrame(rows)


def plot_exergy_balance_breakdown(base):
    """Stack of E_P / E_L / E_D per (pair, T_src) at the best-c_P LS."""
    bal = _load_exergy_balance_per_design(base)
    if bal.empty:
        print("  ! per-design exergoeco CSVs missing — skipping exergy "
              "balance breakdown.")
        return
    sub = _filter_best_ls(bal, base, criterion="epsilon")
    groups = {g: [] for g in EXERGY_BALANCE_GROUPS}   # members irrelevant here
    _plot_components_breakdown(
        sub,
        groups=groups,
        colors=EXERGY_BALANCE_COLORS,
        labels=EXERGY_BALANCE_LABELS,
        ylabel=r"$\dot{E}$  [kW]",
        suptitle=rf"Exergy balance ($\dot{{E}}_F = \dot{{E}}_P + \dot{{E}}_L "
                 rf"+ \dot{{E}}_D$) — best $\mathit{{LS}}$ per "
                 rf"$T_\mathrm{{src,in}}$  "
                 rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fname="exergy_balance_breakdown_per_design.pdf",
    )


def _plot_breakdown_aggregated(cdz, ylabel, title, fname):
    """One bar per fluid pair, stack = mean over all OK designs of that pair.

    A thin black error bar shows the min–max range of the TOTAL across the
    designs in that pair, so the reader can see how much spread sits behind
    the mean.
    """
    pairs = sorted(cdz["pair"].unique())
    means = (cdz.groupby("pair")[list(COMP_GROUPS) + ["TOTAL"]]
                .mean().reindex(pairs))
    tot_min = cdz.groupby("pair")["TOTAL"].min().reindex(pairs).values
    tot_max = cdz.groupby("pair")["TOTAL"].max().reindex(pairs).values
    n_per = cdz.groupby("pair").size().reindex(pairs).values

    x = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, 4.0))

    bottom = np.zeros(len(pairs))
    for group in COMP_GROUPS:
        vals = means[group].values
        ax.bar(x, vals, bottom=bottom, color=GROUP_COLORS[group],
               edgecolor="black", linewidth=0.4, label=group)
        bottom += vals

    # Range whiskers around the mean total
    means_tot = means["TOTAL"].values
    ax.errorbar(
        x, means_tot,
        yerr=[means_tot - tot_min, tot_max - means_tot],
        fmt="none", ecolor="black", elinewidth=1.4, capsize=6, capthick=1.4,
        zorder=5,
    )

    # Mean label above the bar; min/max below the whisker tips
    for i in range(len(pairs)):
        ax.text(i, means_tot[i] + (tot_max.max() * 0.01),
                f"mean {means_tot[i]:.0f}",
                ha="center", va="bottom", fontsize=fs(9), fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n(n={int(n)})" for p, n in zip(pairs, n_per)],
                       fontsize=fs(10))
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, tot_max.max() * 1.15)
    ax.legend(loc="upper right", ncol=2, framealpha=0.95, fontsize=fs(7))
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontweight="bold", fontsize=fs(10))
    fig.tight_layout()
    save_titled_and_paper(fig, PLOTS_DIR, fname.removesuffix(".pdf"))
    plt.close(fig)


def plot_cost_breakdown_aggregated(cdz):
    _plot_breakdown_aggregated(
        cdz,
        ylabel=r"$\dot{C}_D + \dot{Z}$  [EUR/h]",
        title=rf"$\dot{{C}}_D + \dot{{Z}}$ per fluid pair  "
              rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fname="cost_breakdown_aggregated.pdf",
    )


def plot_z_breakdown_aggregated(z):
    _plot_breakdown_aggregated(
        z,
        ylabel=r"$\dot{Z}$  [EUR/h]",
        title=rf"$\dot{{Z}}$ per fluid pair  "
              rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fname="z_breakdown_aggregated.pdf",
    )


# ── Exergoeconomic-specific insight plots ─────────────────────────────────
# These two plots show what an exergoeconomic analysis tells you that a pure
# CAPEX/OPEX analysis cannot:
#   1. Tsatsaronis improvement-priority quadrant (f-factor vs C_D + Z): tells
#      whether each component should be improved by buying-cheaper (high f)
#      or by improving-efficiency (low f).
#   2. Economic-only vs exergoeconomic ranking: shows components whose
#      economic importance is hidden in CAPEX-only analyses but visible once
#      C_D (cost of irreversibility) is included.

def _read_exergoeco_components(pair, ls, T_src):
    """Read per-component exergoeco CSV for a single design (drop TOT row)."""
    pair_dir = pair.replace("/", "_")
    ls_pct = int(round(ls * 100))
    path = os.path.join(
        DESIGNS_DIR, pair_dir, f"LS{ls_pct}_Tsrc{int(T_src)}",
        "exergoeco_components.csv",
    )
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["Component"] != "TOT"].copy()
    return df


def _component_group(name):
    """Map component name to its plotting group + color."""
    for g, members in COMP_GROUPS.items():
        if name in members:
            return g, GROUP_COLORS[g]
    return "OTHER", "#888888"


# Component grouping for the Tsatsaronis quadrant and rank-flip plots:
# COMP+MOT of each cycle are bundled, every other component (VAL1, VAL2,
# SRC_HX, IHX, SNK_HX) is shown individually. Pulled from the central
# plot_common.GROUP_STYLE so the visual language is consistent with the
# other breakdown plots.
INSIGHT_AGGREGATION = {
    g: (GROUP_MEMBERS[g], GROUP_COLORS[g]) for g in GROUP_ORDER
}


def _aggregate_for_insight(comps):
    """Sum Z, C_D, E_D over the drive-package aggregations and re-derive f.

    f-factor and ranking metrics are invariant under bundling only when the
    aggregation matches the cost-row granularity Ommen actually published.
    """
    rows = []
    for label, (members, color) in INSIGHT_AGGREGATION.items():
        sub = comps[comps["Component"].isin(members)]
        if sub.empty:
            continue
        Z   = float(sub["Z [EUR/h]"].sum())
        CD  = float(sub["C_D [EUR/h]"].sum())
        ED  = float(sub["E_D [kW]"].sum())
        f   = (Z / (Z + CD) * 100.0) if (Z + CD) > 0 else 0.0
        rows.append({
            "Component":   label,
            "members":     "+".join(members),
            "Z [EUR/h]":   Z,
            "C_D [EUR/h]": CD,
            "E_D [kW]":    ED,
            "f [%]":       f,
            "color":       color,
        })
    return pd.DataFrame(rows)


def _best_design_records(base):
    """Lowest-c_P design within each fluid pair, as a list of dicts.

    Distinct from ``_best_design_per_pair`` (which returns a DataFrame for the
    FLH/price-sensitivity plots); this list-of-dicts form is what the
    component-insight plots iterate over.
    """
    rows = base.loc[base.groupby("pair")["c_P [EUR/GJ]"].idxmin()]
    return rows.sort_values("pair").to_dict(orient="records")


def plot_tsatsaronis_quadrant(base):
    """Tsatsaronis improvement-priority quadrant — one panel per fluid pair.

    For the lowest-c_P design within each pair, scatter every component on
    (f-factor, C_D+Z). Marker size scales with C_D so dissipative items pop.
    Quadrant guidance lines at f=50% and at half the panel's C_D+Z range.
    """
    bests = _best_design_records(base)
    n = len(bests)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(COL_DOUBLE_IN, 3.2 * nrows),
                              squeeze=False)

    # Pre-collect all aggregated components to set a common y-axis upper bound.
    all_cdz = []
    panels = []
    for r in bests:
        comps = _read_exergoeco_components(r["pair"], r["ls"], r["T_src"])
        if comps is None:
            panels.append(None); continue
        agg = _aggregate_for_insight(comps)
        cdz = agg["C_D [EUR/h]"].values + agg["Z [EUR/h]"].values
        all_cdz.append(cdz)
        panels.append((r, agg, cdz))
    cdz_max = float(np.concatenate(all_cdz).max()) * 1.18 if all_cdz else 1.0

    for idx, panel in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        if panel is None:
            ax.set_visible(False); continue
        r, agg, cdz = panel
        f_pct = agg["f [%]"].values
        c_d   = agg["C_D [EUR/h]"].values
        names = agg["Component"].values
        colors = list(agg["color"].values)

        # Marker size (in pts²) scales with C_D within this panel
        cd_max_panel = max(c_d.max(), 1e-3)
        sizes = 60 + 600 * (c_d / cd_max_panel)

        # Quadrant grid lines (50% f, half y-range)
        ax.axvline(50, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axhline(cdz_max / 2, color="gray", linestyle="--", alpha=0.5,
                   linewidth=0.8)

        ax.scatter(f_pct, cdz, s=sizes, c=colors, edgecolor="black",
                   linewidth=0.6, alpha=0.85, zorder=3)
        for fp, cz, name in zip(f_pct, cdz, names):
            ax.annotate(name, (fp, cz),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=fs(7), alpha=0.9)

        # Quadrant tags
        ax.text(0.02, 0.97, "improve\nefficiency",
                transform=ax.transAxes, fontsize=fs(8), color="#7e1b1b",
                va="top", ha="left", alpha=0.7, fontweight="bold")
        ax.text(0.98, 0.97, "buy\ncheaper",
                transform=ax.transAxes, fontsize=fs(8), color="#1b3d7e",
                va="top", ha="right", alpha=0.7, fontweight="bold")
        ax.text(0.50, 0.02, "low priority",
                transform=ax.transAxes, fontsize=fs(8), color="gray",
                va="bottom", ha="center", alpha=0.7, fontstyle="italic")

        ax.set_xlim(-3, 103)
        ax.set_ylim(0, cdz_max)
        ax.set_xlabel(r"f-factor = $\dot{Z} / (\dot{Z} + \dot{C}_D)$  [%]")
        if idx % ncols == 0:
            ax.set_ylabel(r"$\dot{C}_D + \dot{Z}$  [EUR/h]")
        ax.set_title(
            rf"{r['pair']}  $\mathit{{LS}}={int(r['ls']*100)}$ %  "
            rf"$T_\mathrm{{src,in}}={int(r['T_src'])}$ °C  "
            rf"$c_P={r['c_P [EUR/GJ]']:.1f}$ EUR/GJ$_{{ex}}$",
            fontsize=fs(10),
        )
        ax.grid(alpha=0.25)

    # Legend (component groups + size meaning)
    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g)
               for g in COMP_GROUPS]
    handles.append(plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor="lightgray",
                               markeredgecolor="black", markersize=10,
                               label=r"marker size $\propto \dot{C}_D$"))
    fig.legend(handles=handles, loc="lower center",
               ncol=min(7, len(handles)), fontsize=fs(9), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        rf"Tsatsaronis quadrant — best design per fluid pair  "
        rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fontsize=fs(10), fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    save_titled_and_paper(fig, PLOTS_DIR, "tsatsaronis_quadrant")
    plt.close(fig)


def plot_lift_share_vs_T_src(base):
    """One panel per fluid pair: c_P versus T_src, with one line per LS value.

    Helps answer: for each fluid combo, is it better to push more or less of
    the total temperature lift onto the lower cycle? And does that answer
    change with source temperature?
    """
    try:
        gas_cP = _gas_reference_cP()
    except Exception:
        gas_cP = None

    LS_STYLES = {
        0.30: dict(color="#1f77b4", marker="o", label=r"$\mathit{LS} = 30$ %"),
        0.40: dict(color="#2ca02c", marker="s", label=r"$\mathit{LS} = 40$ %"),
        0.50: dict(color="#d62728", marker="^", label=r"$\mathit{LS} = 50$ %"),
    }

    pairs = sorted(base["pair"].unique())
    n = len(pairs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(COL_DOUBLE_IN, 3.0 * nrows),
                              squeeze=False, sharey=True, sharex=True)

    y_lo = float(base["c_P [EUR/GJ]"].min()) * 0.95
    y_hi = float(base["c_P [EUR/GJ]"].max()) * 1.03
    T_src_all = sorted(base["T_src"].unique())

    for idx, pair in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        sub = base[base["pair"] == pair]

        for ls in sorted(sub["ls"].unique()):
            ls_sub = sub[sub["ls"] == ls].sort_values("T_src")
            style = LS_STYLES.get(round(ls, 2), {})
            ax.plot(ls_sub["T_src"], ls_sub["c_P [EUR/GJ]"],
                    color=style.get("color", "gray"),
                    marker=style.get("marker", "o"),
                    markersize=8, linewidth=1.8,
                    markeredgecolor="black", markeredgewidth=0.5,
                    label=style.get("label",
                                    rf"$\mathit{{LS}} = {int(ls*100)}$ %"))

        if gas_cP is not None:
            ax.axhline(gas_cP, color="black", linestyle="--", linewidth=1.2,
                       alpha=0.7)

        ax.set_xlim(min(T_src_all) - 3, max(T_src_all) + 3)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(T_src_all)
        ax.grid(alpha=0.3)
        ax.set_title(pair, fontsize=fs(11), fontweight="bold")
        if idx % ncols == 0:
            ax.set_ylabel(r"$c_P$  [EUR/GJ$_{ex}$]")
        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"$T_\mathrm{src,in}$  [°C]")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = []
    for ls, style in LS_STYLES.items():
        handles.append(plt.Line2D([0], [0],
                                    color=style["color"],
                                    marker=style["marker"], markersize=8,
                                    linewidth=1.8, markeredgecolor="black",
                                    label=style["label"]))
    if gas_cP is not None:
        handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                                    linewidth=1.2,
                                    label=f"Gas+CO₂ ref ({gas_cP:.1f})"))

    fig.legend(handles=handles, loc="lower center",
               ncol=min(5, len(handles)), fontsize=fs(10), frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        rf"$c_P$ vs $T_\mathrm{{src,in}}$ per lift share  "
        rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fontsize=fs(10), fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_titled_and_paper(fig, PLOTS_DIR, "lift_share_vs_T_src")
    plt.close(fig)


def plot_economic_vs_exergoeconomic_ranking(base):
    """Side-by-side ranking flip: pure-economic (Z only) vs full exergoeco (C_D + Z).

    For each fluid pair's best design, two horizontal bar charts in the same
    panel: components sorted by Z (left, "what economic analysis sees")
    vs components sorted by C_D+Z (right, "the real cost-rate the system
    incurs"). Components that move up the ranking when C_D is included
    are the ones a pure CAPEX/OPEX analysis would underweight.
    """
    bests = _best_design_records(base)
    n = len(bests)
    nrows = n   # one row per fluid pair
    # Each row hosts 7 component labels on the y-axis; needs ~1.05 in/row
    # at journal scale so labels do not stack into each other.
    fig, axes = plt.subplots(nrows, 2,
                              figsize=(COL_DOUBLE_IN, 1.05 * nrows + 0.8),
                              squeeze=False, sharey=False)

    # Determine common x-axis upper bound across all panels for fair compare
    x_max = 0.0
    panel_data = []
    for r in bests:
        comps = _read_exergoeco_components(r["pair"], r["ls"], r["T_src"])
        if comps is None:
            panel_data.append(None); continue
        agg = _aggregate_for_insight(comps)
        agg = agg.assign(CDZ=agg["C_D [EUR/h]"] + agg["Z [EUR/h]"])
        x_max = max(x_max, float(agg["CDZ"].max()))
        panel_data.append((r, agg))
    x_max *= 1.10

    for idx, item in enumerate(panel_data):
        ax_eco = axes[idx][0]
        ax_xex = axes[idx][1]
        if item is None:
            ax_eco.set_visible(False); ax_xex.set_visible(False); continue
        r, agg = item   # agg = aggregated DataFrame from _aggregate_for_insight

        # LEFT: ranked by Z only (what economic analysis sees)
        eco_sorted = agg.sort_values("Z [EUR/h]", ascending=True)
        ax_eco.barh(range(len(eco_sorted)), eco_sorted["Z [EUR/h]"],
                    color=eco_sorted["color"], edgecolor="black", linewidth=0.4)
        ax_eco.set_yticks(range(len(eco_sorted)))
        ax_eco.set_yticklabels(eco_sorted["Component"], fontsize=fs(6))
        ax_eco.tick_params(axis="x", labelsize=6)
        ax_eco.set_xlim(0, x_max)
        ax_eco.grid(axis="x", alpha=0.25)
        if idx == 0:
            ax_eco.set_title(r"Economic-only  $\dot{Z}$", fontsize=fs(8))

        # RIGHT: ranked by C_D + Z (exergoeconomic, what really costs)
        xex_sorted = agg.sort_values("CDZ", ascending=True)
        ax_xex.barh(range(len(xex_sorted)), xex_sorted["Z [EUR/h]"],
                    color=xex_sorted["color"], edgecolor="black", linewidth=0.4,
                    label=r"$\dot{Z}$")
        ax_xex.barh(range(len(xex_sorted)), xex_sorted["C_D [EUR/h]"],
                    left=xex_sorted["Z [EUR/h]"],
                    color=xex_sorted["color"], edgecolor="black", linewidth=0.4,
                    hatch="////", alpha=0.55,
                    label=r"$\dot{C}_D$")
        ax_xex.set_yticks(range(len(xex_sorted)))
        ax_xex.set_yticklabels(xex_sorted["Component"], fontsize=fs(6))
        ax_xex.tick_params(axis="x", labelsize=6)
        ax_xex.set_xlim(0, x_max)
        ax_xex.grid(axis="x", alpha=0.25)
        if idx == 0:
            ax_xex.set_title(r"Exergoeconomic  $\dot{Z} + \dot{C}_D$",
                              fontsize=fs(8))

        # Single-line pair label to the left of the row
        pair_label = (rf"{r['pair']}  $\mathit{{LS}}={int(r['ls']*100)}$ %  "
                      rf"$T_\mathrm{{src,in}}={int(r['T_src'])}$ °C")
        ax_eco.set_ylabel(pair_label, fontsize=fs(7), rotation=0,
                          ha="right", va="center", labelpad=40)

    axes[-1][0].set_xlabel(r"$\dot{Z}$  [EUR/h]", fontsize=fs(7))
    axes[-1][1].set_xlabel(r"$\dot{Z} + \dot{C}_D$  [EUR/h]", fontsize=fs(7))

    fig.suptitle(
        rf"Economic vs exergoeconomic ranking  "
        rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fontsize=fs(10), fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_titled_and_paper(fig, PLOTS_DIR, "economic_vs_exergoeconomic_ranking")
    plt.close(fig)


# ── Plot 10: ±50 % electricity / gas 2-D price sensitivity ────────────────

def _hthp_cP_at_FLH(sens_flh, pair, ls, T_src, e1_axis, full_load_hours):
    """Linear-fit HTHP c_P(e1) at a given FLH from the FLH sensitivity sweep.

    The FLH sweep was run at base electricity, so we recover the e1 slope
    from the e1 sweep (slope is FLH-independent for a HTHP because the
    fuel-cost term scales with e1 only). The intercept is set so that
    c_P(BASE_E1, FLH) matches the FLH-sweep value.
    """
    return None  # placeholder — implemented inline in the caller


def plot_price_sensitivity_2d(base, sens, sens_flh=None,
                                full_load_hours=None,
                                fname=None):
    """2-D contour map of c_P_HTHP vs (e1, c_gas) with break-even contour.

    Parameters
    ----------
    base : DataFrame
        economics_base rows (used to pick the best design per fluid pair).
    sens : DataFrame
        economics_sensitivity_e1 rows (used to recover c_P slope vs e1).
    sens_flh : DataFrame or None
        economics_sensitivity_FLH rows. Required when ``full_load_hours`` is
        not the base FLH; ignored otherwise.
    full_load_hours : float or None
        If None, plot at ``BASE_FULL_LOAD_HOURS`` and write the canonical
        file ``price_sensitivity_2d.png``. Otherwise plot at this FLH and
        write ``price_sensitivity_2d_FLH<NNNN>.png``.
    fname : str or None
        Override output filename.

    Notes
    -----
    Mirrors Ommen 2015 Fig. 3. One panel per fluid pair, showing the
    cheapest (LS, T_src) design at the **base FLH** (we keep the same
    design across FLH variants so the panels stay comparable).
    Axes (computed from BASE_E1_C and BASE_GAS_C in config.py):
    e1 ∈ [0.5·BASE_E1_C, 1.5·BASE_E1_C] EUR/MWh; gas ∈ [0.5·BASE_GAS_C,
    3.0·BASE_GAS_C] EUR/MWh.

    HTHP c_P is affine in e1 (slope = 1/ε, FLH-independent); the FLH only
    shifts the capital-recovery intercept. Gas c_P is linear in c_gas
    (Z_gas = 0 retrofit) and FLH-independent.
    """
    if full_load_hours is None:
        full_load_hours = BASE_FULL_LOAD_HOURS
    is_base_flh = float(full_load_hours) == float(BASE_FULL_LOAD_HOURS)

    # Per pair: pick the design with the lowest c_P at base FLH/e1. In
    # practice this lands at T_src,in = 60 °C for pairs that have a
    # converged design there (R290/R717-based) and falls back to the
    # warmest feasible source (typically T_src,in = 50 °C) for the
    # critical-fluid-limited R1270-based pairs.
    best_per_pair = (base.sort_values("c_P [EUR/GJ]")
                          .drop_duplicates(subset=["pair"], keep="first"))

    # Absolute axis grids (same for all FLH cases)
    e1_axis = np.linspace(BASE_E1_C * 0.5, BASE_E1_C * 1.5, 21)
    gas_axis = np.linspace(BASE_GAS_C * 0.5, BASE_GAS_C * 3.0, 26)
    gas_cP_1d = np.array([_gas_reference_cP_at(g) for g in gas_axis])

    # HTHP c_P(e1) per design at the requested FLH:
    #   - slope from the e1 sens sweep (FLH-independent)
    #   - intercept anchored so c_P(BASE_E1, FLH) matches the FLH-sweep value
    hthp_cP_per_pair = {}
    for _, r in best_per_pair.iterrows():
        pair = r["pair"]
        sub_e1 = sens[(sens["pair"] == pair)
                       & (sens["ls"] == r["ls"])
                       & (sens["T_src"] == r["T_src"])]
        if sub_e1.empty:
            continue
        slope, _ = np.polyfit(sub_e1["e1_c [EUR/MWh]"],
                              sub_e1["c_P [EUR/GJ]"], 1)

        if is_base_flh or sens_flh is None:
            cP_at_base_e1 = float(r["c_P [EUR/GJ]"])
        else:
            sub_flh = sens_flh[(sens_flh["pair"] == pair)
                                & (sens_flh["ls"] == r["ls"])
                                & (sens_flh["T_src"] == r["T_src"])
                                & (sens_flh["full_load_hours [h/a]"]
                                   == int(full_load_hours))]
            if sub_flh.empty:
                continue
            cP_at_base_e1 = float(sub_flh.iloc[0]["c_P [EUR/GJ]"])

        intercept = cP_at_base_e1 - slope * BASE_E1_C
        hthp_cP_per_pair[pair] = (slope * e1_axis + intercept,
                                   r["ls"], r["T_src"], float(r["COP"]))

    if not hthp_cP_per_pair:
        print(f"  ! no valid pairs for 2-D price sensitivity at FLH="
              f"{full_load_hours} h/a; skipping.")
        return

    all_vals = np.concatenate([v[0] for v in hthp_cP_per_pair.values()])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    levels = np.linspace(vmin, vmax, 12)

    n = len(hthp_cP_per_pair)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(COL_DOUBLE_IN, 3.4 * nrows),
                              squeeze=False)
    axes_flat = axes.flatten()

    X, Y = np.meshgrid(e1_axis, gas_axis)
    gas_field = np.tile(gas_cP_1d[:, None], (1, len(e1_axis)))

    cf = None
    for idx, pair in enumerate(sorted(hthp_cP_per_pair)):
        ax = axes_flat[idx]
        cP_e1, ls, T_src, _ = hthp_cP_per_pair[pair]
        Z = np.tile(cP_e1[None, :], (len(gas_axis), 1))

        cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis",
                         vmin=vmin, vmax=vmax, extend="both")

        diff = Z - gas_field
        try:
            ax.contour(X, Y, diff, levels=[0.0], colors="white", linewidths=2.8)
            ax.contour(X, Y, diff, levels=[0.0], colors="black", linewidths=1.4,
                       linestyles="--")
        except ValueError:
            pass

        ax.plot(BASE_E1_C, BASE_GAS_C, marker="o", color="red", markersize=10,
                markeredgecolor="white", zorder=5)
        ax.axhline(BASE_GAS_C, color="gray", linewidth=0.5, alpha=0.4)
        ax.axvline(BASE_E1_C, color="gray", linewidth=0.5, alpha=0.4)

        ax.set_xlabel(r"$c_\mathrm{el,0}$  [EUR/MWh]")
        ax.set_ylabel(r"$c_\mathrm{gas,0}$  [EUR/MWh]")
        # Cap the gas-price axis at 100 EUR/MWh (the field extends to
        # 3·BASE_GAS_C ≈ 141 but the upper range is not of interest).
        ax.set_ylim(gas_axis.min(), 100.0)
        # Pair name on line 1, operating point on line 2 — at fs(8) the
        # second line fits in the ~2 in panel width without overlap.
        ax.set_title(
            f"{pair}\n"
            rf"$\mathit{{LS}}={int(ls*100)}$ %, "
            rf"$T_\mathrm{{src,in}}={int(T_src)}$ °C",
            fontsize=fs(8), linespacing=1.3,
        )

    for idx in range(n, nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.tight_layout(rect=[0, 0.05, 0.92, 0.91], w_pad=1.8, h_pad=2.5)
    cbar_ax = fig.add_axes([0.94, 0.10, 0.014, 0.78])
    fig.colorbar(cf, cax=cbar_ax,
                 label=r"$c_P^{\,\mathrm{HTHP}}$  [EUR/GJ$_{ex}$]")

    legend_handles = [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.4,
                   label=r"Break-even  ($c_P^{\,\mathrm{HTHP}} = c_P^{\,\mathrm{gas}}$)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markeredgecolor="white", markersize=10,
                   label=rf"Base prices  ($c_\mathrm{{el,0}}={BASE_E1_C:.0f}$ "
                         rf"EUR/MWh, $c_\mathrm{{gas,0}}={BASE_GAS_C:.0f}$ EUR/MWh)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=fs(10), frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        rf"$c_P^{{\,\mathrm{{HTHP}}}}$ vs $c_\mathrm{{el,0}}$, $c_\mathrm{{gas,0}}$  "
        rf"($\tau = {full_load_hours:.0f}$ h/a, "
        rf"$T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fontsize=fs(10), fontweight="bold", y=0.965,
    )

    if fname is None:
        if is_base_flh:
            fname = "price_sensitivity_2d.pdf"
        else:
            fname = f"price_sensitivity_2d_FLH{int(full_load_hours)}.pdf"
    save_titled_and_paper(fig, PLOTS_DIR, fname.removesuffix(".pdf"))
    plt.close(fig)


# ── Plot: globally best design per pair, FLH-coupled sensitivities ───────
#
# Two companion plots, both styled exactly like ``plot_price_sensitivity_2d``
# (2×3 grid of viridis filled contours, shared vmin/vmax across panels,
# white-edged red base marker, dashed break-even contour, right-hand
# colourbar, bottom legend):
#
#   * ``plot_FLH_e1_bestpairs``  — c_P^HTHP(τ, c_el) at base c_gas
#   * ``plot_FLH_gas_bestpairs`` — c_P^HTHP(τ) tiled against c_gas axis
#
# Per-pair design selection mirrors ``plot_price_sensitivity_2d``: the row
# with the lowest c_P across all (LS, T_src,in) combinations at base prices.
# The chosen T_src,in is surfaced in each panel title alongside LS, so the
# reader can see at a glance which source temperature optimises each pair.

def _best_design_per_pair(base: pd.DataFrame) -> pd.DataFrame:
    """Per pair: globally cheapest design at base prices (any LS, any T_src)."""
    return (base.sort_values("c_P [EUR/GJ]")
                 .drop_duplicates(subset=["pair"], keep="first"))


def _affine_flh(sens_flh_sub: pd.DataFrame):
    """Fit c_P(τ) = a + b/τ from a single-design FLH sweep."""
    flh = sens_flh_sub["full_load_hours [h/a]"].astype(float).values
    cP = sens_flh_sub["c_P [EUR/GJ]"].astype(float).values
    b, a = np.polyfit(1.0 / flh, cP, 1)
    return a, b


def plot_FLH_e1_bestpairs(base, sens, sens_flh,
                           fname="sensitivity_FLH_e1_bestpairs"):
    """2×3 grid: c_P^HTHP(τ, c_el) per pair (globally best design)."""
    best = _best_design_per_pair(base)
    if best.empty:
        print(f"  ! no base rows; skipping {fname}.")
        return

    e1_axis = np.linspace(BASE_E1_C * 0.5, BASE_E1_C * 1.5, 21)
    flh_axis = np.linspace(5000.0, 8000.0, 26)
    gas_cP = _gas_reference_cP()      # constant in this 2-D plane

    hthp_per_pair = {}   # pair -> (Z[2D over flh × e1], ls, T_src)
    for _, r in best.iterrows():
        pair = r["pair"]
        ls = float(r["ls"]); T_src = float(r["T_src"])
        sub_e1 = sens[(sens["pair"] == pair)
                       & np.isclose(sens["ls"], ls)
                       & np.isclose(sens["T_src"], T_src)]
        sub_flh = sens_flh[(sens_flh["pair"] == pair)
                            & np.isclose(sens_flh["ls"], ls)
                            & np.isclose(sens_flh["T_src"], T_src)]
        if sub_e1.empty or sub_flh.empty:
            continue
        slope_e1, _ = np.polyfit(sub_e1["e1_c [EUR/MWh]"],
                                  sub_e1["c_P [EUR/GJ]"], 1)
        a_flh, b_flh = _affine_flh(sub_flh)
        cP_tau = a_flh + b_flh / flh_axis
        Z = (cP_tau[:, None]
              + slope_e1 * (e1_axis[None, :] - BASE_E1_C))
        hthp_per_pair[pair] = (Z, ls, T_src)

    if not hthp_per_pair:
        print(f"  ! no usable sensitivity rows for any pair; skipping {fname}.")
        return

    all_vals = np.concatenate([Z.ravel() for Z, _, _ in hthp_per_pair.values()])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    levels = np.linspace(vmin, vmax, 12)

    n = len(hthp_per_pair)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(COL_DOUBLE_IN, 3.4 * nrows),
                              squeeze=False)
    axes_flat = axes.flatten()
    X, Y = np.meshgrid(e1_axis, flh_axis)

    cf = None
    for idx, pair in enumerate(sorted(hthp_per_pair)):
        ax = axes_flat[idx]
        Z, ls, T_src = hthp_per_pair[pair]
        cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis",
                          vmin=vmin, vmax=vmax, extend="both")
        try:
            ax.contour(X, Y, Z - gas_cP, levels=[0.0],
                       colors="white", linewidths=2.8)
            ax.contour(X, Y, Z - gas_cP, levels=[0.0],
                       colors="black", linewidths=1.4, linestyles="--")
        except ValueError:
            pass

        ax.plot(BASE_E1_C, BASE_FULL_LOAD_HOURS, marker="o", color="red",
                markersize=10, markeredgecolor="white", zorder=5)
        ax.axhline(BASE_FULL_LOAD_HOURS, color="gray", linewidth=0.5, alpha=0.4)
        ax.axvline(BASE_E1_C, color="gray", linewidth=0.5, alpha=0.4)

        ax.set_xlabel(r"$c_\mathrm{el,0}$  [EUR/MWh]")
        ax.set_ylabel(r"$\tau$  [h/a]")
        ax.set_title(
            f"{pair}\n"
            rf"$\mathit{{LS}}={int(ls*100)}$ %, "
            rf"$T_\mathrm{{src,in}}={int(T_src)}$ °C",
            fontsize=fs(8), linespacing=1.3,
        )

    for idx in range(n, nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.tight_layout(rect=[0, 0.05, 0.92, 0.91], w_pad=1.8, h_pad=2.5)
    cbar_ax = fig.add_axes([0.94, 0.10, 0.014, 0.78])
    fig.colorbar(cf, cax=cbar_ax,
                 label=r"$c_P^{\,\mathrm{HTHP}}$  [EUR/GJ$_{ex}$]")

    legend_handles = [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.4,
                   label=rf"Break-even  ($c_P^{{\,\mathrm{{HTHP}}}} = "
                         rf"c_P^{{\,\mathrm{{gas}}}}={gas_cP:.1f}$)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markeredgecolor="white", markersize=10,
                   label=rf"Base  ($c_\mathrm{{el,0}}={BASE_E1_C:.0f}$ EUR/MWh, "
                         rf"$\tau={BASE_FULL_LOAD_HOURS:.0f}$ h/a)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=fs(10), frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        rf"$c_P^{{\,\mathrm{{HTHP}}}}$ vs $\tau$, $c_\mathrm{{el,0}}$  "
        rf"(globally cheapest design per pair, "
        rf"$T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fontsize=fs(10), fontweight="bold", y=0.965,
    )
    save_titled_and_paper(fig, PLOTS_DIR, fname)
    plt.close(fig)


def plot_FLH_gas_bestpairs(base, sens_flh,
                            fname="sensitivity_FLH_gas_bestpairs"):
    """2×3 grid: c_P^HTHP(τ) vs c_gas per pair (globally best design).

    c_P^HTHP is FLH-dependent but c_gas-independent (HTHP burns no gas),
    so the heatmap appears as horizontal bands in each panel; the dashed
    break-even contour curves as c_gas rises and the gas-side c_P catches up.
    """
    best = _best_design_per_pair(base)
    if best.empty:
        print(f"  ! no base rows; skipping {fname}.")
        return

    gas_axis = np.linspace(BASE_GAS_C * 0.5, BASE_GAS_C * 3.0, 26)
    flh_axis = np.linspace(5000.0, 8000.0, 26)
    gas_cP_1d = np.array([_gas_reference_cP_at(g) for g in gas_axis])

    hthp_per_pair = {}   # pair -> (cP_tau[len(flh_axis)], ls, T_src)
    for _, r in best.iterrows():
        pair = r["pair"]
        ls = float(r["ls"]); T_src = float(r["T_src"])
        sub_flh = sens_flh[(sens_flh["pair"] == pair)
                            & np.isclose(sens_flh["ls"], ls)
                            & np.isclose(sens_flh["T_src"], T_src)]
        if sub_flh.empty:
            continue
        a_flh, b_flh = _affine_flh(sub_flh)
        cP_tau = a_flh + b_flh / flh_axis
        hthp_per_pair[pair] = (cP_tau, ls, T_src)

    if not hthp_per_pair:
        print(f"  ! no usable FLH sensitivity rows for any pair; "
              f"skipping {fname}.")
        return

    all_vals = np.concatenate([cP for cP, _, _ in hthp_per_pair.values()])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    levels = np.linspace(vmin, vmax, 12)

    n = len(hthp_per_pair)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(COL_DOUBLE_IN, 3.4 * nrows),
                              squeeze=False)
    axes_flat = axes.flatten()
    X, Y = np.meshgrid(gas_axis, flh_axis)
    gas_field = np.tile(gas_cP_1d[None, :], (len(flh_axis), 1))

    cf = None
    for idx, pair in enumerate(sorted(hthp_per_pair)):
        ax = axes_flat[idx]
        cP_tau, ls, T_src = hthp_per_pair[pair]
        Z = np.tile(cP_tau[:, None], (1, len(gas_axis)))

        cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis",
                          vmin=vmin, vmax=vmax, extend="both")
        try:
            ax.contour(X, Y, Z - gas_field, levels=[0.0],
                       colors="white", linewidths=2.8)
            ax.contour(X, Y, Z - gas_field, levels=[0.0],
                       colors="black", linewidths=1.4, linestyles="--")
        except ValueError:
            pass

        ax.plot(BASE_GAS_C, BASE_FULL_LOAD_HOURS, marker="o", color="red",
                markersize=10, markeredgecolor="white", zorder=5)
        ax.axhline(BASE_FULL_LOAD_HOURS, color="gray", linewidth=0.5, alpha=0.4)
        ax.axvline(BASE_GAS_C, color="gray", linewidth=0.5, alpha=0.4)

        ax.set_xlabel(r"$c_\mathrm{gas,0}$  [EUR/MWh]")
        ax.set_ylabel(r"$\tau$  [h/a]")
        ax.set_title(
            f"{pair}\n"
            rf"$\mathit{{LS}}={int(ls*100)}$ %, "
            rf"$T_\mathrm{{src,in}}={int(T_src)}$ °C",
            fontsize=fs(8), linespacing=1.3,
        )

    for idx in range(n, nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.tight_layout(rect=[0, 0.05, 0.92, 0.91], w_pad=1.8, h_pad=2.5)
    cbar_ax = fig.add_axes([0.94, 0.10, 0.014, 0.78])
    fig.colorbar(cf, cax=cbar_ax,
                 label=r"$c_P^{\,\mathrm{HTHP}}$  [EUR/GJ$_{ex}$]")

    legend_handles = [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.4,
                   label=r"Break-even  ($c_P^{\,\mathrm{HTHP}} = c_P^{\,\mathrm{gas}}$)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markeredgecolor="white", markersize=10,
                   label=rf"Base  ($c_\mathrm{{gas,0}}={BASE_GAS_C:.0f}$ EUR/MWh, "
                         rf"$\tau={BASE_FULL_LOAD_HOURS:.0f}$ h/a)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=fs(10), frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        rf"$c_P^{{\,\mathrm{{HTHP}}}}$ vs $\tau$, $c_\mathrm{{gas,0}}$  "
        rf"(globally cheapest design per pair, "
        rf"$T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fontsize=fs(10), fontweight="bold", y=0.965,
    )
    save_titled_and_paper(fig, PLOTS_DIR, fname)
    plt.close(fig)


# ── Plot: c_P vs (E_D + E_L) faceted by T_src ──────────────────────────────

def plot_cP_vs_ED_EL_by_Tsrc(base):
    """Scatter c_P vs total exergy losses (E_D + E_L), one panel per T_src.

    System exergy balance: E_F = E_P + E_D + E_L. The non-product exergy
    flow is therefore E_D + E_L = E_F − E_P (computed directly from the base
    CSV columns). For closed-cycle HTHPs E_L ≈ 0, so this is dominated by
    component irreversibilities, but the formulation is general.

    Each panel shows ONE T_src value because designs running with different
    source-water inlet temperatures sit on different thermodynamic envelopes
    (different T_evap_c1, different volumetric flows, different cost
    correlations). Cross-panel comparisons mix two different "case studies"
    and are not meaningful — the suptitle calls this out explicitly.
    """
    df = base.copy()
    df["ED_EL [kW]"] = df["E_F [kW]"] - df["E_P [kW]"]

    T_src_vals = sorted(df["T_src"].unique())
    pairs = sorted(df["pair"].unique())
    pair_colors = {p: c for p, c in zip(pairs,
                                          plt.cm.tab10(np.linspace(0, 1, 10)))}
    ls_markers = {0.30: "o", 0.40: "^", 0.50: "s"}

    n = len(T_src_vals)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(COL_DOUBLE_IN, 3.0 * nrows),
                              squeeze=False, sharey=True)
    axes_flat = axes.flatten()

    x_min, x_max = df["ED_EL [kW]"].min(), df["ED_EL [kW]"].max()
    y_min, y_max = df["c_P [EUR/GJ]"].min(), df["c_P [EUR/GJ]"].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.06

    for idx, T_src in enumerate(T_src_vals):
        ax = axes_flat[idx]
        sub = df[df["T_src"] == T_src]
        for _, r in sub.iterrows():
            ax.scatter(r["ED_EL [kW]"], r["c_P [EUR/GJ]"],
                       color=pair_colors[r["pair"]],
                       marker=ls_markers.get(round(float(r["ls"]), 2), "o"),
                       s=85, edgecolor="black", linewidth=0.4, alpha=0.9,
                       zorder=3)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_title(rf"$T_\mathrm{{src,in}} = {int(T_src)}$ °C  (n = {len(sub)})",
                     fontsize=fs(11), fontweight="bold")
        ax.set_xlabel(r"$\dot{E}_D + \dot{E}_L$  [kW]")
        if idx % ncols == 0:
            ax.set_ylabel(r"$c_P$  [EUR/GJ$_{ex}$]")
        ax.grid(alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes_flat[idx].set_visible(False)

    pair_handles = [mpatches.Patch(color=pair_colors[p], label=p) for p in pairs]
    ls_handles = [
        plt.Line2D([0], [0], marker=m, color="black", linestyle="None",
                   markerfacecolor="lightgray", markersize=9,
                   label=rf"$\mathit{{LS}} = {int(round(ls*100))}$ %")
        for ls, m in ls_markers.items()
    ]
    fig.legend(handles=pair_handles + ls_handles,
               loc="lower center", ncol=len(pair_handles) + len(ls_handles),
               fontsize=fs(9), frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        rf"$c_P$ vs $\dot{{E}}_D + \dot{{E}}_L$ per $T_\mathrm{{src,in}}$  "
        rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fontsize=fs(10), fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    save_titled_and_paper(fig, PLOTS_DIR, "cP_vs_ED_EL_by_Tsrc")
    plt.close(fig)


# ── Plot: best configuration per T_src — KPI dashboard ────────────────────

def plot_best_designs_per_T_src(base, cdz):
    """For each T_src, show the cheapest (lowest c_P) design and its KPIs.

    Four-panel dashboard arranged as 2×2:
      top-left     (1) COP         — bar per T_src
      top-right    (2) ε [%]       — bar per T_src
      bottom-left  (3) TCI/kW      — bar per T_src (Annex 58 / Project 68 band overlaid)
      bottom-right (4) C_D + Z     — stacked bar per T_src, components
                                      COMP1+MOT1, SRC_HX, VAL1, IHX,
                                      COMP2+MOT2, SNK_HX, VAL2

    x-tick labels: ``T_src=X °C\\npair LS=Y%`` so the "winning" design is
    visible at a glance for each T_src case.
    """
    T_src_vals = sorted(base["T_src"].unique())
    best = (base.sort_values("c_P [EUR/GJ]")
                 .drop_duplicates(subset=["T_src"], keep="first")
                 .set_index("T_src")
                 .loc[T_src_vals])
    cdz_idx = (cdz.set_index(["pair", "ls", "T_src"])
               if cdz is not None and not cdz.empty else None)

    x = np.arange(len(T_src_vals))
    # Three short lines per tick keep labels narrow enough that adjacent
    # bars in the 2×2 layout do not run into each other horizontally.
    labels = [rf"$T_\mathrm{{src,in}}={int(T)}$ °C" + "\n"
              + rf"{best.loc[T, 'pair']}" + "\n"
              + rf"$\mathit{{LS}}={int(round(best.loc[T, 'ls']*100))}$ %"
              for T in T_src_vals]
    pairs_in_best = list(best["pair"])
    pair_colors = {p: c for p, c in zip(sorted(set(pairs_in_best)),
                                          plt.cm.tab10(np.linspace(0, 1, 10)))}
    bar_colors = [pair_colors[p] for p in pairs_in_best]

    fig, axes = plt.subplots(2, 2, figsize=(COL_DOUBLE_IN, 6.5))
    axes = axes.flatten()

    # Panel 1: COP
    ax = axes[0]
    cops = best["COP"].values
    ax.bar(x, cops, color=bar_colors, edgecolor="black", linewidth=0.4)
    for k, v in enumerate(cops):
        ax.text(k, v + 0.04, f"{v:.2f}", ha="center", va="bottom",
                fontsize=fs(9), fontweight="bold")
    ax.set_ylabel(r"$\mathrm{COP}$  [-]")
    ax.set_title("Coefficient of performance", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fs(7))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(cops) * 1.18)

    # Panel 2: epsilon (%)
    ax = axes[1]
    eps = best["epsilon"].astype(float).values * 100.0
    ax.bar(x, eps, color=bar_colors, edgecolor="black", linewidth=0.4)
    for k, v in enumerate(eps):
        ax.text(k, v + 0.6, f"{v:.1f}", ha="center", va="bottom",
                fontsize=fs(9), fontweight="bold")
    ax.set_ylabel(r"$\varepsilon_\mathrm{tot}$  [%]")
    ax.set_title("Exergetic efficiency", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fs(7))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(eps) * 1.20)

    # Panel 3: TCI / kW with Annex 58 / Project 68 band
    ax = axes[2]
    tci = best["PEC [EUR/kW]"].values
    ax.bar(x, tci, color=bar_colors, edgecolor="black", linewidth=0.4)
    ax.axhspan(ANNEX58_BAND_LOW, ANNEX58_BAND_HIGH,
               color="#2e7d32", alpha=0.15, zorder=0,
               label=f"Annex 58 / Project 68 band (no integration, "
                     f"{ANNEX58_BAND_LOW:.0f}–{ANNEX58_BAND_HIGH:.0f} EUR/kW)")
    for k, v in enumerate(tci):
        ax.text(k, v + 12, f"{v:.0f}", ha="center", va="bottom",
                fontsize=fs(9), fontweight="bold")
    ax.set_ylabel(r"TCI / $\dot{Q}_\mathrm{H}$  [EUR/kW]")
    ax.set_title("Total capital investment per kW heat", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fs(7))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(tci) * 1.40)
    ax.legend(loc="upper right", fontsize=fs(7), framealpha=0.95)

    # Panel 4: C_D + Z stacked breakdown
    ax = axes[3]
    if cdz_idx is None:
        ax.text(0.5, 0.5, "per-design exergoeco CSVs missing",
                ha="center", va="center", transform=ax.transAxes,
                color="grey", fontsize=fs(10))
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        bottom = np.zeros(len(T_src_vals))
        for group in COMP_GROUPS:
            vals = []
            for T in T_src_vals:
                key = (best.loc[T, "pair"], float(best.loc[T, "ls"]), float(T))
                vals.append(float(cdz_idx.loc[key, group])
                             if key in cdz_idx.index else 0.0)
            vals = np.array(vals)
            ax.bar(x, vals, bottom=bottom, color=GROUP_COLORS[group],
                   edgecolor="black", linewidth=0.3, label=group)
            bottom += vals
        for k, v in enumerate(bottom):
            ax.text(k, v + 6, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=fs(9), fontweight="bold")
        ax.set_ylabel(r"$\dot{C}_D + \dot{Z}$  [EUR/h]")
        ax.set_title("Component cost-rate breakdown", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=fs(7))
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, bottom.max() * 1.45)
        ax.legend(loc="upper right", ncol=3, fontsize=fs(7), framealpha=0.95)

    fig.suptitle(
        rf"Best design per $T_\mathrm{{src,in}}$  "
        rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fontsize=fs(10), fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_titled_and_paper(fig, PLOTS_DIR, "best_designs_per_T_src")
    plt.close(fig)


# ── LS-criterion comparison plots ──────────────────────────────────────────
# Three figures that visualize *how much* the choice of "best LS" criterion
# (highest ε vs lowest Σ Ż vs lowest c_P) actually moves the design:
#   1. plot_LS_choice_agreement  — categorical map: which LS each criterion
#                                  picks per (pair, T_src) cell.
#   2. plot_LS_pareto_trade_off  — (ε, c_P) Pareto scatter, three LS points
#                                  per T_src connected by a line per pair.
#   3. plot_LS_regret_bars       — % efficiency lost / % cost added when the
#                                  "wrong" criterion is used to pick LS.

_LS_PALETTE = {0.30: "#1f77b4", 0.40: "#2ca02c", 0.50: "#d62728"}
_LS_MARKER  = {0.30: "o",        0.40: "s",        0.50: "^"}


def _criterion_lookup(base, z_only):
    """Merge ``base`` with the per-design Σ Ż from ``z_only`` on (pair, ls,
    T_src). Returns a DataFrame with columns pair, ls, T_src, epsilon, COP,
    c_P [EUR/GJ], Z_sum. ls is rounded to 2 decimals before the join to
    avoid float-precision merge misses."""
    b = base.copy()
    b["_ls_r"] = b["ls"].round(2)
    z = z_only[["pair", "ls", "T_src", "TOTAL"]].copy()
    z["_ls_r"] = z["ls"].round(2)
    z = z.rename(columns={"TOTAL": "Z_sum"}).drop(columns=["ls"])
    return b.merge(z, on=["pair", "_ls_r", "T_src"], how="left")


def _best_ls_per_cell(merged):
    """For each (pair, T_src) cell, return the LS picked by each of the three
    criteria and the corresponding metric values. One row per cell, with
    columns ls_eps, ls_Z, ls_cP plus the (eps|cP|Z) value evaluated at each
    of those three LS choices."""
    rows = []
    for (pair, T_src), grp in merged.groupby(["pair", "T_src"]):
        ie = grp["epsilon"].idxmax()
        iz = grp["Z_sum"].idxmin()
        ic = grp["c_P [EUR/GJ]"].idxmin()
        rows.append({
            "pair": pair, "T_src": T_src,
            "ls_eps": float(grp.loc[ie, "ls"]),
            "ls_Z":   float(grp.loc[iz, "ls"]),
            "ls_cP":  float(grp.loc[ic, "ls"]),
            "eps_eps": float(grp.loc[ie, "epsilon"]),
            "eps_Z":   float(grp.loc[iz, "epsilon"]),
            "eps_cP":  float(grp.loc[ic, "epsilon"]),
            "Z_eps":   float(grp.loc[ie, "Z_sum"]),
            "Z_Z":     float(grp.loc[iz, "Z_sum"]),
            "Z_cP":    float(grp.loc[ic, "Z_sum"]),
            "cP_eps":  float(grp.loc[ie, "c_P [EUR/GJ]"]),
            "cP_Z":    float(grp.loc[iz, "c_P [EUR/GJ]"]),
            "cP_cP":   float(grp.loc[ic, "c_P [EUR/GJ]"]),
        })
    return pd.DataFrame(rows)


def plot_LS_choice_agreement(base, z_only):
    """One small panel per fluid pair. Y-axis = three criteria (ε, Σ Ż, c_P);
    x-axis = T_src; cell colour = LS picked by that criterion (blue 30, green
    40, red 50). Where all three rows of a column share a colour, the
    criteria agree; differing colours flag the disagreement at a glance."""
    from matplotlib.colors import ListedColormap

    merged = _criterion_lookup(base, z_only)
    bdf = _best_ls_per_cell(merged)
    pairs = sorted(bdf["pair"].unique())
    n = len(pairs); ncols = 3; nrows = (n + ncols - 1) // ncols
    T_src_axis = list(T_SOURCE_IN_RANGE)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(COL_DOUBLE_IN, 2.4 * nrows + 0.8),
                              squeeze=False)
    # 4-colour map: 3 LS choices + index 3 reserved for "infeasible".
    cmap = ListedColormap([_LS_PALETTE[0.30], _LS_PALETTE[0.40],
                            _LS_PALETTE[0.50], "#cccccc"])
    code_map = {0.30: 0, 0.40: 1, 0.50: 2}
    crit_keys   = ["ls_eps", "ls_cP", "ls_Z"]
    crit_labels = [r"$\max\,\varepsilon$",
                   r"$\min\,c_P$",
                   r"$\min\,\dot{Z}_\mathrm{tot}$"]

    for idx, pair in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        sub = bdf[bdf["pair"] == pair].set_index("T_src")
        # Build 3-row × len(T_src_axis) grid; code 3 = infeasible.
        grid = np.full((len(crit_keys), len(T_src_axis)), np.nan)
        codes = np.full(grid.shape, 3, dtype=int)
        for j, T in enumerate(T_src_axis):
            if T in sub.index:
                for i, c in enumerate(crit_keys):
                    ls_val = round(float(sub.loc[T, c]), 2)
                    grid[i, j] = ls_val
                    codes[i, j] = code_map[ls_val]
        ax.imshow(codes, cmap=cmap, aspect="auto", vmin=0, vmax=3)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if codes[i, j] == 3:
                    ax.text(j, i, "n/a", ha="center", va="center",
                            fontsize=fs(7), color="#666", fontweight="bold")
                else:
                    ax.text(j, i, f"{int(round(grid[i, j] * 100))}",
                            ha="center", va="center",
                            fontsize=fs(9), color="white", fontweight="bold")
        ax.set_xticks(range(len(T_src_axis)))
        ax.set_xticklabels([f"{int(t)}" for t in T_src_axis],
                            fontsize=fs(8))
        ax.set_yticks(range(len(crit_keys)))
        ax.set_yticklabels(crit_labels, fontsize=fs(8))
        ax.set_title(pair, fontweight="bold", fontsize=fs(9))
        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"$T_\mathrm{src,in}$  [°C]")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    legend = [mpatches.Patch(color=_LS_PALETTE[ls],
                              label=rf"$\mathit{{LS}} = {int(ls*100)}$ %")
              for ls in (0.30, 0.40, 0.50)]
    legend.append(mpatches.Patch(color="#cccccc", label="infeasible"))
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=fs(9),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(rf"Best $\mathit{{LS}}$ by criterion  "
                  rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
                  fontsize=fs(10), fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    save_titled_and_paper(fig, PLOTS_DIR, "LS_choice_agreement")
    plt.close(fig)


def plot_LS_pareto_trade_off(base, z_only):
    """One panel per fluid pair. Each T_src contributes three points (one per
    LS) connected by a thin line — a mini-Pareto front in (ε, c_P) space.
    Marker shape encodes LS, colour encodes T_src so the reader can see both
    the per-T_src trade-off curve and how it shifts across source
    temperatures."""
    merged = _criterion_lookup(base, z_only)
    pairs = sorted(merged["pair"].unique())
    n = len(pairs); ncols = 3; nrows = (n + ncols - 1) // ncols

    T_src_all = sorted(merged["T_src"].unique())
    cmap = plt.get_cmap("viridis")
    T_colors = {T: cmap(i / max(len(T_src_all) - 1, 1))
                for i, T in enumerate(T_src_all)}

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(COL_DOUBLE_IN, 3.0 * nrows + 0.8),
                              squeeze=False)

    for idx, pair in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        sub = merged[merged["pair"] == pair]
        for T_src in sorted(sub["T_src"].unique()):
            cell = sub[sub["T_src"] == T_src].sort_values("ls")
            xs = cell["epsilon"].values * 100.0
            ys = cell["c_P [EUR/GJ]"].values
            ax.plot(xs, ys, "-", color=T_colors[T_src],
                    alpha=0.45, linewidth=0.9, zorder=2)
            for _, r in cell.iterrows():
                ax.scatter(r["epsilon"] * 100.0, r["c_P [EUR/GJ]"],
                           marker=_LS_MARKER[round(r["ls"], 2)],
                           color=T_colors[T_src], s=55,
                           edgecolor="black", linewidth=0.4, zorder=3)
        ax.set_title(pair, fontweight="bold", fontsize=fs(9))
        ax.grid(alpha=0.3)
        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"$\varepsilon_\mathrm{tot}$  [%]")
        if idx % ncols == 0:
            ax.set_ylabel(r"$c_P$  [EUR/GJ$_{ex}$]")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    ls_handles = [plt.Line2D([0], [0], marker=_LS_MARKER[ls], color="gray",
                              linestyle="None", markersize=8,
                              markeredgecolor="black",
                              label=rf"$\mathit{{LS}} = {int(ls*100)}$ %")
                  for ls in (0.30, 0.40, 0.50)]
    T_handles = [mpatches.Patch(color=T_colors[T],
                                  label=rf"$T_\mathrm{{src,in}} = {int(T)}$ °C")
                 for T in T_src_all]
    fig.legend(handles=ls_handles + T_handles, loc="lower center",
               ncol=len(ls_handles) + len(T_handles), fontsize=fs(7),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(rf"Pareto trade-off: $c_P$ vs $\varepsilon_\mathrm{{tot}}$  "
                  rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
                  fontsize=fs(10), fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    save_titled_and_paper(fig, PLOTS_DIR, "LS_pareto_trade_off")
    plt.close(fig)


def plot_LS_regret_bars(base, z_only):
    """For each (pair, T_src) compute two regrets (always ≥ 0):
       * ε regret [%]  = (ε(ε*) − ε(c_P*)) / ε(ε*) × 100
                          — efficiency lost if you adopt the c_P-best LS
                          instead of the ε-best LS.
       * c_P regret [%] = (c_P(ε*) − c_P(c_P*)) / c_P(c_P*) × 100
                          — extra specific cost paid if you adopt the ε-best
                          LS instead of the c_P-best LS.
    One panel per fluid pair, two grouped bars per T_src."""
    merged = _criterion_lookup(base, z_only)
    bdf = _best_ls_per_cell(merged)
    pairs = sorted(bdf["pair"].unique())
    n = len(pairs); ncols = 3; nrows = (n + ncols - 1) // ncols
    T_src_axis = list(T_SOURCE_IN_RANGE)

    # Shared y-max so panels are visually comparable; needs the full set of
    # regret values across all pairs.
    reg_eps_all = (bdf["eps_eps"] - bdf["eps_cP"]) / bdf["eps_eps"] * 100.0
    reg_cP_all  = (bdf["cP_eps"]  - bdf["cP_cP"])  / bdf["cP_cP"]  * 100.0
    y_max = max(float(reg_eps_all.max()), float(reg_cP_all.max()), 0.5) * 1.25

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(COL_DOUBLE_IN, 2.8 * nrows + 0.8),
                              squeeze=False)

    for idx, pair in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        sub = bdf[bdf["pair"] == pair].set_index("T_src")
        reg_eps = []
        reg_cP  = []
        feasible = []
        for T in T_src_axis:
            if T in sub.index:
                r = sub.loc[T]
                reg_eps.append((r["eps_eps"] - r["eps_cP"]) / r["eps_eps"] * 100.0)
                reg_cP.append( (r["cP_eps"]  - r["cP_cP"])  / r["cP_cP"]  * 100.0)
                feasible.append(True)
            else:
                reg_eps.append(0.0); reg_cP.append(0.0); feasible.append(False)
        reg_eps = np.array(reg_eps); reg_cP = np.array(reg_cP)
        x = np.arange(len(T_src_axis))
        w = 0.38
        ax.bar(x - w / 2, reg_eps, width=w, color="#1f77b4",
                edgecolor="black", linewidth=0.3)
        ax.bar(x + w / 2, reg_cP,  width=w, color="#d62728",
                edgecolor="black", linewidth=0.3)
        for k, T in enumerate(T_src_axis):
            if not feasible[k]:
                ax.add_patch(plt.Rectangle(
                    (k - 0.4, 0), 0.8, y_max,
                    fill=True, color="#eeeeee", zorder=0,
                ))
                ax.text(k, y_max * 0.04, "infeasible",
                        ha="center", va="bottom", fontsize=fs(7),
                        rotation=90, color="#888")
                continue
            if reg_eps[k] > 0.05:
                ax.text(k - w / 2, reg_eps[k] + y_max * 0.01,
                        f"{reg_eps[k]:.1f}", ha="center", va="bottom",
                        fontsize=fs(6))
            if reg_cP[k] > 0.05:
                ax.text(k + w / 2, reg_cP[k] + y_max * 0.01,
                        f"{reg_cP[k]:.1f}", ha="center", va="bottom",
                        fontsize=fs(6))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(T)}" for T in T_src_axis], fontsize=fs(8))
        ax.set_ylim(0, y_max)
        ax.set_title(pair, fontweight="bold", fontsize=fs(9))
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.4)
        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"$T_\mathrm{src,in}$  [°C]")
        if idx % ncols == 0:
            ax.set_ylabel("Regret  [%]")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [
        mpatches.Patch(color="#1f77b4",
                        label=r"$\varepsilon$ lost using $c_P$-best $\mathit{LS}$"),
        mpatches.Patch(color="#d62728",
                        label=r"$c_P$ added using $\varepsilon$-best $\mathit{LS}$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=fs(8),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(rf"Regret from choosing the 'wrong' criterion  "
                  rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
                  fontsize=fs(10), fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    save_titled_and_paper(fig, PLOTS_DIR, "LS_regret_bars")
    plt.close(fig)


def main(T_steam=None):
    """Render the full economics plot set for one T_steam case study."""
    if T_steam is None:
        T_steam = T_STEAM_CASE_DEFAULT
    _set_paths_for_T_steam(T_steam)
    if not all(os.path.exists(p) for p in [BASE_CSV, BREAKDOWN_CSV, SENS_CSV]):
        print(f"Missing economics CSVs in {DATA_DIR}/. "
              f"Run `python main.py --t-steam {int(T_steam)}` first.")
        sys.exit(1)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    base, bk, sens, sens_flh = _load()

    print("Plotting PEC component breakdown ...")
    plot_pec_breakdown(base, bk)
    print("Plotting c_P vs electricity price ...")
    plot_cP_vs_e1(base, sens)
    print("Plotting c_P vs full-load hours ...")
    plot_cP_vs_FLH(base, sens_flh)
    print("Plotting c_P vs gas reference ...")
    plot_cP_vs_gas(base)
    print("Plotting c_P sorted per T_source case study ...")
    plot_cP_sorted_by_T_src(base)
    print("Plotting c_P vs (E_D + E_L) faceted by T_src ...")
    plot_cP_vs_ED_EL_by_Tsrc(base)

    cdz = _load_cdz_per_design(base)
    if cdz.empty:
        print("  ! per-design exergoeco CSVs missing — skipping cost-breakdown "
              "plots. Run main.py to regenerate per-design exports.")
    else:
        print(f"Plotting C_D+Z per-design breakdown (n={len(cdz)}) ...")
        plot_cost_breakdown_per_design(base, cdz)
        print("Plotting C_D+Z aggregated by fluid pair ...")
        plot_cost_breakdown_aggregated(cdz)
        z_only = _load_cdz_per_design(base, value_col="Z [EUR/h]")
        print("Plotting Z per-design breakdown ...")
        plot_z_breakdown_per_design(base, z_only)
        print("Plotting Z aggregated by fluid pair ...")
        plot_z_breakdown_aggregated(z_only)
        # E_D is loaded with the SPLIT-motor grouping so the breakdown
        # plot renders MOT1 / MOT2 as their own stack segments.
        ed = _load_cdz_per_design(
            base, value_col="E_D [kW]",
            groups=GROUP_MEMBERS_SPLIT_MOT,
        )
        print("Plotting components breakdown (best LS per T_src) — C_D+Z, Z, E_D ...")
        plot_CDZ_components_breakdown(base, cdz)
        plot_z_components_breakdown(base, z_only)
        plot_ED_components_breakdown(base, ed)
        print("Plotting system exergy balance (E_P / E_L / E_D) ...")
        plot_exergy_balance_breakdown(base)
        print("Plotting Tsatsaronis improvement-priority quadrant per fluid pair ...")
        try:
            plot_tsatsaronis_quadrant(base)
        except Exception as e:
            print(f"  ! plot_tsatsaronis_quadrant failed ({type(e).__name__}: {e}); skipping.")
        print("Plotting economic vs exergoeconomic ranking ...")
        try:
            plot_economic_vs_exergoeconomic_ranking(base)
        except Exception as e:
            print(f"  ! plot_economic_vs_exergoeconomic_ranking failed ({type(e).__name__}: {e}); skipping.")
        print("Plotting best designs per T_src dashboard ...")
        try:
            plot_best_designs_per_T_src(base, cdz)
        except Exception as e:
            print(f"  ! plot_best_designs_per_T_src failed ({type(e).__name__}: {e}); skipping.")
        print("Plotting LS-criterion comparison (agreement / Pareto / regret) ...")
        try:
            plot_LS_choice_agreement(base, z_only)
            plot_LS_pareto_trade_off(base, z_only)
            plot_LS_regret_bars(base, z_only)
        except Exception as e:
            print(f"  ! LS-criterion plots failed ({type(e).__name__}: {e}); skipping.")
    print("Plotting lift-share trends vs T_src per fluid pair ...")
    plot_lift_share_vs_T_src(base)

    print("Plotting ±50 % e1/gas 2-D price sensitivity with break-even ...")
    plot_price_sensitivity_2d(base, sens)

    print("Plotting best-design-per-pair sensitivity (τ, c_el) ...")
    plot_FLH_e1_bestpairs(base, sens, sens_flh)
    print("Plotting best-design-per-pair sensitivity (τ, c_gas) ...")
    plot_FLH_gas_bestpairs(base, sens_flh)

    if sens_flh is not None and not sens_flh.empty:
        flh_values = sorted(sens_flh["full_load_hours [h/a]"].unique())
        for flh in flh_values:
            if int(flh) == int(BASE_FULL_LOAD_HOURS):
                continue  # already produced above
            print(f"  ... 2-D price sensitivity at FLH = {int(flh)} h/a ...")
            plot_price_sensitivity_2d(base, sens, sens_flh=sens_flh,
                                       full_load_hours=int(flh))

    print(f"\nGas reference at base prices: c_P_gas = "
          f"{_gas_reference_cP():.1f} EUR/GJ_ex  "
          f"(at {BASE_GAS_C:.0f} EUR/MWh gas, η_boiler=0.90, exergy basis)")
    print(f"Figures written to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
