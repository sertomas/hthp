"""Plotting module for the case-study economics and cross-T_steam results.

Entry points: ``economics_main`` (per-T_steam economics figure set) and
``compare_main`` (cross-T_steam comparison). The per-design and (T_src, T_steam)
heatmaps run via the CLI (``python plots.py cdz`` and ``python plots.py cP_tsrc``).
Shared helpers live in common.py.
"""
from __future__ import annotations
from common import (
    COL_DOUBLE_IN,
    COL_SINGLE_IN,
    GROUP_COLORS,
    GROUP_COLORS_SPLIT_MOT,
    GROUP_LABELS,
    GROUP_MEMBERS,
    GROUP_MEMBERS_SPLIT_MOT,
    GROUP_ORDER,
    GROUP_ORDER_SPLIT_MOT,
    GROUP_STYLE,
    STATUS_COLORS,
    STATUS_ORDER,
    classify_status,
    fs,
    nosolve_short,
    reason_short,
    save_titled_and_paper,
    slice_grid,
)


# ===================== per-T_steam economics figures =====================
# Visualise the exergoeconomic results for one case-study T_steam
# (LS in {0.30, 0.40, 0.50}). ``economics_main`` is invoked once per
# T_steam in T_STEAMS_TO_RUN; paths are repointed at runtime via
# ``_set_paths_economics``. Reads the economics CSVs and gas_heater.json
# under results/case_steam_<int(T_steam)>/ and writes the economics figures
# back to the same case folder.


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


def _op_string():
    """Operating-assumption tag used in every economics plot title."""
    return (f"FLH = {BASE_FULL_LOAD_HOURS:.0f} h/a, "
            f"e1 = {BASE_E1_C:.0f} EUR/MWh, "
            f"gas = {BASE_GAS_C:.0f} EUR/MWh")


# Gas reference: computed via the exergy-based gas heater model so the c_P
# axis stays on the same exergy basis (per E_P_steam) as the HTHP c_P,
# keeping the two directly comparable.

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


# Module-level path globals are rewritten by ``_set_paths_economics`` at
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


def _set_paths_economics(T_steam):
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
# Colours / labels / member list come from ``common.GROUP_STYLE``.
COMP_GROUPS = dict(GROUP_MEMBERS)

# Supplier "per unit, no integration" cost band for 0.5-3 MWth, 110-150 °C
# (Annex 58 reference range).
ANNEX58_BAND_LOW = 400.0   # EUR/kW
ANNEX58_BAND_HIGH = 700.0  # EUR/kW


def _load():
    base = pd.read_csv(BASE_CSV)
    bk = pd.read_csv(BREAKDOWN_CSV)
    sens = pd.read_csv(SENS_CSV)
    sens_flh = (pd.read_csv(SENS_FLH_CSV)
                if os.path.exists(SENS_FLH_CSV) else None)

    # Add components-only PEC (purchase equipment cost, no installation)
    # for fair comparison to the Annex 58 supplier EUR/kW which excludes
    # integration.
    base["PEC_components_only [EUR/kW]"] = (
        base["PEC [EUR/kW]"] / F_INSTALL
    ).round(1)
    return base, bk, sens, sens_flh


# ── C_D + Z component breakdown per design ────────────────────────────────

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
# common.GROUP_STYLE so the visual language is consistent with the
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
        # Pair name on line 1, operating point on line 2 — at fs(9) the
        # subscripts clear the 6 pt Elsevier floor and the second line
        # still fits the ~2 in panel width without overlap.
        ax.set_title(
            f"{pair}\n"
            rf"$\mathit{{LS}}={int(ls*100)}$ %, "
            rf"$T_\mathrm{{src,in}}={int(T_src)}$ °C",
            fontsize=fs(9), linespacing=1.3,
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
            fontsize=fs(9), linespacing=1.3,
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
            fontsize=fs(9), linespacing=1.3,
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


# ── Plot: best configuration per T_src — KPI dashboard ────────────────────

def plot_best_designs_per_T_src(base, cdz):
    """For each T_src, show the cheapest (lowest c_P) design and its KPIs.

    Four-panel dashboard arranged as 2×2:
      top-left     (1) COP         — bar per T_src
      top-right    (2) ε [%]       — bar per T_src
      bottom-left  (3) TCI/kW      — bar per T_src (Annex 58 band overlaid)
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
    # bars in the 2×2 layout do not run into each other horizontally. The
    # first line is just the value + unit (the suptitle already names the
    # axis as T_src,in) so the wide "T_src,in =" prefix does not overflow.
    labels = [rf"${int(T)}$ °C" + "\n"
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

    # Panel 3: TCI / kW with Annex 58 band
    ax = axes[2]
    tci = best["PEC [EUR/kW]"].values
    ax.bar(x, tci, color=bar_colors, edgecolor="black", linewidth=0.4)
    ax.axhspan(ANNEX58_BAND_LOW, ANNEX58_BAND_HIGH,
               color="#2e7d32", alpha=0.15, zorder=0,
               label=f"Annex 58 band (no integration, "
                     f"{ANNEX58_BAND_LOW:.0f}-{ANNEX58_BAND_HIGH:.0f} EUR/kW)")
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
        # Extra headroom so the (wide, 3-column) legend sits above the
        # tallest bar's value label instead of covering it.
        ax.set_ylim(0, bottom.max() * 1.75)
        ax.legend(loc="upper right", ncol=3, fontsize=fs(7), framealpha=0.95)

    fig.suptitle(
        rf"Best design per $T_\mathrm{{src,in}}$  "
        rf"($T_\mathrm{{steam}} = {_T_STEAM_CURRENT:.0f}$ °C)",
        fontsize=fs(10), fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_titled_and_paper(fig, PLOTS_DIR, "best_designs_per_T_src")
    plt.close(fig)


# ── LS-criterion comparison plot ───────────────────────────────────────────
# Visualizes *how much* the choice of "best LS" criterion (highest ε vs lowest
# Σ Ż vs lowest c_P) actually moves the design:
#   plot_LS_choice_agreement  — categorical map: which LS each criterion picks
#                               per (pair, T_src) cell.

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


def economics_main(T_steam=None):
    """Render the full economics plot set for one T_steam case study."""
    if T_steam is None:
        T_steam = T_STEAM_CASE_DEFAULT
    _set_paths_economics(T_steam)
    if not all(os.path.exists(p) for p in [BASE_CSV, BREAKDOWN_CSV, SENS_CSV]):
        print(f"Missing economics CSVs in {DATA_DIR}/. "
              f"Run `python main.py --t-steam {int(T_steam)}` first.")
        sys.exit(1)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    base, _, sens, sens_flh = _load()

    cdz = _load_cdz_per_design(base)
    if cdz.empty:
        print("  ! per-design exergoeco CSVs missing — skipping cost-breakdown "
              "plots. Run main.py to regenerate per-design exports.")
    else:
        z_only = _load_cdz_per_design(base, value_col="Z [EUR/h]")
        # E_D is loaded with the SPLIT-motor grouping so the breakdown plot
        # renders MOT1 / MOT2 as their own stack segments.
        ed = _load_cdz_per_design(base, value_col="E_D [kW]",
                                  groups=GROUP_MEMBERS_SPLIT_MOT)
        print("Plotting components breakdown (best LS per T_src) — C_D+Z, Z, E_D ...")
        plot_CDZ_components_breakdown(base, cdz)
        plot_z_components_breakdown(base, z_only)
        plot_ED_components_breakdown(base, ed)
        print("Plotting system exergy balance (E_P / E_L / E_D) ...")
        plot_exergy_balance_breakdown(base)
        print("Plotting best designs per T_src dashboard ...")
        try:
            plot_best_designs_per_T_src(base, cdz)
        except Exception as e:
            print(f"  ! plot_best_designs_per_T_src failed ({type(e).__name__}: {e}); skipping.")
        print("Plotting LS-criterion comparison (agreement) ...")
        try:
            plot_LS_choice_agreement(base, z_only)
        except Exception as e:
            print(f"  ! LS-criterion plots failed ({type(e).__name__}: {e}); skipping.")

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




# ===================== cross-T_steam comparison figures =====================
# Reads per-T_steam economics outputs from results/case_steam_<T>/ and writes
# the comparison figures into results/case_steam_compare/:
#   exergy_destruction_base : fleet-wide component E_D at the base case
#                             (one panel per T_steam, bars per fluid pair).
#   sensitivity_FLH_cel     : 2-D c_P map over (FLH, c_el) for the reference
#                             design, with the gas-heater break-even contour.


import json
import logging
import os
import sys
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    BASE_E1_C, BASE_FULL_LOAD_HOURS, BASE_GAS_C,
    case_data_dir, case_gas_heater_dir, case_results_dir,
    t_steam_compare_dir, T_STEAMS_TO_RUN,
)


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


def _load_economics_for(T_steam: float):
    """Load `economics_base.csv` and gas-heater c_P for one T_steam.

    Returns (df, gas_cP) or (None, None) if the data is missing.
    """
    base_csv = os.path.join(case_data_dir(T_steam), "economics_base.csv")
    gas_json = os.path.join(case_gas_heater_dir(T_steam),
                            "gas_heater.json")

    if not os.path.exists(base_csv):
        print(f"  ! economics_base.csv missing for T_steam = {T_steam:.0f} °C")
        return None, None

    df = pd.read_csv(base_csv)
    gas_cP = None
    if os.path.exists(gas_json):
        try:
            gas_cP = float(json.load(open(gas_json))["economics"]["c_P"])
        except Exception as e:
            print(f"  ! could not read gas-heater c_P at "
                  f"T_steam={T_steam:.0f}: {e}")
    return df, gas_cP


def _best_per_pair(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pair lowest c_P design (one row per fluid pair)."""
    return df.loc[df.groupby("pair")["c_P [EUR/GJ]"].idxmin()] \
             .sort_values("pair") \
             .reset_index(drop=True)


# ── Fleet exergy destruction at base case (T_src=50, LS=0.30) ─────────────

# Per-component breakdown. Colours and labels come from the central
# palette in common.GROUP_STYLE so every component plot across the
# project (this one + the cost / Z breakdowns in plot_case_steam_economics)
# shares the same visual language: COMP+MOT bundled per cycle, every other
# component (VAL1, VAL2, SRC_HX, IHX, SNK_HX) shown individually.

# Pair plotting order (cycle-1 fluid first); kept consistent across panels
_PAIR_ORDER = ["R717/R600", "R717/R600a",
               "R290/R600", "R290/R600a",
               "R1270/R600", "R1270/R600a"]


def _read_exergy_components(T_steam: float, pair: str,
                             ls: float = 0.50, T_src: float = 60.0):
    """Return per-component E_D [kW] for the (T_steam, pair) base-case design.

    Returns a dict {component_name: E_D_kW} or None when the design folder
    is missing (i.e. the (T_steam, pair, ls, T_src) combination did not
    converge or was discarded by the feasibility screen).
    """
    f1, f2 = pair.split("/")
    pair_dir = f"{f1}_{f2}"
    ls_pct = int(round(ls * 100))
    path = os.path.join(
        case_results_dir(T_steam), "designs", pair_dir,
        f"LS{ls_pct}_Tsrc{int(T_src)}", "exergoeco_components.csv",
    )
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["Component"] != "TOT"]
    return dict(zip(df["Component"], df["E_D [kW]"].astype(float)))


def plot_fleet_exergy_destruction_base(out_dir: str,
                                        ls: float = 0.30,
                                        T_src: float = 50.0):
    """Fleet-wide E_D component breakdown at the base case.

    Three panels (one per T_steam). x-axis = fluid pair (fixed order),
    each bar stacked by component group. Infeasible (pair, T_steam) cells
    are drawn as an empty slot with an "infeasible" annotation so the
    reader can locate the feasibility envelope at a glance.
    """
    Ts = sorted(T_STEAMS_TO_RUN)
    n_T = len(Ts)
    n_pairs = len(_PAIR_ORDER)

    # Gather data + compute a common y-axis upper bound across panels.
    # Each (T_steam, pair) entry holds a {group -> summed E_D in kW} dict
    # (raw COMP1 / MOT1 / ... are summed into the bundled COMP+MOT groups
    # at read time so the plotting loop only sees the bundled view).
    panel_data = {}    # T_steam -> {pair -> {group -> kW} | None}
    y_max = 0.0
    for T in Ts:
        per_pair = {}
        for pair in _PAIR_ORDER:
            comps = _read_exergy_components(T, pair, ls=ls, T_src=T_src)
            if comps is None:
                per_pair[pair] = None
                continue
            per_group = {
                g: sum(float(comps.get(m, 0.0)) for m in GROUP_MEMBERS[g])
                for g in GROUP_ORDER
            }
            per_pair[pair] = per_group
            y_max = max(y_max, sum(per_group.values()))
        panel_data[T] = per_pair

    y_max = y_max * 1.20 if y_max > 0 else 1.0

    fig, axes = plt.subplots(1, n_T, figsize=(COL_DOUBLE_IN, 3.6),
                              sharey=True, squeeze=False)

    x = np.arange(n_pairs)
    for j, T in enumerate(Ts):
        ax = axes[0][j]
        per_pair = panel_data[T]
        bottoms = np.zeros(n_pairs)

        for group in GROUP_ORDER:
            vals = np.array([
                per_pair[p][group] if per_pair[p] is not None else 0.0
                for p in _PAIR_ORDER
            ])
            ax.bar(x, vals, bottom=bottoms, color=GROUP_COLORS[group],
                   edgecolor="black", linewidth=0.4)
            bottoms += vals

        # Total label above each feasible bar; infeasible label inside slot
        for i, p in enumerate(_PAIR_ORDER):
            if per_pair[p] is None:
                ax.text(i, y_max * 0.04, "infeasible",
                        ha="center", va="bottom", fontsize=fs(8),
                        rotation=90, color="#888")
                ax.add_patch(plt.Rectangle((i - 0.4, 0), 0.8, y_max,
                                            fill=True, color="#eeeeee",
                                            zorder=0))
            else:
                ax.text(i, bottoms[i] + y_max * 0.01,
                        f"{bottoms[i]:.0f}",
                        ha="center", va="bottom", fontsize=fs(8),
                        fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(_PAIR_ORDER, rotation=30, ha="right", fontsize=fs(9))
        ax.set_title(rf"$T_\mathrm{{steam}} = {int(T)}$ °C",
                     fontsize=fs(11), fontweight="bold")
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=0.25)
        if j == 0:
            ax.set_ylabel(r"Exergy destruction  $\dot E_\mathrm{D}$  [kW]")

    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g])
               for g in GROUP_ORDER]
    fig.legend(handles=handles, loc="lower center",
               ncol=5, fontsize=fs(6), frameon=False,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        rf"Component $\dot{{E}}_\mathrm{{D}}$ at base case "
        rf"($\mathit{{LS}} = {ls*100:.0f}$ %, "
        rf"$T_\mathrm{{src,in}} = {int(T_src)}$ °C)",
        fontsize=fs(10), fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])
    save_titled_and_paper(fig, out_dir, "exergy_destruction_base")


# ── 2-D (FLH, c_el) sensitivity for the reference design ──────────────────

def plot_sensitivity_FLH_cel(out_dir: str,
                              ref_pair: str = "R717/R600",
                              ref_ls: float = 0.50,
                              ref_T_src: float = 60.0,
                              T_steam_for_break_even: float = 110.0):
    """Combined 2-D c_P map over (FLH, c_el) for the reference design.

    Uses the already-cached 1-D sweeps in
    ``economics_sensitivity_FLH.csv`` (FLH axis, at base c_el) and
    ``economics_sensitivity_e1.csv`` (c_el axis, at base FLH) to
    reconstruct c_P(FLH, c_el) under the affine model
        c_P(FLH, c_el) ≈ c_P_FLH(FLH) + slope_e1 · (c_el - BASE_E1_C)
    which holds because the HTHP fuel-cost term is linear in c_el and the
    capital-recovery term is linear in 1/FLH but independent of c_el.
    The break-even contour against the gas reference is overlaid at
    T_steam = ``T_steam_for_break_even`` (gas c_P is FLH-independent
    under the brownfield retrofit assumption).
    """
    data_dir = case_data_dir(T_steam_for_break_even)
    base_csv = os.path.join(data_dir, "economics_base.csv")
    flh_csv = os.path.join(data_dir, "economics_sensitivity_FLH.csv")
    e1_csv = os.path.join(data_dir, "economics_sensitivity_e1.csv")
    gas_json = os.path.join(case_gas_heater_dir(T_steam_for_break_even),
                             "gas_heater.json")
    for p in (base_csv, flh_csv, e1_csv):
        if not os.path.exists(p):
            print(f"  ! {p} missing — skipping sensitivity_FLH_cel.pdf")
            return

    base = pd.read_csv(base_csv)
    flh_df = pd.read_csv(flh_csv)
    e1_df = pd.read_csv(e1_csv)

    sel = ((base["pair"] == ref_pair)
           & (base["ls"] == ref_ls)
           & (base["T_src"] == ref_T_src))
    if not sel.any():
        print(f"  ! reference design {ref_pair} LS={ref_ls} T_src={ref_T_src} "
              f"missing at T_steam={T_steam_for_break_even} — skipping plot.")
        return
    cP_base = float(base.loc[sel, "c_P [EUR/GJ]"].iloc[0])

    flh_sub = flh_df[(flh_df["pair"] == ref_pair)
                      & (flh_df["ls"] == ref_ls)
                      & (flh_df["T_src"] == ref_T_src)] \
                 .sort_values("full_load_hours [h/a]")
    e1_sub = e1_df[(e1_df["pair"] == ref_pair)
                    & (e1_df["ls"] == ref_ls)
                    & (e1_df["T_src"] == ref_T_src)] \
                .sort_values("e1_c [EUR/MWh]")
    if flh_sub.empty or e1_sub.empty:
        print(f"  ! sensitivity rows missing for reference design "
              f"{ref_pair} LS={ref_ls} T_src={ref_T_src}; skipping plot.")
        return

    # Linear fit of c_P vs e1 (slope is FLH-independent for HTHP under the
    # affine cost model)
    slope_e1, _ = np.polyfit(e1_sub["e1_c [EUR/MWh]"],
                              e1_sub["c_P [EUR/GJ]"], 1)

    # FLH sweep is one-sided (5000-7500 h/a, with the base sitting at the
    # upper edge). Fit the affine-in-1/FLH model
    #     c_P(FLH) = a + b/FLH
    # — which holds because the capital-recovery term in the levelised-cost
    # formula scales as 1/FLH and every other term is FLH-independent — and
    # extrapolate by ±500 h/a so the base operating point sits interior
    # rather than on the upper border of the heatmap.
    flh_data = flh_sub["full_load_hours [h/a]"].astype(float).values
    cP_data = flh_sub["c_P [EUR/GJ]"].astype(float).values
    b_flh, a_flh = np.polyfit(1.0 / flh_data, cP_data, 1)

    flh_axis = np.linspace(flh_data.min() - 250.0,
                            flh_data.max() + 500.0, 60)
    cP_at_base_e1 = a_flh + b_flh / flh_axis
    e1_axis = np.linspace(BASE_E1_C * 0.5, BASE_E1_C * 1.5, 41)

    # cP[i,j] = cP_at_base_e1[i] + slope * (e1_axis[j] - BASE_E1_C)
    cP = (cP_at_base_e1[:, None]
          + slope_e1 * (e1_axis[None, :] - BASE_E1_C))

    # Gas reference c_P (FLH-independent under retrofit, T_steam-dependent)
    gas_cP = None
    if os.path.exists(gas_json):
        try:
            gas_cP = float(json.load(open(gas_json))["economics"]["c_P"])
        except Exception:
            gas_cP = None

    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, 4.0))
    vmin, vmax = float(np.nanmin(cP)), float(np.nanmax(cP))
    levels = np.linspace(vmin, vmax, 14)
    X, Y = np.meshgrid(e1_axis, flh_axis)
    cf = ax.contourf(X, Y, cP, levels=levels, cmap="viridis", extend="both")

    if gas_cP is not None:
        try:
            ax.contour(X, Y, cP - gas_cP, levels=[0.0],
                       colors="white", linewidths=2.8)
            ax.contour(X, Y, cP - gas_cP, levels=[0.0],
                       colors="black", linewidths=1.4, linestyles="--")
        except ValueError:
            pass

    # Base-case marker
    ax.plot(BASE_E1_C, BASE_FULL_LOAD_HOURS, marker="o", color="red",
            markersize=11, markeredgecolor="white",
            label=rf"Base ($c_\mathrm{{el,0}}={BASE_E1_C:.0f}$ EUR/MWh, "
                  rf"$\tau={BASE_FULL_LOAD_HOURS:.0f}$ h/a)  "
                  rf"$c_P={cP_base:.1f}$")
    ax.axhline(BASE_FULL_LOAD_HOURS, color="grey", linewidth=0.5, alpha=0.4)
    ax.axvline(BASE_E1_C, color="grey", linewidth=0.5, alpha=0.4)

    ax.set_xlabel(r"Electricity price $c_\mathrm{el,0}$  [EUR/MWh]")
    ax.set_ylabel(r"Full-load hours $\tau$  [h/a]")
    ax.set_xlim(e1_axis.min(), e1_axis.max())
    ax.set_ylim(flh_axis.min(), flh_axis.max())

    cbar = fig.colorbar(cf, ax=ax,
                         label=r"$c_P^{\,\mathrm{HTHP}}$  [EUR/GJ$_{ex}$]",
                         pad=0.02, fraction=0.045)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markeredgecolor="white", markersize=10,
                   label="Base case"),
    ]
    if gas_cP is not None:
        handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                                    linewidth=1.4,
                                    label=r"Break-even vs gas"))
    ax.legend(handles=handles, loc="lower right", fontsize=fs(8), framealpha=0.92)

    fig.suptitle(
        rf"$c_P^{{\mathrm{{HTHP}}}}(\tau, c_\mathrm{{el,0}})$ — "
        rf"{ref_pair}, $\mathit{{LS}}={int(ref_ls*100)}$ %, "
        rf"$T_\mathrm{{src,in}}={int(ref_T_src)}$ °C, "
        rf"$T_\mathrm{{steam}}={int(T_steam_for_break_even)}$ °C",
        fontsize=fs(10), fontweight="bold",
    )
    fig.tight_layout()
    save_titled_and_paper(fig, out_dir, "sensitivity_FLH_cel")


# ── Entry point ───────────────────────────────────────────────────────────

def compare_main():
    out_dir = t_steam_compare_dir()
    os.makedirs(out_dir, exist_ok=True)

    per_T = {}
    for T in T_STEAMS_TO_RUN:
        df, gas_cP = _load_economics_for(T)
        per_T[T] = {"df": df, "gas_cP": gas_cP}

    if all(per_T[T]["df"] is None for T in per_T):
        print("No per-T_steam economics data found — nothing to compare.")
        return

    print(f"Cross-T_steam comparison over T_steam ∈ "
          f"{[int(T) for T in T_STEAMS_TO_RUN]} °C")
    for T in T_STEAMS_TO_RUN:
        df = per_T[T]["df"]
        gas = per_T[T]["gas_cP"]
        if df is None:
            print(f"  T_steam = {int(T)} °C: MISSING")
            continue
        n = len(df)
        cP_min = df["c_P [EUR/GJ]"].min()
        gas_str = f"{gas:.2f}" if gas is not None else "?"
        print(f"  T_steam = {int(T)} °C: {n} designs, "
              f"best HTHP c_P = {cP_min:.2f} EUR/GJ_ex, "
              f"gas+CO₂ ref = {gas_str} EUR/GJ_ex")

    print()
    print("Plotting exergy_destruction_base.png ...")
    plot_fleet_exergy_destruction_base(out_dir)
    print("Plotting sensitivity_FLH_cel.png ...")
    plot_sensitivity_FLH_cel(out_dir, ref_pair="R717/R600")
    print(f"\nFigures written to {out_dir}/")




# ===================== c_P heatmap over (T_src,in, T_steam) =====================
# For every (pair, T_src,in, T_steam) combination, the lift share with the
# lowest c_P is selected from economics_base.csv. The result is a 2x3 grid
# (one panel per fluid pair, shared colour scale, T_src,in on x and T_steam
# on y) written to results/case_steam_compare/cP_heatmap_Tsrc_Tsteam.pdf.
# Reached via ``python plots.py cP_tsrc``.


import os
import sys
import warnings
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    case_data_dir, t_steam_compare_dir, T_STEAMS_TO_RUN,
)


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


# Same alphabetical order as ``_discover_pairs`` (sorted on-disk folder
# names), so each fluid pair lands in the same panel position as in the
# per-design ``cP_heatmap.pdf`` / ``TCI_per_kW_heatmap.pdf`` heatmaps.
PAIR_ORDER = ["R1270/R600", "R1270/R600a",
              "R290/R600",  "R290/R600a",
              "R717/R600",  "R717/R600a"]


def _load_all() -> pd.DataFrame:
    frames = []
    for T in T_STEAMS_TO_RUN:
        csv = os.path.join(case_data_dir(T), "economics_base.csv")
        if not os.path.exists(csv):
            print(f"  ! missing {csv}")
            continue
        frames.append(pd.read_csv(csv))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _best_cP_grid(df: pd.DataFrame, pair: str,
                  t_src_vals: list[float], t_steam_vals: list[float]):
    """Return (cP, ls) 2-D arrays indexed by (T_steam row, T_src column).

    cP[i, j] = lowest c_P across all LS for (T_steam=t_steam_vals[i],
                                              T_src=t_src_vals[j]).
    ls[i, j] = the LS value (fraction in [0, 1]) that attains that minimum.
    Missing combinations are returned as NaN.
    """
    n_rows = len(t_steam_vals)
    n_cols = len(t_src_vals)
    cP = np.full((n_rows, n_cols), np.nan)
    ls = np.full((n_rows, n_cols), np.nan)
    for i, T_steam in enumerate(t_steam_vals):
        sub_T = df[(df["pair"] == pair) & (df["T_steam"] == T_steam)]
        for j, T_src in enumerate(t_src_vals):
            sub = sub_T[sub_T["T_src"] == T_src]
            if sub.empty:
                continue
            row = sub.loc[sub["c_P [EUR/GJ]"].idxmin()]
            cP[i, j] = row["c_P [EUR/GJ]"]
            ls[i, j] = row["ls"]
    return cP, ls


def _draw_heatmap(ax, cP, ls, t_src_vals, t_steam_vals,
                  vmin, vmax, cmap, fontsize=fs(7)):
    """Render one (T_steam × T_src) heatmap into ``ax`` with cell text
    ``c_P / LS=xx`` whose colour is chosen from cell luminance."""
    masked = np.ma.masked_invalid(cP)
    # origin="lower" puts the first (lowest) T_steam row at the bottom and
    # the highest T_steam at the top; the index-based y-tick labels follow.
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="equal", origin="lower")
    for i in range(len(t_steam_vals)):
        for j in range(len(t_src_vals)):
            if np.isnan(cP[i, j]):
                continue
            norm = (cP[i, j] - vmin) / max(vmax - vmin, 1e-6)
            r, g, b, _ = cmap(norm)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            txt_color = "white" if luminance < 0.5 else "black"
            ax.text(j, i, f"{cP[i, j]:.1f}",
                    ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=txt_color)
    ax.set_xticks(range(len(t_src_vals)))
    ax.set_xticklabels([f"{T:g}" for T in t_src_vals])
    ax.set_yticks(range(len(t_steam_vals)))
    ax.set_yticklabels([f"{T:g}" for T in t_steam_vals])
    return im


def plot_cP_heatmap_all_pairs(df: pd.DataFrame, out_dir: str):
    """2x3 grid: one heatmap per fluid pair, shared colour scale."""
    t_src_vals = sorted(df["T_src"].unique())
    t_steam_vals = sorted(df["T_steam"].unique())

    grids = {pair: _best_cP_grid(df, pair, t_src_vals, t_steam_vals)
             for pair in PAIR_ORDER}
    cP_all = np.concatenate([g[0].ravel() for g in grids.values()])
    vmin = float(np.nanmin(cP_all))
    vmax = float(np.nanmax(cP_all))

    cmap = plt.get_cmap("plasma_r").copy()
    cmap.set_bad(color="#cccccc")

    ncols = 3
    nrows = (len(PAIR_ORDER) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(COL_DOUBLE_IN, 3.6),
                             squeeze=False)

    last_im = None
    for idx, pair in enumerate(PAIR_ORDER):
        ax = axes[idx // ncols][idx % ncols]
        cP, ls = grids[pair]
        last_im = _draw_heatmap(ax, cP, ls, t_src_vals, t_steam_vals,
                                vmin, vmax, cmap, fontsize=fs(7))
        ax.set_title(pair, fontsize=fs(9), fontweight="bold")
        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"$T_\mathrm{src,in}$  [°C]")
        if idx % ncols == 0:
            ax.set_ylabel(r"$T_\mathrm{steam}$  [°C]")

    if last_im is not None:
        fig.tight_layout(rect=[0, 0, 0.93, 0.93])
        cbar_ax = fig.add_axes([0.945, 0.10, 0.012, 0.80])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label(r"$c_P$  [EUR/GJ$_{ex}$]")

    fig.suptitle(
        r"$c_P$ heatmap over $(T_\mathrm{src,in},\,T_\mathrm{steam})$ "
        r"— best $\mathit{LS}$ per cell",
        fontsize=fs(10), fontweight="bold",
    )
    base = "cP_heatmap_Tsrc_Tsteam"
    save_titled_and_paper(fig, out_dir, base)
    print(f"  wrote {os.path.join(out_dir, base)}.pdf")


def cP_tsrc_main():
    out_dir = t_steam_compare_dir()
    os.makedirs(out_dir, exist_ok=True)
    df = _load_all()
    if df.empty:
        print("No economics_base.csv files found.")
        return
    plot_cP_heatmap_all_pairs(df, out_dir)




# ===================== per-design Z+C_D sensitivity and heatmaps =====================
# Per-design metric heatmaps and the feasibility map, written into the case
# plots folder. Reached via ``python plots.py cdz [T ...]``.


import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from matplotlib.colors import ListedColormap

from config import (
    BASE_E1_C, BASE_FULL_LOAD_HOURS, BASE_GAS_C, R_N_EL, T_STEAMS_TO_RUN,
    case_data_dir, case_plots_dir, case_results_dir, m_steam_label,
)
from economics import F_INSTALL, _celf

# Levelized first-year electricity price [EUR/MWh]. Multiplying c_el,0 by the
# CELF at the electricity escalation rate r_n,el yields the constant equivalent
# annual price over the 20-year horizon; this is the value used in the LCOH
# definition stated in the paper.
C_EL_LEV = BASE_E1_C * _celf(R_N_EL)

# Component grouping pulled from the central common dict. COMP+MOT of
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

    By default NOSOLVE cells (no convergence / over-critical) come back NaN,
    so cost / pressure / temperature heatmaps cover every thermodynamically-
    feasible design. Pass ``accept_states=None`` to keep NOSOLVE rows too.
    """
    enriched_csv = os.path.join(case_dir,
                                f"case_steam_{int(T_steam)}_enriched.csv")
    if not os.path.exists(enriched_csv):
        return None
    df = pd.read_csv(enriched_csv)
    df = df[df["pair"] == pair]
    if accept_states is not None and "status_4state" in df.columns:
        df = df[df["status_4state"] != "NOSOLVE"]
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

        LCOH [EUR/MWh_th] = (1000 · Ż_sum / Q_H) + c_el,lev / COP

    where c_el,lev = c_el,0 · CELF(r_n,el) is the levelized electricity
    price (constant equivalent annual value over the 20-year horizon).
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
            grid[i, j] = 1000.0 * Z_sum / Q_H + C_EL_LEV / COP
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
    # origin="lower" puts the first LS row (smallest, 30 %) at the bottom
    # and the last (50 %) at the top; the index-based y-tick labels follow.
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="equal", origin="lower")
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
# Feasibility heatmap coded by failure cause: envelope status (OK / V_ONLY /
# HARD) combined with thermodynamic-solver status (none / T_crit_c1 /
# T_crit_c2 / Pcrit_c1 / convergence). Built on an integer-coded grid paired
# with an index→(color, label) map.

# "Why does it fail" palette for the feasibility heatmap. Each cell is
# coloured by its dominant failure cause: thermodynamic (no solution) reasons
# take precedence over envelope ones.
FAIL_PALETTE = {
    0: ("#66bb6a", "feasible"),
    1: ("#ef6c00", "T_crit cycle 1 (over-critical T)"),
    2: ("#d84315", "T_crit cycle 2 (over-critical T)"),
    3: ("#c62828", "p_crit cycle 1 (over-critical p)"),
    4: ("#6d4c41", "no convergence"),
    5: ("#5e35b1", r"$p_\mathrm{high}$ > limit"),
    6: ("#00838f", r"$T_\mathrm{disch}$ > limit"),
    7: ("#fbc02d", r"$\dot{V}$ out of range"),
}

FAIL_THERMO_TO_CODE = {
    "": 0, "T_crit_c1": 1, "T_crit_c2": 2, "Pcrit_c1": 3, "convergence": 4,
}


def _csv_bool(v):
    """True iff a value (possibly a 'True'/'False' string from a CSV) is true."""
    return v is True or str(v).strip().lower() in ("true", "1", "1.0", "yes")


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _fail_code(row):
    """(code, label) for one design row in the Ommen feasibility heatmap.

    ``code`` (0-7, see ``FAIL_PALETTE``) is the dominant reason; ``label`` is
    the value of the exceeded quantity (bar / °C / m³/h) for the envelope
    categories, "" otherwise. A design is only ever *excluded* on
    thermodynamic grounds (``no_solve_class``); the Ommen pressure /
    discharge-T / volume-flow limits are reported here for information only,
    in the precedence pressure > discharge-T > volume flow.
    """
    nsc = str(row.get("no_solve_class") or "").strip()
    if nsc and nsc.lower() != "nan":
        return FAIL_THERMO_TO_CODE.get(nsc, 4), ""
    ok = lambda c: _csv_bool(row.get(c))
    p_ok = ok("p_OK_c1") and ok("p_OK_c2")
    T_ok = ok("T_OK_c1") and ok("T_OK_c2")
    V_ok = ok("V_OK_c1") and ok("V_OK_c2")
    if p_ok and T_ok and V_ok:
        return 0, ""
    if not p_ok:
        ps = [_num(row.get(f"p_high_c{c} [bar]"))
              for c in (1, 2) if not ok(f"p_OK_c{c}")]
        return 5, f"{max(ps):.0f}"
    if not T_ok:
        Ts = [_num(row.get(f"T_disch_c{c} [°C]"))
              for c in (1, 2) if not ok(f"T_OK_c{c}")]
        return 6, f"{max(Ts):.0f}"
    Vs = [_num(row.get(f"V_dot_c{c}_target [m3/h]"))
          for c in (1, 2) if not ok(f"V_OK_c{c}")]
    return 7, f"{max(Vs):.0f}"


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
    # origin="lower" puts the first LS row (smallest, 30 %) at the bottom
    # and the last (50 %) at the top; the index-based y-tick labels follow.
    im = ax.imshow(masked, cmap=cmap, vmin=-0.5, vmax=n_codes - 0.5,
                   aspect="equal", origin="lower")
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


def _load_fail_grid(case_dir, T_steam, pair, ls_vals, t_src_vals):
    """Build the (LS × T_src) Ommen failure-reason grid for one pair.

    Returns ``(grid, cell_labels)``: ``grid`` is integer-coded per
    ``FAIL_PALETTE``; ``cell_labels`` is a matching 2-D list of value strings
    (the exceeded p/T/V̇ magnitude) drawn on the heatmap. Reads the enriched
    screen CSV.
    """
    enriched_csv = os.path.join(case_dir,
                                f"case_steam_{int(T_steam)}_enriched.csv")
    if not os.path.exists(enriched_csv):
        return None
    df = pd.read_csv(enriched_csv)
    df = df[df["pair"] == pair]
    grid = np.full((len(ls_vals), len(t_src_vals)), np.nan)
    labels = [["" for _ in t_src_vals] for _ in ls_vals]
    for i, ls_pct in enumerate(ls_vals):
        for j, T in enumerate(t_src_vals):
            row = df[(np.isclose(df["ls"], ls_pct / 100.0))
                     & (np.isclose(df["T_src"], T))]
            if row.empty:
                continue
            code, label = _fail_code(row.iloc[0])
            grid[i, j] = code
            labels[i][j] = label
    return grid, labels


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
        grids[pair] = g   # (grid, cell_labels)
    if not grids:
        return None

    codes = sorted(palette.keys())
    colors = [palette[c][0] for c in codes]
    cmap = ListedColormap(colors)

    n = len(pairs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    # The panels are aspect="equal" (fixed height ~1.5 in each), so figure
    # height only buys inter-row gap. 5.2 in gives the two rows enough room
    # that the bottom-row titles clear the top-row x-axis labels, plus the
    # bottom-legend reservation.
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(COL_DOUBLE_IN, 5.2),
                              squeeze=False)

    for idx, pair in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        if pair not in grids:
            ax.set_title(pair, fontsize=fs(9), fontweight="bold")
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#888888", fontsize=fs(8))
            ax.set_xticks([]); ax.set_yticks([])
            continue
        g, cell_labels = grids[pair]
        _draw_categorical_heatmap(ax, g, ls_axis, tsrc_axis, cmap,
                                   cell_labels=cell_labels, fontsize=fs(7))
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
               ncol=min(4, len(legend_handles)),
               fontsize=fs(7), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(rf"{suptitle}  ($T_\mathrm{{steam}}$ = {T_steam:.0f} °C)",
                  fontsize=fs(10), fontweight="bold")
    save_titled_and_paper(fig, plots_dir, base_name)
    return os.path.join(plots_dir, f"{base_name}.pdf")


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
# Single source of truth for the 13 continuous per-design heatmaps. Each
# entry is consumed by ``run_all`` and produces a paper variant plus a
# titled variant via ``save_titled_and_paper``.

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
            fmt="{:.1f}", cmap_name="plasma_r")),
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

        # 13 continuous-metric per-design heatmaps (paper + titled variants).
        for label, kwargs in _build_continuous_metrics(
                data_dir, designs_dir, case_dir, T_steam):
            out = _plot_metric_grid(designs_dir, plots_dir, pairs, T_steam,
                                     **kwargs)
            if out:
                print(f"  [ok] T_steam={T_steam:g} C  {label:<12s} heatmap "
                      f"-> {out}")

        # 1 categorical "why does it fail" heatmap (single Ommen screen).
        # Each cell is coloured by its dominant reason and annotated with the
        # exceeded magnitude; only thermodynamic reasons exclude a design.
        out = _plot_categorical_grid(
            designs_dir, plots_dir, pairs, T_steam,
            loader=lambda p, ls, t: _load_fail_grid(case_dir, T_steam, p, ls, t),
            palette=FAIL_PALETTE,
            suptitle="Feasibility per design (Ommen 2015 envelope)",
            base_name="feasibility_ommen")
        if out:
            print(f"  [ok] T_steam={T_steam:g} C  feasibility_ommen -> {out}")




# ===================== standalone CLI dispatch =====================
if __name__ == "__main__":
    import sys
    _arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if _arg == "cdz":
        _Ts = [float(x) for x in sys.argv[2:]] or None
        run_all(_Ts)
    elif _arg == "cP_tsrc":
        cP_tsrc_main()
    else:
        print("usage: python plots.py [cdz [T ...] | cP_tsrc]")
