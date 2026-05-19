"""
plot_compare_T_steam.py — Cross-T_steam comparison plots.

Reads per-T_steam economics outputs (one set under
`results/case_steam_<T>/economics/` for each of T ∈ {100, 110, 120} °C) and
writes four comparison figures into `results/case_steam_compare/`:

  1. `compare_cP_best_vs_T_steam.png` — for each fluid pair, the lowest c_P
     achievable at each steam temperature, with the gas+CO₂ reference line
     at each T_steam (it shifts because the steam exergy E_P depends on T).
     Tells you whether the HTHP-vs-gas gap widens or narrows with T_steam.

  2. `compare_cP_heatmap.png` — 6 fluid pairs × 3 T_steam columns; cell
     colour = best c_P [EUR/GJ_ex] across all (LS, T_src) for that combo,
     with cell text showing the winning (LS, T_src). Designs that beat
     the gas+CO₂ reference at the corresponding T_steam are highlighted
     with a green frame.

  3. `exergy_destruction_base.png` — fleet-wide component-level E_D at the
     base case (LS = 0.50, T_src = 60 °C). Three panels (T_steam ∈ {100,
     110, 120} °C); x-axis = fluid pair, bars stacked by component. Bars
     for infeasible (pair, T_steam) combinations at the base case are
     left empty with an "infeasible" label so the reader can see the
     fluid-side feasibility envelope at a glance.

  4. `sensitivity_FLH_cel.png` — single 2-D c_P heatmap over (FLH, c_el)
     for the reference R717/R600a design at base case, with the gas-
     heater break-even contour overlaid at T_steam = 110 °C. Replaces
     the two separate 1-D plots (cP_vs_FLH, cP_vs_e1c) in the paper.

Run after `main.py` has populated all three temperatures' economics CSVs.
"""

from __future__ import annotations

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
from plot_common import (
    COL_DOUBLE_IN, COL_SINGLE_IN,
    GROUP_STYLE, GROUP_ORDER, GROUP_COLORS, GROUP_LABELS, GROUP_MEMBERS,
    save_titled_and_paper,
    fs,
)


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


# Per-fluid-pair colours — re-use a palette consistent with cP_vs_gas.png
PAIR_COLORS = {
    "R1270/R600":  "#1f77b4",
    "R1270/R600a": "#ff7f0e",
    "R290/R600":   "#2ca02c",
    "R290/R600a":  "#d62728",
    "R717/R600":   "#9467bd",
    "R717/R600a":  "#8c564b",
}


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


# ── Plot 1: c_P best per pair vs T_steam ──────────────────────────────────

def plot_cP_best_vs_T_steam(per_T: dict[float, dict], out_dir: str):
    """One panel: line per fluid pair, gas+CO₂ ref line, all temperatures."""
    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, 4.0))

    pairs_seen = set()
    for T, payload in sorted(per_T.items()):
        if payload["df"] is None:
            continue
        for p in payload["df"]["pair"].unique():
            pairs_seen.add(p)

    pairs = sorted(pairs_seen)

    # One line per fluid pair
    for pair in pairs:
        Ts = []
        cPs = []
        for T in sorted(per_T):
            df = per_T[T]["df"]
            if df is None: continue
            sub = df[df["pair"] == pair]
            if sub.empty: continue
            Ts.append(T)
            cPs.append(sub["c_P [EUR/GJ]"].min())
        if not Ts: continue
        ax.plot(Ts, cPs,
                color=PAIR_COLORS.get(pair, "gray"),
                marker="o", markersize=10, linewidth=2.0,
                markeredgecolor="black", markeredgewidth=0.6,
                label=pair)

    # Gas+CO2 reference (one point per T_steam, dashed line connecting)
    Ts_gas = []
    cPs_gas = []
    for T in sorted(per_T):
        gas_cP = per_T[T]["gas_cP"]
        if gas_cP is None: continue
        Ts_gas.append(T); cPs_gas.append(gas_cP)
    if Ts_gas:
        ax.plot(Ts_gas, cPs_gas,
                color="black", linestyle="--", linewidth=2.0,
                marker="s", markersize=10, markerfacecolor="none",
                markeredgecolor="black", markeredgewidth=1.5,
                label=rf"Gas+CO₂ ref ($c_\mathrm{{gas}}={BASE_GAS_C:.0f}$ EUR/MWh)")

    ax.set_xticks(sorted(per_T))
    ax.set_xlabel(r"Steam temperature $T_\mathrm{steam}$  [°C]")
    ax.set_ylabel(r"$c_P$  [EUR/GJ$_{ex}$]")
    fig.suptitle(r"Best $c_P$ per fluid pair vs $T_\mathrm{steam}$",
                  fontsize=fs(10), fontweight="bold")
    ax.legend(loc="upper left", fontsize=fs(7), frameon=True)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    save_titled_and_paper(fig, out_dir, "compare_cP_best_vs_T_steam")


# ── Plot 2: c_P heatmap (pair × T_steam) ──────────────────────────────────

def plot_cP_heatmap(per_T: dict[float, dict], out_dir: str):
    """Heatmap with one row per fluid pair, one column per T_steam.

    Cell colour = best c_P; cell text = best (LS, T_src) annotation;
    green frame around cells where best c_P < gas+CO₂ reference at the
    corresponding T_steam.
    """
    Ts = sorted(per_T)

    pairs_seen = set()
    for T in Ts:
        if per_T[T]["df"] is None: continue
        pairs_seen.update(per_T[T]["df"]["pair"].unique())
    pairs = sorted(pairs_seen)

    n_pairs = len(pairs)
    n_T = len(Ts)
    cP = np.full((n_pairs, n_T), np.nan)
    annotations = [["" for _ in range(n_T)] for _ in range(n_pairs)]
    beats_gas = np.zeros((n_pairs, n_T), dtype=bool)

    for j, T in enumerate(Ts):
        df = per_T[T]["df"]
        gas_cP = per_T[T]["gas_cP"] or np.inf
        if df is None: continue
        for i, pair in enumerate(pairs):
            sub = df[df["pair"] == pair]
            if sub.empty: continue
            best = sub.loc[sub["c_P [EUR/GJ]"].idxmin()]
            cP[i, j] = best["c_P [EUR/GJ]"]
            annotations[i][j] = (
                rf"${best['c_P [EUR/GJ]']:.1f}$" + "\n"
                + rf"$\mathit{{LS}}={int(best['ls']*100)}$ %  "
                + rf"$T_\mathrm{{src,in}}={int(best['T_src'])}$ °C"
            )
            if best["c_P [EUR/GJ]"] < gas_cP:
                beats_gas[i, j] = True

    fig, ax = plt.subplots(figsize=(COL_DOUBLE_IN, 0.55 * n_pairs + 1.5))
    cmap = plt.get_cmap("YlGnBu_r")
    cmap.set_bad(color="#cccccc")
    im = ax.imshow(np.ma.masked_invalid(cP), cmap=cmap, aspect="auto")

    for i in range(n_pairs):
        for j in range(n_T):
            if np.isnan(cP[i, j]): continue
            ax.text(j, i, annotations[i][j],
                    ha="center", va="center", fontsize=fs(9),
                    color=("white" if cP[i, j] > np.nanmean(cP) else "black"))
            if beats_gas[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False, edgecolor="#2e7d32",
                                            linewidth=3.5))

    ax.set_xticks(range(n_T))
    ax.set_xticklabels([f"{int(T)} °C" for T in Ts], fontsize=fs(11))
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pairs, fontsize=fs(11))
    ax.set_xlabel(r"Steam temperature $T_\mathrm{steam}$", fontsize=fs(11))
    ax.set_ylabel("Fluid pair", fontsize=fs(11))

    cbar = fig.colorbar(im, ax=ax, label=r"best $c_P$  [EUR/GJ$_{ex}$]",
                        pad=0.02, fraction=0.04)

    # Gas reference annotation as a footnote
    gas_str = "  ".join(
        f"{int(T)}°C: {per_T[T]['gas_cP']:.1f}" if per_T[T]["gas_cP"] else f"{int(T)}°C: ?"
        for T in Ts
    )

    fig.suptitle(
        r"Best $c_P$ per (fluid pair × $T_\mathrm{steam}$)",
        fontsize=fs(10), fontweight="bold",
    )
    fig.tight_layout()
    save_titled_and_paper(fig, out_dir, "compare_cP_heatmap")


# ── Plot 3: fleet exergy destruction at base case (T_src=50, LS=0.30) ─────

# Per-component breakdown. Colours and labels come from the central
# palette in plot_common.GROUP_STYLE so every component plot across the
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


# ── Plot 4: 2-D (FLH, c_el) sensitivity for the reference design ──────────

def plot_sensitivity_FLH_cel(out_dir: str,
                              ref_pair: str = "R717/R600a",
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
            label=rf"Base ($c_\mathrm{{el}}={BASE_E1_C:.0f}$ EUR/MWh, "
                  rf"$\tau={BASE_FULL_LOAD_HOURS:.0f}$ h/a)  "
                  rf"$c_P={cP_base:.1f}$")
    ax.axhline(BASE_FULL_LOAD_HOURS, color="grey", linewidth=0.5, alpha=0.4)
    ax.axvline(BASE_E1_C, color="grey", linewidth=0.5, alpha=0.4)

    ax.set_xlabel(r"Electricity price $c_\mathrm{el}$  [EUR/MWh]")
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
        rf"$c_P^{{\mathrm{{HTHP}}}}(\tau, c_\mathrm{{el}})$ — "
        rf"{ref_pair}, $\mathit{{LS}}={int(ref_ls*100)}$ %, "
        rf"$T_\mathrm{{src,in}}={int(ref_T_src)}$ °C, "
        rf"$T_\mathrm{{steam}}={int(T_steam_for_break_even)}$ °C",
        fontsize=fs(10), fontweight="bold",
    )
    fig.tight_layout()
    save_titled_and_paper(fig, out_dir, "sensitivity_FLH_cel")


# ── Entry point ───────────────────────────────────────────────────────────

def main():
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
    print("Plotting compare_cP_best_vs_T_steam.png ...")
    plot_cP_best_vs_T_steam(per_T, out_dir)
    print("Plotting compare_cP_heatmap.png ...")
    plot_cP_heatmap(per_T, out_dir)
    print("Plotting exergy_destruction_base.png ...")
    plot_fleet_exergy_destruction_base(out_dir)
    print("Plotting sensitivity_FLH_cel.png ...")
    plot_sensitivity_FLH_cel(out_dir)
    print(f"\nFigures written to {out_dir}/")


if __name__ == "__main__":
    main()
