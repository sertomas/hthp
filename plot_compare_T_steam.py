"""
plot_compare_T_steam.py — Cross-T_steam comparison plots.

Reads per-T_steam economics outputs (one set under
`results/case_steam_<T>/economics/` for each of T ∈ {100, 110, 120} °C) and
writes two comparison figures into `results/case_steam_compare/`:

  1. `compare_cP_best_vs_T_steam.png` — for each fluid pair, the lowest c_P
     achievable at each steam temperature, with the gas+CO₂ reference line
     at each T_steam (it shifts because the steam exergy E_P depends on T).
     Tells you whether the HTHP-vs-gas gap widens or narrows with T_steam.

  2. `compare_cP_heatmap.png` — 6 fluid pairs × 3 T_steam columns; cell
     colour = best c_P [EUR/GJ_ex] across all (LS, T_src) for that combo,
     with cell text showing the winning (LS, T_src). Designs that beat
     the gas+CO₂ reference at the corresponding T_steam are highlighted
     with a green frame.

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
    case_results_dir, t_steam_compare_dir,
    T_STEAMS_TO_RUN,
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
    case_dir = case_results_dir(T_steam)
    base_csv = os.path.join(case_dir, "economics", "economics_base.csv")
    gas_json = os.path.join(case_dir, "economics", "gas_heater",
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
    fig, ax = plt.subplots(figsize=(10.0, 6.5))

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
                label=f"Gas+CO₂ ref (gas={BASE_GAS_C:.0f} EUR/MWh)")

    ax.set_xticks(sorted(per_T))
    ax.set_xlabel("Steam temperature  [°C]", fontsize=11)
    ax.set_ylabel(r"$c_P$  [EUR/GJ$_{ex}$]", fontsize=11)
    ax.set_title(
        "Best HTHP $c_P$ per fluid pair vs steam temperature\n"
        f"(FLH = {BASE_FULL_LOAD_HOURS} h/a, e1 = {BASE_E1_C:.0f} EUR/MWh — "
        "lowest c_P design selected per pair)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_cP_best_vs_T_steam.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


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
                f"{best['c_P [EUR/GJ]']:.1f}\n"
                f"LS={int(best['ls']*100)}%  T={int(best['T_src'])}°C"
            )
            if best["c_P [EUR/GJ]"] < gas_cP:
                beats_gas[i, j] = True

    fig, ax = plt.subplots(figsize=(2.2 * n_T + 4, 0.9 * n_pairs + 2))
    cmap = plt.get_cmap("YlGnBu_r")
    cmap.set_bad(color="#cccccc")
    im = ax.imshow(np.ma.masked_invalid(cP), cmap=cmap, aspect="auto")

    for i in range(n_pairs):
        for j in range(n_T):
            if np.isnan(cP[i, j]): continue
            ax.text(j, i, annotations[i][j],
                    ha="center", va="center", fontsize=9,
                    color=("white" if cP[i, j] > np.nanmean(cP) else "black"))
            if beats_gas[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False, edgecolor="#2e7d32",
                                            linewidth=3.5))

    ax.set_xticks(range(n_T))
    ax.set_xticklabels([f"{int(T)} °C" for T in Ts], fontsize=11)
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pairs, fontsize=11)
    ax.set_xlabel("Steam temperature", fontsize=11)
    ax.set_ylabel("Fluid pair", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, label=r"best $c_P$  [EUR/GJ$_{ex}$]",
                        pad=0.02, fraction=0.04)

    # Gas reference annotation as a footnote
    gas_str = "  ".join(
        f"{int(T)}°C: {per_T[T]['gas_cP']:.1f}" if per_T[T]["gas_cP"] else f"{int(T)}°C: ?"
        for T in Ts
    )

    ax.set_title(
        "Cross-T_steam best $c_P$ per fluid pair  "
        f"(FLH={BASE_FULL_LOAD_HOURS} h/a, e1={BASE_E1_C:.0f} EUR/MWh)\n"
        f"Green frame: HTHP beats gas+CO₂ at that T_steam   "
        f"(gas+CO₂ refs:  {gas_str})",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_cP_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    print(f"\nFigures written to {out_dir}/")


if __name__ == "__main__":
    main()
