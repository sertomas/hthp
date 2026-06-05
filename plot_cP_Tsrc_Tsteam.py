"""
plot_cP_Tsrc_Tsteam.py — c_P heatmap over (T_src,in, T_steam) for each fluid pair.

For every (pair, T_src,in, T_steam) combination, the lift share with the
lowest c_P is selected from ``economics_base.csv``. Each cell shows that
best c_P and is annotated with the winning LS underneath. Style matches
the per-design cP / TCI / PEC heatmaps in ``plot_cdz_sensitivity_per_design``
(plasma_r colormap, equal-aspect cells, luminance-based text colour,
shared colorbar). Two outputs are written to ``results/case_steam_compare/``:

  * ``cP_heatmap_Tsrc_Tsteam.pdf`` — 2x3 grid, one panel per fluid pair,
    shared colour scale (T_src,in on x, T_steam on y).
  * ``cP_heatmap_Tsrc_Tsteam_R717_R600.pdf`` — single panel for the
    cost-optimal pair.

Standalone script — run directly:
    python plot_cP_Tsrc_Tsteam.py
"""

from __future__ import annotations

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
from plot_common import (
    COL_DOUBLE_IN, COL_SINGLE_IN, save_titled_and_paper, fs,
)


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


# Same alphabetical order as ``_discover_pairs`` in
# ``plot_cdz_sensitivity_per_design.py`` (sorted on-disk folder names), so
# each fluid pair lands in the same panel position as in ``cP_heatmap.pdf``,
# ``TCI_per_kW_heatmap.pdf``, etc.
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


def plot_cP_heatmap_single_pair(df: pd.DataFrame, out_dir: str,
                                 pair: str = "R717/R600"):
    """Single-panel heatmap for one fluid pair."""
    t_src_vals = sorted(df["T_src"].unique())
    t_steam_vals = sorted(df["T_steam"].unique())
    cP, ls = _best_cP_grid(df, pair, t_src_vals, t_steam_vals)
    vmin = float(np.nanmin(cP))
    vmax = float(np.nanmax(cP))
    cmap = plt.get_cmap("plasma_r").copy()
    cmap.set_bad(color="#cccccc")

    fig, ax = plt.subplots(figsize=(COL_SINGLE_IN, 2.8))
    im = _draw_heatmap(ax, cP, ls, t_src_vals, t_steam_vals,
                       vmin, vmax, cmap, fontsize=fs(7))
    ax.set_xlabel(r"$T_\mathrm{src,in}$  [°C]")
    ax.set_ylabel(r"$T_\mathrm{steam}$  [°C]")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$c_P$  [EUR/GJ$_{ex}$]")
    fig.suptitle(
        rf"$c_P$ — {pair}  (best $\mathit{{LS}}$ per cell)",
        fontsize=fs(9), fontweight="bold",
    )
    fig.tight_layout()
    base = f"cP_heatmap_Tsrc_Tsteam_{pair.replace('/', '_')}"
    save_titled_and_paper(fig, out_dir, base)
    print(f"  wrote {os.path.join(out_dir, base)}.pdf")


def main():
    out_dir = t_steam_compare_dir()
    os.makedirs(out_dir, exist_ok=True)
    df = _load_all()
    if df.empty:
        print("No economics_base.csv files found.")
        return
    plot_cP_heatmap_all_pairs(df, out_dir)
    plot_cP_heatmap_single_pair(df, out_dir, pair="R717/R600")


if __name__ == "__main__":
    main()
