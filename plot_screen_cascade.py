"""
plot_screen_cascade.py — Visualise the cascaded-HTHP feasibility screening
produced by ``screen_cascade.py``. Reads
``results/screen_cascade/cascade_screen.csv`` and writes figures to the
same folder.

Figures:
    1. feasibility_grid.png    — 3 × 3 panel grid (T_steam × LS), each panel
                                  is fluid-pair × T_src colored by feasibility
                                  (green/red), annotated with COP.
    2. cop_heatmap.png         — same layout, viridis on COP, NaN/grey if
                                  infeasible.
    3. steam_kgph_heatmap.png  — same layout, plasma on per-machine kg/h.
    4. p_high_constraints.png  — p_high vs T_steam at default LS, panel per
                                  cycle (1, 2), envelope lines (28/50/140 bar).
    5. T_disch_constraints.png — T_disch vs T_steam at default LS, panel per
                                  cycle, 180 °C limit line.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


CSV_PATH = os.path.join("results", "screen_cascade", "cascade_screen.csv")
OUT_DIR = os.path.join("results", "screen_cascade")


def _is_true(v) -> bool:
    return v is True or v == "True" or v == 1 or v == 1.0


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    for col in ["COP", "m_steam [kg/h]", "p_high_c1 [bar]", "p_high_c2 [bar]",
                "T_disch_c1 [°C]", "T_disch_c2 [°C]"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["feasible_bool"] = df["feasible"].apply(_is_true)
    df["pair"] = df["f1"] + "/" + df["f2"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helper: pivot a metric for a given T_steam, LS slice into a fluid-pair × T_src grid
# ─────────────────────────────────────────────────────────────────────────────

def _slice_grid(df, T_steam, ls, value_col, *, feasible_only=True):
    sub = df[(df["T_steam"] == T_steam) & (df["ls"] == ls)]
    if feasible_only:
        sub = sub[sub["feasible_bool"]]
    pair_order = sorted(df["pair"].unique())
    T_src_order = sorted(df["T_src"].unique())
    grid = np.full((len(pair_order), len(T_src_order)), np.nan)
    for i, pair in enumerate(pair_order):
        for j, T_src in enumerate(T_src_order):
            row = sub[(sub["pair"] == pair) & (sub["T_src"] == T_src)]
            if not row.empty:
                grid[i, j] = row.iloc[0][value_col]
    return grid, pair_order, T_src_order


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: feasibility grid (per T_steam, LS) — green/red with COP annotation
# ─────────────────────────────────────────────────────────────────────────────

def plot_feasibility_grid(df: pd.DataFrame):
    T_steam_vals = sorted(df["T_steam"].unique())
    ls_vals = sorted(df["ls"].unique())
    pair_order = sorted(df["pair"].unique())
    T_src_order = sorted(df["T_src"].unique())

    cmap = ListedColormap(["#cccccc", "#e57373", "#66bb6a"])

    n_rows = len(ls_vals)
    n_cols = len(T_steam_vals)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 4 * n_rows),
                              squeeze=False)

    for i, ls in enumerate(ls_vals):
        for j, T_steam in enumerate(T_steam_vals):
            ax = axes[i][j]
            sub = df[(df["T_steam"] == T_steam) & (df["ls"] == ls)]

            Z = np.zeros((len(pair_order), len(T_src_order)), dtype=int)
            COP = np.full_like(Z, np.nan, dtype=float)
            reasons = np.full(Z.shape, "", dtype=object)
            no_solve_class = np.full(Z.shape, "", dtype=object)

            for ii, pair in enumerate(pair_order):
                for jj, T_src in enumerate(T_src_order):
                    row = sub[(sub["pair"] == pair) & (sub["T_src"] == T_src)]
                    if row.empty:
                        Z[ii, jj] = 0
                        continue
                    r = row.iloc[0]
                    if _is_true(r["feasible"]):
                        Z[ii, jj] = 2
                        COP[ii, jj] = r["COP"]
                    elif r.get("reason") == "no_solve":
                        Z[ii, jj] = 0
                        no_solve_class[ii, jj] = str(r.get("no_solve_class", ""))
                    else:
                        Z[ii, jj] = 1
                        reasons[ii, jj] = str(r.get("reason", ""))

            ax.imshow(Z, cmap=cmap, vmin=0, vmax=2, aspect="auto")
            for ii in range(len(pair_order)):
                for jj in range(len(T_src_order)):
                    if Z[ii, jj] == 2:
                        txt = f"{COP[ii, jj]:.2f}"
                        color = "white"
                    elif Z[ii, jj] == 1:
                        txt = (str(reasons[ii, jj]).replace("p_c1", "p₁")
                                                  .replace("p_c2", "p₂")
                                                  .replace("T_c1", "T₁")
                                                  .replace("T_c2", "T₂")
                                                  .replace("V_c1", "V₁")
                                                  .replace("V_c2", "V₂"))
                        color = "black"
                    else:
                        # no-solve cells: show classified reason
                        cls = no_solve_class[ii, jj]
                        txt_map = {
                            "T_crit_c1":   "T_crit\nc1",
                            "T_crit_c2":   "T_crit\nc2",
                            "Pcrit_c1":    "P_crit\nc1",
                            "convergence": "TESPy\nconv.",
                        }
                        txt = txt_map.get(cls, "no\nsolve")
                        color = "#444444"
                    ax.text(jj, ii, txt, ha="center", va="center",
                            fontsize=8, color=color)

            ax.set_xticks(range(len(T_src_order)))
            ax.set_xticklabels([f"{T:g}" for T in T_src_order])
            ax.set_yticks(range(len(pair_order)))
            ax.set_yticklabels(pair_order, fontsize=9)
            ax.set_xlabel("T_source_in [°C]")
            if j == 0:
                ax.set_ylabel(f"LS = {ls*100:.0f}%", fontsize=11, fontweight="bold")
            ax.set_title(f"T_steam = {T_steam:.0f} °C", fontsize=10)

    fig.suptitle(
        "Cascade feasibility (Ommen 2015 Table 3 envelope, V̇ = V_max binding)\n"
        "green = feasible (cell shows COP); red = constraint failure (p₁/p₂/T₁/T₂/V₁/V₂);  "
        "grey = no solve (T_crit_c1/c2 = above critical, P_crit_c1 = ≥ 95 % p_crit, TESPy conv.)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, "feasibility_grid.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2/3: heatmap of COP / m_steam (feasible cells only, gray otherwise)
# ─────────────────────────────────────────────────────────────────────────────

def plot_value_heatmap(df: pd.DataFrame, value_col: str, title: str,
                       fname: str, fmt: str = "{:.2f}", cmap: str = "viridis"):
    T_steam_vals = sorted(df["T_steam"].unique())
    ls_vals = sorted(df["ls"].unique())

    feasible_vals = df.loc[df["feasible_bool"], value_col].dropna()
    if feasible_vals.empty:
        print(f"  ! no feasible data for {value_col}; skipping {fname}")
        return
    vmin = float(feasible_vals.min())
    vmax = float(feasible_vals.max())

    n_rows, n_cols = len(ls_vals), len(T_steam_vals)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 4 * n_rows),
                              squeeze=False)
    last_im = None

    for i, ls in enumerate(ls_vals):
        for j, T_steam in enumerate(T_steam_vals):
            ax = axes[i][j]
            grid, pair_order, T_src_order = _slice_grid(
                df, T_steam, ls, value_col, feasible_only=True
            )
            cmap_obj = plt.cm.get_cmap(cmap).copy()
            cmap_obj.set_bad(color="#cccccc")
            masked = np.ma.masked_invalid(grid)
            last_im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax,
                                aspect="auto")

            for ii in range(grid.shape[0]):
                for jj in range(grid.shape[1]):
                    if not np.isnan(grid[ii, jj]):
                        ax.text(jj, ii, fmt.format(grid[ii, jj]),
                                ha="center", va="center", fontsize=8,
                                color="white" if (grid[ii, jj] - vmin) /
                                                  max(vmax - vmin, 1e-6) > 0.5
                                       else "black")

            ax.set_xticks(range(len(T_src_order)))
            ax.set_xticklabels([f"{T:g}" for T in T_src_order])
            ax.set_yticks(range(len(pair_order)))
            ax.set_yticklabels(pair_order, fontsize=9)
            ax.set_xlabel("T_source_in [°C]")
            if j == 0:
                ax.set_ylabel(f"LS = {ls*100:.0f}%", fontsize=11, fontweight="bold")
            ax.set_title(f"T_steam = {T_steam:.0f} °C", fontsize=10)

    if last_im is not None:
        fig.tight_layout(rect=[0, 0, 0.93, 0.94])
        cbar_ax = fig.add_axes([0.95, 0.10, 0.012, 0.80])
        fig.colorbar(last_im, cax=cbar_ax, label=value_col)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4/5: constraint diagnostic plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_constraint_lines(df: pd.DataFrame, value_col: str, fname: str,
                          ylabel: str, title: str,
                          horizontal_lines: list[tuple[float, str]] | None = None,
                          y_log: bool = False):
    """Line plot of `value_col` vs T_steam (one panel per LS), faceted by
    fluid pair, with horizontal envelope lines.
    """
    ls_vals = sorted(df["ls"].unique())
    T_steam_vals = sorted(df["T_steam"].unique())
    pairs = sorted(df["pair"].unique())

    # default T_src = the median (sorted middle) value
    T_src_default = sorted(df["T_src"].unique())[len(df["T_src"].unique()) // 2]

    n_cols = len(ls_vals)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5),
                              squeeze=False, sharey=True)

    pair_colors = plt.cm.tab10(np.linspace(0, 1, len(pairs)))

    for i, ls in enumerate(ls_vals):
        ax = axes[0][i]
        for k, pair in enumerate(pairs):
            sub = df[(df["pair"] == pair) & (df["ls"] == ls)
                     & (df["T_src"] == T_src_default)
                     & (df["status"] == "ok")].sort_values("T_steam")
            if sub.empty:
                continue
            ax.plot(sub["T_steam"], sub[value_col],
                    marker="o", markersize=5,
                    color=pair_colors[k], label=pair)

        if horizontal_lines:
            for y, label in horizontal_lines:
                ax.axhline(y, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
                ax.text(T_steam_vals[0], y, f" {label}",
                        fontsize=8, va="bottom", ha="left", alpha=0.7)

        if y_log:
            ax.set_yscale("log")
        ax.set_xlabel("T_steam [°C]")
        if i == 0:
            ax.set_ylabel(ylabel)
        ax.set_title(f"LS = {ls*100:.0f}%")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f"{title}  (T_source_in = {T_src_default} °C)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Could not find {CSV_PATH}. Run screen_cascade.py first.")
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()

    print("Plotting feasibility grid ...")
    plot_feasibility_grid(df)

    print("Plotting COP heatmap ...")
    plot_value_heatmap(df, "COP",
                       title="COP — feasible cascade designs only "
                             "(V̇ binding @ V_max)",
                       fname="cop_heatmap.png", fmt="{:.2f}")

    print("Plotting steam mass-flow heatmap ...")
    plot_value_heatmap(df, "m_steam [kg/h]",
                       title="Steam mass flow per machine [kg/h] "
                             "— feasible cascades only",
                       fname="steam_kgph_heatmap.png",
                       fmt="{:.0f}", cmap="plasma")

    print("Plotting p_high constraints (cycle 1) ...")
    plot_constraint_lines(
        df, "p_high_c1 [bar]", "p_high_c1_constraints.png",
        ylabel="p_high cycle 1 [bar]  (log scale)",
        title="Cycle-1 high-side pressure vs T_steam",
        horizontal_lines=[(28, "28 bar (HC / R717-LP)"),
                          (50, "50 bar (R717-HP)"),
                          (140, "140 bar (R744)")],
        y_log=True,
    )

    print("Plotting p_high constraints (cycle 2) ...")
    plot_constraint_lines(
        df, "p_high_c2 [bar]", "p_high_c2_constraints.png",
        ylabel="p_high cycle 2 [bar]  (log scale)",
        title="Cycle-2 high-side pressure vs T_steam",
        horizontal_lines=[(28, "28 bar (HC / R717-LP)"),
                          (50, "50 bar (R717-HP)"),
                          (140, "140 bar (R744)")],
        y_log=True,
    )

    print("Plotting T_disch constraints (cycle 1) ...")
    plot_constraint_lines(
        df, "T_disch_c1 [°C]", "T_disch_c1_constraints.png",
        ylabel="T_disch cycle 1 [°C]",
        title="Cycle-1 compressor discharge T vs T_steam",
        horizontal_lines=[(180, "180 °C oil limit")],
    )

    print("Plotting T_disch constraints (cycle 2) ...")
    plot_constraint_lines(
        df, "T_disch_c2 [°C]", "T_disch_c2_constraints.png",
        ylabel="T_disch cycle 2 [°C]",
        title="Cycle-2 compressor discharge T vs T_steam",
        horizontal_lines=[(180, "180 °C oil limit")],
    )

    print(f"\nFigures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
