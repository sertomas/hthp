"""
plot_case_steam.py — Visualise the cascade screening for one T_steam.

Reads either the raw screen CSV
``results/case_steam_<int(T_steam)>/case_steam_<int(T_steam)>.csv`` (Ommen-
strict mode) or the modern-envelope reclassified CSV
``case_steam_<int(T_steam)>_enriched.csv`` (modern mode), and writes five
PNGs to the same folder. The mode is selected via the module-level
``ENVELOPE_MODE`` ('ommen' or 'modern'); main.py runs both in turn so each
case-study folder ends up with both the bare and ``_modern``-suffixed sets.

Status colour legend (4-state, defined in ``plot_common.STATUS_COLORS``):

    GREEN  (OK)      — within Ommen 2015 envelope on p, T, V̇.
    AMBER  (V_ONLY)  — V̇ outside Ommen V_min/V_max but p, T compliant;
                       operable per IEA HPT Annex 58 supplier evidence.
    RED    (HARD)    — pressure or T_disch violation; genuinely problematic.
    GREY   (NOSOLVE) — thermodynamic infeasibility or TESPy non-convergence.

Plots:
    1. feasibility_grid.png — 4-state colour, COP / failure-flag / no-solve
                              annotations. Panels per lift share.
    2. cop_grid.png         — COP heatmap, V_ONLY hatched.
    3. Tdisch_grid.png      — max(T_disch_c1, T_disch_c2) per cell, 180 °C
                              reference contour.
    4. p_high_grid.png      — max(p_high_c1, p_high_c2) per cell, log scale,
                              28 / 50 bar reference contours.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, LogNorm

from config import T_STEAM_CASE_DEFAULT, m_steam_label
from plot_common import (
    STATUS_COLORS,
    STATUS_ORDER,
    classify_status,
    nosolve_short,
    reason_short,
    slice_grid,
)


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Two envelope modes:
#   default  → Ommen 2015 strict; reads case_steam_<int(T)>.csv,
#              status col = status_4state, output filename suffix = ""
#   --modern → Project 68 modern; reads case_steam_<int(T)>_enriched.csv,
#              status col = status_modern, output filename suffix = "_modern"
ENVELOPE_MODE = "ommen"  # overridden by CLI

# Module-level path globals — rewritten by ``_set_paths_for_T_steam`` at
# the start of ``main`` for each case-study T_steam; the placeholders below
# only matter if the helpers below are called before that override.
CSV_PATH = ""
CSV_PATH_MODERN = ""
OUT_DIR = ""
_T_STEAM_CURRENT = 0.0   # set at the same time, used by suptitle strings


def _set_paths_for_T_steam(T_steam):
    """Rewrite the module-level path globals for a different T_steam."""
    global CSV_PATH, CSV_PATH_MODERN, OUT_DIR, _T_STEAM_CURRENT
    from config import case_results_dir
    out_dir = case_results_dir(T_steam)
    tag = f"case_steam_{int(T_steam)}"
    CSV_PATH = os.path.join(out_dir, f"{tag}.csv")
    CSV_PATH_MODERN = os.path.join(out_dir, f"{tag}_enriched.csv")
    OUT_DIR = out_dir
    _T_STEAM_CURRENT = T_steam


def _csv_path() -> str:
    return CSV_PATH_MODERN if ENVELOPE_MODE == "modern" else CSV_PATH


def _status_col() -> str:
    return "status_modern" if ENVELOPE_MODE == "modern" else "status_4state"


def _reason_col() -> str:
    return "reason_modern" if ENVELOPE_MODE == "modern" else "reason"


def _suffix() -> str:
    return "_modern" if ENVELOPE_MODE == "modern" else ""


def _envelope_label_text() -> str:
    return ("Project 68 modern envelope (T_disch ≤ 200 °C, R717-HP ≤ 76 bar, "
            "V̇ ≤ 1500 m³/h)"
            if ENVELOPE_MODE == "modern"
            else "Ommen 2015 strict envelope (T_disch ≤ 180 °C, "
                 "R717-HP ≤ 50 bar, V̇ ≤ 280 m³/h Type-2)")


# ── Load & validate ─────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    df = pd.read_csv(_csv_path())
    for col in ["COP", "eta_Lorenz", "Q_H_kW", "m_steam [kg/h]",
                "p_high_c1 [bar]", "p_high_c2 [bar]",
                "T_disch_c1 [°C]", "T_disch_c2 [°C]",
                "V_dot_c1_target [m3/h]", "V_dot_c2_target [m3/h]"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["pair"] = df["f1"] + "/" + df["f2"]

    # Re-derive status_4state to be safe against CSV round-trip surprises
    if _status_col() not in df.columns:
        df[_status_col()] = df.apply(classify_status, axis=1)

    # Q_H consistency check (skip NOSOLVE rows): every solved design must
    # produce the same native Q_H (config.M_STEAM × Δh_vap, depending only
    # on T_steam — so the median Q_H_kW across solved rows is the reference).
    nn = df[df[_status_col()] != "NOSOLVE"]
    if not nn.empty:
        Q_H_native = nn["Q_H_kW"].median()
        max_dev = (nn["Q_H_kW"] - Q_H_native).abs().max()
        if max_dev > 0.1:
            print(f"  ! Q_H consistency: max deviation {max_dev:.3f} kW from "
                  f"native {Q_H_native:.0f} kW > 0.1 kW")

    df["pair_max_p"] = df[["p_high_c1 [bar]", "p_high_c2 [bar]"]].max(axis=1)
    df["pair_max_T"] = df[["T_disch_c1 [°C]", "T_disch_c2 [°C]"]].max(axis=1)
    return df


# ── Helpers for panel layout ────────────────────────────────────────────────

def _panel_axes(ls_vals, n_cols=None):
    """One row × len(ls_vals) cols of panels, sized for readability.

    Figure is sized to leave room above for a multi-line suptitle (3 lines:
    metric / mode / envelope spec) and below for the legend without
    collisions with the per-panel ``LS = X%`` titles.
    """
    n = len(ls_vals)
    cols = n if n_cols is None else n_cols
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(4.0 * cols, 4.6 * rows + 1.4),
        squeeze=False,
    )
    return fig, axes, rows, cols


def _set_panel_ticks(ax, pair_order, T_src_order, *, show_y=True):
    ax.set_xticks(range(len(T_src_order)))
    ax.set_xticklabels([f"{T:g}" for T in T_src_order])
    ax.set_xlabel("T_source_in [°C]")
    if show_y:
        ax.set_yticks(range(len(pair_order)))
        ax.set_yticklabels(pair_order, fontsize=9)
    else:
        ax.set_yticks(range(len(pair_order)))
        ax.set_yticklabels([])


# ── Plot 1: feasibility grid (4-state) ──────────────────────────────────────

def plot_feasibility_grid(df: pd.DataFrame):
    ls_vals = sorted(df["ls"].unique())
    pair_order = sorted(df["pair"].unique())
    T_src_order = sorted(df["T_src"].unique())

    state_to_idx = {s: i for i, s in enumerate(STATUS_ORDER)}  # OK=0, V=1, H=2, N=3
    cmap = ListedColormap([STATUS_COLORS[s] for s in STATUS_ORDER])

    fig, axes, _, n_cols = _panel_axes(ls_vals)

    for j, ls in enumerate(ls_vals):
        ax = axes[0][j]
        sub = df[df["ls"] == ls]

        Z = np.full((len(pair_order), len(T_src_order)), state_to_idx["NOSOLVE"], dtype=int)
        annot = np.full(Z.shape, "", dtype=object)
        annot_color = np.full(Z.shape, "black", dtype=object)

        for ii, pair in enumerate(pair_order):
            for jj, T_src in enumerate(T_src_order):
                row = sub[(sub["pair"] == pair) & (sub["T_src"] == T_src)]
                if row.empty:
                    Z[ii, jj] = state_to_idx["NOSOLVE"]
                    annot[ii, jj] = "—"
                    continue
                r = row.iloc[0]
                state = r[_status_col()]
                Z[ii, jj] = state_to_idx[state]
                if state == "OK":
                    annot[ii, jj] = f"{r['COP']:.2f}"
                    annot_color[ii, jj] = "white"
                elif state == "V_ONLY":
                    annot[ii, jj] = f"{r['COP']:.2f}\n{reason_short(r[_reason_col()])}"
                    annot_color[ii, jj] = "black"
                elif state == "HARD":
                    annot[ii, jj] = reason_short(r[_reason_col()])
                    annot_color[ii, jj] = "black"
                else:  # NOSOLVE
                    annot[ii, jj] = nosolve_short(r.get("no_solve_class", ""))
                    annot_color[ii, jj] = "#444444"

        ax.imshow(Z, cmap=cmap, vmin=0, vmax=len(STATUS_ORDER) - 1, aspect="auto")
        for ii in range(len(pair_order)):
            for jj in range(len(T_src_order)):
                ax.text(jj, ii, annot[ii, jj],
                        ha="center", va="center",
                        fontsize=7.5, color=annot_color[ii, jj])

        _set_panel_ticks(ax, pair_order, T_src_order, show_y=(j == 0))
        ax.set_title(f"LS = {ls*100:.0f}%", fontsize=11)
        if j == 0:
            ax.set_ylabel("Cycle-1 / Cycle-2 fluid pair", fontsize=10)

    # Legend
    ok_label = ("OK — within Project 68 modern envelope"
                if ENVELOPE_MODE == "modern"
                else "OK — within Ommen 2015 envelope")
    legend_handles = [
        mpatches.Patch(color=STATUS_COLORS["OK"], label=ok_label),
        mpatches.Patch(color=STATUS_COLORS["V_ONLY"],
                       label="V_ONLY — V̇ outside envelope, p/T OK"),
        mpatches.Patch(color=STATUS_COLORS["HARD"],
                       label="HARD — p or T_disch violation"),
        mpatches.Patch(color=STATUS_COLORS["NOSOLVE"],
                       label="NOSOLVE — thermodynamic infeasibility / TESPy"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Cascade feasibility — steam at {_T_STEAM_CURRENT:.0f} °C, native scale ({m_steam_label()})\n"
        f"{_envelope_label_text()}\n"
        f"Cells annotated with COP (OK / V_ONLY) or failed-flag list "
        f"(HARD: p₁/p₂/T₁/T₂; V_ONLY: V₁/V₂)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.85])
    fig.savefig(os.path.join(OUT_DIR, f"feasibility_grid{_suffix()}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 2: COP heatmap with V_ONLY hatch ────────────────────────────────

def plot_value_heatmap(df, value_col, fname, title, *,
                       fmt="{:.2f}", cmap="viridis", vmin=None, vmax=None):
    ls_vals = sorted(df["ls"].unique())
    pair_order = sorted(df["pair"].unique())
    T_src_order = sorted(df["T_src"].unique())

    accepted = df[df[_status_col()].isin(["OK", "V_ONLY"])]
    if accepted.empty or accepted[value_col].dropna().empty:
        print(f"  ! no valid data for {value_col}; skipping {fname}")
        return
    if vmin is None:
        vmin = float(accepted[value_col].min())
    if vmax is None:
        vmax = float(accepted[value_col].max())

    fig, axes, _, _ = _panel_axes(ls_vals)
    last_im = None

    for j, ls in enumerate(ls_vals):
        ax = axes[0][j]
        sub = df[df["ls"] == ls]

        grid = np.full((len(pair_order), len(T_src_order)), np.nan)
        is_v_only = np.zeros_like(grid, dtype=bool)

        for ii, pair in enumerate(pair_order):
            for jj, T_src in enumerate(T_src_order):
                row = sub[(sub["pair"] == pair) & (sub["T_src"] == T_src)]
                if row.empty:
                    continue
                r = row.iloc[0]
                state = r[_status_col()]
                if state in ("OK", "V_ONLY"):
                    val = r[value_col]
                    if pd.notna(val):
                        grid[ii, jj] = val
                if state == "V_ONLY":
                    is_v_only[ii, jj] = True

        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="#cccccc")
        masked = np.ma.masked_invalid(grid)
        last_im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax,
                            aspect="auto")

        # V_ONLY hatch overlay
        for ii in range(len(pair_order)):
            for jj in range(len(T_src_order)):
                if is_v_only[ii, jj]:
                    ax.add_patch(plt.Rectangle(
                        (jj - 0.5, ii - 0.5), 1, 1,
                        fill=False, hatch="///", edgecolor="white",
                        linewidth=0.0, alpha=0.85,
                    ))
                if not np.isnan(grid[ii, jj]):
                    norm_val = (grid[ii, jj] - vmin) / max(vmax - vmin, 1e-6)
                    text_color = "white" if norm_val > 0.5 else "black"
                    ax.text(jj, ii, fmt.format(grid[ii, jj]),
                            ha="center", va="center", fontsize=8,
                            color=text_color)

        _set_panel_ticks(ax, pair_order, T_src_order, show_y=(j == 0))
        ax.set_title(f"LS = {ls*100:.0f}%", fontsize=11)
        if j == 0:
            ax.set_ylabel("Cycle-1 / Cycle-2 fluid pair", fontsize=10)

    if last_im is not None:
        fig.tight_layout(rect=[0, 0.07, 0.93, 0.85])
        cbar_ax = fig.add_axes([0.945, 0.13, 0.012, 0.65])
        fig.colorbar(last_im, cax=cbar_ax, label=value_col)

    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black",
                       hatch="///", label="V_ONLY — V̇ outside Ommen (operable per Annex 58)"),
        mpatches.Patch(color="#cccccc", label="HARD / NOSOLVE — masked"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{title}\n{_envelope_label_text()}",
                 fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 4: T_disch grid with 180 °C reference ──────────────────────────────

def plot_Tdisch_grid(df: pd.DataFrame):
    ls_vals = sorted(df["ls"].unique())
    pair_order = sorted(df["pair"].unique())
    T_src_order = sorted(df["T_src"].unique())

    accepted = df[df[_status_col()].isin(["OK", "V_ONLY", "HARD"])]
    if accepted.empty:
        print("  ! no T_disch data; skipping Tdisch_grid")
        return
    vmin = float(accepted["pair_max_T"].min())
    vmax = max(200.0, float(accepted["pair_max_T"].max()))

    fig, axes, _, _ = _panel_axes(ls_vals)
    last_im = None

    for j, ls in enumerate(ls_vals):
        ax = axes[0][j]
        sub = df[df["ls"] == ls]
        grid = np.full((len(pair_order), len(T_src_order)), np.nan)
        is_hot = np.zeros_like(grid, dtype=bool)

        for ii, pair in enumerate(pair_order):
            for jj, T_src in enumerate(T_src_order):
                row = sub[(sub["pair"] == pair) & (sub["T_src"] == T_src)]
                if row.empty:
                    continue
                r = row.iloc[0]
                if r[_status_col()] in ("OK", "V_ONLY", "HARD"):
                    val = r["pair_max_T"]
                    if pd.notna(val):
                        grid[ii, jj] = val
                        if val > 180.0:
                            is_hot[ii, jj] = True

        cmap_obj = plt.get_cmap("RdYlGn_r").copy()
        cmap_obj.set_bad(color="#cccccc")
        masked = np.ma.masked_invalid(grid)
        last_im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax,
                            aspect="auto")

        for ii in range(len(pair_order)):
            for jj in range(len(T_src_order)):
                if not np.isnan(grid[ii, jj]):
                    txt_color = "black" if grid[ii, jj] < 160 else "white"
                    ax.text(jj, ii, f"{grid[ii, jj]:.0f}",
                            ha="center", va="center",
                            fontsize=8, color=txt_color)
                if is_hot[ii, jj]:
                    ax.add_patch(plt.Rectangle(
                        (jj - 0.5, ii - 0.5), 1, 1,
                        fill=False, edgecolor="black", linewidth=1.5,
                    ))

        _set_panel_ticks(ax, pair_order, T_src_order, show_y=(j == 0))
        ax.set_title(f"LS = {ls*100:.0f}%", fontsize=11)
        if j == 0:
            ax.set_ylabel("Cycle-1 / Cycle-2 fluid pair", fontsize=10)

    if last_im is not None:
        fig.tight_layout(rect=[0, 0.07, 0.93, 0.85])
        cbar_ax = fig.add_axes([0.945, 0.13, 0.012, 0.65])
        cbar = fig.colorbar(last_im, cax=cbar_ax,
                            label="max(T_disch_c1, T_disch_c2) [°C]")
        cbar.ax.axhline(180, color="black", linewidth=1.5)

    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black",
                       label="black border = T_disch > 180 °C (Ommen oil-degradation limit)"),
        mpatches.Patch(color="#cccccc", label="NOSOLVE — masked"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Compressor discharge temperature — max(T_disch_c1, T_disch_c2)\n"
        f"Steam at {_T_STEAM_CURRENT:.0f} °C, native scale ({m_steam_label()})",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(os.path.join(OUT_DIR, f"Tdisch_grid{_suffix()}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 5: p_high grid with 28 / 50 bar reference contours ─────────────────

def plot_p_high_grid(df: pd.DataFrame):
    ls_vals = sorted(df["ls"].unique())
    pair_order = sorted(df["pair"].unique())
    T_src_order = sorted(df["T_src"].unique())

    accepted = df[df[_status_col()].isin(["OK", "V_ONLY", "HARD"])]
    if accepted.empty:
        print("  ! no p_high data; skipping p_high_grid")
        return
    vmin = max(1.0, float(accepted["pair_max_p"].min()))
    vmax = float(accepted["pair_max_p"].max())

    fig, axes, _, _ = _panel_axes(ls_vals)
    last_im = None

    for j, ls in enumerate(ls_vals):
        ax = axes[0][j]
        sub = df[df["ls"] == ls]
        grid = np.full((len(pair_order), len(T_src_order)), np.nan)

        for ii, pair in enumerate(pair_order):
            for jj, T_src in enumerate(T_src_order):
                row = sub[(sub["pair"] == pair) & (sub["T_src"] == T_src)]
                if row.empty:
                    continue
                r = row.iloc[0]
                if r[_status_col()] in ("OK", "V_ONLY", "HARD"):
                    val = r["pair_max_p"]
                    if pd.notna(val):
                        grid[ii, jj] = val

        cmap_obj = plt.get_cmap("plasma").copy()
        cmap_obj.set_bad(color="#cccccc")
        masked = np.ma.masked_invalid(grid)
        norm = LogNorm(vmin=max(vmin, 1.0), vmax=max(vmax, 10.0))
        last_im = ax.imshow(masked, cmap=cmap_obj, norm=norm, aspect="auto")

        for ii in range(len(pair_order)):
            for jj in range(len(T_src_order)):
                if not np.isnan(grid[ii, jj]):
                    norm_val = (np.log10(grid[ii, jj]) - np.log10(max(vmin, 1.0))) \
                               / max(np.log10(max(vmax, 10.0)) - np.log10(max(vmin, 1.0)), 1e-6)
                    txt_color = "black" if norm_val < 0.55 else "white"
                    ax.text(jj, ii, f"{grid[ii, jj]:.0f}",
                            ha="center", va="center",
                            fontsize=8, color=txt_color)

        _set_panel_ticks(ax, pair_order, T_src_order, show_y=(j == 0))
        ax.set_title(f"LS = {ls*100:.0f}%", fontsize=11)
        if j == 0:
            ax.set_ylabel("Cycle-1 / Cycle-2 fluid pair", fontsize=10)

    if last_im is not None:
        fig.tight_layout(rect=[0, 0.07, 0.93, 0.85])
        cbar_ax = fig.add_axes([0.945, 0.13, 0.012, 0.65])
        cbar = fig.colorbar(last_im, cax=cbar_ax,
                            label="max(p_high_c1, p_high_c2) [bar]  (log)")
        for p_ref in (28.0, 50.0):
            cbar.ax.axhline(p_ref, color="black", linewidth=1.0, linestyle="--")
            cbar.ax.text(1.05, p_ref, f" {p_ref:.0f} bar",
                         transform=cbar.ax.get_yaxis_transform(),
                         fontsize=8, va="center")

    legend_handles = [
        mpatches.Patch(facecolor="none", edgecolor="black",
                       label="Reference contours on colorbar: 28 bar (Ommen Type-2 / R717-LP), 50 bar (R717-HP)"),
        mpatches.Patch(color="#cccccc", label="NOSOLVE — masked"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=1, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"High-side pressure — max(p_high_c1, p_high_c2)\n"
        f"Steam at {_T_STEAM_CURRENT:.0f} °C, native scale ({m_steam_label()})",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(os.path.join(OUT_DIR, f"p_high_grid{_suffix()}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Driver ──────────────────────────────────────────────────────────────────

def main(T_steam=None):
    """Render feasibility plots for a given T_steam case-study folder."""
    if T_steam is None:
        T_steam = T_STEAM_CASE_DEFAULT
    _set_paths_for_T_steam(T_steam)
    csv = _csv_path()
    if not os.path.exists(csv):
        print(f"Could not find {csv}. "
              f"Run `python main.py --t-steam {int(T_steam)}` first.")
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Envelope mode: {ENVELOPE_MODE.upper()}  ({_envelope_label_text()})")
    print(f"Reading {csv}")
    df = load()

    print("Plotting feasibility grid (4-state) ...")
    plot_feasibility_grid(df)

    print("Plotting COP heatmap ...")
    plot_value_heatmap(
        df, "COP", f"cop_grid{_suffix()}.png",
        title=f"COP — steam at {_T_STEAM_CURRENT:.0f} °C, native scale ({m_steam_label()})\n"
              "OK + V_ONLY shown; V_ONLY hatched",
        fmt="{:.2f}", cmap="viridis",
    )

    print("Plotting T_disch grid ...")
    plot_Tdisch_grid(df)

    print("Plotting p_high grid ...")
    plot_p_high_grid(df)

    print(f"\nFigures written to {OUT_DIR}/  (suffix '{_suffix()}')")


if __name__ == "__main__":
    if "--modern" in sys.argv:
        # Module-level ENVELOPE_MODE is read by the helper functions
        # (_csv_path, _status_col, etc.); rebind it here.
        globals()["ENVELOPE_MODE"] = "modern"
    main()
