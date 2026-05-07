"""
plot_case_steam_110_economics.py — Visualise the exergoeconomic results
for the steam-at-110 °C case study (modern envelope, LS ∈ {0.30, 0.40, 0.50}).

Reads:
    results/case_steam_110/economics/economics_base.csv
    results/case_steam_110/economics/economics_pec_breakdown.csv
    results/case_steam_110/economics/economics_sensitivity_e1.csv

Writes (to the same folder):
    cP_grid.png            — c_P heatmap  (panels per LS, fluid pair × T_src)
    PEC_per_kW_grid.png    — PEC at 1 MWth, EUR/kW (TCI), with Annex 58 band
    PEC_components_only.png — Components-only PEC = TCI / F_INSTALL, with band
    PEC_breakdown.png      — stacked-bar component breakdown for top-12 by c_P
    cP_vs_e1c.png          — c_P sensitivity to electricity price
    cP_vs_gas.png          — c_P at base electricity price vs gas reference
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
    PRICE_SENS_FRAC_RANGE, T_STEAM_CASE_110,
)
from economics import F_INSTALL


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


ECON_DIR = os.path.join("results", "case_steam_110", "economics")
BASE_CSV = os.path.join(ECON_DIR, "economics_base.csv")
BREAKDOWN_CSV = os.path.join(ECON_DIR, "economics_pec_breakdown.csv")
SENS_CSV = os.path.join(ECON_DIR, "economics_sensitivity_e1.csv")
SENS_FLH_CSV = os.path.join(ECON_DIR, "economics_sensitivity_FLH.csv")
DESIGNS_DIR = os.path.join("results", "case_steam_110", "designs")
_T_STEAM_CURRENT = 110.0


def _set_paths_for_T_steam(T_steam):
    """Rewrite the module-level path globals for a given T_steam.

    Also invalidates the gas-reference c_P cache (it depends on T_steam
    because steam exergy E_P is T-dependent).
    """
    global ECON_DIR, BASE_CSV, BREAKDOWN_CSV, SENS_CSV, SENS_FLH_CSV
    global DESIGNS_DIR, _T_STEAM_CURRENT, _gas_reference_cache
    from config import case_results_dir
    case_dir = case_results_dir(T_steam)
    ECON_DIR = os.path.join(case_dir, "economics")
    BASE_CSV = os.path.join(ECON_DIR, "economics_base.csv")
    BREAKDOWN_CSV = os.path.join(ECON_DIR, "economics_pec_breakdown.csv")
    SENS_CSV = os.path.join(ECON_DIR, "economics_sensitivity_e1.csv")
    SENS_FLH_CSV = os.path.join(ECON_DIR, "economics_sensitivity_FLH.csv")
    DESIGNS_DIR = os.path.join(case_dir, "designs")
    _T_STEAM_CURRENT = T_steam
    # Force the gas-reference cache to recompute at the new T_steam
    _gas_reference_cache = {}

# Component groups + colors mirror plot_comparison.py from hthp_optimization
# (COMP+MOT blue, COND red, EVAP green, CASC orange, ECO yellow → reused for
# pumps, VAL pink). For the cascade HTHP, IHX *is* the cascade HX so it gets
# the orange CASC color; SRC_HX is the evaporator (green); SNK_HX is the
# steam-generating condenser (red).
COMP_GROUPS = {
    "COMP+MOT":  ["COMP1", "COMP2", "MOT1", "MOT2"],
    "SRC_HX":    ["SRC_HX"],
    "IHX":       ["IHX"],
    "SNK_HX":    ["SNK_HX"],
    "VAL":       ["VAL1", "VAL2"],
    "PUMP":      ["SRC_PUMP", "SNK_PUMP", "MOT3", "MOT4"],
}
GROUP_COLORS = {
    "COMP+MOT": "#4C72B0",
    "SRC_HX":   "#55A868",
    "IHX":      "#D4A039",
    "SNK_HX":   "#C44E52",
    "VAL":      "#E8A0BF",
    "PUMP":     "#CCB974",
}

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
    # for fair comparison to Annex 58 supplier €/kW which excludes integration.
    base["PEC_1MW_components_only [EUR/kW]"] = (
        base["PEC_1MW [EUR/kW]"] / F_INSTALL
    ).round(1)
    base["PEC_native_components_only [EUR/kW]"] = (
        base["PEC_native [EUR/kW]"] / F_INSTALL
    ).round(1)
    return base, bk, sens, sens_flh


# ── Plot 1: c_P heatmap (LS panels × fluid pair × T_src) ────────────────────

def plot_cP_grid(base):
    ls_vals = sorted(base["ls"].unique())
    pair_order = sorted(base["pair"].unique())
    T_src_order = sorted(base["T_src"].unique())

    vmin = float(base["c_P [EUR/GJ]"].min())
    vmax = float(base["c_P [EUR/GJ]"].max())

    fig, axes = plt.subplots(1, len(ls_vals),
                              figsize=(4.6 * len(ls_vals), 4.5),
                              squeeze=False)
    last_im = None
    for j, ls in enumerate(ls_vals):
        ax = axes[0][j]
        sub = base[base["ls"] == ls]
        grid = np.full((len(pair_order), len(T_src_order)), np.nan)
        for ii, pair in enumerate(pair_order):
            for jj, T_src in enumerate(T_src_order):
                row = sub[(sub["pair"] == pair) & (sub["T_src"] == T_src)]
                if not row.empty:
                    grid[ii, jj] = row.iloc[0]["c_P [EUR/GJ]"]
        cmap = plt.get_cmap("plasma_r").copy()
        cmap.set_bad(color="#cccccc")
        masked = np.ma.masked_invalid(grid)
        last_im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for ii in range(len(pair_order)):
            for jj in range(len(T_src_order)):
                if not np.isnan(grid[ii, jj]):
                    norm_val = (grid[ii, jj] - vmin) / max(vmax - vmin, 1e-6)
                    txt_color = "white" if norm_val > 0.5 else "black"
                    ax.text(jj, ii, f"{grid[ii, jj]:.0f}",
                            ha="center", va="center", fontsize=8,
                            color=txt_color)
        ax.set_xticks(range(len(T_src_order)))
        ax.set_xticklabels([f"{T:g}" for T in T_src_order])
        ax.set_yticks(range(len(pair_order)))
        ax.set_yticklabels(pair_order if j == 0 else [], fontsize=9)
        ax.set_xlabel("T_source_in [°C]")
        ax.set_title(f"LS = {ls*100:.0f}%", fontsize=11)
        if j == 0:
            ax.set_ylabel("Fluid pair", fontsize=10)

    if last_im is not None:
        fig.tight_layout(rect=[0, 0.03, 0.93, 0.93])
        cbar_ax = fig.add_axes([0.945, 0.10, 0.012, 0.80])
        fig.colorbar(last_im, cax=cbar_ax,
                     label=r"$c_P$  [EUR/GJ$_{ex}$]")

    fig.suptitle(
        f"Specific product cost c_P  (Q_H = 1 MWth, T_steam = 110 °C)\n"
        f"{_op_string()}  |  modern-envelope OK designs — lower c_P = cheaper steam exergy",
        fontsize=11, fontweight="bold",
    )
    fig.savefig(os.path.join(ECON_DIR, "cP_grid.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 2 & 3: PEC per kW vs Annex 58 band (TCI and components-only) ──────

def _plot_pec_per_kW(base, value_col, fname, title):
    ls_vals = sorted(base["ls"].unique())
    pair_order = sorted(base["pair"].unique())
    T_src_order = sorted(base["T_src"].unique())

    vmin = float(base[value_col].min())
    vmax = float(base[value_col].max())

    fig, axes = plt.subplots(1, len(ls_vals),
                              figsize=(4.6 * len(ls_vals), 4.5),
                              squeeze=False)
    last_im = None
    for j, ls in enumerate(ls_vals):
        ax = axes[0][j]
        sub = base[base["ls"] == ls]
        grid = np.full((len(pair_order), len(T_src_order)), np.nan)
        for ii, pair in enumerate(pair_order):
            for jj, T_src in enumerate(T_src_order):
                row = sub[(sub["pair"] == pair) & (sub["T_src"] == T_src)]
                if not row.empty:
                    grid[ii, jj] = row.iloc[0][value_col]
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad(color="#cccccc")
        last_im = ax.imshow(np.ma.masked_invalid(grid),
                            cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for ii in range(len(pair_order)):
            for jj in range(len(T_src_order)):
                if not np.isnan(grid[ii, jj]):
                    norm_val = (grid[ii, jj] - vmin) / max(vmax - vmin, 1e-6)
                    txt_color = "white" if norm_val > 0.5 else "black"
                    ax.text(jj, ii, f"{grid[ii, jj]:.0f}",
                            ha="center", va="center", fontsize=8,
                            color=txt_color)
                    # mark cells inside Annex 58 band with a green frame
                    if ANNEX58_BAND_LOW <= grid[ii, jj] <= ANNEX58_BAND_HIGH:
                        ax.add_patch(plt.Rectangle(
                            (jj - 0.5, ii - 0.5), 1, 1,
                            fill=False, edgecolor="#2e7d32", linewidth=2.0))
        ax.set_xticks(range(len(T_src_order)))
        ax.set_xticklabels([f"{T:g}" for T in T_src_order])
        ax.set_yticks(range(len(pair_order)))
        ax.set_yticklabels(pair_order if j == 0 else [], fontsize=9)
        ax.set_xlabel("T_source_in [°C]")
        ax.set_title(f"LS = {ls*100:.0f}%", fontsize=11)
        if j == 0:
            ax.set_ylabel("Fluid pair", fontsize=10)

    if last_im is not None:
        fig.tight_layout(rect=[0, 0.03, 0.93, 0.92])
        cbar_ax = fig.add_axes([0.945, 0.10, 0.012, 0.80])
        cbar = fig.colorbar(last_im, cax=cbar_ax, label="EUR/kW")
        # Annex 58 band marker on the colorbar. Only draw the marker if the
        # band edge actually falls within the cbar data range — otherwise the
        # axhline/text get rendered far outside the visible cbar, and with
        # bbox_inches="tight" they blow up the figure bounding box (this was
        # the layout glitch where PEC_components_only had 70% empty space
        # above the heatmap because vmax≈308 ≪ 700).
        for band_edge in (ANNEX58_BAND_LOW, ANNEX58_BAND_HIGH):
            if vmin <= band_edge <= vmax:
                cbar.ax.axhline(band_edge, color="#2e7d32", linewidth=1.5)
                cbar.ax.text(1.05, band_edge, f" {band_edge:.0f}",
                             transform=cbar.ax.get_yaxis_transform(),
                             fontsize=8, va="center", color="#2e7d32")

    legend = [mpatches.Patch(facecolor="none", edgecolor="#2e7d32",
                             linewidth=2,
                             label=f"Inside Annex 58 / Project 68 band "
                                   f"({ANNEX58_BAND_LOW:.0f}–{ANNEX58_BAND_HIGH:.0f} EUR/kW, 0.5–3 MWth, 110–150 °C)")]
    fig.legend(handles=legend, loc="lower center",
               ncol=1, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(ECON_DIR, fname),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pec_per_kW_TCI(base):
    _plot_pec_per_kW(
        base, "PEC_1MW [EUR/kW]", "PEC_per_kW_grid.png",
        title=(f"Total Capital Investment (TCI = F_INSTALL × PEC) per kW "
               f"at Q_H = 1 MWth\n"
               f"F_INSTALL = {F_INSTALL:.2f}; capital only (independent of "
               f"FLH / e1 / gas)"),
    )


def plot_pec_per_kW_components(base):
    _plot_pec_per_kW(
        base, "PEC_1MW_components_only [EUR/kW]", "PEC_components_only.png",
        title=(f"Purchase Equipment Cost (PEC) per kW at Q_H = 1 MWth\n"
               f"Components only (TCI / F_INSTALL = {F_INSTALL:.2f}); "
               f"capital only (independent of FLH / e1 / gas) — comparable "
               f"to Annex 58 / Project 68 band"),
    )


# ── Plot 4: PEC component breakdown for the cheapest 12 designs by c_P ──────

def plot_pec_breakdown(base, bk):
    top = base.nsmallest(12, "c_P [EUR/GJ]").copy()
    top["label"] = (top["pair"] + "\nLS=" + (top["ls"]*100).astype(int).astype(str)
                    + "%, T_src=" + top["T_src"].astype(int).astype(str) + "°C\n"
                    + "c_P=" + top["c_P [EUR/GJ]"].round(0).astype(int).astype(str)
                    + r" EUR/GJ$_{ex}$")

    components = ["COMP1", "COMP2", "MOT1", "MOT2",
                  "SRC_HX", "IHX", "SNK_HX", "SRC_PUMP", "SNK_PUMP"]
    comp_colors = plt.cm.tab10(np.linspace(0, 1, len(components)))

    fig, ax = plt.subplots(figsize=(13, 7))
    bottom = np.zeros(len(top))
    for k, comp in enumerate(components):
        vals = []
        for _, r in top.iterrows():
            sub = bk[(bk["pair"] == r["pair"])
                     & (bk["ls"] == r["ls"])
                     & (bk["T_src"] == r["T_src"])
                     & (bk["component"] == comp)]
            v = float(sub.iloc[0]["PEC_1MW [EUR/kW]"]) if not sub.empty else 0.0
            vals.append(v)
        ax.bar(range(len(top)), vals, bottom=bottom,
               color=comp_colors[k], label=comp, edgecolor="white", linewidth=0.3)
        bottom += np.array(vals)

    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top["label"], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("PEC per kW heating  [EUR/kW]")
    ax.axhspan(ANNEX58_BAND_LOW * F_INSTALL, ANNEX58_BAND_HIGH * F_INSTALL,
               color="#2e7d32", alpha=0.10, zorder=0,
               label=f"Annex 58 band × F_INSTALL ({F_INSTALL:.1f})")
    ax.set_title(f"PEC component breakdown — 12 cheapest designs by c_P "
                 f"(Q_H = 1 MWth, TCI scale)\n"
                 f"Ranking by c_P at {_op_string()}; bar = TCI/kW (capital "
                 f"only, FLH/e1/gas affect ranking but not bar heights)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", ncol=2, fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ECON_DIR, "PEC_breakdown.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 5: c_P sensitivity to electricity price ───────────────────────────

def plot_cP_vs_e1(base, sens):
    # Pick the 12 best designs (by base c_P) for clarity
    best12 = base.nsmallest(12, "c_P [EUR/GJ]")
    keys = list(zip(best12["pair"], best12["ls"], best12["T_src"]))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    # Tab10 + tab10b extension for 12 distinguishable colors
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[:12]
    for k, (pair, ls, T_src) in enumerate(keys):
        sub = sens[(sens["pair"] == pair)
                   & (sens["ls"] == ls)
                   & (sens["T_src"] == T_src)].sort_values("e1_c [EUR/MWh]")
        if sub.empty:
            continue
        label = f"{pair} LS={ls*100:.0f}% T_src={T_src:.0f}°C"
        ax.plot(sub["e1_c [EUR/MWh]"], sub["c_P [EUR/GJ]"],
                marker="o", markersize=4, linewidth=1.4,
                color=colors[k], label=label)

    # Gas heater horizontal reference: exergy-based c_P at base gas price
    gas_c_eur_GJ = _gas_reference_cP()
    ax.axhline(gas_c_eur_GJ, color="black", linestyle="--", linewidth=1.5,
               label=rf"Gas heater @ {BASE_GAS_C:.0f} EUR/MWh, η=0.90"
                     rf"  ($c_P$ = {gas_c_eur_GJ:.0f} EUR/GJ$_{{ex}}$)")
    ax.axvline(BASE_E1_C, color="grey", linestyle=":", alpha=0.6,
               label=f"Base e1 = {BASE_E1_C:.0f} EUR/MWh")

    ax.set_xlabel("Electricity price e1 [EUR/MWh]")
    ax.set_ylabel(r"Specific product cost $c_P$  [EUR/GJ$_{ex}$]")
    ax.set_title("c_P sensitivity to electricity price — 12 cheapest designs\n"
                 f"Steam at 110 °C, Q_H = 1 MWth, modern envelope, "
                 f"LS ∈ {{0.30, 0.40, 0.50}}  |  "
                 f"FLH = {BASE_FULL_LOAD_HOURS:.0f} h/a, "
                 f"gas = {BASE_GAS_C:.0f} EUR/MWh",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.90)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ECON_DIR, "cP_vs_e1c.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot: c_P sensitivity to full-load hours ───────────────────────────────

def plot_cP_vs_FLH(base, sens_flh):
    if sens_flh is None or sens_flh.empty:
        print("  ! no FLH sensitivity data; skipping cP_vs_FLH")
        return
    best12 = base.nsmallest(12, "c_P [EUR/GJ]")
    keys = list(zip(best12["pair"], best12["ls"], best12["T_src"]))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[:12]
    for k, (pair, ls, T_src) in enumerate(keys):
        sub = sens_flh[(sens_flh["pair"] == pair)
                        & (sens_flh["ls"] == ls)
                        & (sens_flh["T_src"] == T_src)
                        ].sort_values("full_load_hours [h/a]")
        if sub.empty:
            continue
        label = f"{pair} LS={ls*100:.0f}% T_src={T_src:.0f}°C"
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
               label=f"Base FLH = {BASE_FULL_LOAD_HOURS:.0f} h/a")

    ax.set_xlabel("Full-load operating hours [h/a]")
    ax.set_ylabel(r"Specific product cost $c_P$  [EUR/GJ$_{ex}$]")
    ax.set_title("c_P sensitivity to full-load hours — 12 cheapest designs\n"
                 f"Steam at 110 °C, Q_H = 1 MWth, modern envelope, "
                 f"LS ∈ {{0.30, 0.40, 0.50}}  |  "
                 f"e1 = {BASE_E1_C:.0f} EUR/MWh, "
                 f"gas = {BASE_GAS_C:.0f} EUR/MWh",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ECON_DIR, "cP_vs_FLH.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 6: c_P at base price vs gas reference ─────────────────────────────

def plot_cP_vs_gas(base):
    # Sort by c_P ascending; color by fluid pair
    df = base.sort_values("c_P [EUR/GJ]").reset_index(drop=True)
    pair_colors = {p: c for p, c in zip(sorted(df["pair"].unique()),
                                          plt.cm.tab10(np.linspace(0, 1, 10)))}

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for k, (_, r) in enumerate(df.iterrows()):
        ax.bar(k, r["c_P [EUR/GJ]"], color=pair_colors[r["pair"]],
               edgecolor="black", linewidth=0.2)

    gas_c_eur_GJ = _gas_reference_cP()
    ax.axhline(gas_c_eur_GJ, color="black", linestyle="--", linewidth=1.5,
               label=rf"Gas heater (retrofit) at {BASE_GAS_C:.0f} EUR/MWh, "
                     rf"η=0.90: $c_P$ = {gas_c_eur_GJ:.0f} EUR/GJ$_{{ex}}$")

    ax.set_xlabel(r"Designs (sorted by $c_P$, lowest left)")
    ax.set_ylabel(r"$c_P$  [EUR/GJ$_{ex}$]")
    ax.set_title(f"c_P — all OK designs sorted by cheapness, color by fluid pair\n"
                 f"{_op_string()}",
                 fontsize=11, fontweight="bold")
    handles = [mpatches.Patch(color=c, label=p) for p, c in pair_colors.items()]
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                              label=rf"Gas reference $c_P$ = {gas_c_eur_GJ:.0f} EUR/GJ$_{{ex}}$"))
    ax.legend(handles=handles, ncol=5, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ECON_DIR, "cP_vs_gas.png"),
                dpi=150, bbox_inches="tight")
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
    y_max = float(base["c_P [EUR/GJ]"].max()) * 1.08
    y_max = max(y_max, gas_c_eur_GJ * 1.10)

    n = len(T_src_vals)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 5.5),
                             sharey=True, squeeze=False)
    for j, T_src in enumerate(T_src_vals):
        ax = axes[0][j]
        sub = (base[base["T_src"] == T_src]
               .sort_values("c_P [EUR/GJ]")
               .reset_index(drop=True))
        if sub.empty:
            ax.set_title(f"T_src = {T_src:.0f} °C\n(no OK designs)",
                         fontsize=10)
            ax.axis("off")
            continue

        labels = [f"{r['pair']}  LS={r['ls']*100:.0f}%"
                  for _, r in sub.iterrows()]
        colors = [pair_colors[r["pair"]] for _, r in sub.iterrows()]
        x = np.arange(len(sub))
        ax.bar(x, sub["c_P [EUR/GJ]"], color=colors,
               edgecolor="black", linewidth=0.3)
        ax.axhline(gas_c_eur_GJ, color="black", linestyle="--", linewidth=1.2)

        # value labels above each bar
        for k, v in enumerate(sub["c_P [EUR/GJ]"]):
            ax.text(k, v + 0.5, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=60, ha="right")
        ax.set_title(f"T_src = {T_src:.0f} °C  (n={len(sub)})",
                     fontsize=10, fontweight="bold")
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=0.3)
        if j == 0:
            ax.set_ylabel(r"$c_P$  [EUR/GJ$_{ex}$]")

    handles = [mpatches.Patch(color=c, label=p)
               for p, c in pair_colors.items()
               if p in base["pair"].unique()]
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                              label=rf"Gas heater  $c_P$ = {gas_c_eur_GJ:.0f} EUR/GJ$_{{ex}}$"))
    fig.legend(handles=handles, ncol=len(handles),
               loc="lower center", fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Specific product cost c_P, ranked within each source-water case "
        f"study  (Q_H = 1 MWth, T_steam = 110 °C)\n{_op_string()}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(os.path.join(ECON_DIR, "cP_sorted_by_T_src.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 8 & 9: C_D + Z component breakdown per design ────────────────────

def _design_dir(pair, ls, T_src):
    f1, f2 = pair.split("/")
    ls_pct = int(round(ls * 100))
    return os.path.join(DESIGNS_DIR, f"{f1}_{f2}",
                         f"LS{ls_pct}_Tsrc{int(T_src)}")


def _load_cdz_per_design(base, value_col="C_D+Z [EUR/h]"):
    """Per-design cost-rate metric [EUR/h] grouped by COMP_GROUPS.

    Returns a DataFrame with columns: pair, ls, T_src, <group>... (one column
    per group in COMP_GROUPS) and a TOTAL column. One row per OK design.
    """
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
        for group, members in COMP_GROUPS.items():
            row[group] = sum(cdz.get(m, 0.0) for m in members)
        row["TOTAL"] = sum(row[g] for g in COMP_GROUPS)
        rows.append(row)
    return pd.DataFrame(rows)


# Designs with T_src ≤ this temperature operate the SRC_HX in the dissipative
# regime (T_evap_c1 < T0 = 20 °C, refrigerant evaporates below the dead state).
# Both source water and refrigerant lose thermal exergy across the SRC_HX,
# so it produces no useful exergy gain → no F-P split → C_D inflates and is
# no longer a fair "cost-of-irreversibility" indicator at the component level.
# These bars are shaded with a hatch pattern in the breakdown plots.
DISSIPATIVE_T_SRC_MAX_C = 30   # °C, inclusive


def _plot_breakdown_per_design(cdz, ylabel, suptitle, fname):
    """One panel per fluid pair; bars per (T_src × LS) within the panel."""
    pairs = sorted(cdz["pair"].unique())
    n = len(pairs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6.0 * ncols, 5.5 * nrows),
                              squeeze=False)

    y_max = float(cdz["TOTAL"].max()) * 1.18

    for idx, pair in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        sub = (cdz[cdz["pair"] == pair]
               .sort_values(["T_src", "ls"])
               .reset_index(drop=True))
        labels = [f"T{int(r['T_src'])}\nLS{int(r['ls']*100)}"
                  for _, r in sub.iterrows()]
        # Mark designs where SRC_HX operates dissipatively (T_evap_c1 < T0).
        is_diss = sub["T_src"].values <= DISSIPATIVE_T_SRC_MAX_C
        x = np.arange(len(sub))
        bottom = np.zeros(len(sub))
        for group in COMP_GROUPS:
            vals = sub[group].values
            # Plot productive (no hatch) and dissipative (hatched) bars
            # in two separate calls to give different hatch attributes.
            for mask, hatch in ((~is_diss, None), (is_diss, "////")):
                if not mask.any():
                    continue
                ax.bar(x[mask], vals[mask], bottom=bottom[mask],
                       color=GROUP_COLORS[group],
                       edgecolor="black", linewidth=0.3,
                       hatch=hatch,
                       label=group if (idx == 0 and hatch is None) else None)
            bottom += vals
        # Background tint behind dissipative columns (very subtle).
        for k, dissipative in enumerate(is_diss):
            if dissipative:
                ax.axvspan(k - 0.45, k + 0.45, color="red", alpha=0.06,
                           zorder=0)
        # Total label on top
        for k, v in enumerate(sub["TOTAL"]):
            ax.text(k, v + y_max * 0.01, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, y_max)
        ax.set_title(pair, fontweight="bold", fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        if idx % ncols == 0:
            ax.set_ylabel(ylabel)

    # Hide any unused panels
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g)
               for g in COMP_GROUPS]
    handles.append(mpatches.Patch(facecolor="white", edgecolor="black",
                                   hatch="////",
                                   label=(f"hatched: T_src ≤ {DISSIPATIVE_T_SRC_MAX_C} °C "
                                          f"→ T_evap_c1 < T0 → SRC_HX dissipative "
                                          f"(C_D inflated, see notes)")))
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(os.path.join(ECON_DIR, fname),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost_breakdown_per_design(base, cdz):
    _plot_breakdown_per_design(
        cdz,
        ylabel=r"$\dot{C}_D + \dot{Z}$  [EUR/h]",
        suptitle=(r"$\dot{C}_D + \dot{Z}$ component breakdown per design "
                  f"(Q_H = 1 MWth, T_steam = 110 °C)\n{_op_string()}"),
        fname="cost_breakdown_per_design.png",
    )


def plot_z_breakdown_per_design(base, z):
    _plot_breakdown_per_design(
        z,
        ylabel=r"$\dot{Z}$  [EUR/h]",
        suptitle=(r"$\dot{Z}$ (capital cost-rate) component breakdown per design "
                  f"(Q_H = 1 MWth, T_steam = 110 °C)\n{_op_string()}"),
        fname="z_breakdown_per_design.png",
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
    fig, ax = plt.subplots(figsize=(11, 6.5))

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
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n(n={int(n)})" for p, n in zip(pairs, n_per)],
                       fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, tot_max.max() * 1.15)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.legend(loc="upper right", ncol=2, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ECON_DIR, fname),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost_breakdown_aggregated(cdz):
    _plot_breakdown_aggregated(
        cdz,
        ylabel=r"$\dot{C}_D + \dot{Z}$  [EUR/h]",
        title=("Mean component cost-rate breakdown per fluid pair  "
               "(whiskers = min–max across LS × T_src)\n"
               f"{_op_string()}"),
        fname="cost_breakdown_aggregated.png",
    )


def plot_z_breakdown_aggregated(z):
    _plot_breakdown_aggregated(
        z,
        ylabel=r"$\dot{Z}$  [EUR/h]",
        title=("Mean component capital cost-rate (Z) breakdown per fluid pair  "
               "(whiskers = min–max across LS × T_src)\n"
               f"{_op_string()}"),
        fname="z_breakdown_aggregated.png",
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


# Methodology fix: COMP/MOT and PUMP/MOT cost-row bundling is inconsistent
# across fluid families in Ommen Table 4 (HC: motor lumped into compressor
# PEC; R717: motor priced separately; pump motors: always lumped into pump).
# This makes the per-component f-factor (= Z / (Z+C_D)) misleading: e.g.,
# MOT1 has f=0 % for HC fluids (its Z is hidden inside COMP1) but f=34 % for
# R717 (its Z is exposed). To produce *cross-fluid-consistent* improvement-
# priority and ranking plots, we aggregate at the level the cost rows are
# actually published — drive packages (compressor+motor, pump+motor).
# The Tsatsaronis quadrant and ranking-flip plots use these aggregates;
# the breakdown plots can still show the fine-grained components.
INSIGHT_AGGREGATION = {
    "COMP+MOT C1":  (["COMP1", "MOT1"],          GROUP_COLORS["COMP+MOT"]),
    "COMP+MOT C2":  (["COMP2", "MOT2"],          GROUP_COLORS["COMP+MOT"]),
    "PUMP+MOT SRC": (["SRC_PUMP", "MOT3"],       GROUP_COLORS["PUMP"]),
    "PUMP+MOT SNK": (["SNK_PUMP", "MOT4"],       GROUP_COLORS["PUMP"]),
    "SRC_HX":       (["SRC_HX"],                 GROUP_COLORS["SRC_HX"]),
    "IHX":          (["IHX"],                    GROUP_COLORS["IHX"]),
    "SNK_HX":       (["SNK_HX"],                 GROUP_COLORS["SNK_HX"]),
    "VAL1":         (["VAL1"],                   GROUP_COLORS["VAL"]),
    "VAL2":         (["VAL2"],                   GROUP_COLORS["VAL"]),
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


def _best_design_per_pair(base):
    """Lowest-c_P design within each fluid pair, as a list of dicts."""
    rows = base.loc[base.groupby("pair")["c_P [EUR/GJ]"].idxmin()]
    return rows.sort_values("pair").to_dict(orient="records")


def plot_tsatsaronis_quadrant(base):
    """Tsatsaronis improvement-priority quadrant — one panel per fluid pair.

    For the lowest-c_P design within each pair, scatter every component on
    (f-factor, C_D+Z). Marker size scales with C_D so dissipative items pop.
    Quadrant guidance lines at f=50% and at half the panel's C_D+Z range.
    """
    bests = _best_design_per_pair(base)
    n = len(bests)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.6 * ncols, 5.0 * nrows),
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
                        fontsize=7, alpha=0.9)

        # Quadrant tags
        ax.text(0.02, 0.97, "improve\nefficiency",
                transform=ax.transAxes, fontsize=8, color="#7e1b1b",
                va="top", ha="left", alpha=0.7, fontweight="bold")
        ax.text(0.98, 0.97, "buy\ncheaper",
                transform=ax.transAxes, fontsize=8, color="#1b3d7e",
                va="top", ha="right", alpha=0.7, fontweight="bold")
        ax.text(0.50, 0.02, "low priority",
                transform=ax.transAxes, fontsize=8, color="gray",
                va="bottom", ha="center", alpha=0.7, fontstyle="italic")

        ax.set_xlim(-3, 103)
        ax.set_ylim(0, cdz_max)
        ax.set_xlabel(r"f-factor = $\dot{Z} / (\dot{Z} + \dot{C}_D)$  [%]")
        if idx % ncols == 0:
            ax.set_ylabel(r"$\dot{C}_D + \dot{Z}$  [EUR/h]")
        ax.set_title(
            f"{r['pair']}  LS={int(r['ls']*100)}%  T_src={int(r['T_src'])} °C  "
            rf"$c_P$={r['c_P [EUR/GJ]']:.1f} EUR/GJ$_{{ex}}$",
            fontsize=10,
        )
        ax.grid(alpha=0.25)

    # Legend (component groups + size meaning)
    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g)
               for g in COMP_GROUPS]
    handles.append(plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor="lightgray",
                               markeredgecolor="black", markersize=10,
                               label="marker size ∝ C_D"))
    fig.legend(handles=handles, loc="lower center",
               ncol=min(7, len(handles)), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Tsatsaronis improvement-priority quadrant — best design of each fluid pair\n"
        "low f → fix by efficiency; high f → fix by buying cheaper; "
        "size ∝ exergy-destruction cost  "
        "(COMP+MOT and PUMP+MOT aggregated to match Ommen Tab. 4 cost-row granularity)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(os.path.join(ECON_DIR, "tsatsaronis_quadrant.png"),
                dpi=150, bbox_inches="tight")
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
        0.30: dict(color="#1f77b4", marker="o", label="LS = 30 %"),
        0.40: dict(color="#2ca02c", marker="s", label="LS = 40 %"),
        0.50: dict(color="#d62728", marker="^", label="LS = 50 %"),
    }

    pairs = sorted(base["pair"].unique())
    n = len(pairs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.4 * ncols, 4.4 * nrows),
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
                    label=style.get("label", f"LS = {int(ls*100)} %"))

        if gas_cP is not None:
            ax.axhline(gas_cP, color="black", linestyle="--", linewidth=1.2,
                       alpha=0.7)

        ax.set_xlim(min(T_src_all) - 3, max(T_src_all) + 3)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(T_src_all)
        ax.grid(alpha=0.3)
        ax.set_title(pair, fontsize=11, fontweight="bold")
        if idx % ncols == 0:
            ax.set_ylabel(r"$c_P$  [EUR/GJ$_{ex}$]")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("T_source_in  [°C]")

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
               ncol=min(5, len(handles)), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Lift-share preference vs source temperature — one panel per fluid pair  "
        f"({_op_string()})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(ECON_DIR, "lift_share_vs_T_src.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_economic_vs_exergoeconomic_ranking(base):
    """Side-by-side ranking flip: pure-economic (Z only) vs full exergoeco (C_D + Z).

    For each fluid pair's best design, two horizontal bar charts in the same
    panel: components sorted by Z (left, "what economic analysis sees")
    vs components sorted by C_D+Z (right, "the real cost-rate the system
    incurs"). Components that move up the ranking when C_D is included
    are the ones a pure CAPEX/OPEX analysis would underweight.
    """
    bests = _best_design_per_pair(base)
    n = len(bests)
    nrows = n   # one row per fluid pair
    fig, axes = plt.subplots(nrows, 2,
                              figsize=(13.0, 2.2 * nrows + 1.2),
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
        ax_eco.set_yticklabels(eco_sorted["Component"], fontsize=8)
        ax_eco.set_xlim(0, x_max)
        ax_eco.grid(axis="x", alpha=0.25)
        if idx == 0:
            ax_eco.set_title(r"Economic-only ranking by  $\dot{Z}$ (capital)",
                              fontsize=10)

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
        ax_xex.set_yticklabels(xex_sorted["Component"], fontsize=8)
        ax_xex.set_xlim(0, x_max)
        ax_xex.grid(axis="x", alpha=0.25)
        if idx == 0:
            ax_xex.set_title(r"Exergoeconomic ranking by  $\dot{Z} + \dot{C}_D$",
                              fontsize=10)

        # Pair label on left side
        pair_label = (f"{r['pair']}\nLS={int(r['ls']*100)}%, "
                      f"T_src={int(r['T_src'])} °C\n"
                      rf"$c_P$={r['c_P [EUR/GJ]']:.1f}")
        ax_eco.set_ylabel(pair_label, fontsize=9, rotation=0,
                          ha="right", va="center", labelpad=50)

    axes[-1][0].set_xlabel(r"$\dot{Z}$  [EUR/h]")
    axes[-1][1].set_xlabel(r"$\dot{Z} + \dot{C}_D$  [EUR/h]  (solid = Z, hatched = C_D)")

    fig.suptitle(
        "What an exergoeconomic analysis catches that a pure CAPEX ranking misses\n"
        "Components shifted UP on the right are under-weighted by economic-only analysis  "
        "(COMP+MOT and PUMP+MOT aggregated to match Ommen Tab. 4 cost-row granularity)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(ECON_DIR, "economic_vs_exergoeconomic_ranking.png"),
                dpi=150, bbox_inches="tight")
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
    Mirrors Ommen 2015 Fig. 3 / the broader-pipeline plot.py. One panel per
    fluid pair, showing the cheapest (LS, T_src) design at the **base FLH**
    (we keep the same design across FLH variants so the panels stay
    comparable). Axes: e1 ∈ [37.5, 112.5] EUR/MWh; gas ∈ [25, 150] EUR/MWh.

    HTHP c_P is affine in e1 (slope = 1/ε, FLH-independent); the FLH only
    shifts the capital-recovery intercept. Gas c_P is linear in c_gas
    (Z_gas = 0 retrofit) and FLH-independent.
    """
    if full_load_hours is None:
        full_load_hours = BASE_FULL_LOAD_HOURS
    is_base_flh = float(full_load_hours) == float(BASE_FULL_LOAD_HOURS)

    # Per pair: pick the design with the lowest c_P at base FLH/e1
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.8 * nrows),
                              squeeze=False)
    axes_flat = axes.flatten()

    X, Y = np.meshgrid(e1_axis, gas_axis)
    gas_field = np.tile(gas_cP_1d[:, None], (1, len(e1_axis)))

    cf = None
    for idx, pair in enumerate(sorted(hthp_cP_per_pair)):
        ax = axes_flat[idx]
        cP_e1, ls, T_src, cop = hthp_cP_per_pair[pair]
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

        ax.set_xlabel(r"$c_{e1}$ [EUR/MWh]")
        ax.set_ylabel(r"$c_{gas}$ [EUR/MWh]")
        ax.set_title(f"{pair}  LS={int(ls*100)}%  T_src={int(T_src)} °C  "
                     f"COP={cop:.2f}", fontsize=10)

    for idx in range(n, nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.tight_layout(rect=[0, 0.04, 0.92, 0.94])
    cbar_ax = fig.add_axes([0.94, 0.10, 0.014, 0.78])
    fig.colorbar(cf, cax=cbar_ax,
                 label=r"$c_P^{\,\mathrm{HTHP}}$  [EUR/GJ$_{ex}$]")

    legend_handles = [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.4,
                   label=r"Break-even  ($c_P^{\,\mathrm{HTHP}} = c_P^{\,\mathrm{gas}}$)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markeredgecolor="white", markersize=10,
                   label=f"Base prices  ({BASE_E1_C:.0f} EUR/MWh el., "
                         f"{BASE_GAS_C:.0f} EUR/MWh gas)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        r"$c_P^{\,\mathrm{HTHP}}$ vs electricity & gas prices, with break-even "
        r"contour  "
        f"(best design per fluid pair, FLH = {full_load_hours:.0f} h/a)\n"
        f"Above the dashed line: HTHP cheaper than gas; below: gas cheaper.",
        fontsize=12, fontweight="bold", y=0.98,
    )

    if fname is None:
        if is_base_flh:
            fname = "price_sensitivity_2d.png"
        else:
            fname = f"price_sensitivity_2d_FLH{int(full_load_hours)}.png"
    fig.savefig(os.path.join(ECON_DIR, fname),
                dpi=150, bbox_inches="tight")
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
                              figsize=(5.0 * ncols, 4.5 * nrows),
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
        ax.set_title(f"T_src = {int(T_src)} °C  (n = {len(sub)})",
                     fontsize=11, fontweight="bold")
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
                   label=f"LS = {int(round(ls*100))}%")
        for ls, m in ls_markers.items()
    ]
    fig.legend(handles=pair_handles + ls_handles,
               loc="lower center", ncol=len(pair_handles) + len(ls_handles),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        r"$c_P$ vs total system exergy losses ($\dot{E}_D + \dot{E}_L$) "
        f"per source-water case study  (Q_H = 1 MWth, T_steam = 110 °C)\n"
        f"{_op_string()}  |  Only designs sharing the same T_src are directly "
        f"comparable.",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(os.path.join(ECON_DIR, "cP_vs_ED_EL_by_Tsrc.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot: best configuration per T_src — KPI dashboard ────────────────────

def plot_best_designs_per_T_src(base, cdz):
    """For each T_src, show the cheapest (lowest c_P) design and its KPIs.

    Four-panel dashboard:
      (1) COP        — bar per T_src
      (2) ε [%]      — bar per T_src
      (3) TCI/kW     — bar per T_src (Annex 58 / Project 68 band overlaid)
      (4) C_D + Z    — stacked bar per T_src, components COMP+MOT, SRC_HX,
                        IHX, SNK_HX, VAL, PUMP

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
    labels = [f"T_src={int(T)} °C\n{best.loc[T, 'pair']}  "
              f"LS={int(round(best.loc[T, 'ls']*100))}%"
              for T in T_src_vals]
    pairs_in_best = list(best["pair"])
    pair_colors = {p: c for p, c in zip(sorted(set(pairs_in_best)),
                                          plt.cm.tab10(np.linspace(0, 1, 10)))}
    bar_colors = [pair_colors[p] for p in pairs_in_best]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

    # Panel 1: COP
    ax = axes[0]
    cops = best["COP"].values
    ax.bar(x, cops, color=bar_colors, edgecolor="black", linewidth=0.4)
    for k, v in enumerate(cops):
        ax.text(k, v + 0.04, f"{v:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylabel("COP  [-]")
    ax.set_title("Coefficient of performance", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(cops) * 1.18)

    # Panel 2: epsilon (%)
    ax = axes[1]
    eps = best["epsilon"].astype(float).values * 100.0
    ax.bar(x, eps, color=bar_colors, edgecolor="black", linewidth=0.4)
    for k, v in enumerate(eps):
        ax.text(k, v + 0.6, f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylabel(r"$\varepsilon$  [%]")
    ax.set_title("Exergetic efficiency", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(eps) * 1.20)

    # Panel 3: TCI / kW with Annex 58 / Project 68 band
    ax = axes[2]
    tci = best["PEC_1MW [EUR/kW]"].values
    ax.bar(x, tci, color=bar_colors, edgecolor="black", linewidth=0.4)
    ax.axhspan(ANNEX58_BAND_LOW, ANNEX58_BAND_HIGH,
               color="#2e7d32", alpha=0.15, zorder=0,
               label=f"Annex 58 / Project 68 band (no integration, "
                     f"{ANNEX58_BAND_LOW:.0f}–{ANNEX58_BAND_HIGH:.0f} EUR/kW)")
    for k, v in enumerate(tci):
        ax.text(k, v + 12, f"{v:.0f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylabel(r"TCI/kW  [EUR/kW] (F$_{install}$ × PEC × CI)")
    ax.set_title("Total capital investment per kW heat", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(tci) * 1.20)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.95)

    # Panel 4: C_D + Z stacked breakdown
    ax = axes[3]
    if cdz_idx is None:
        ax.text(0.5, 0.5, "per-design exergoeco CSVs missing",
                ha="center", va="center", transform=ax.transAxes,
                color="grey", fontsize=10)
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
                    fontsize=9, fontweight="bold")
        ax.set_ylabel(r"$\dot{C}_D + \dot{Z}$  [EUR/h]")
        ax.set_title("Component cost-rate breakdown", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, bottom.max() * 1.20)
        ax.legend(loc="upper right", ncol=2, fontsize=7, framealpha=0.95)

    fig.suptitle(
        f"Best (cheapest c_P) design per source-water case study  "
        f"(Q_H = 1 MWth, T_steam = 110 °C)\n{_op_string()}  |  "
        f"x-axis: T_src; bar = winning fluid pair / LS at that T_src",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(ECON_DIR, "best_designs_per_T_src.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(T_steam=None):
    """Render the full economics plot set for one T_steam case study."""
    if T_steam is not None:
        _set_paths_for_T_steam(T_steam)
    if not all(os.path.exists(p) for p in [BASE_CSV, BREAKDOWN_CSV, SENS_CSV]):
        print("Missing economics CSVs. Run case_steam_110_economics.py first.")
        sys.exit(1)
    os.makedirs(ECON_DIR, exist_ok=True)
    base, bk, sens, sens_flh = _load()

    print("Plotting c_P heatmap ...")
    plot_cP_grid(base)
    print("Plotting PEC/kW (TCI) with Annex 58 band ...")
    plot_pec_per_kW_TCI(base)
    print("Plotting PEC/kW (components only) with Annex 58 band ...")
    plot_pec_per_kW_components(base)
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
        print("Plotting Tsatsaronis improvement-priority quadrant per fluid pair ...")
        plot_tsatsaronis_quadrant(base)
        print("Plotting economic vs exergoeconomic ranking ...")
        plot_economic_vs_exergoeconomic_ranking(base)
        print("Plotting best designs per T_src dashboard ...")
        plot_best_designs_per_T_src(base, cdz)
    print("Plotting lift-share trends vs T_src per fluid pair ...")
    plot_lift_share_vs_T_src(base)

    print("Plotting ±50 % e1/gas 2-D price sensitivity with break-even ...")
    plot_price_sensitivity_2d(base, sens)

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
    print(f"Figures written to {ECON_DIR}/")


if __name__ == "__main__":
    main()
