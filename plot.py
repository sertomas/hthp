"""
Stage 3 — Generate all figures from cached analysis and simulation data.

Each public ``plot_*`` function produces one figure (or a family of figures
for per-scenario diagrams) and saves it under ``results/``.  The module
can be run standalone or called programmatically via ``generate_all_plots``.

Usage
-----
::

    python plot.py               # generate all plots
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fluprodia import FluidPropertyDiagram

from analyze import load_analysis
from config import (
    BASE_E1_C,
    BASE_FULL_LOAD_HOURS,
    E1_C_RANGE,
    FLUIDS_C1,
    FLUIDS_C2,
    LIFT_SHARE_DEFAULT,
    LIFT_SHARE_RANGE,
    RESULTS_DIR,
    SOURCE_DELTA_T,
    SOURCE_MASS_FLOW,
    SOURCE_MASS_FLOW_RANGE,
    T_SOURCE_IN_DEFAULT,
    T_SOURCE_IN_RANGE,
    lift_share_to_T34,
    ls_label,
)
from simulate import load_simulations


# ── Helpers ──────────────────────────────────────────────────────────────────

def _out_path(*parts):
    """Build a path under ``RESULTS_DIR``, creating parent directories."""
    p = os.path.join(RESULTS_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


METRICS = {
    "c_P [EUR/GJ]": ("$c_P$ [EUR/GJ]", "c_P"),
    "Z_sum [EUR/h]": ("$\\sum \\dot{Z}$ [EUR/h]", "Z_sum"),
    "COP [-]": ("COP [-]", "COP"),
    "epsilon [%]": ("$\\varepsilon$ [%]", "epsilon"),
}

COLORS_C2 = {"R600a": "#1f77b4", "R600": "#ff7f0e", "R717": "#2ca02c"}

# Lift share x-axis values as percentages
_LS_PCT = [ls * 100 for ls in LIFT_SHARE_RANGE]


def _valid_scenario_keys(analysis):
    """Return set of (f1, f2, ls, T_source_in) keys with successful economics."""
    valid = set()
    sens_ls = analysis["sensitivity_lift_share"]
    for (f1, f2, ls), res in sens_ls.items():
        if res is not None and not np.isnan(res.get("c_P", np.nan)):
            valid.add((f1, f2, ls, T_SOURCE_IN_DEFAULT))
    sens_T = analysis["sensitivity_T_source_in"]
    for (f1, f2, T_src), res in sens_T.items():
        if res is not None and not np.isnan(res.get("c_P", np.nan)):
            valid.add((f1, f2, LIFT_SHARE_DEFAULT, T_src))
    return valid


# ── Results summary table ────────────────────────────────────────────────────

def print_summary_table(analysis):
    """Print a formatted table of base-case results to stdout."""
    results = analysis["base_results"]
    heater = analysis["heater_ref"]

    rows = []
    for (f1, f2), res in results.items():
        row = {"Cycle 1": f1, "Cycle 2": f2}
        if res is not None:
            row.update({
                "COP [-]": round(res["COP"], 3),
                "epsilon [%]": round(res["epsilon"] * 100, 2),
                "c_P [EUR/GJ]": round(res["c_P"], 2),
                "Z_sum [EUR/h]": round(res["Z_sum"], 2),
                "E_F [kW]": round(res["E_F"], 2),
                "E_P [kW]": round(res["E_P"], 2),
                "E_D [kW]": round(res["E_D"], 2),
            })
        else:
            row.update({k: None for k in [
                "COP [-]", "epsilon [%]", "c_P [EUR/GJ]",
                "Z_sum [EUR/h]", "E_F [kW]", "E_P [kW]", "E_D [kW]",
            ]})
        rows.append(row)

    rows.append({
        "Cycle 1": "Heater", "Cycle 2": "(el. ref)",
        "COP [-]": round(heater["COP"], 3),
        "epsilon [%]": round(heater["epsilon"] * 100, 2),
        "c_P [EUR/GJ]": round(heater["c_P"], 2),
        "Z_sum [EUR/h]": round(heater["Z_sum"], 2),
        "E_F [kW]": round(heater["E_F"] / 1000, 2),
        "E_P [kW]": round(heater["E_P"] / 1000, 2),
        "E_D [kW]": round(heater["E_D"] / 1000, 2),
    })

    gas_heater = analysis["gas_heater_ref"]
    rows.append({
        "Cycle 1": "Heater", "Cycle 2": "(gas ref)",
        "COP [-]": round(gas_heater["COP"], 3),
        "epsilon [%]": round(gas_heater["epsilon"] * 100, 2),
        "c_P [EUR/GJ]": round(gas_heater["c_P"], 2),
        "Z_sum [EUR/h]": round(gas_heater["Z_sum"], 2),
        "E_F [kW]": round(gas_heater["E_F"] / 1000, 2),
        "E_P [kW]": round(gas_heater["E_P"] / 1000, 2),
        "E_D [kW]": round(gas_heater["E_D"] / 1000, 2),
    })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (fixed_delta_T mode)")
    print("=" * 80)
    print(df.to_string(index=False))

    # Fixed mass flow summary
    results_fm = analysis.get("base_results_fm", {})
    if results_fm:
        rows_fm = []
        for (f1, f2), res in results_fm.items():
            row = {"Cycle 1": f1, "Cycle 2": f2}
            if res is not None:
                row.update({
                    "COP [-]": round(res["COP"], 3),
                    "epsilon [%]": round(res["epsilon"] * 100, 2),
                    "c_P [EUR/GJ]": round(res["c_P"], 2),
                    "T_src_out [°C]": round(res.get("T_source_out", 0), 1) if res.get("T_source_out") else "N/A",
                    "m_src [kg/s]": round(res.get("m_source", 0), 3) if res.get("m_source") else "N/A",
                })
            else:
                row.update({k: None for k in [
                    "COP [-]", "epsilon [%]", "c_P [EUR/GJ]",
                    "T_src_out [°C]", "m_src [kg/s]",
                ]})
            rows_fm.append(row)
        df_fm = pd.DataFrame(rows_fm)
        print("\n" + "=" * 80)
        print(f"RESULTS SUMMARY (fixed_mass_flow mode, m={SOURCE_MASS_FLOW} kg/s)")
        print("=" * 80)
        print(df_fm.to_string(index=False))


# ── Figure 1: Grouped bar charts ─────────────────────────────────────────────

def plot_comparison_bars(analysis):
    """Grouped bar chart of COP, epsilon, c_P and Z_sum for all fluid combos."""
    results = analysis["base_results"]
    heater = analysis["heater_ref"]
    gas_heater = analysis["gas_heater_ref"]
    x = np.arange(len(FLUIDS_C1))
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("HTHP Fluid Combination Comparison", fontsize=14, fontweight="bold")

    for ax, (col_name, (ylabel, key)) in zip(axes.flat, METRICS.items()):
        for i, f2 in enumerate(FLUIDS_C2):
            vals = []
            for f1 in FLUIDS_C1:
                res = results.get((f1, f2))
                if res is not None:
                    vals.append(res[key] * 100 if key == "epsilon" else res[key])
                else:
                    vals.append(0)
            ax.bar(x + i * width, vals, width, label=f"Cycle 2: {f2}", color=COLORS_C2[f2])
            for j, v in enumerate(vals):
                if results.get((FLUIDS_C1[j], f2)) is None:
                    ax.text(x[j] + i * width, 0, "N/A", ha="center", va="bottom", fontsize=8, color="red")

        ref_val = heater[key] * 100 if key == "epsilon" else heater[key]
        ax.axhline(y=ref_val, color="red", linestyle="--", linewidth=1.5, label="El. heater (ref)")
        gas_ref_val = gas_heater[key] * 100 if key == "epsilon" else gas_heater[key]
        ax.axhline(y=gas_ref_val, color="green", linestyle="--", linewidth=1.5, label="Gas heater (ref)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"C1: {f}" for f in FLUIDS_C1])
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(_out_path("overview", "comparison_bars.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Figure 2: Heatmaps ──────────────────────────────────────────────────────

def plot_comparison_heatmaps(analysis):
    """Heatmap matrix of COP, epsilon, c_P and Z_sum (C1 vs C2 fluids)."""
    results = analysis["base_results"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("HTHP Fluid Combination Comparison — Heatmaps", fontsize=14, fontweight="bold")

    for ax, (col_name, (ylabel, key)) in zip(axes.flat, METRICS.items()):
        data = np.full((len(FLUIDS_C1), len(FLUIDS_C2)), np.nan)
        for i, f1 in enumerate(FLUIDS_C1):
            for j, f2 in enumerate(FLUIDS_C2):
                res = results.get((f1, f2))
                if res is not None:
                    data[i, j] = res[key] * 100 if key == "epsilon" else res[key]

        masked = np.ma.masked_invalid(data)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="lightgray")
        im = ax.imshow(masked, cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)

        for i in range(len(FLUIDS_C1)):
            for j in range(len(FLUIDS_C2)):
                if np.isnan(data[i, j]):
                    ax.text(j, i, "N/A", ha="center", va="center", color="gray", fontsize=10)
                else:
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                            color="white", fontsize=9, fontweight="bold")

        ax.set_xticks(range(len(FLUIDS_C2)))
        ax.set_xticklabels(FLUIDS_C2)
        ax.set_yticks(range(len(FLUIDS_C1)))
        ax.set_yticklabels(FLUIDS_C1)
        ax.set_xlabel("Cycle 2 (upper)")
        ax.set_ylabel("Cycle 1 (lower)")
        ax.set_title(ylabel)

    fig.tight_layout()
    fig.savefig(_out_path("overview", "comparison_heatmaps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Sensitivity line plots ───────────────────────────────────────────────────

def _sensitivity_line_plot(x_range, data, heater_data, combos, xlabel, ylabel, title, path,
                           gas_heater_data=None):
    """Generic line plot for a 1-D sensitivity sweep."""
    combo_colors = plt.cm.tab10(np.linspace(0, 1, len(combos)))
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, key in enumerate(combos):
        ax.plot(x_range, data[key], marker="o", markersize=4,
                color=combo_colors[i], label=f"{key[0]}/{key[1]}")
    ax.plot(x_range, heater_data, linestyle="--", linewidth=2, color="red", label="El. heater (ref)")
    if gas_heater_data is not None:
        ax.plot(x_range, gas_heater_data, linestyle="--", linewidth=2, color="green", label="Gas heater (ref)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Cycle 1 / Cycle 2", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_e1c(analysis):
    """c_P vs electricity price at base full-load hours."""
    _sensitivity_line_plot(
        E1_C_RANGE,
        analysis["sensitivity_e1c"],
        analysis["heater_sens_e1c"],
        analysis["valid_combos"],
        "Electricity price $c_{e1}$ [ct/kWh]",
        "$c_P$ [EUR/GJ]",
        f"Sensitivity: $c_P$ vs. electricity price (at {BASE_FULL_LOAD_HOURS} h/a)",
        _out_path("overview", "sensitivity_e1c.png"),
        gas_heater_data=analysis["gas_heater_sens_e1c"],
    )


# ── Lift share sensitivity line plots ────────────────────────────────────────

def plot_sensitivity_lift_share_COP(analysis):
    """COP vs lift share for all valid fluid combinations."""
    valid = analysis["valid_combos"]
    sens = analysis["sensitivity_lift_share"]
    combo_colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (f1, f2) in enumerate(valid):
        vals = [sens.get((f1, f2, ls))["COP"] if sens.get((f1, f2, ls)) else np.nan
                for ls in LIFT_SHARE_RANGE]
        ax.plot(_LS_PCT, vals, marker="o", markersize=5,
                color=combo_colors[i], label=f"{f1}/{f2}")
    ax.set_xlabel("Lower cycle lift share [%]")
    ax.set_ylabel("COP [-]")
    ax.set_title("Sensitivity: COP vs. lift share")
    ax.legend(title="Cycle 1 / Cycle 2", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(_out_path("overview", "sensitivity_lift_share_COP.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_lift_share_cP(analysis):
    """c_P vs lift share for all valid fluid combinations."""
    valid = analysis["valid_combos"]
    heater = analysis["heater_ref"]
    gas_heater = analysis["gas_heater_ref"]
    sens = analysis["sensitivity_lift_share"]
    combo_colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (f1, f2) in enumerate(valid):
        vals = [sens.get((f1, f2, ls))["c_P"] if sens.get((f1, f2, ls)) else np.nan
                for ls in LIFT_SHARE_RANGE]
        ax.plot(_LS_PCT, vals, marker="s", markersize=5,
                color=combo_colors[i], label=f"{f1}/{f2}")
    ax.axhline(y=heater["c_P"], color="red", linestyle="--", linewidth=1.5, label="El. heater (ref)")
    ax.axhline(y=gas_heater["c_P"], color="green", linestyle="--", linewidth=1.5, label="Gas heater (ref)")
    ax.set_xlabel("Lower cycle lift share [%]")
    ax.set_ylabel("$c_P$ [EUR/GJ]")
    ax.set_title(f"Sensitivity: $c_P$ vs. lift share "
                 f"(at {BASE_E1_C:.0f} ct/kWh, {BASE_FULL_LOAD_HOURS} h/a)")
    ax.legend(title="Cycle 1 / Cycle 2", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(_out_path("overview", "sensitivity_lift_share_cP.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_lift_share_epsilon(analysis):
    """Exergetic efficiency vs lift share for all valid fluid combinations."""
    valid = analysis["valid_combos"]
    sens = analysis["sensitivity_lift_share"]
    combo_colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (f1, f2) in enumerate(valid):
        vals = [sens.get((f1, f2, ls))["epsilon"] * 100 if sens.get((f1, f2, ls)) else np.nan
                for ls in LIFT_SHARE_RANGE]
        ax.plot(_LS_PCT, vals, marker="^", markersize=5,
                color=combo_colors[i], label=f"{f1}/{f2}")
    ax.set_xlabel("Lower cycle lift share [%]")
    ax.set_ylabel("$\\varepsilon$ [%]")
    ax.set_title("Sensitivity: $\\varepsilon$ vs. lift share")
    ax.legend(title="Cycle 1 / Cycle 2", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(_out_path("overview", "sensitivity_lift_share_epsilon.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Lift share heatmap ──────────────────────────────────────────────────────

def plot_sensitivity_lift_share_heatmap(analysis):
    """Heatmap of COP, epsilon and c_P across all combos and lift share values."""
    sens = analysis["sensitivity_lift_share"]
    all_combos = [(f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2]
    all_labels = [f"{f1}/{f2}" for f1, f2 in all_combos]

    heatmap_metrics = {
        "COP [-]": "COP",
        "$\\varepsilon$ [%]": "epsilon",
        "$c_P$ [EUR/GJ]": "c_P",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Sensitivity: lift share — all fluid combinations", fontsize=14, fontweight="bold")

    for ax, (ylabel, key) in zip(axes, heatmap_metrics.items()):
        data = np.full((len(all_combos), len(LIFT_SHARE_RANGE)), np.nan)
        for i, (f1, f2) in enumerate(all_combos):
            for j, ls in enumerate(LIFT_SHARE_RANGE):
                res = sens.get((f1, f2, ls))
                if res is not None:
                    val = res[key]
                    if key == "epsilon":
                        val *= 100
                    data[i, j] = val

        masked = np.ma.masked_invalid(data)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="lightgray")
        im = ax.imshow(masked, cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)

        for i in range(len(all_combos)):
            for j in range(len(LIFT_SHARE_RANGE)):
                if np.isnan(data[i, j]):
                    ax.text(j, i, "N/A", ha="center", va="center", color="gray", fontsize=9)
                else:
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                            color="white", fontsize=8, fontweight="bold")

        ax.set_xticks(range(len(LIFT_SHARE_RANGE)))
        ax.set_xticklabels([f"{ls_label(ls)}%" for ls in LIFT_SHARE_RANGE])
        ax.set_yticks(range(len(all_combos)))
        ax.set_yticklabels(all_labels)
        ax.set_xlabel("Lift share (lower/upper) [%]")
        ax.set_title(ylabel)

    fig.tight_layout()
    fig.savefig(_out_path("overview", "sensitivity_lift_share_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Economic sensitivities by lift share ─────────────────────────────────────

def plot_sensitivity_e1c_by_lift_share(analysis):
    """c_P vs electricity price — one subplot per lift share value."""
    sens = analysis["sensitivity_e1c_by_lift_share"]
    heater = analysis["heater_sens_e1c"]
    gas_heater = analysis["gas_heater_sens_e1c"]
    all_possible = [(f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2]
    color_map = {c: plt.cm.tab10(i / len(all_possible)) for i, c in enumerate(all_possible)}

    n_vals = len(LIFT_SHARE_RANGE)
    ncols = 3
    nrows = (n_vals + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle(f"$c_P$ vs. electricity price — by lift share (at {BASE_FULL_LOAD_HOURS} h/a)",
                 fontsize=14, fontweight="bold")

    axes_flat = axes.flat
    for idx, ls in enumerate(LIFT_SHARE_RANGE):
        ax = axes_flat[idx]
        combos = analysis["valid_combos_lift_share"][ls]
        for f1, f2 in combos:
            ax.plot(E1_C_RANGE, sens[ls][(f1, f2)],
                    marker="s", markersize=3, color=color_map[(f1, f2)], label=f"{f1}/{f2}")
        ax.plot(E1_C_RANGE, heater,
                linestyle="--", linewidth=2, color="red", label="El. heater (ref)")
        ax.plot(E1_C_RANGE, gas_heater,
                linestyle="--", linewidth=2, color="green", label="Gas heater (ref)")
        ax.set_xlabel("Electricity price $c_{e1}$ [ct/kWh]")
        ax.set_ylabel("$c_P$ [EUR/GJ]")
        ax.set_title(f"LS = {ls_label(ls)}%")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)

    for idx in range(n_vals, nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(_out_path("overview", "sensitivity_e1c_by_lift_share.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── T_source_in sensitivity line plots ───────────────────────────────────────

def plot_sensitivity_T_source_in_COP(analysis):
    """COP vs T_source_in for all valid fluid combinations."""
    valid = analysis["valid_combos"]
    sens = analysis["sensitivity_T_source_in"]
    combo_colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (f1, f2) in enumerate(valid):
        vals = [sens.get((f1, f2, T))["COP"] if sens.get((f1, f2, T)) else np.nan
                for T in T_SOURCE_IN_RANGE]
        ax.plot(T_SOURCE_IN_RANGE, vals, marker="o", markersize=5,
                color=combo_colors[i], label=f"{f1}/{f2}")
    ax.set_xlabel("$T_{\\mathrm{source,in}}$ [°C]")
    ax.set_ylabel("COP [-]")
    ax.set_title(f"Sensitivity: COP vs. $T_{{\\mathrm{{source,in}}}}$ "
                 f"(LS = {ls_label(LIFT_SHARE_DEFAULT)}%, "
                 f"$\\Delta T_{{src}}$ = {SOURCE_DELTA_T} K)")
    ax.legend(title="Cycle 1 / Cycle 2", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(_out_path("overview", "sensitivity_T_source_in_COP.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_T_source_in_cP(analysis):
    """c_P vs T_source_in for all valid fluid combinations."""
    valid = analysis["valid_combos"]
    heater = analysis["heater_ref"]
    gas_heater = analysis["gas_heater_ref"]
    sens = analysis["sensitivity_T_source_in"]
    combo_colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (f1, f2) in enumerate(valid):
        vals = [sens.get((f1, f2, T))["c_P"] if sens.get((f1, f2, T)) else np.nan
                for T in T_SOURCE_IN_RANGE]
        ax.plot(T_SOURCE_IN_RANGE, vals, marker="s", markersize=5,
                color=combo_colors[i], label=f"{f1}/{f2}")
    ax.axhline(y=heater["c_P"], color="red", linestyle="--", linewidth=1.5, label="El. heater (ref)")
    ax.axhline(y=gas_heater["c_P"], color="green", linestyle="--", linewidth=1.5, label="Gas heater (ref)")
    ax.set_xlabel("$T_{\\mathrm{source,in}}$ [°C]")
    ax.set_ylabel("$c_P$ [EUR/GJ]")
    ax.set_title(f"Sensitivity: $c_P$ vs. $T_{{\\mathrm{{source,in}}}}$ "
                 f"(LS = {ls_label(LIFT_SHARE_DEFAULT)}%, "
                 f"{BASE_E1_C:.0f} ct/kWh, {BASE_FULL_LOAD_HOURS} h/a)")
    ax.legend(title="Cycle 1 / Cycle 2", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(_out_path("overview", "sensitivity_T_source_in_cP.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_T_source_in_epsilon(analysis):
    """Exergetic efficiency vs T_source_in for all valid fluid combinations."""
    valid = analysis["valid_combos"]
    sens = analysis["sensitivity_T_source_in"]
    combo_colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (f1, f2) in enumerate(valid):
        vals = [sens.get((f1, f2, T))["epsilon"] * 100 if sens.get((f1, f2, T)) else np.nan
                for T in T_SOURCE_IN_RANGE]
        ax.plot(T_SOURCE_IN_RANGE, vals, marker="^", markersize=5,
                color=combo_colors[i], label=f"{f1}/{f2}")
    ax.set_xlabel("$T_{\\mathrm{source,in}}$ [°C]")
    ax.set_ylabel("$\\varepsilon$ [%]")
    ax.set_title(f"Sensitivity: $\\varepsilon$ vs. $T_{{\\mathrm{{source,in}}}}$ "
                 f"(LS = {ls_label(LIFT_SHARE_DEFAULT)}%)")
    ax.legend(title="Cycle 1 / Cycle 2", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(_out_path("overview", "sensitivity_T_source_in_epsilon.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── T_source_in heatmap ─────────────────────────────────────────────────────

def plot_sensitivity_T_source_in_heatmap(analysis):
    """Heatmap of COP, epsilon and c_P across all combos and T_source_in values."""
    sens = analysis["sensitivity_T_source_in"]
    all_combos = [(f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2]
    all_labels = [f"{f1}/{f2}" for f1, f2 in all_combos]

    heatmap_metrics = {
        "COP [-]": "COP",
        "$\\varepsilon$ [%]": "epsilon",
        "$c_P$ [EUR/GJ]": "c_P",
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Sensitivity: $T_{{\\mathrm{{source,in}}}}$ — all fluid combinations "
                 f"(LS = {ls_label(LIFT_SHARE_DEFAULT)}%, "
                 f"$\\Delta T_{{src}}$ = {SOURCE_DELTA_T} K)",
                 fontsize=14, fontweight="bold")

    for ax, (ylabel, key) in zip(axes, heatmap_metrics.items()):
        data = np.full((len(all_combos), len(T_SOURCE_IN_RANGE)), np.nan)
        for i, (f1, f2) in enumerate(all_combos):
            for j, T_val in enumerate(T_SOURCE_IN_RANGE):
                res = sens.get((f1, f2, T_val))
                if res is not None:
                    val = res[key]
                    if key == "epsilon":
                        val *= 100
                    data[i, j] = val

        masked = np.ma.masked_invalid(data)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="lightgray")
        im = ax.imshow(masked, cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)

        for i in range(len(all_combos)):
            for j in range(len(T_SOURCE_IN_RANGE)):
                if np.isnan(data[i, j]):
                    ax.text(j, i, "N/A", ha="center", va="center", color="gray", fontsize=8)
                else:
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                            color="white", fontsize=7, fontweight="bold")

        ax.set_xticks(range(len(T_SOURCE_IN_RANGE)))
        ax.set_xticklabels([f"{T}°C" for T in T_SOURCE_IN_RANGE], fontsize=8)
        ax.set_yticks(range(len(all_combos)))
        ax.set_yticklabels(all_labels)
        ax.set_xlabel("$T_{\\mathrm{source,in}}$ [°C]")
        ax.set_title(ylabel)

    fig.tight_layout()
    fig.savefig(_out_path("overview", "sensitivity_T_source_in_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_T_source_in_heatmap_by_lift_share(analysis):
    """Heatmap of COP, epsilon and c_P vs T_source_in — one figure per lift share."""
    sens_by_ls = analysis["sensitivity_T_source_in_by_lift_share"]
    all_combos = [(f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2]
    all_labels = [f"{f1}/{f2}" for f1, f2 in all_combos]

    heatmap_metrics = {
        "COP [-]": "COP",
        "$\\varepsilon$ [%]": "epsilon",
        "$c_P$ [EUR/GJ]": "c_P",
    }

    for ls in LIFT_SHARE_RANGE:
        sens = sens_by_ls[ls]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f"Sensitivity: $T_{{\\mathrm{{source,in}}}}$ — all fluid combinations "
                     f"(LS = {ls_label(ls)}%)",
                     fontsize=14, fontweight="bold")

        for ax, (ylabel, key) in zip(axes, heatmap_metrics.items()):
            data = np.full((len(all_combos), len(T_SOURCE_IN_RANGE)), np.nan)
            for i, (f1, f2) in enumerate(all_combos):
                for j, T_val in enumerate(T_SOURCE_IN_RANGE):
                    res = sens.get((f1, f2, T_val))
                    if res is not None:
                        val = res[key]
                        if key == "epsilon":
                            val *= 100
                        data[i, j] = val

            masked = np.ma.masked_invalid(data)
            cmap = plt.cm.viridis.copy()
            cmap.set_bad(color="lightgray")
            im = ax.imshow(masked, cmap=cmap, aspect="auto")
            fig.colorbar(im, ax=ax, shrink=0.8)

            for i in range(len(all_combos)):
                for j in range(len(T_SOURCE_IN_RANGE)):
                    if np.isnan(data[i, j]):
                        ax.text(j, i, "N/A", ha="center", va="center", color="gray", fontsize=8)
                    else:
                        ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                                color="white", fontsize=7, fontweight="bold")

            ax.set_xticks(range(len(T_SOURCE_IN_RANGE)))
            ax.set_xticklabels([f"{T}°C" for T in T_SOURCE_IN_RANGE], fontsize=8)
            ax.set_yticks(range(len(all_combos)))
            ax.set_yticklabels(all_labels)
            ax.set_xlabel("$T_{\\mathrm{source,in}}$ [°C]")
            ax.set_title(ylabel)

        fig.tight_layout()
        ls_pct = int(round(ls * 100))
        fig.savefig(_out_path("overview", f"sensitivity_T_source_in_heatmap_LS_{ls_pct}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


# ── Economic sensitivities by T_source_in ────────────────────────────────────

def plot_sensitivity_e1c_by_T_source_in(analysis):
    """c_P vs electricity price — one subplot per T_source_in value."""
    sens = analysis["sensitivity_e1c_by_T_source_in"]
    heater = analysis["heater_sens_e1c"]
    gas_heater = analysis["gas_heater_sens_e1c"]
    all_possible = [(f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2]
    color_map = {c: plt.cm.tab10(i / len(all_possible)) for i, c in enumerate(all_possible)}

    n_vals = len(T_SOURCE_IN_RANGE)
    ncols = 3
    nrows = (n_vals + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle(f"$c_P$ vs. electricity price — by $T_{{\\mathrm{{source,in}}}}$ "
                 f"(at {BASE_FULL_LOAD_HOURS} h/a, LS = {ls_label(LIFT_SHARE_DEFAULT)}%)",
                 fontsize=14, fontweight="bold")

    axes_flat = axes.flat
    for idx, T_src_val in enumerate(T_SOURCE_IN_RANGE):
        ax = axes_flat[idx]
        combos = analysis["valid_combos_T_source_in"][T_src_val]
        for f1, f2 in combos:
            ax.plot(E1_C_RANGE, sens[T_src_val][(f1, f2)],
                    marker="s", markersize=3, color=color_map[(f1, f2)], label=f"{f1}/{f2}")
        ax.plot(E1_C_RANGE, heater,
                linestyle="--", linewidth=2, color="red", label="El. heater (ref)")
        ax.plot(E1_C_RANGE, gas_heater,
                linestyle="--", linewidth=2, color="green", label="Gas heater (ref)")
        ax.set_xlabel("Electricity price $c_{e1}$ [ct/kWh]")
        ax.set_ylabel("$c_P$ [EUR/GJ]")
        ax.set_title(f"$T_{{\\mathrm{{source,in}}}}$ = {T_src_val} °C")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)

    for idx in range(n_vals, nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(_out_path("overview", "sensitivity_e1c_by_T_source_in.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Fixed mass flow mode plots ───────────────────────────────────────────────

def plot_sensitivity_T_source_in_COP_fm(analysis):
    """COP vs T_source_in for fixed_mass_flow mode."""
    sens = analysis.get("sensitivity_T_source_in_fm", {})
    if not sens:
        return

    all_combos = [(f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2]
    valid = [c for c in all_combos
             if any(sens.get((c[0], c[1], T)) is not None for T in T_SOURCE_IN_RANGE)]
    if not valid:
        return

    combo_colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (f1, f2) in enumerate(valid):
        vals = [sens.get((f1, f2, T))["COP"] if sens.get((f1, f2, T)) else np.nan
                for T in T_SOURCE_IN_RANGE]
        ax.plot(T_SOURCE_IN_RANGE, vals, marker="o", markersize=5,
                color=combo_colors[i], label=f"{f1}/{f2}")
    ax.set_xlabel("$T_{\\mathrm{source,in}}$ [°C]")
    ax.set_ylabel("COP [-]")
    ax.set_title(f"COP vs. $T_{{\\mathrm{{source,in}}}}$ — fixed mass flow "
                 f"($\\dot{{m}}_{{src}}$ = {SOURCE_MASS_FLOW} kg/s, "
                 f"LS = {ls_label(LIFT_SHARE_DEFAULT)}%)")
    ax.legend(title="Cycle 1 / Cycle 2", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(_out_path("overview", "sensitivity_T_source_in_COP_fm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_T_source_in_cP_fm(analysis):
    """c_P vs T_source_in for fixed_mass_flow mode."""
    sens = analysis.get("sensitivity_T_source_in_fm", {})
    if not sens:
        return

    heater = analysis["heater_ref"]
    gas_heater = analysis["gas_heater_ref"]

    all_combos = [(f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2]
    valid = [c for c in all_combos
             if any(sens.get((c[0], c[1], T)) is not None for T in T_SOURCE_IN_RANGE)]
    if not valid:
        return

    combo_colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (f1, f2) in enumerate(valid):
        vals = [sens.get((f1, f2, T))["c_P"] if sens.get((f1, f2, T)) else np.nan
                for T in T_SOURCE_IN_RANGE]
        ax.plot(T_SOURCE_IN_RANGE, vals, marker="s", markersize=5,
                color=combo_colors[i], label=f"{f1}/{f2}")
    ax.axhline(y=heater["c_P"], color="red", linestyle="--", linewidth=1.5, label="El. heater (ref)")
    ax.axhline(y=gas_heater["c_P"], color="green", linestyle="--", linewidth=1.5, label="Gas heater (ref)")
    ax.set_xlabel("$T_{\\mathrm{source,in}}$ [°C]")
    ax.set_ylabel("$c_P$ [EUR/GJ]")
    ax.set_title(f"$c_P$ vs. $T_{{\\mathrm{{source,in}}}}$ — fixed mass flow "
                 f"($\\dot{{m}}_{{src}}$ = {SOURCE_MASS_FLOW} kg/s, "
                 f"{BASE_E1_C:.0f} ct/kWh, {BASE_FULL_LOAD_HOURS} h/a)")
    ax.legend(title="Cycle 1 / Cycle 2", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(_out_path("overview", "sensitivity_T_source_in_cP_fm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_T_source_in_epsilon_fm(analysis):
    """Exergetic efficiency vs T_source_in for fixed_mass_flow mode."""
    sens = analysis.get("sensitivity_T_source_in_fm", {})
    if not sens:
        return

    all_combos = [(f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2]
    valid = [c for c in all_combos
             if any(sens.get((c[0], c[1], T)) is not None for T in T_SOURCE_IN_RANGE)]
    if not valid:
        return

    combo_colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (f1, f2) in enumerate(valid):
        vals = [sens.get((f1, f2, T))["epsilon"] * 100 if sens.get((f1, f2, T)) else np.nan
                for T in T_SOURCE_IN_RANGE]
        ax.plot(T_SOURCE_IN_RANGE, vals, marker="^", markersize=5,
                color=combo_colors[i], label=f"{f1}/{f2}")
    ax.set_xlabel("$T_{\\mathrm{source,in}}$ [°C]")
    ax.set_ylabel("$\\varepsilon$ [%]")
    ax.set_title(f"$\\varepsilon$ vs. $T_{{\\mathrm{{source,in}}}}$ — fixed mass flow "
                 f"($\\dot{{m}}_{{src}}$ = {SOURCE_MASS_FLOW} kg/s)")
    ax.legend(title="Cycle 1 / Cycle 2", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(_out_path("overview", "sensitivity_T_source_in_epsilon_fm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_modes_COP(analysis):
    """Side-by-side COP comparison: fixed_delta_T vs fixed_mass_flow."""
    sens_dt = analysis["sensitivity_T_source_in"]
    sens_fm = analysis.get("sensitivity_T_source_in_fm", {})
    if not sens_fm:
        return

    all_combos = [(f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2]
    valid = [c for c in all_combos
             if any(sens_dt.get((c[0], c[1], T)) is not None for T in T_SOURCE_IN_RANGE)]
    if not valid:
        return

    combo_colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(f"COP vs. $T_{{\\mathrm{{source,in}}}}$ — mode comparison "
                 f"(LS = {ls_label(LIFT_SHARE_DEFAULT)}%)",
                 fontsize=14, fontweight="bold")

    for i, (f1, f2) in enumerate(valid):
        vals_dt = [sens_dt.get((f1, f2, T))["COP"] if sens_dt.get((f1, f2, T)) else np.nan
                   for T in T_SOURCE_IN_RANGE]
        ax1.plot(T_SOURCE_IN_RANGE, vals_dt, marker="o", markersize=5,
                 color=combo_colors[i], label=f"{f1}/{f2}")

        vals_fm = [sens_fm.get((f1, f2, T))["COP"] if sens_fm.get((f1, f2, T)) else np.nan
                   for T in T_SOURCE_IN_RANGE]
        ax2.plot(T_SOURCE_IN_RANGE, vals_fm, marker="o", markersize=5,
                 color=combo_colors[i], label=f"{f1}/{f2}")

    for ax, title in [(ax1, f"fixed $\\Delta T$ = {SOURCE_DELTA_T} K"),
                       (ax2, f"fixed $\\dot{{m}}$ = {SOURCE_MASS_FLOW} kg/s")]:
        ax.set_xlabel("$T_{\\mathrm{source,in}}$ [°C]")
        ax.set_ylabel("COP [-]")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(_out_path("overview", "comparison_modes_COP.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_modes_cP(analysis):
    """Side-by-side c_P comparison: fixed_delta_T vs fixed_mass_flow."""
    sens_dt = analysis["sensitivity_T_source_in"]
    sens_fm = analysis.get("sensitivity_T_source_in_fm", {})
    if not sens_fm:
        return

    heater = analysis["heater_ref"]
    gas_heater = analysis["gas_heater_ref"]

    all_combos = [(f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2]
    valid = [c for c in all_combos
             if any(sens_dt.get((c[0], c[1], T)) is not None for T in T_SOURCE_IN_RANGE)]
    if not valid:
        return

    combo_colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(f"$c_P$ vs. $T_{{\\mathrm{{source,in}}}}$ — mode comparison "
                 f"(LS = {ls_label(LIFT_SHARE_DEFAULT)}%, "
                 f"{BASE_E1_C:.0f} ct/kWh, {BASE_FULL_LOAD_HOURS} h/a)",
                 fontsize=14, fontweight="bold")

    for i, (f1, f2) in enumerate(valid):
        vals_dt = [sens_dt.get((f1, f2, T))["c_P"] if sens_dt.get((f1, f2, T)) else np.nan
                   for T in T_SOURCE_IN_RANGE]
        ax1.plot(T_SOURCE_IN_RANGE, vals_dt, marker="s", markersize=5,
                 color=combo_colors[i], label=f"{f1}/{f2}")

        vals_fm = [sens_fm.get((f1, f2, T))["c_P"] if sens_fm.get((f1, f2, T)) else np.nan
                   for T in T_SOURCE_IN_RANGE]
        ax2.plot(T_SOURCE_IN_RANGE, vals_fm, marker="s", markersize=5,
                 color=combo_colors[i], label=f"{f1}/{f2}")

    for ax, title in [(ax1, f"fixed $\\Delta T$ = {SOURCE_DELTA_T} K"),
                       (ax2, f"fixed $\\dot{{m}}$ = {SOURCE_MASS_FLOW} kg/s")]:
        ax.axhline(y=heater["c_P"], color="red", linestyle="--", linewidth=1.5, label="El. heater (ref)")
        ax.axhline(y=gas_heater["c_P"], color="green", linestyle="--", linewidth=1.5, label="Gas heater (ref)")
        ax.set_xlabel("$T_{\\mathrm{source,in}}$ [°C]")
        ax.set_ylabel("$c_P$ [EUR/GJ]")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(_out_path("overview", "comparison_modes_cP.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Mass flow sensitivity plots ──────────────────────────────────────────────

def plot_sensitivity_mass_flow_COP(analysis):
    """COP vs T_source_in for each mass flow value."""
    sens = analysis.get("sensitivity_mass_flow", {})
    if not sens:
        return

    all_combos = [(f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2]
    valid = [c for c in all_combos
             if any(sens.get((c[0], c[1], T_SOURCE_IN_DEFAULT, m)) is not None
                    for m in SOURCE_MASS_FLOW_RANGE)]
    if not valid:
        return

    n_combos = len(valid)
    fig, axes = plt.subplots(1, n_combos, figsize=(6 * n_combos, 5), sharey=True, squeeze=False)
    axes = axes[0]
    m_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(SOURCE_MASS_FLOW_RANGE)))

    for idx, (f1, f2) in enumerate(valid):
        ax = axes[idx]
        for j, m_val in enumerate(SOURCE_MASS_FLOW_RANGE):
            vals = [sens.get((f1, f2, T, m_val))["COP"]
                    if sens.get((f1, f2, T, m_val)) else np.nan
                    for T in T_SOURCE_IN_RANGE]
            ax.plot(T_SOURCE_IN_RANGE, vals, marker="o", markersize=5,
                    color=m_colors[j], label=f"$\\dot{{m}}$ = {m_val} kg/s")
        ax.set_xlabel("$T_{\\mathrm{source,in}}$ [°C]")
        if idx == 0:
            ax.set_ylabel("COP [-]")
        ax.set_title(f"{f1}/{f2}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(f"COP vs. $T_{{\\mathrm{{source,in}}}}$ — mass flow sensitivity "
                 f"(LS = {ls_label(LIFT_SHARE_DEFAULT)}%)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(_out_path("overview", "sensitivity_mass_flow_COP.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_mass_flow_cP(analysis):
    """c_P vs T_source_in for each mass flow value."""
    sens = analysis.get("sensitivity_mass_flow", {})
    if not sens:
        return

    heater = analysis["heater_ref"]
    gas_heater = analysis["gas_heater_ref"]

    all_combos = [(f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2]
    valid = [c for c in all_combos
             if any(sens.get((c[0], c[1], T_SOURCE_IN_DEFAULT, m)) is not None
                    for m in SOURCE_MASS_FLOW_RANGE)]
    if not valid:
        return

    n_combos = len(valid)
    fig, axes = plt.subplots(1, n_combos, figsize=(6 * n_combos, 5), sharey=True, squeeze=False)
    axes = axes[0]
    m_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(SOURCE_MASS_FLOW_RANGE)))

    for idx, (f1, f2) in enumerate(valid):
        ax = axes[idx]
        for j, m_val in enumerate(SOURCE_MASS_FLOW_RANGE):
            vals = [sens.get((f1, f2, T, m_val))["c_P"]
                    if sens.get((f1, f2, T, m_val)) else np.nan
                    for T in T_SOURCE_IN_RANGE]
            ax.plot(T_SOURCE_IN_RANGE, vals, marker="s", markersize=5,
                    color=m_colors[j], label=f"$\\dot{{m}}$ = {m_val} kg/s")
        ax.axhline(y=heater["c_P"], color="red", linestyle="--", linewidth=1.5, label="El. heater")
        ax.axhline(y=gas_heater["c_P"], color="green", linestyle="--", linewidth=1.5, label="Gas heater")
        ax.set_xlabel("$T_{\\mathrm{source,in}}$ [°C]")
        if idx == 0:
            ax.set_ylabel("$c_P$ [EUR/GJ]")
        ax.set_title(f"{f1}/{f2}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(f"$c_P$ vs. $T_{{\\mathrm{{source,in}}}}$ — mass flow sensitivity "
                 f"(LS = {ls_label(LIFT_SHARE_DEFAULT)}%, "
                 f"{BASE_E1_C:.0f} ct/kWh, {BASE_FULL_LOAD_HOURS} h/a)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(_out_path("overview", "sensitivity_mass_flow_cP.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_mass_flow_epsilon(analysis):
    """Exergetic efficiency vs T_source_in for each mass flow value."""
    sens = analysis.get("sensitivity_mass_flow", {})
    if not sens:
        return

    all_combos = [(f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2]
    valid = [c for c in all_combos
             if any(sens.get((c[0], c[1], T_SOURCE_IN_DEFAULT, m)) is not None
                    for m in SOURCE_MASS_FLOW_RANGE)]
    if not valid:
        return

    n_combos = len(valid)
    fig, axes = plt.subplots(1, n_combos, figsize=(6 * n_combos, 5), sharey=True, squeeze=False)
    axes = axes[0]
    m_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(SOURCE_MASS_FLOW_RANGE)))

    for idx, (f1, f2) in enumerate(valid):
        ax = axes[idx]
        for j, m_val in enumerate(SOURCE_MASS_FLOW_RANGE):
            vals = [sens.get((f1, f2, T, m_val))["epsilon"] * 100
                    if sens.get((f1, f2, T, m_val)) else np.nan
                    for T in T_SOURCE_IN_RANGE]
            ax.plot(T_SOURCE_IN_RANGE, vals, marker="^", markersize=5,
                    color=m_colors[j], label=f"$\\dot{{m}}$ = {m_val} kg/s")
        ax.set_xlabel("$T_{\\mathrm{source,in}}$ [°C]")
        if idx == 0:
            ax.set_ylabel("$\\varepsilon$ [%]")
        ax.set_title(f"{f1}/{f2}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(f"$\\varepsilon$ vs. $T_{{\\mathrm{{source,in}}}}$ — mass flow sensitivity "
                 f"(LS = {ls_label(LIFT_SHARE_DEFAULT)}%)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(_out_path("overview", "sensitivity_mass_flow_epsilon.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Q-T Diagrams ─────────────────────────────────────────────────────────────

HX_NAMES = ["SRC_HX", "IHX", "SNK_HX"]
HX_TITLES = {"SRC_HX": "Source Heat Exchanger", "IHX": "Internal Heat Exchanger", "SNK_HX": "Sink Heat Exchanger"}


def plot_qt_diagrams(simulations, analysis):
    """Q-T diagrams for every scenario with successful economics."""
    valid_keys = _valid_scenario_keys(analysis)
    hthp = simulations["hthp"]
    for (f1, f2, ls, T_src_val), sim in hthp.items():
        if sim is None or (f1, f2, ls, T_src_val) not in valid_keys:
            continue

        T34 = lift_share_to_T34(ls, T_src_val)
        ls_pct = int(round(ls * 100))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Q–T Diagrams: {f1} / {f2} — LS = {ls_label(ls)}%, "
                     f"$T_{{\\mathrm{{source,in}}}}$ = {T_src_val} °C "
                     f"($T_{{34}}$ = {T34:.1f} °C)",
                     fontsize=13, fontweight="bold")

        for ax, hx_name in zip(axes, HX_NAMES):
            qt = sim["qt_sections"][hx_name]
            Q_kW = qt["Q"]
            T_hot = qt["T_hot"]
            T_cold = qt["T_cold"]
            ax.plot(Q_kW, T_hot, "r-o", markersize=5, label="Hot side")
            ax.plot(Q_kW, T_cold, "b-o", markersize=5, label="Cold side")
            ax.fill_between(Q_kW, T_cold, T_hot, alpha=0.1, color="gray")
            ax.set_xlabel("Q [kW]")
            ax.set_ylabel("T [°C]")
            ax.set_title(HX_TITLES[hx_name])
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(_out_path(f"{f1}_{f2}", f"LS_{ls_pct}_Tsrc_{T_src_val}", "QT_diagram.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


# ── Log(p)-h Diagrams ────────────────────────────────────────────────────────

_diagram_cache = {}


def _get_diagram(fluid):
    """Return a cached FluidPropertyDiagram (computed on first call)."""
    if fluid not in _diagram_cache:
        print(f"  Computing isolines for {fluid} ...")
        d = FluidPropertyDiagram(fluid)
        d.set_unit_system(T="°C", p="bar", h="kJ/kg", s="kJ/kgK")
        T_crit = d.convert_from_SI(d.T_crit, "T")
        d.set_isolines_subcritical(T_min=-40, T_max=T_crit - 2)
        d.calc_isolines()
        _diagram_cache[fluid] = d
    return _diagram_cache[fluid]


def plot_logph_diagrams(simulations, analysis):
    """Log(p)-h diagrams for every scenario with successful economics."""
    valid_keys = _valid_scenario_keys(analysis)
    hthp = simulations["hthp"]
    for (f1, f2, ls, T_src_val), sim in hthp.items():
        if sim is None or (f1, f2, ls, T_src_val) not in valid_keys:
            continue

        T34 = lift_share_to_T34(ls, T_src_val)
        ls_pct = int(round(ls * 100))
        states = sim["cycle_states"]

        fig, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(f"Log(p)–h Diagram: {f1} / {f2} — LS = {ls_label(ls)}%, "
                     f"$T_{{\\mathrm{{source,in}}}}$ = {T_src_val} °C "
                     f"($T_{{34}}$ = {T34:.1f} °C)",
                     fontsize=13, fontweight="bold")

        for ax, cycle_key, cycle_label in [
            (ax_c1, "cycle1", "Cycle 1 (lower)"),
            (ax_c2, "cycle2", "Cycle 2 (upper)"),
        ]:
            cycle = states[cycle_key]
            fluid = cycle["fluid"]
            points = cycle["points"]
            diagram = _get_diagram(fluid)

            h_vals = [pt["h"] for pt in points]
            p_vals = [pt["p"] for pt in points]
            h_margin = (max(h_vals) - min(h_vals)) * 0.3

            diagram.draw_isolines(
                fig=fig, ax=ax, diagram_type="logph",
                x_min=min(h_vals) - h_margin, x_max=max(h_vals) + h_margin,
                y_min=min(p_vals) / 3, y_max=max(p_vals) * 3,
            )

            h_cycle = h_vals + [h_vals[0]]
            p_cycle = p_vals + [p_vals[0]]
            ax.plot(h_cycle, p_cycle, "r-", linewidth=2.5, zorder=3)
            ax.plot(h_vals, p_vals, "ko", markersize=8, zorder=4)

            for pt in points:
                ax.annotate(pt["label"], (pt["h"], pt["p"]),
                            textcoords="offset points", xytext=(8, 8),
                            fontsize=10, fontweight="bold")
            ax.set_title(f"{cycle_label}: {fluid}")

        fig.tight_layout()
        fig.savefig(_out_path(f"{f1}_{f2}", f"LS_{ls_pct}_Tsrc_{T_src_val}", "logph_diagram.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


# ── Generate all ─────────────────────────────────────────────────────────────

def generate_all_plots(analysis, simulations):
    """Generate and save every figure in the study."""
    print("  Comparison bars ...")
    plot_comparison_bars(analysis)
    print("  Comparison heatmaps ...")
    plot_comparison_heatmaps(analysis)
    print("  Sensitivity: e1c ...")
    plot_sensitivity_e1c(analysis)
    print("  Sensitivity: lift share COP ...")
    plot_sensitivity_lift_share_COP(analysis)
    print("  Sensitivity: lift share c_P ...")
    plot_sensitivity_lift_share_cP(analysis)
    print("  Sensitivity: lift share epsilon ...")
    plot_sensitivity_lift_share_epsilon(analysis)
    print("  Sensitivity: lift share heatmap ...")
    plot_sensitivity_lift_share_heatmap(analysis)
    print("  Sensitivity: e1c by lift share ...")
    plot_sensitivity_e1c_by_lift_share(analysis)
    print("  Sensitivity: T_source_in COP ...")
    plot_sensitivity_T_source_in_COP(analysis)
    print("  Sensitivity: T_source_in c_P ...")
    plot_sensitivity_T_source_in_cP(analysis)
    print("  Sensitivity: T_source_in epsilon ...")
    plot_sensitivity_T_source_in_epsilon(analysis)
    print("  Sensitivity: T_source_in heatmap ...")
    plot_sensitivity_T_source_in_heatmap(analysis)
    print("  Sensitivity: T_source_in heatmap by lift share ...")
    plot_sensitivity_T_source_in_heatmap_by_lift_share(analysis)
    print("  Sensitivity: e1c by T_source_in ...")
    plot_sensitivity_e1c_by_T_source_in(analysis)
    # Fixed mass flow mode plots
    print("  Fixed mass flow: T_source_in COP ...")
    plot_sensitivity_T_source_in_COP_fm(analysis)
    print("  Fixed mass flow: T_source_in c_P ...")
    plot_sensitivity_T_source_in_cP_fm(analysis)
    print("  Fixed mass flow: T_source_in epsilon ...")
    plot_sensitivity_T_source_in_epsilon_fm(analysis)
    print("  Mode comparison: COP ...")
    plot_comparison_modes_COP(analysis)
    print("  Mode comparison: c_P ...")
    plot_comparison_modes_cP(analysis)
    print("  Mass flow sensitivity: COP ...")
    plot_sensitivity_mass_flow_COP(analysis)
    print("  Mass flow sensitivity: c_P ...")
    plot_sensitivity_mass_flow_cP(analysis)
    print("  Mass flow sensitivity: epsilon ...")
    plot_sensitivity_mass_flow_epsilon(analysis)
    # Per-scenario diagrams
    print("  Q-T diagrams ...")
    plot_qt_diagrams(simulations, analysis)
    print("  Log(p)-h diagrams ...")
    plot_logph_diagrams(simulations, analysis)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Loading cached data ...")
    sims = load_simulations()
    analysis = load_analysis()
    print_summary_table(analysis)
    print("\nGenerating plots ...")
    generate_all_plots(analysis, sims)
    print("Done.")


if __name__ == "__main__":
    main()
