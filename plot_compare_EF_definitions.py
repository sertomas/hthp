"""
plot_compare_EF_definitions.py — Compare the two exergy-fuel definitions.

Reads the economics pipeline outputs of BOTH results trees
(``results/ef_outlet_loss/`` and ``results/ef_outlet_fuel/``, see
``config.EF_DEFINITIONS``) and quantifies what the choice of system
boundary for the source-water outlet (stream 12) does to the exergy
bookkeeping:

  * ``outlet_loss`` — 12 is always a LOSS:        E_F = E_e1 + E_11
  * ``outlet_fuel`` — 12 is a FUEL OUTPUT while it leaves at or above
    ambient (net water fuel):                     E_F = E_e1 + E_11 − E_12

For each T_steam one column figure is written (plus a merged summary CSV)
to ``results/ef_definition_compare/``:

  ``ef_compare_metrics_<T>`` — seven metric panels, each with paired
  columns per fluid pair (cost-optimal design): hatched = 12 → loss,
  solid = 12 → fuel output. E_D, C_D, Z and C_D+Z are stacked by
  component with the same palette as ``exergy_destruction_base``;
  ε_TOT, c_P TOT and LCOH are single columns. Only ε_TOT moves with the
  definition — the six identical column pairs make the invariance of the
  destruction/cost side visible at a glance.

Usage:  python plot_compare_EF_definitions.py
"""

import logging
import os
import sys
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    BASE_E1_C, R_N_EL, T_STEAMS_TO_RUN,
    ef_compare_dir, ef_results_dir,
)
from economics import _celf
from plot_common import (
    COL_DOUBLE_IN,
    GROUP_COLORS, GROUP_LABELS, GROUP_MEMBERS, GROUP_ORDER,
    save_titled_and_paper,
    fs,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


DEF_OLD = "outlet_loss"
DEF_NEW = "outlet_fuel"
DEF_LABELS = {
    DEF_OLD: "12 → loss   ($E_F = E_{e1} + E_{11}$)",
    DEF_NEW: "12 → fuel output   ($E_F = E_{e1} + E_{11} - E_{12}$)",
}
HATCH_OLD = "///"     # hatched columns = old definition, solid = new

# Same fleet order as plot_compare_T_steam / exergy_destruction_base.
PAIR_ORDER = ["R290/R600", "R290/R600a", "R1270/R600", "R1270/R600a",
              "R717/R600", "R717/R600a"]

# Levelized first-year electricity price [EUR/MWh] — same LCOH definition
# as plot_cdz_sensitivity_per_design.py:
#   LCOH [EUR/MWh_th] = 1000 · Ż_sum / Q_H + c_el,lev / COP
C_EL_LEV = BASE_E1_C * _celf(R_N_EL)

# Stacked panels: (title, exergoeco_components.csv column, y-label)
STACKED_METRICS = [
    ("$\\dot E_D$",       "E_D [kW]",       "$\\dot E_D$  [kW]"),
    ("$\\dot C_D$",       "C_D [EUR/h]",    "$\\dot C_D$  [EUR/h]"),
    ("$\\dot Z$",         "Z [EUR/h]",      "$\\dot Z$  [EUR/h]"),
    ("$\\dot C_D+\\dot Z$", "C_D+Z [EUR/h]", "$\\dot C_D+\\dot Z$  [EUR/h]"),
]
# Scalar panels: (title, row-extractor over the merged best-design row,
# y-label, annotation format)
SCALAR_METRICS = [
    ("$\\varepsilon_{TOT}$",
     lambda r, sfx: 100 * r[f"epsilon{sfx}"],
     "$\\varepsilon_{TOT}$  [%]", "{:.1f}"),
    ("$c_{P,TOT}$",
     lambda r, sfx: r[f"c_P [EUR/GJ]{sfx}"],
     "$c_{P,TOT}$  [EUR/GJ]", "{:.1f}"),
    ("LCOH",
     lambda r, sfx: (1000.0 * r[f"Z_sum [EUR/h]{sfx}"] / r[f"Q_H_kW{sfx}"]
                     + C_EL_LEV / r[f"COP{sfx}"]),
     "LCOH  [EUR/MWh$_{th}$]", "{:.0f}"),
]

# Quantities that must NOT change with the fuel definition.
INVARIANT_COLS = ["COP", "E_P [kW]", "E_D [kW]",
                  "c_P [EUR/GJ]", "Z_sum [EUR/h]", "PEC_total [kEUR]"]

MERGE_KEYS = ["f1", "f2", "pair", "ls", "T_src", "T_steam"]


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_base(defn, T_steam):
    """economics_base.csv of one definition tree at one T_steam (or None)."""
    path = os.path.join(ef_results_dir(defn), f"case_steam_{int(T_steam)}",
                        "data", "economics_base.csv")
    if not os.path.exists(path):
        print(f"  ! missing {path}")
        return None
    return pd.read_csv(path)


def load_merged():
    """Row-by-row merge of both definition trees over all T_steam."""
    frames = []
    for T in T_STEAMS_TO_RUN:
        old, new = _load_base(DEF_OLD, T), _load_base(DEF_NEW, T)
        if old is None or new is None:
            continue
        frames.append(old.merge(new, on=MERGE_KEYS, suffixes=("_old", "_new"),
                                validate="one_to_one"))
    if not frames:
        raise SystemExit("No overlapping economics_base.csv data found — "
                         "populate both results/ef_* trees first.")
    df = pd.concat(frames, ignore_index=True)
    df["d_eps_pp"] = 100 * (df["epsilon_new"] - df["epsilon_old"])
    df["E12_kW"] = df["E_F [kW]_old"] - df["E_F [kW]_new"]
    return df


def check_invariants(df):
    """Verify E_P, E_D and every cost metric is definition-independent."""
    print("Invariance check (quantities that must not move with E_F def):")
    for col in INVARIANT_COLS:
        dev = (df[f"{col}_old"] - df[f"{col}_new"]).abs().max()
        flag = "OK " if dev < 1e-6 else "FAIL"
        print(f"  {flag} max |Δ {col}| = {dev:.3g}")


def best_per_pair(df, T_steam):
    """Cheapest (lowest c_P) design per pair at one T_steam, fleet order."""
    sub = df[df["T_steam"] == T_steam]
    if sub.empty:
        return sub
    rows = sub.loc[sub.groupby("pair")["c_P [EUR/GJ]_old"].idxmin()]
    order = {p: i for i, p in enumerate(PAIR_ORDER)}
    return rows.sort_values("pair", key=lambda s: s.map(order))


def _load_components(defn, T_steam, r):
    """exergoeco_components.csv of one design in one tree (indexed, no TOT)."""
    path = os.path.join(ef_results_dir(defn), f"case_steam_{int(T_steam)}",
                        "designs", f"{r['f1']}_{r['f2']}",
                        f"LS{int(round(r['ls'] * 100))}_Tsrc{int(r['T_src'])}",
                        "exergoeco_components.csv")
    if not os.path.exists(path):
        return None
    comps = pd.read_csv(path, index_col="Component")
    return comps[comps.index != "TOT"]


# ── Column figure ────────────────────────────────────────────────────────────

def _paired_x(n_pairs):
    """x positions of the (old, new) column pair per fluid pair."""
    x = np.arange(n_pairs, dtype=float)
    w = 0.38
    return x - w / 2, x + w / 2, w


def _draw_stacked(ax, bests, comps_old, comps_new, col):
    """One stacked metric: hatched old / solid new columns per pair."""
    x_old, x_new, w = _paired_x(len(bests))
    tops = np.zeros(len(bests))
    for xs, comps_by_pair, hatch in ((x_old, comps_old, HATCH_OLD),
                                     (x_new, comps_new, None)):
        bottoms = np.zeros(len(bests))
        for group in GROUP_ORDER:
            vals = np.array([
                sum(float(c.loc[m, col]) for m in GROUP_MEMBERS[group]
                    if c is not None and m in c.index)
                if c is not None else 0.0
                for c in comps_by_pair
            ])
            ax.bar(xs, vals, w, bottom=bottoms, color=GROUP_COLORS[group],
                   edgecolor="black", linewidth=0.3, hatch=hatch)
            bottoms += vals
        tops = np.maximum(tops, bottoms)
    y_max = tops.max() * 1.18 if tops.max() > 0 else 1.0
    for xi, t in zip(x_new, tops):
        ax.text(xi, t + y_max * 0.015, f"{t:.0f}", ha="center", va="bottom",
                fontsize=fs(6))
    ax.set_ylim(0, y_max)


def _draw_scalar(ax, bests, value_fn, fmt):
    """One scalar metric: neutral hatched/solid column pair per fluid pair.

    The new-definition value is always annotated; the old one only when it
    differs (i.e. on the ε panel), so invariant metrics stay uncluttered.
    """
    x_old, x_new, w = _paired_x(len(bests))
    v_old = np.array([value_fn(r, "_old") for _, r in bests.iterrows()])
    v_new = np.array([value_fn(r, "_new") for _, r in bests.iterrows()])
    ax.bar(x_old, v_old, w, color="#b0b0b0", edgecolor="black",
           linewidth=0.3, hatch=HATCH_OLD)
    ax.bar(x_new, v_new, w, color="#b0b0b0", edgecolor="black",
           linewidth=0.3)
    # Vertical paired labels when the metric responds to the definition,
    # one horizontal label per pair when it is invariant.
    differs = np.abs(v_new - v_old) > 0.05 * np.maximum(np.abs(v_new), 1e-12)
    y_max = max(v_old.max(), v_new.max()) * (1.30 if differs.any() else 1.18)
    for xo, xn, vo, vn, d in zip(x_old, x_new, v_old, v_new, differs):
        if d:
            ax.text(xn, vn + y_max * 0.015, fmt.format(vn), ha="center",
                    va="bottom", fontsize=fs(6), rotation=90)
            ax.text(xo, vo + y_max * 0.015, fmt.format(vo), ha="center",
                    va="bottom", fontsize=fs(6), rotation=90, color="0.45")
        else:
            ax.text(xn, vn + y_max * 0.015, fmt.format(vn), ha="center",
                    va="bottom", fontsize=fs(6))
    ax.set_ylim(0, y_max)


def plot_metric_columns(df, T_steam):
    """Seven metric panels of paired (old | new definition) columns."""
    bests = best_per_pair(df, T_steam)
    if bests.empty:
        return
    comps_old = [_load_components(DEF_OLD, T_steam, r)
                 for _, r in bests.iterrows()]
    comps_new = [_load_components(DEF_NEW, T_steam, r)
                 for _, r in bests.iterrows()]
    xticklabels = [f"{r['pair']}\nLS{r['ls']:.1f} / {r['T_src']:.0f} °C"
                   for _, r in bests.iterrows()]
    x = np.arange(len(bests))

    fig, axes = plt.subplots(2, 4, figsize=(COL_DOUBLE_IN, 5.4),
                             constrained_layout=True)

    for ax, (title, col, ylabel) in zip(axes.flat[:4], STACKED_METRICS):
        _draw_stacked(ax, bests, comps_old, comps_new, col)
        ax.set_title(title, fontsize=fs(9))
        ax.set_ylabel(ylabel, fontsize=fs(8))

    for ax, (title, value_fn, ylabel, fmt) in zip(axes.flat[4:7],
                                                  SCALAR_METRICS):
        _draw_scalar(ax, bests, value_fn, fmt)
        ax.set_title(title, fontsize=fs(9))
        ax.set_ylabel(ylabel, fontsize=fs(8))

    for ax in axes.flat[:7]:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=fs(6))
        ax.grid(axis="y", alpha=0.25, lw=0.4)
        ax.tick_params(axis="y", labelsize=fs(7))

    # Legend panel: component palette + definition hatch key
    leg_ax = axes.flat[7]
    leg_ax.set_axis_off()
    comp_handles = [mpatches.Patch(color=GROUP_COLORS[g],
                                   label=GROUP_LABELS[g])
                    for g in GROUP_ORDER]
    def_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.4,
                       hatch=HATCH_OLD, label=DEF_LABELS[DEF_OLD]),
        mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.4,
                       label=DEF_LABELS[DEF_NEW]),
    ]
    leg1 = leg_ax.legend(handles=def_handles, loc="upper left",
                         fontsize=fs(6.5), frameon=False,
                         title="$E_F$ definition", title_fontsize=fs(7),
                         alignment="left")
    leg_ax.add_artist(leg1)
    leg_ax.legend(handles=comp_handles, loc="lower left", ncols=2,
                  fontsize=fs(6.5), frameon=False,
                  title="Component", title_fontsize=fs(7),
                  alignment="left")

    fig.suptitle(
        rf"$E_F$ definition comparison — cost-optimal design per pair, "
        rf"$T_\mathrm{{steam}} = {int(T_steam)}$ °C "
        rf"(only $\varepsilon_{{TOT}}$ responds; all destruction and cost "
        rf"metrics are invariant)",
        fontsize=fs(9))
    save_titled_and_paper(fig, ef_compare_dir(),
                          f"ef_compare_metrics_{int(T_steam)}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir = ef_compare_dir()
    os.makedirs(out_dir, exist_ok=True)

    df = load_merged()
    print(f"Merged {len(df)} designs over T_steam = "
          f"{[int(t) for t in sorted(df['T_steam'].unique())]}")
    check_invariants(df)
    print(f"Δε range: +{df['d_eps_pp'].min():.1f} … "
          f"+{df['d_eps_pp'].max():.1f} pp "
          f"(E_12 up to {df['E12_kW'].max():.0f} kW).")

    summary_cols = MERGE_KEYS + [
        "epsilon_old", "epsilon_new", "d_eps_pp",
        "E_F [kW]_old", "E_F [kW]_new", "E12_kW",
        "E_P [kW]_old", "E_D [kW]_old", "c_P [EUR/GJ]_old", "COP_old",
    ]
    summary = df[summary_cols].rename(columns={
        "E_P [kW]_old": "E_P [kW]", "E_D [kW]_old": "E_D [kW]",
        "c_P [EUR/GJ]_old": "c_P [EUR/GJ]", "COP_old": "COP"})
    summary_path = os.path.join(out_dir, "ef_compare_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary CSV → {summary_path}")

    for T in sorted(df["T_steam"].unique()):
        print(f"Plotting ef_compare_metrics_{int(T)} ...")
        plot_metric_columns(df, T)
    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
