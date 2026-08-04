"""
export_design_details.py — Per-design TESPy + exergoeconomic export.

For every feasible design at LS ∈ {0.30, 0.40, 0.50} of the case study,
at the requested T_steam, re-run simulate_hthp and dump:

    results/case_steam_<int(T_steam)>/designs/<f1>_<f2>/LS<XX>_Tsrc<YY>/
        connections.csv                    — TESPy state per labeled conn
                                              (m, T, p, h, s, v, e_T/e_M/e_PH)
        components.csv                     — TESPy component parameters
                                              (P, Q, pr, eta_s, kA, ttd, ...)
        qt_diagram.pdf                     — Q-T profiles for SRC_HX, IHX, SNK_HX
        logph_diagram.pdf                  — log(p)-h via fluprodia (1-200 bar)
        exergoeco_components.csv           — C_F, C_P, C_D, Z, c_F, c_P, f, r
        exergoeco_connections_material.csv — exergy + cost (C^T, C^M, C^TOT, c^T,...)
        exergoeco_connections_nonmat.csv   — power/heat connections (C^TOT, c^TOT)

Economics is run at native simulation scale (m_steam = ``config.M_STEAM``
kg/s; Q_H = M_STEAM · Δh_sat,water(T_steam)) with BASE_FULL_LOAD_HOURS /
BASE_E1_C from config — same scale that case_steam_economics.py uses for
the headline c_P numbers.

T_steam defaults to T_STEAM_CASE_DEFAULT (110 °C) when ``main(T_steam=None)``
is invoked.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    BASE_E1_C, BASE_FULL_LOAD_HOURS,
    T_STEAM_CASE_DEFAULT, lift_share_to_T34,
)
from economics import run_economics
from models import simulate_hthp


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


# Path globals set by ``_set_paths_for_T_steam`` for each T_steam case.
CASE_DIR = ""
ENRICHED_CSV = ""
DESIGNS_DIR = ""
_T_STEAM_CURRENT = 0.0


def _set_paths_for_T_steam(T_steam):
    """Rewrite the module-level path globals for a given T_steam."""
    global CASE_DIR, ENRICHED_CSV, DESIGNS_DIR, _T_STEAM_CURRENT
    from config import case_results_dir
    CASE_DIR = case_results_dir(T_steam)
    ENRICHED_CSV = os.path.join(CASE_DIR, f"case_steam_{int(T_steam)}_enriched.csv")
    DESIGNS_DIR = os.path.join(CASE_DIR, "designs")
    _T_STEAM_CURRENT = T_steam

# Map connection label → fluid (used for header annotation in CSV)
def _fluid_for_label(label, f1, f2):
    s = str(label)
    if s.startswith("1") or s in ("11", "12", "13"):
        return "water (source)"
    if s.startswith("2"):
        return f1
    if s.startswith("3"):
        return f2
    if s.startswith("4"):
        return "water (sink)"
    return ""


# ── Connections export ──────────────────────────────────────────────────────

def _connections_dataframe(sim):
    f1 = sim["fluid_cycle1"]
    f2 = sim["fluid_cycle2"]
    conns = sim["exerpy_data"]["connections"]
    rows = []
    for label, c in conns.items():
        if c.get("kind") != "material":
            # power connections, control etc. — skipped
            continue
        T_K = c.get("T")
        p_pa = c.get("p")
        h_J = c.get("h")
        s_J = c.get("s")
        v_m3kg = c.get("v")
        m = c.get("m")
        rows.append({
            "label": str(label),
            "fluid": _fluid_for_label(label, f1, f2),
            "from": c.get("source_component", ""),
            "to": c.get("target_component", ""),
            "m [kg/s]": round(m, 5) if m is not None else "",
            "T [°C]": round(T_K - 273.15, 2) if T_K is not None else "",
            "p [bar]": round(p_pa / 1e5, 4) if p_pa is not None else "",
            "h [kJ/kg]": round(h_J / 1e3, 3) if h_J is not None else "",
            "s [kJ/(kg·K)]": round(s_J / 1e3, 4) if s_J is not None else "",
            "v [m³/kg]": round(v_m3kg, 6) if v_m3kg is not None else "",
            "rho [kg/m³]": round(1.0 / v_m3kg, 3) if v_m3kg else "",
            "V_dot [m³/h]": round(m * v_m3kg * 3600, 2) if (m and v_m3kg) else "",
            "e_T [J/kg]": round(c.get("e_T", 0.0), 2),
            "e_M [J/kg]": round(c.get("e_M", 0.0), 2),
            "e_PH [J/kg]": round(c.get("e_PH", 0.0), 2),
        })
    df = pd.DataFrame(rows)
    # Sort by label numerically where possible (cycle1 first, then cycle2)
    def _sort_key(s):
        try:
            return (int(str(s).rstrip("c")), 1 if str(s).endswith("c") else 0)
        except Exception:
            return (10**6, 0)
    df["_k"] = df["label"].apply(_sort_key)
    df = df.sort_values("_k").drop(columns=["_k"])
    return df


# ── Components export ───────────────────────────────────────────────────────

# Parameters worth reporting per component type (parameter, divisor for unit
# normalisation, output column header).
_COMP_PARAMS = {
    "Compressor": [
        ("P",       1e3, "P [kW]"),
        ("pr",      1.0, "pressure ratio"),
        ("dp",      1.0, "Δp [bar]"),
        ("eta_s",   1.0, "η_s"),
    ],
    "HeatExchanger": [
        ("Q",       1e3, "Q [kW]"),
        ("kA",      1e3, "kA [kW/K]"),
        ("UA",      1e3, "UA [kW/K]"),
        ("ttd_u",   1.0, "ΔT_upper [K]"),
        ("ttd_l",   1.0, "ΔT_lower [K]"),
        ("td_pinch",1.0, "ΔT_pinch [K]"),
        ("td_log",  1.0, "LMTD [K]"),
        ("pr1",     1.0, "pr_hot"),
        ("pr2",     1.0, "pr_cold"),
        ("eff_hot", 1.0, "ε_hot"),
        ("eff_cold",1.0, "ε_cold"),
    ],
    "Motor": [
        ("P_in",    1e3, "P_in [kW]"),
        ("P_out",   1e3, "P_out [kW]"),
        ("eta",     1.0, "η"),
    ],
    "Pump": [
        ("P",       1e3, "P [kW]"),
        ("pr",      1.0, "pressure ratio"),
        ("dp",      1.0, "Δp [bar]"),
        ("eta_s",   1.0, "η_s"),
    ],
    "Valve": [
        ("pr",      1.0, "pressure ratio"),
        ("dp",      1.0, "Δp [bar]"),
    ],
    "PowerBus": [
        ("P_in_total",  1e3, "P_in_total [kW]"),
        ("P_out_total", 1e3, "P_out_total [kW]"),
    ],
}


def _components_dataframe(sim):
    comps = sim["exerpy_data"]["components"]
    rows = []
    for ctype, units in comps.items():
        cfg = _COMP_PARAMS.get(ctype)
        if cfg is None:
            continue
        for name, comp in units.items():
            params = comp.get("parameters", {})
            row = {"name": name, "type": ctype}
            for key, divisor, header in cfg:
                v = params.get(key)
                if v is None:
                    row[header] = ""
                    continue
                try:
                    row[header] = round(float(v) / divisor, 4)
                except Exception:
                    row[header] = v
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Order: Compressor, Pump, HeatExchanger, Motor, Valve, others
    type_order = ["Compressor", "Pump", "HeatExchanger", "Motor", "Valve",
                  "PowerBus", "CycleCloser"]
    df["_k"] = df["type"].map({t: i for i, t in enumerate(type_order)}).fillna(99)
    df = df.sort_values(["_k", "name"]).drop(columns=["_k"])
    return df


# ── Q-T plot ────────────────────────────────────────────────────────────────

HX_NAMES = ["SRC_HX", "IHX", "SNK_HX"]
# Two-line titles so each fits a ~2.5 in panel at full-page print width.
HX_TITLES = {
    "SRC_HX": "Source HX\n(cycle-1 evaporator)",
    "IHX":    "Internal HX\n(cycle-1 cond. / cycle-2 evap.)",
    "SNK_HX": "Sink HX\n(cycle-2 condenser, steam)",
}


# log(p)-h plotting via fluprodia. Subcritical isoline set (saturation dome,
# isobars, isotherms, isenthalps, isentropes) with the y-axis pinned to a fixed
# window (LOGPH_P_MIN_BAR - LOGPH_P_MAX_BAR = 1-200 bar) so every design plots
# on the same scale.

LOGPH_P_MIN_BAR = 1.0
LOGPH_P_MAX_BAR = 200.0

_diagram_cache = {}


def _get_diagram(fluid):
    """Return a cached FluidPropertyDiagram (computed on first call)."""
    from fluprodia import FluidPropertyDiagram
    if fluid not in _diagram_cache:
        d = FluidPropertyDiagram(fluid)
        d.set_unit_system(T="°C", p="bar", h="kJ/kg", s="kJ/kgK")
        T_crit = d.convert_from_SI(d.T_crit, "T")
        d.set_isolines_subcritical(T_min=-40, T_max=200)
        d.calc_isolines()
        _diagram_cache[fluid] = d
    return _diagram_cache[fluid]


def _plot_logph_one_cycle(fig, ax, fluid, points, cycle_label):
    """Draw fluprodia isolines + cycle loop on one log(p)-h axis."""
    diagram = _get_diagram(fluid)
    h_vals = [pt["h"] for pt in points]
    p_vals = [pt["p"] for pt in points]
    h_margin = (max(h_vals) - min(h_vals)) * 0.3

    diagram.draw_isolines(
        fig=fig, ax=ax, diagram_type="logph",
        x_min=min(h_vals) - h_margin, x_max=max(h_vals) + h_margin,
        y_min=LOGPH_P_MIN_BAR, y_max=LOGPH_P_MAX_BAR,
    )

    # The figure is authored at the final print width (see _plot_logph), so
    # these point sizes are the printed sizes. Isoline labels sit at the 6 pt
    # floor and the cycle annotations get the 9 pt body size.
    for txt in ax.texts:
        txt.set_fontsize(6)

    # Closed cycle loop (last point → first)
    h_cycle = h_vals + [h_vals[0]]
    p_cycle = p_vals + [p_vals[0]]
    ax.plot(h_cycle, p_cycle, "r-", linewidth=1.3, zorder=3)
    ax.plot(h_vals, p_vals, "ko", markersize=4, zorder=4)

    for pt in points:
        ax.annotate(pt["label"], (pt["h"], pt["p"]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=9, fontweight="bold")

    ax.set_ylim(LOGPH_P_MIN_BAR, LOGPH_P_MAX_BAR)
    # Journal point sizes (figure is authored at print width, so 1:1).
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.set_title(cycle_label, fontsize=10, pad=6)


def _plot_logph(sim, out_path, title_extra=""):
    """Two-panel log(p)-h diagram (cycle 1 LP / cycle 2 HP) via fluprodia."""
    f1 = sim["fluid_cycle1"]
    f2 = sim["fluid_cycle2"]
    states = sim["cycle_states"]

    # Author at full-page width (190 mm = 7.48 in) so the figure imports
    # into LaTeX at 1:1 and the point sizes set below are the printed sizes.
    fig, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(7.48, 3.6))
    _plot_logph_one_cycle(fig, ax_c1, f1, states["cycle1"]["points"],
                          f"Cycle 1 ({f1})")
    _plot_logph_one_cycle(fig, ax_c2, f2, states["cycle2"]["points"],
                          f"Cycle 2 ({f2})")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_qt(sim, out_path, title_extra=""):
    # Authored at full-page print width (190 mm = 7.48 in) so point sizes
    # are the printed sizes.
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.9), squeeze=False)
    axes = axes[0]
    for ax, hx_name in zip(axes, HX_NAMES):
        qt = sim["qt_sections"][hx_name]
        Q = np.asarray(qt["Q"])
        T_hot = np.asarray(qt["T_hot"])
        T_cold = np.asarray(qt["T_cold"])
        ax.plot(Q, T_hot, "r-o", markersize=3, linewidth=1.0, label="Hot side")
        ax.plot(Q, T_cold, "b-o", markersize=3, linewidth=1.0, label="Cold side")
        ax.fill_between(Q, T_cold, T_hot, alpha=0.10, color="gray")
        ax.set_xlabel("Q [kW]", fontsize=9)
        ax.set_ylabel("T [°C]", fontsize=9)
        ax.set_title(HX_TITLES[hx_name], fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Q-T diagrams — {title_extra}", fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ── Exergoeconomic export ───────────────────────────────────────────────────

def _export_exergoeco_csvs(sim, out_dir):
    """Run exergoeconomics at native scale and dump 3 CSVs.

    Returns the (c_P, Z_sum) headline pair for logging, or (None, None)
    if economics could not be solved for this design.
    """
    eco = run_economics(sim, BASE_FULL_LOAD_HOURS, BASE_E1_C / 10.0)
    if eco is None:
        return None, None
    exergoeco = eco["exergoeco"]
    df_comp, df_mat1, df_mat2, df_non_mat = exergoeco.exergoeconomic_results(
        print_results=False
    )

    # Merge df_mat1 (state + exergy) and df_mat2 (cost) on Connection so that
    # all per-connection material info sits in one CSV. Drop duplicated
    # columns (E, e^PH, ...).
    dup_cols = [c for c in df_mat2.columns
                if c in df_mat1.columns and c != "Connection"]
    df_mat = df_mat1.merge(df_mat2.drop(columns=dup_cols),
                           on="Connection", how="left")

    df_comp.to_csv(os.path.join(out_dir, "exergoeco_components.csv"),
                   index=False, float_format="%.4g")
    df_mat.to_csv(os.path.join(out_dir, "exergoeco_connections_material.csv"),
                  index=False, float_format="%.4g")
    df_non_mat.to_csv(os.path.join(out_dir, "exergoeco_connections_nonmat.csv"),
                       index=False, float_format="%.4g")
    return eco["c_P"], eco["Z_sum"]


# ── Main loop ───────────────────────────────────────────────────────────────

def main(T_steam=None):
    """Export per-design TESPy + exergoeconomic details for one T_steam case."""
    if T_steam is None:
        T_steam = T_STEAM_CASE_DEFAULT
    _set_paths_for_T_steam(T_steam)

    if not os.path.exists(ENRICHED_CSV):
        print(f"Could not find {ENRICHED_CSV}. "
              f"Run `python main.py --t-steam {int(T_steam)}` first.")
        sys.exit(1)
    os.makedirs(DESIGNS_DIR, exist_ok=True)

    df = pd.read_csv(ENRICHED_CSV)
    # Every thermodynamically-feasible design (status_4state != NOSOLVE). The
    # Ommen p / T_disch / V̇ limits are informational only and do not exclude;
    # this matches the set case_steam_economics.py runs economics on.
    designs = df[(df["status_4state"] != "NOSOLVE")
                 & (df["ls"].isin([0.30, 0.40, 0.50]))].copy()
    designs = designs.sort_values(by=["f1", "f2", "ls", "T_src"]).reset_index(drop=True)
    n = len(designs)
    print(f"Exporting per-design TESPy details for {n} thermodynamically-"
          f"feasible designs at LS ∈ {{0.30, 0.40, 0.50}} "
          f"(T_steam = {T_steam:.0f} °C)\n")

    n_ok = 0
    n_fail = 0
    n_eco_fail = 0
    for i, design in designs.iterrows():
        f1, f2 = design["f1"], design["f2"]
        ls = float(design["ls"])
        T_src = float(design["T_src"])
        ls_pct = int(round(ls * 100))
        tag = f"[{i+1:>3d}/{n}] {f1}/{f2:<6s} LS={ls_pct:>2d}% T_src={int(T_src):>2d}"

        out_dir = os.path.join(DESIGNS_DIR, f"{f1}_{f2}",
                               f"LS{ls_pct}_Tsrc{int(T_src)}")
        os.makedirs(out_dir, exist_ok=True)

        T34 = lift_share_to_T34(ls, T_src, T_steam=T_steam)
        sim = simulate_hthp(
            f1, f2,
            T_evap_c2_override=T34,
            T_source_in_override=T_src,
            T_steam_override=T_steam,
            source_mode="fixed_mass_flow",
            skip_ommen_check=True,
        )
        if sim is None:
            print(f"{tag}  SIM FAILED (skipped)")
            n_fail += 1
            continue

        try:
            conn_df = _connections_dataframe(sim)
            conn_df.to_csv(os.path.join(out_dir, "connections.csv"), index=False)

            comp_df = _components_dataframe(sim)
            comp_df.to_csv(os.path.join(out_dir, "components.csv"), index=False)

            title = (f"{f1}/{f2}, LS = {ls_pct}%, T_src = {int(T_src)} °C, "
                     f"T_steam = {int(T_steam)} °C  |  COP = {sim['COP']:.2f}, "
                     f"ε = {sim['epsilon']:.3f}")
            _plot_qt(sim, os.path.join(out_dir, "qt_diagram.pdf"), title_extra=title)
            _plot_logph(sim, os.path.join(out_dir, "logph_diagram.pdf"), title_extra=title)

            c_P, Z_sum = _export_exergoeco_csvs(sim, out_dir)
            if c_P is None:
                n_eco_fail += 1

            n_ok += 1
            if (i + 1) % 12 == 0 or (i + 1) == n:
                eco_str = (f"  c_P={c_P:.1f} EUR/GJ  Z={Z_sum:.2f} EUR/h"
                            if c_P is not None else "  (eco failed)")
                print(f"{tag}  [OK]  COP={sim['COP']:.2f}{eco_str}  -> {out_dir}")
        except Exception as e:
            print(f"{tag}  EXPORT FAILED: {e}")
            n_fail += 1

    print()
    print("=" * 80)
    print(f"Designs exported successfully: {n_ok}")
    print(f"Designs failed (sim/export)   : {n_fail}")
    print(f"Designs with eco failure      : {n_eco_fail}")
    print(f"Output root                   : {DESIGNS_DIR}/")
    print("Per design: connections.csv, components.csv, qt_diagram.pdf, "
          "logph_diagram.pdf,")
    print("           exergoeco_components.csv, "
          "exergoeco_connections_material.csv,")
    print("           exergoeco_connections_nonmat.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
