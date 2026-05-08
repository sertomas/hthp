"""
Orchestrator — case-study pipeline for steam at 100 / 110 / 120 °C, native scale
(m_steam = ``config.M_STEAM`` kg/s, Q_H = M_STEAM · Δh_vap of water at T_steam).

Two modes:

1. Full / partial pipeline (cache-aware):

       python main.py                       # full pipeline, all stages
       python main.py --skip-screen         # reuse cached screen CSV
       python main.py --skip-economics      # reuse cached economics CSV
       python main.py --skip-export         # skip per-design dumps
       python main.py --only-plots          # only regenerate plots
       python main.py --no-export           # alias for --skip-export

2. Single-design fast path (~5 s):

       python main.py --design R290 R600 0.40 40
                                            # f1, f2, LS (0–1), T_src (°C)
       python main.py --list-designs        # print all available OK designs

Stages (each can be cached / skipped):
    1. Screen        — case_steam.py            (TESPy 150-case sweep, ~5 min)
    2. Reclassify    — reclassify_modern.py     (post-process, instant)
    3. Feasibility   — plot_case_steam.py       (Ommen + modern, instant)
    4. Economics     — case_steam_economics.py  (~3 min)
    5. Per-design    — export_design_details.py (~5 min)
                       (must run before Stage 6 — produces per-design
                       exergoeco_components.csv that several economics plots
                       depend on)
    6. Eco plots     — plot_case_steam_economics.py (instant)
    7. Cross-T       — plot_compare_T_steam.py  (after all T_steam cases, instant)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ── Per-T_steam path computation ──────────────────────────────────────────────
# The pipeline now supports multiple steam temperatures. Path constants are
# computed dynamically from the per-stage T_steam — see _paths_for(T_steam)
# below. Module-level constants are kept for backwards compatibility (e.g. the
# `--design` single-design fast path defaults to 110 °C).

def _paths_for(T_steam: float) -> dict:
    """Return all cached-file paths for a given case-study T_steam."""
    case_dir = os.path.join("results", f"case_steam_{int(T_steam)}")
    tag = f"case_steam_{int(T_steam)}"
    return {
        "CASE_DIR":      case_dir,
        "SCREEN_CSV":    os.path.join(case_dir, f"{tag}.csv"),
        "ENRICHED_CSV":  os.path.join(case_dir, f"{tag}_enriched.csv"),
        "ECON_BASE_CSV": os.path.join(case_dir, "economics", "economics_base.csv"),
        "DESIGNS_DIR":   os.path.join(case_dir, "designs"),
    }


# Default (legacy) 110 °C paths used by the single-design fast path
_default_paths  = _paths_for(110.0)
CASE_DIR        = _default_paths["CASE_DIR"]
SCREEN_CSV      = _default_paths["SCREEN_CSV"]
ENRICHED_CSV    = _default_paths["ENRICHED_CSV"]
ECON_BASE_CSV   = _default_paths["ECON_BASE_CSV"]
DESIGNS_DIR     = _default_paths["DESIGNS_DIR"]


# ── Stage runners (thin wrappers around each script's main()) ────────────────

def _t() -> float:
    return time.perf_counter()


def _banner(title: str) -> None:
    bar = "═" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def stage_screen(T_steam: float = 110.0):
    """1. TESPy 150-case feasibility screen."""
    from case_steam import main as fn
    fn(T_steam=T_steam)


def stage_reclassify(T_steam: float = 110.0):
    """2. Modern envelope reclassification + sorted CSVs."""
    from reclassify_modern import main as fn
    fn(T_steam=T_steam)


def stage_feasibility_plots(T_steam: float = 110.0):
    """3. Feasibility plots in both Ommen-strict and modern modes."""
    import importlib
    import plot_case_steam as plot_mod
    importlib.reload(plot_mod)
    plot_mod.ENVELOPE_MODE = "ommen"
    plot_mod.main(T_steam=T_steam)
    plot_mod.ENVELOPE_MODE = "modern"
    plot_mod.main(T_steam=T_steam)


def stage_economics(T_steam: float = 110.0):
    """4. Exergoeconomic analysis on OK designs at LS ∈ {0.30, 0.40, 0.50}."""
    from case_steam_economics import main as fn
    fn(T_steam=T_steam)


def stage_export(T_steam: float = 110.0):
    """5. Per-design connections / components / Q-T / log(p)-h.

    Runs BEFORE the economics plots so that the per-design
    exergoeco_components.csv files exist when stage_economics_plots tries
    to read them — otherwise the cost-breakdown / Tsatsaronis / ranking
    plots silently fall back to a placeholder.
    """
    from export_design_details import main as fn
    fn(T_steam=T_steam)


def stage_economics_plots(T_steam: float = 110.0):
    """6. c_P heatmap, PEC vs Annex 58, cost breakdown, sensitivities."""
    from plot_case_steam_economics import main as fn
    fn(T_steam=T_steam)


def stage_compare():
    """7. Cross-T_steam comparison plots (after all temperatures have run)."""
    from plot_compare_T_steam import main as fn
    fn()


def _run_stage(label: str, fn, force_skip: bool = False, skip_reason: str = "",
               *fn_args, **fn_kwargs):
    """Run a stage with timing and clear console framing."""
    if force_skip:
        print(f"\n── {label}  ❯  SKIPPED ({skip_reason})")
        return
    print(f"\n── {label}")
    t0 = _t()
    fn(*fn_args, **fn_kwargs)
    print(f"     done in {_t() - t0:.1f} s")


# ── Single-design fast path ──────────────────────────────────────────────────

def run_single_design(f1: str, f2: str, ls: float, T_src: float,
                      T_steam: float | None = None,
                      run_economics: bool = True,
                      verbose: bool = True) -> dict | None:
    """Run ONE design end-to-end and return a brief summary dict.

    Output: per-design folder with connections.csv, components.csv,
    qt_diagram.png, logph_diagram.png. If ``run_economics`` is True, also
    runs the exergoeconomic balance and prints c_P / Z_sum / TCI.
    """
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    from CoolProp.CoolProp import PropsSI

    from config import (
        BASE_E1_C, BASE_FULL_LOAD_HOURS, M_STEAM, T_STEAM_CASE_DEFAULT,
        lift_share_to_T34, p_water_for_T_steam,
    )
    from models import simulate_hthp
    from export_design_details import (
        _connections_dataframe, _components_dataframe,
        _plot_qt, _plot_logph,
    )

    if T_steam is None:
        T_steam = T_STEAM_CASE_DEFAULT

    ls_pct = int(round(ls * 100))
    out_dir = os.path.join(DESIGNS_DIR, f"{f1}_{f2}",
                           f"LS{ls_pct}_Tsrc{int(T_src)}")
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        _banner(f"Single design: {f1}/{f2}  LS={ls_pct}%  T_src={int(T_src)} °C  "
                f"T_steam={int(T_steam)} °C")

    t0 = _t()
    sim = simulate_hthp(
        f1, f2,
        T_evap_c2_override=lift_share_to_T34(ls, T_src, T_steam=T_steam),
        T_source_in_override=T_src,
        T_steam_override=T_steam,
        source_mode="fixed_mass_flow",
        skip_ommen_check=True,
    )
    if sim is None:
        print("  ! TESPy did not converge or design is thermodynamically infeasible.")
        return None
    sim_dt = _t() - t0

    title = (f"{f1}/{f2}, LS = {ls_pct}%, T_src = {int(T_src)} °C, "
             f"T_steam = {int(T_steam)} °C  |  COP = {sim['COP']:.2f}, "
             f"ε = {sim['epsilon']:.3f}")

    _connections_dataframe(sim).to_csv(
        os.path.join(out_dir, "connections.csv"), index=False)
    _components_dataframe(sim).to_csv(
        os.path.join(out_dir, "components.csv"), index=False)
    _plot_qt(sim, os.path.join(out_dir, "qt_diagram.png"), title_extra=title)
    _plot_logph(sim, os.path.join(out_dir, "logph_diagram.png"), title_extra=title)

    summary = {
        "f1": f1, "f2": f2, "ls": ls, "T_src": T_src, "T_steam": T_steam,
        "COP": sim["COP"], "epsilon": sim["epsilon"],
        "T_source_out": sim["T_source_out"],
        "out_dir": out_dir,
        "sim_seconds": sim_dt,
    }

    if run_economics:
        from economics import run_economics as _run_eco
        eco = _run_eco(sim, BASE_FULL_LOAD_HOURS, BASE_E1_C / 10.0)
        if eco is not None:
            # Native-scale TCI/kW (Q_H = M_STEAM · Δh_vap)
            p_pa = p_water_for_T_steam(T_steam) * 1e5
            dh = (PropsSI("H", "P", p_pa, "Q", 1, "water")
                  - PropsSI("H", "P", p_pa, "Q", 0, "water")) / 1e3
            Q_H_native_kW = dh * M_STEAM
            from case_steam_economics import _pec_native
            pec = _pec_native(sim)
            summary["c_P"] = eco["c_P"]
            summary["Z_sum"] = eco["Z_sum"]
            summary["Q_H_kW"] = Q_H_native_kW
            summary["TCI [EUR/kW]"] = pec["TOTAL"] / Q_H_native_kW

    if verbose:
        print(f"  TESPy converged ({sim_dt:.1f}s)  COP={sim['COP']:.3f}  "
              f"ε={sim['epsilon']:.3f}  T_src_out={sim['T_source_out']:.1f} °C")
        print(f"  Files written: {out_dir}/")
        print(f"    connections.csv  components.csv  qt_diagram.png  logph_diagram.png")
        if run_economics and "c_P" in summary:
            print(f"  c_P = {summary['c_P']:.2f} EUR/GJ   "
                  f"Z_sum = {summary['Z_sum']:.2f} EUR/h   "
                  f"Q_H = {summary['Q_H_kW']:.0f} kW   "
                  f"TCI/kW = {summary['TCI [EUR/kW]']:.0f} EUR/kW   "
                  f"(@ e1={BASE_E1_C} EUR/MWh, FLH={BASE_FULL_LOAD_HOURS} h/a)")

    return summary


def list_designs() -> None:
    """Print a table of all OK (modern) designs from the cached economics CSV."""
    import pandas as pd
    if not os.path.exists(ECON_BASE_CSV):
        print(f"! {ECON_BASE_CSV} missing — run the pipeline first.")
        sys.exit(1)
    df = pd.read_csv(ECON_BASE_CSV).sort_values(by=["c_P [EUR/GJ]"])
    cols = ["pair", "ls", "T_src", "COP", "eta_Lorenz",
            "c_P [EUR/GJ]", "PEC [EUR/kW]"]
    available = [c for c in cols if c in df.columns]
    print(f"\nAll OK (modern) designs at LS ∈ {{0.30, 0.40, 0.50}} — "
          f"{len(df)} total, sorted by c_P:\n")
    print(df[available].to_string(index=False))


# ── Pipeline orchestrator ────────────────────────────────────────────────────

def run_pipeline_for_T(args, T_steam: float) -> None:
    """Run the six-stage pipeline once for a single steam temperature."""
    paths = _paths_for(T_steam)
    SCREEN_CSV    = paths["SCREEN_CSV"]
    ENRICHED_CSV  = paths["ENRICHED_CSV"]
    ECON_BASE_CSV = paths["ECON_BASE_CSV"]

    from config import m_steam_label
    _banner(f"HTHP CASE STUDY — steam at {T_steam:.0f} °C, "
            f"native scale ({m_steam_label()})")

    if args.only_plots:
        if not os.path.exists(ENRICHED_CSV):
            print(f"! {ENRICHED_CSV} missing — run without --only-plots first.")
            return
        _run_stage("Stage 3: Feasibility plots", stage_feasibility_plots,
                   False, "", T_steam=T_steam)
        if os.path.exists(ECON_BASE_CSV):
            _run_stage("Stage 6: Economics plots", stage_economics_plots,
                       False, "", T_steam=T_steam)
        else:
            print("\n── Stage 6 skipped (no cached economics CSV)")
        return

    # Stage 1: TESPy 150-case screen
    skip_screen = args.skip_screen and os.path.exists(SCREEN_CSV)
    _run_stage("Stage 1: Feasibility screen (150 TESPy cases, ~5 min)",
               stage_screen,
               skip_screen, "cached CSV present", T_steam=T_steam)

    # Stage 2: reclassify (always; instant)
    _run_stage("Stage 2: Reclassify with modern envelope (instant)",
               stage_reclassify, False, "", T_steam=T_steam)

    # Stage 3: feasibility plots
    _run_stage("Stage 3: Feasibility plots — Ommen + modern (instant)",
               stage_feasibility_plots, False, "", T_steam=T_steam)

    # Stage 4: economics
    skip_eco = args.skip_economics and os.path.exists(ECON_BASE_CSV)
    _run_stage("Stage 4: Exergoeconomic analysis "
               "(OK designs at LS ∈ {0.30, 0.40, 0.50}, ~3 min)",
               stage_economics,
               skip_eco, "cached economics CSV present", T_steam=T_steam)

    # Stage 5: per-design export (run BEFORE plots so the cost-breakdown
    # / Tsatsaronis / ranking plots have access to the per-design
    # exergoeco_components.csv files; otherwise those plots get skipped).
    skip_export = args.skip_export or args.no_export
    _run_stage("Stage 5: Per-design exports — connections / components / "
               "qt / log(p)-h (~5 min)",
               stage_export,
               skip_export, "--skip-export / --no-export", T_steam=T_steam)

    # Stage 6: economics plots (depends on per-design exergoeco CSVs from
    # Stage 5 for the C_D + Z, Z-only, Tsatsaronis, ranking, and best-design
    # plots; falls back gracefully if Stage 5 was skipped).
    _run_stage("Stage 6: Economics plots (instant)",
               stage_economics_plots, False, "", T_steam=T_steam)


def run_pipeline(args) -> None:
    """Top-level driver. Loops over the requested T_steam values and, when
    all three are present, builds the cross-T_steam comparison plots."""
    from config import T_STEAMS_TO_RUN

    if args.t_steam == "all":
        t_list = list(T_STEAMS_TO_RUN)
    else:
        t_list = [float(args.t_steam)]

    for T in t_list:
        run_pipeline_for_T(args, T)

    # Cross-T_steam comparison only when all three temperatures are now
    # present on disk (either ran them now or they were already cached).
    if all(os.path.exists(_paths_for(T)["ECON_BASE_CSV"]) for T in T_STEAMS_TO_RUN):
        _banner("Cross-T_steam comparison")
        _run_stage("Stage 7: Cross-T_steam comparison plots", stage_compare)
    else:
        missing = [int(T) for T in T_STEAMS_TO_RUN
                   if not os.path.exists(_paths_for(T)["ECON_BASE_CSV"])]
        print(f"\n── Stage 7 (comparison) skipped — economics CSV missing for "
              f"T_steam ∈ {missing}. Re-run with --t-steam all to populate.")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    from config import m_steam_label as _msl
    parser = argparse.ArgumentParser(
        description="HTHP case-study pipeline orchestrator "
                    f"(steam ∈ {{100, 110, 120}} °C, native scale {_msl()}, "
                    "modern envelope).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples
--------
  python main.py                                     all 3 temperatures (~24 min)
  python main.py --t-steam 110                       only the 110 °C case (~8 min)
  python main.py --t-steam all --skip-screen         all 3, reuse cached screens
  python main.py --skip-economics                    reuse cached economics
  python main.py --skip-export                       skip per-design dumps
  python main.py --only-plots                        plots only, from cache
  python main.py --design R290 R600 0.40 40          single design (~5 s, T=110)
  python main.py --list-designs                      list all OK designs (T=110)
""",
    )

    # Single-design / list mode (mutually exclusive with pipeline flags)
    parser.add_argument("--design", nargs=4,
                        metavar=("F1", "F2", "LS", "T_SRC"),
                        help="Run ONE design end-to-end (~5 s) and exit.")
    parser.add_argument("--list-designs", action="store_true",
                        help="Print the table of all OK (modern) designs and exit.")

    # Per-T_steam selection
    parser.add_argument("--t-steam", default="all",
                        choices=["100", "110", "120", "all"],
                        help="Which steam temperature(s) to run. "
                             "Default 'all' loops over 100/110/120 °C and "
                             "writes the cross-T_steam comparison plots.")

    # Pipeline stage flags
    parser.add_argument("--skip-screen", action="store_true",
                        help="Reuse cached 150-case screen CSV.")
    parser.add_argument("--skip-economics", action="store_true",
                        help="Reuse cached economics CSVs.")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip the per-design export stage.")
    parser.add_argument("--no-export", action="store_true",
                        help="Alias for --skip-export.")
    parser.add_argument("--only-plots", action="store_true",
                        help="Regenerate plots only, from cached CSVs.")

    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ── Single-design fast path ────────────────────────────────────────────
    if args.list_designs:
        list_designs()
        return

    if args.design:
        f1, f2, ls_str, T_src_str = args.design
        try:
            ls_val = float(ls_str)
            T_src_val = float(T_src_str)
        except ValueError:
            print(f"! Could not parse LS={ls_str} or T_src={T_src_str} as numbers.")
            sys.exit(1)
        if not 0.0 < ls_val < 1.0:
            print(f"! LS must be in (0, 1); got {ls_val}. "
                  f"For a 40 % lift share use 0.40, not 40.")
            sys.exit(1)
        run_single_design(f1, f2, ls_val, T_src_val)
        return

    # ── Full / partial pipeline ────────────────────────────────────────────
    t0 = _t()
    run_pipeline(args)
    print(f"\nTotal pipeline runtime: {(_t() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
