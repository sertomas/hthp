"""
Stage 2 — Run economic / exergoeconomic analysis on cached simulations.

Loads the simulation JSON produced by ``simulate.py``, evaluates cost
balances for every combination under the base-case and alternative
economic parameters, computes all sensitivity sweeps, and exports
exergoeconomic CSV tables and an analysis JSON for the plotting stage.

Usage
-----
::

    python analyze.py            # uses cached simulations + default config
"""

import json
import os

import numpy as np

from config import (
    ANALYSIS_JSON_FILE,
    BASE_CO2_PRICE,
    BASE_E1_C,
    BASE_FULL_LOAD_HOURS,
    BASE_GAS_C,
    E1_C_RANGE,
    FLUIDS_C1,
    FLUIDS_C2,
    LIFT_SHARE_DEFAULT,
    LIFT_SHARE_RANGE,
    NumpyEncoder,
    PRICE_SENS_FRAC_RANGE,
    RESULTS_DIR,
    SOURCE_MASS_FLOW,
    SOURCE_MASS_FLOW_RANGE,
    T_SOURCE_IN_DEFAULT,
    T_SOURCE_IN_RANGE,
    T_STEAM_RANGE,
    lift_share_to_T34,
)
from economics import run_economics, run_economics_gas_heater
from simulate import load_simulations


# ── Helpers ──────────────────────────────────────────────────────────────────

def _out_path(*parts):
    """Build a path under ``RESULTS_DIR``, creating parent directories."""
    p = os.path.join(RESULTS_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _scenario_folder(f1, f2, ls, T_source_in):
    """Return the two-level folder path for a specific scenario.

    ``int(T_source_in)`` mirrors :func:`config.sim_cache_folder` so that the
    folder name is stable across runs — without the cast, a JSON roundtrip
    converts ``20`` to ``20.0`` and produces a duplicate sibling directory.
    """
    ls_pct = int(round(ls * 100))
    return (f"{f1}_{f2}", f"LS_{ls_pct}_Tsrc_{int(T_source_in)}")


# ── Main analysis routine ───────────────────────────────────────────────────

def run_all_analysis(simulations):
    """
    Run economics and sensitivity sweeps for all cached simulations.

    Parameters
    ----------
    simulations : dict
        Output of ``run_all_simulations``.

    Returns
    -------
    dict
        Analysis results for both source modes.
    """
    hthp = simulations["hthp"]
    hthp_fm = simulations.get("hthp_fm", {})
    gas_heater_sim = simulations["gas_heater"]

    # ── Gas heater reference ─────────────────────────────────────────────
    gas_heater_base = run_economics_gas_heater(gas_heater_sim, BASE_FULL_LOAD_HOURS, BASE_GAS_C,
                                                co2_price_eur_per_t=BASE_CO2_PRICE)
    gas_heater_ref = {
        "c_P": gas_heater_base["c_P"],
        "Z_sum": gas_heater_base["Z_sum"],
        "COP": gas_heater_sim["COP"],
        "epsilon": gas_heater_sim["epsilon"],
        "E_F": gas_heater_sim["E_F"],
        "E_P": gas_heater_sim["E_P"],
        "E_D": gas_heater_sim["E_D"],
    }
    print(f"  Gas Heater (ref): COP={gas_heater_ref['COP']:.3f}  "
          f"epsilon={gas_heater_ref['epsilon']:.4f}  c_P={gas_heater_ref['c_P']:.2f}")

    # ── Gas heater sensitivity: c_P vs electricity price ─────────────────
    # Gas heater c_P depends only on the gas price (retrofit ⇒ Z = 0), so the
    # curve is flat across the electricity-price sweep.
    gas_heater_sens_e1c_val = run_economics_gas_heater(gas_heater_sim, BASE_FULL_LOAD_HOURS, BASE_GAS_C,
                                                        co2_price_eur_per_t=BASE_CO2_PRICE)["c_P"]
    gas_heater_sens_e1c = [gas_heater_sens_e1c_val] * len(E1_C_RANGE)

    # ── Base-case results (default lift share, default T_source_in) ───────
    base_results = {}
    for f1 in FLUIDS_C1:
        for f2 in FLUIDS_C2:
            sim = hthp.get((f1, f2, LIFT_SHARE_DEFAULT, T_SOURCE_IN_DEFAULT))
            if sim is None:
                base_results[(f1, f2)] = None
                continue
            eco = run_economics(sim, BASE_FULL_LOAD_HOURS, BASE_E1_C)
            if eco is None:
                base_results[(f1, f2)] = None
                continue
            base_results[(f1, f2)] = {
                "c_P": eco["c_P"],
                "Z_sum": eco["Z_sum"],
                "COP": sim["COP"],
                "epsilon": sim["epsilon"],
                "E_F": sim["E_F"],
                "E_P": sim["E_P"],
                "E_D": sim["E_D"],
            }
            print(f"  {f1}/{f2}: COP={sim['COP']:.3f}  epsilon={sim['epsilon']:.4f}  "
                  f"c_P={eco['c_P']:.2f}  Z_sum={eco['Z_sum']:.2f}")

    # ── Valid combos (base lift share, base T_source_in) ─────────────────
    valid_combos = [
        (f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2
        if hthp.get((f1, f2, LIFT_SHARE_DEFAULT, T_SOURCE_IN_DEFAULT)) is not None
    ]

    # ── Sensitivity: c_P vs electricity price (base hours, fixed) ────────
    print("  Sensitivity: c_P vs electricity price ...")
    sens_e1c = {k: [] for k in valid_combos}
    for e1c in E1_C_RANGE:
        for key in valid_combos:
            eco = run_economics(hthp[(key[0], key[1], LIFT_SHARE_DEFAULT, T_SOURCE_IN_DEFAULT)], BASE_FULL_LOAD_HOURS, e1c)
            sens_e1c[key].append(eco["c_P"] if eco else np.nan)

    # ── Lift share sensitivity (at default T_source_in) ───────────────────
    print("  Lift share sensitivity ...")
    sens_lift_share = {}
    for ls in LIFT_SHARE_RANGE:
        T34 = lift_share_to_T34(ls, T_SOURCE_IN_DEFAULT)
        for f1 in FLUIDS_C1:
            for f2 in FLUIDS_C2:
                sim = hthp.get((f1, f2, ls, T_SOURCE_IN_DEFAULT))
                if sim is None:
                    sens_lift_share[(f1, f2, ls)] = None
                    continue
                eco = run_economics(sim, BASE_FULL_LOAD_HOURS, BASE_E1_C)
                sens_lift_share[(f1, f2, ls)] = {
                    "COP": sim["COP"],
                    "epsilon": sim["epsilon"],
                    "c_P": eco["c_P"] if eco else np.nan,
                    "T34": T34,
                }

                # Save exergoeconomic CSV tables
                if eco is not None:
                    scenario = _scenario_folder(f1, f2, ls, T_SOURCE_IN_DEFAULT)
                    df_comp, df_mat1, df_mat2, df_nmat = eco["exergoeco"].exergoeconomic_results(print_results=False)
                    df_comp.to_csv(_out_path(*scenario, "components.csv"), index=False)
                    df_mat1.to_csv(_out_path(*scenario, "connections_exergy.csv"), index=False)
                    df_mat2.to_csv(_out_path(*scenario, "connections_costs.csv"), index=False)
                    df_nmat.to_csv(_out_path(*scenario, "nonmaterial.csv"), index=False)

    # ── Lift share × economic sensitivities (at default T_source_in) ──────
    print("  Lift share × economic sweeps ...")
    valid_combos_lift_share = {}
    for ls in LIFT_SHARE_RANGE:
        valid_combos_lift_share[ls] = [
            (f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2
            if (sens_lift_share.get((f1, f2, ls)) is not None
                and not np.isnan(sens_lift_share[(f1, f2, ls)].get("c_P", np.nan)))
        ]

    sens_e1c_by_lift_share = {}
    for ls in LIFT_SHARE_RANGE:
        combos = valid_combos_lift_share[ls]
        e1c_data = {k: [] for k in combos}
        for e1c in E1_C_RANGE:
            for key in combos:
                eco = run_economics(hthp[(key[0], key[1], ls, T_SOURCE_IN_DEFAULT)], BASE_FULL_LOAD_HOURS, e1c)
                e1c_data[key].append(eco["c_P"] if eco else np.nan)
        sens_e1c_by_lift_share[ls] = e1c_data

    # ── T_source_in sensitivity (at default lift share) ───────────────────
    print("  T_source_in sensitivity ...")
    sens_T_source_in = {}
    for T_src_val in T_SOURCE_IN_RANGE:
        T34 = lift_share_to_T34(LIFT_SHARE_DEFAULT, T_src_val)
        for f1 in FLUIDS_C1:
            for f2 in FLUIDS_C2:
                sim = hthp.get((f1, f2, LIFT_SHARE_DEFAULT, T_src_val))
                if sim is None:
                    sens_T_source_in[(f1, f2, T_src_val)] = None
                    continue
                eco = run_economics(sim, BASE_FULL_LOAD_HOURS, BASE_E1_C)
                sens_T_source_in[(f1, f2, T_src_val)] = {
                    "COP": sim["COP"],
                    "epsilon": sim["epsilon"],
                    "c_P": eco["c_P"] if eco else np.nan,
                    "T34": T34,
                }

                # Save exergoeconomic CSV tables
                if eco is not None:
                    scenario = _scenario_folder(f1, f2, LIFT_SHARE_DEFAULT, T_src_val)
                    df_comp, df_mat1, df_mat2, df_nmat = eco["exergoeco"].exergoeconomic_results(print_results=False)
                    df_comp.to_csv(_out_path(*scenario, "components.csv"), index=False)
                    df_mat1.to_csv(_out_path(*scenario, "connections_exergy.csv"), index=False)
                    df_mat2.to_csv(_out_path(*scenario, "connections_costs.csv"), index=False)
                    df_nmat.to_csv(_out_path(*scenario, "nonmaterial.csv"), index=False)

    # ── T_source_in × economic sensitivities (at default lift share) ──────
    print("  T_source_in × economic sweeps ...")
    valid_combos_T_source_in = {}
    for T_src_val in T_SOURCE_IN_RANGE:
        valid_combos_T_source_in[T_src_val] = [
            (f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2
            if (sens_T_source_in.get((f1, f2, T_src_val)) is not None
                and not np.isnan(sens_T_source_in[(f1, f2, T_src_val)].get("c_P", np.nan)))
        ]

    sens_e1c_by_T_source_in = {}
    for T_src_val in T_SOURCE_IN_RANGE:
        combos = valid_combos_T_source_in[T_src_val]
        e1c_data = {k: [] for k in combos}
        for e1c in E1_C_RANGE:
            for key in combos:
                eco = run_economics(hthp[(key[0], key[1], LIFT_SHARE_DEFAULT, T_src_val)], BASE_FULL_LOAD_HOURS, e1c)
                e1c_data[key].append(eco["c_P"] if eco else np.nan)
        sens_e1c_by_T_source_in[T_src_val] = e1c_data

    # ── T_source_in sensitivity by lift share ─────────────────────────────
    print("  T_source_in sensitivity by lift share ...")
    sens_T_source_in_by_lift_share = {}
    for ls in LIFT_SHARE_RANGE:
        ls_data = {}
        for T_src_val in T_SOURCE_IN_RANGE:
            T34 = lift_share_to_T34(ls, T_src_val)
            for f1 in FLUIDS_C1:
                for f2 in FLUIDS_C2:
                    sim = hthp.get((f1, f2, ls, T_src_val))
                    if sim is None:
                        ls_data[(f1, f2, T_src_val)] = None
                        continue
                    eco = run_economics(sim, BASE_FULL_LOAD_HOURS, BASE_E1_C)
                    ls_data[(f1, f2, T_src_val)] = {
                        "COP": sim["COP"],
                        "epsilon": sim["epsilon"],
                        "c_P": eco["c_P"] if eco else np.nan,
                        "T34": T34,
                    }
        sens_T_source_in_by_lift_share[ls] = ls_data

    # ── Fixed mass flow mode ─────────────────────────────────────────────
    # hthp_fm is nested: {m_val: {(f1, f2, ls, T_src): sim, ...}}
    # Use SOURCE_MASS_FLOW (40 kg/s) as the base case.
    hthp_fm_base = hthp_fm.get(SOURCE_MASS_FLOW, {})

    # ── Fixed mass flow: T_source_in sensitivity (base mass flow) ──────
    print("  Fixed mass flow: T_source_in sensitivity ...")
    sens_T_source_in_fm = {}
    if hthp_fm_base:
        for T_src_val in T_SOURCE_IN_RANGE:
            T34 = lift_share_to_T34(LIFT_SHARE_DEFAULT, T_src_val)
            for f1 in FLUIDS_C1:
                for f2 in FLUIDS_C2:
                    sim = hthp_fm_base.get((f1, f2, LIFT_SHARE_DEFAULT, T_src_val))
                    if sim is None:
                        sens_T_source_in_fm[(f1, f2, T_src_val)] = None
                        continue
                    eco = run_economics(sim, BASE_FULL_LOAD_HOURS, BASE_E1_C)
                    sens_T_source_in_fm[(f1, f2, T_src_val)] = {
                        "COP": sim["COP"],
                        "epsilon": sim["epsilon"],
                        "c_P": eco["c_P"] if eco else np.nan,
                        "T34": T34,
                        "T_source_out": sim.get("T_source_out"),
                        "m_source": sim.get("m_source"),
                    }

    valid_combos_T_source_in_fm = {}
    for T_src_val in T_SOURCE_IN_RANGE:
        valid_combos_T_source_in_fm[T_src_val] = [
            (f1, f2) for f1 in FLUIDS_C1 for f2 in FLUIDS_C2
            if (sens_T_source_in_fm.get((f1, f2, T_src_val)) is not None
                and not np.isnan(sens_T_source_in_fm[(f1, f2, T_src_val)].get("c_P", np.nan)))
        ]

    # ── Fixed mass flow: e1c sensitivity (base mass flow) ─────────────
    print("  Fixed mass flow: e1c sweeps ...")
    sens_e1c_by_T_source_in_fm = {}
    if hthp_fm_base:
        for T_src_val in T_SOURCE_IN_RANGE:
            combos = valid_combos_T_source_in_fm[T_src_val]
            e1c_data = {k: [] for k in combos}
            for e1c in E1_C_RANGE:
                for key in combos:
                    eco = run_economics(hthp_fm_base[(key[0], key[1], LIFT_SHARE_DEFAULT, T_src_val)], BASE_FULL_LOAD_HOURS, e1c)
                    e1c_data[key].append(eco["c_P"] if eco else np.nan)
            sens_e1c_by_T_source_in_fm[T_src_val] = e1c_data

    # ── Fixed mass flow: base-case results (base mass flow) ───────────
    base_results_fm = {}
    if hthp_fm_base:
        for f1 in FLUIDS_C1:
            for f2 in FLUIDS_C2:
                sim = hthp_fm_base.get((f1, f2, LIFT_SHARE_DEFAULT, T_SOURCE_IN_DEFAULT))
                if sim is None:
                    base_results_fm[(f1, f2)] = None
                    continue
                eco = run_economics(sim, BASE_FULL_LOAD_HOURS, BASE_E1_C)
                if eco is None:
                    base_results_fm[(f1, f2)] = None
                    continue
                base_results_fm[(f1, f2)] = {
                    "c_P": eco["c_P"],
                    "Z_sum": eco["Z_sum"],
                    "COP": sim["COP"],
                    "epsilon": sim["epsilon"],
                    "E_F": sim["E_F"],
                    "E_P": sim["E_P"],
                    "E_D": sim["E_D"],
                    "T_source_out": sim.get("T_source_out"),
                    "m_source": sim.get("m_source"),
                }

    # ── Fixed mass flow: lift share sensitivity (base mass flow) ──────
    print("  Fixed mass flow: lift share sensitivity ...")
    sens_lift_share_fm = {}
    if hthp_fm_base:
        for ls in LIFT_SHARE_RANGE:
            T34 = lift_share_to_T34(ls, T_SOURCE_IN_DEFAULT)
            for f1 in FLUIDS_C1:
                for f2 in FLUIDS_C2:
                    sim = hthp_fm_base.get((f1, f2, ls, T_SOURCE_IN_DEFAULT))
                    if sim is None:
                        sens_lift_share_fm[(f1, f2, ls)] = None
                        continue
                    eco = run_economics(sim, BASE_FULL_LOAD_HOURS, BASE_E1_C)
                    sens_lift_share_fm[(f1, f2, ls)] = {
                        "COP": sim["COP"],
                        "epsilon": sim["epsilon"],
                        "c_P": eco["c_P"] if eco else np.nan,
                        "T34": T34,
                        "T_source_out": sim.get("T_source_out"),
                        "m_source": sim.get("m_source"),
                    }

    # ── ±50 % electricity / gas price 2-D sensitivity ─────────────────────
    # Mirrors Ommen et al. (2015), Fig. 3a/b: vary c_el and c_gas around their
    # base values in [-50 %, +50 %] and record c_P for HTHP combos and the
    # gas-heater reference. Used downstream to plot a contour map and the
    # break-even (c_P_HTHP = c_P_gas) curve.
    print("  ±50 % price 2-D sensitivity ...")
    e1c_grid = [BASE_E1_C * (1 + f) for f in PRICE_SENS_FRAC_RANGE]
    gas_grid = [BASE_GAS_C * (1 + f) for f in PRICE_SENS_FRAC_RANGE]

    # Gas heater c_P depends only on c_gas (Z = 0 ⇒ no electricity term).
    # CO2 charge is held fixed at BASE_CO2_PRICE across the gas-price sweep
    # so the curve isolates the gas-fuel sensitivity.
    gas_heater_grid_2d = []
    for gc in gas_grid:
        row = run_economics_gas_heater(gas_heater_sim, BASE_FULL_LOAD_HOURS, gc,
                                        co2_price_eur_per_t=BASE_CO2_PRICE)
        gas_heater_grid_2d.append(row["c_P"])

    # HTHP c_P depends only on c_el (gas price doesn't enter run_economics),
    # so we only need a 1-D sweep over electricity prices per fluid combo.
    # We still expose a 2-D-shaped result for plotting symmetry with the
    # gas heater grid.
    hthp_cP_grid_2d = {}  # {(f1, f2): [c_P over e1c_grid]}
    for f1, f2 in valid_combos:
        sim = hthp[(f1, f2, LIFT_SHARE_DEFAULT, T_SOURCE_IN_DEFAULT)]
        cps = []
        for e1c in e1c_grid:
            eco = run_economics(sim, BASE_FULL_LOAD_HOURS, e1c)
            cps.append(eco["c_P"] if eco else np.nan)
        hthp_cP_grid_2d[(f1, f2)] = cps

    price_sens_2d = {
        "frac_range": list(PRICE_SENS_FRAC_RANGE),
        "e1c_grid": e1c_grid,
        "gas_grid": gas_grid,
        "gas_heater_cP": gas_heater_grid_2d,   # length = len(gas_grid)
        "hthp_cP": hthp_cP_grid_2d,            # {(f1, f2): [len(e1c_grid)]}
        "base_e1c": BASE_E1_C,
        "base_gas": BASE_GAS_C,
    }

    # ── T_steam sensitivity (at default LS / T_source_in) ────────────────
    # Reads sims["hthp_T_steam"] (keyed by (f1, f2, T_steam)) and per-T_steam
    # gas-heater results, computes c_P and gas-heater c_P at base prices.
    print("  T_steam sensitivity ...")
    hthp_T_steam_sims = simulations.get("hthp_T_steam", {})
    gas_T_steam_sims = simulations.get("gas_heater_T_steam", {})
    sens_T_steam = {}
    gas_heater_T_steam = {}
    for T_steam in T_STEAM_RANGE:
        print(f"  --- T_steam = {T_steam:.0f} °C ---")
        gh_sim = gas_T_steam_sims.get(T_steam)
        if gh_sim is not None:
            gh_eco = run_economics_gas_heater(gh_sim, BASE_FULL_LOAD_HOURS, BASE_GAS_C,
                                                co2_price_eur_per_t=BASE_CO2_PRICE)
            gas_heater_T_steam[T_steam] = {
                "c_P": gh_eco["c_P"],
                "Z_sum": gh_eco["Z_sum"],
                "COP": gh_sim["COP"],
                "epsilon": gh_sim["epsilon"],
            }
            print(f"    Gas heater (ref): COP={gh_sim['COP']:.3f}  "
                  f"epsilon={gh_sim['epsilon']:.4f}  "
                  f"c_P={gh_eco['c_P']:.2f}")
        for f1 in FLUIDS_C1:
            for f2 in FLUIDS_C2:
                sim = hthp_T_steam_sims.get((f1, f2, T_steam))
                if sim is None:
                    sens_T_steam[(f1, f2, T_steam)] = None
                    continue
                eco = run_economics(sim, BASE_FULL_LOAD_HOURS, BASE_E1_C)
                sens_T_steam[(f1, f2, T_steam)] = {
                    "COP": sim["COP"],
                    "epsilon": sim["epsilon"],
                    "c_P": eco["c_P"] if eco else np.nan,
                }
                cp = eco["c_P"] if eco else float("nan")
                print(f"    {f1}/{f2:6s}: COP={sim['COP']:.3f}  "
                      f"epsilon={sim['epsilon']:.4f}  c_P={cp:.2f}")

    # ── Mass flow sensitivity (across SOURCE_MASS_FLOW_RANGE) ─────────
    print("  Mass flow sensitivity ...")
    sens_mass_flow = {}
    for m_val in SOURCE_MASS_FLOW_RANGE:
        hthp_m = hthp_fm.get(m_val, {})
        if not hthp_m:
            continue
        for T_src_val in T_SOURCE_IN_RANGE:
            T34 = lift_share_to_T34(LIFT_SHARE_DEFAULT, T_src_val)
            for f1 in FLUIDS_C1:
                for f2 in FLUIDS_C2:
                    sim = hthp_m.get((f1, f2, LIFT_SHARE_DEFAULT, T_src_val))
                    if sim is None:
                        sens_mass_flow[(f1, f2, T_src_val, m_val)] = None
                        continue
                    eco = run_economics(sim, BASE_FULL_LOAD_HOURS, BASE_E1_C)
                    sens_mass_flow[(f1, f2, T_src_val, m_val)] = {
                        "COP": sim["COP"],
                        "epsilon": sim["epsilon"],
                        "c_P": eco["c_P"] if eco else np.nan,
                        "T34": T34,
                        "T_source_out": sim.get("T_source_out"),
                        "m_source": sim.get("m_source"),
                    }

    # ── Pack everything ──────────────────────────────────────────────────
    return {
        "gas_heater_ref": gas_heater_ref,
        "base_results": base_results,
        "valid_combos": valid_combos,
        "sensitivity_e1c": sens_e1c,
        "gas_heater_sens_e1c": gas_heater_sens_e1c,
        "sensitivity_lift_share": sens_lift_share,
        "valid_combos_lift_share": valid_combos_lift_share,
        "sensitivity_e1c_by_lift_share": sens_e1c_by_lift_share,
        "sensitivity_T_source_in": sens_T_source_in,
        "valid_combos_T_source_in": valid_combos_T_source_in,
        "sensitivity_e1c_by_T_source_in": sens_e1c_by_T_source_in,
        "sensitivity_T_source_in_by_lift_share": sens_T_source_in_by_lift_share,
        # Fixed mass flow mode
        "base_results_fm": base_results_fm,
        "sensitivity_T_source_in_fm": sens_T_source_in_fm,
        "valid_combos_T_source_in_fm": valid_combos_T_source_in_fm,
        "sensitivity_e1c_by_T_source_in_fm": sens_e1c_by_T_source_in_fm,
        "sensitivity_lift_share_fm": sens_lift_share_fm,
        "sensitivity_mass_flow": sens_mass_flow,
        "price_sens_2d": price_sens_2d,
        "sensitivity_T_steam": sens_T_steam,
        "gas_heater_T_steam": gas_heater_T_steam,
    }


# ── JSON key encoding helpers ────────────────────────────────────────────────

def _encode_tuple_key(k):
    """Encode a tuple key as a pipe-separated string for JSON."""
    return "|".join(str(x) for x in k)


def _decode_pair_key(s):
    """Decode a ``"f1|f2"`` string back to a ``(f1, f2)`` tuple."""
    parts = s.split("|")
    return (parts[0], parts[1])


def _decode_triple_key(s):
    """Decode a ``"f1|f2|ls"`` string back to a ``(f1, f2, float)`` tuple."""
    parts = s.split("|")
    return (parts[0], parts[1], float(parts[2]))


def _encode_dict_keys(d):
    """Convert a dict with tuple keys to string keys."""
    return {_encode_tuple_key(k): v for k, v in d.items()}


def _decode_dict_pair_keys(d):
    """Convert a dict with ``"f1|f2"`` string keys back to tuple keys."""
    return {_decode_pair_key(k): v for k, v in d.items()}


def _decode_dict_triple_keys(d):
    """Convert a dict with ``"f1|f2|val"`` string keys back to tuple keys."""
    return {_decode_triple_key(k): v for k, v in d.items()}


def _decode_quad_key(s):
    """Decode a ``"f1|f2|T_src|m"`` string back to a 4-tuple of (str, str, float, float)."""
    parts = s.split("|")
    return (parts[0], parts[1], float(parts[2]), float(parts[3]))


def _decode_dict_quad_keys(d):
    """Convert a dict with ``"f1|f2|val1|val2"`` string keys back to tuple keys."""
    return {_decode_quad_key(k): v for k, v in d.items()}


def _decode_list_of_pairs(lst):
    """Decode a list of ``["f1", "f2"]`` lists back to tuples."""
    return [(x[0], x[1]) for x in lst]


# ── Persistence ──────────────────────────────────────────────────────────────

def save_analysis(data):
    """
    Serialise analysis results to a JSON file.
    """
    out = {}

    # Scalars / simple dicts
    out["gas_heater_ref"] = data["gas_heater_ref"]
    out["gas_heater_sens_e1c"] = data["gas_heater_sens_e1c"]

    # (f1, f2) keyed dicts
    out["base_results"] = _encode_dict_keys(data["base_results"])
    out["valid_combos"] = [list(c) for c in data["valid_combos"]]
    out["sensitivity_e1c"] = _encode_dict_keys(data["sensitivity_e1c"])

    # (f1, f2, ls) keyed dicts
    out["sensitivity_lift_share"] = _encode_dict_keys(data["sensitivity_lift_share"])

    # {ls: [(f1,f2), ...]}
    out["valid_combos_lift_share"] = {
        str(ls): [list(c) for c in combos]
        for ls, combos in data["valid_combos_lift_share"].items()
    }

    # {ls: {(f1,f2): [...]}}
    out["sensitivity_e1c_by_lift_share"] = {
        str(ls): _encode_dict_keys(inner)
        for ls, inner in data["sensitivity_e1c_by_lift_share"].items()
    }

    # (f1, f2, T_source_in) keyed dicts
    out["sensitivity_T_source_in"] = _encode_dict_keys(data["sensitivity_T_source_in"])

    # {T_src: [(f1,f2), ...]}
    out["valid_combos_T_source_in"] = {
        str(T): [list(c) for c in combos]
        for T, combos in data["valid_combos_T_source_in"].items()
    }
    out["sensitivity_e1c_by_T_source_in"] = {
        str(T): _encode_dict_keys(inner)
        for T, inner in data["sensitivity_e1c_by_T_source_in"].items()
    }

    # {ls: {(f1, f2, T_src): result}}
    out["sensitivity_T_source_in_by_lift_share"] = {
        str(ls): _encode_dict_keys(inner)
        for ls, inner in data["sensitivity_T_source_in_by_lift_share"].items()
    }

    # Fixed mass flow mode
    out["base_results_fm"] = _encode_dict_keys(data["base_results_fm"])
    out["sensitivity_T_source_in_fm"] = _encode_dict_keys(data["sensitivity_T_source_in_fm"])
    out["valid_combos_T_source_in_fm"] = {
        str(T): [list(c) for c in combos]
        for T, combos in data["valid_combos_T_source_in_fm"].items()
    }
    out["sensitivity_e1c_by_T_source_in_fm"] = {
        str(T): _encode_dict_keys(inner)
        for T, inner in data["sensitivity_e1c_by_T_source_in_fm"].items()
    }
    out["sensitivity_lift_share_fm"] = _encode_dict_keys(data["sensitivity_lift_share_fm"])
    out["sensitivity_mass_flow"] = _encode_dict_keys(data["sensitivity_mass_flow"])

    # T_steam sensitivity (at default LS / T_source_in)
    out["sensitivity_T_steam"] = _encode_dict_keys(data["sensitivity_T_steam"])
    out["gas_heater_T_steam"] = {str(T): v for T, v in data["gas_heater_T_steam"].items()}

    # 2-D price sensitivity (±50 %)
    ps = data["price_sens_2d"]
    out["price_sens_2d"] = {
        "frac_range": ps["frac_range"],
        "e1c_grid": ps["e1c_grid"],
        "gas_grid": ps["gas_grid"],
        "gas_heater_cP": ps["gas_heater_cP"],
        "hthp_cP": _encode_dict_keys(ps["hthp_cP"]),
        "base_e1c": ps["base_e1c"],
        "base_gas": ps["base_gas"],
    }

    os.makedirs(os.path.dirname(ANALYSIS_JSON_FILE), exist_ok=True)
    with open(ANALYSIS_JSON_FILE, "w") as f:
        json.dump(out, f, cls=NumpyEncoder, indent=2)
    print(f"Analysis saved -> {ANALYSIS_JSON_FILE}")


def load_analysis():
    """
    Load cached analysis results from a JSON file.
    """
    with open(ANALYSIS_JSON_FILE) as f:
        raw = json.load(f)

    data = {}
    data["gas_heater_ref"] = raw["gas_heater_ref"]
    data["gas_heater_sens_e1c"] = raw["gas_heater_sens_e1c"]

    data["base_results"] = _decode_dict_pair_keys(raw["base_results"])
    data["valid_combos"] = _decode_list_of_pairs(raw["valid_combos"])
    data["sensitivity_e1c"] = _decode_dict_pair_keys(raw["sensitivity_e1c"])

    data["sensitivity_lift_share"] = _decode_dict_triple_keys(raw["sensitivity_lift_share"])

    data["valid_combos_lift_share"] = {
        float(ls): _decode_list_of_pairs(combos)
        for ls, combos in raw["valid_combos_lift_share"].items()
    }
    data["sensitivity_e1c_by_lift_share"] = {
        float(ls): _decode_dict_pair_keys(inner)
        for ls, inner in raw["sensitivity_e1c_by_lift_share"].items()
    }

    data["sensitivity_T_source_in"] = _decode_dict_triple_keys(raw["sensitivity_T_source_in"])
    data["valid_combos_T_source_in"] = {
        float(T): _decode_list_of_pairs(combos)
        for T, combos in raw["valid_combos_T_source_in"].items()
    }
    data["sensitivity_e1c_by_T_source_in"] = {
        float(T): _decode_dict_pair_keys(inner)
        for T, inner in raw["sensitivity_e1c_by_T_source_in"].items()
    }

    data["sensitivity_T_source_in_by_lift_share"] = {
        float(ls): _decode_dict_triple_keys(inner)
        for ls, inner in raw["sensitivity_T_source_in_by_lift_share"].items()
    }

    # Fixed mass flow mode
    data["base_results_fm"] = _decode_dict_pair_keys(raw.get("base_results_fm", {}))
    data["sensitivity_T_source_in_fm"] = _decode_dict_triple_keys(raw.get("sensitivity_T_source_in_fm", {}))
    data["valid_combos_T_source_in_fm"] = {
        float(T): _decode_list_of_pairs(combos)
        for T, combos in raw.get("valid_combos_T_source_in_fm", {}).items()
    }
    data["sensitivity_e1c_by_T_source_in_fm"] = {
        float(T): _decode_dict_pair_keys(inner)
        for T, inner in raw.get("sensitivity_e1c_by_T_source_in_fm", {}).items()
    }
    data["sensitivity_lift_share_fm"] = _decode_dict_triple_keys(raw.get("sensitivity_lift_share_fm", {}))
    data["sensitivity_mass_flow"] = _decode_dict_quad_keys(raw.get("sensitivity_mass_flow", {}))
    data["sensitivity_T_steam"] = _decode_dict_triple_keys(raw.get("sensitivity_T_steam", {}))
    data["gas_heater_T_steam"] = {float(k): v for k, v in raw.get("gas_heater_T_steam", {}).items()}

    # 2-D price sensitivity (±50 %) — may be missing from older analysis JSONs
    ps_raw = raw.get("price_sens_2d")
    if ps_raw is not None:
        data["price_sens_2d"] = {
            "frac_range": ps_raw["frac_range"],
            "e1c_grid": ps_raw["e1c_grid"],
            "gas_grid": ps_raw["gas_grid"],
            "gas_heater_cP": ps_raw["gas_heater_cP"],
            "hthp_cP": _decode_dict_pair_keys(ps_raw["hthp_cP"]),
            "base_e1c": ps_raw["base_e1c"],
            "base_gas": ps_raw["base_gas"],
        }
    else:
        data["price_sens_2d"] = None

    print(f"Analysis loaded <- {ANALYSIS_JSON_FILE}")
    return data


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Loading cached simulations ...")
    sims = load_simulations()
    print("Running analysis ...")
    analysis = run_all_analysis(sims)
    save_analysis(analysis)
    print("Done.")


if __name__ == "__main__":
    main()
