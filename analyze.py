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
    BASE_E1_C,
    BASE_FULL_LOAD_HOURS,
    BASE_GAS_C,
    E1_C_RANGE,
    FLUIDS_C1,
    FLUIDS_C2,
    LIFT_SHARE_DEFAULT,
    LIFT_SHARE_RANGE,
    NumpyEncoder,
    RESULTS_DIR,
    SOURCE_MASS_FLOW,
    SOURCE_MASS_FLOW_RANGE,
    T_SOURCE_IN_DEFAULT,
    T_SOURCE_IN_RANGE,
    lift_share_to_T34,
)
from economics import run_economics, run_economics_gas_heater, run_economics_heater
from simulate import load_simulations


# ── Helpers ──────────────────────────────────────────────────────────────────

def _out_path(*parts):
    """Build a path under ``RESULTS_DIR``, creating parent directories."""
    p = os.path.join(RESULTS_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _scenario_folder(f1, f2, ls, T_source_in):
    """Return the two-level folder path for a specific scenario."""
    ls_pct = int(round(ls * 100))
    return (f"{f1}_{f2}", f"LS_{ls_pct}_Tsrc_{T_source_in}")


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
    heater_sim = simulations["heater"]
    gas_heater_sim = simulations["gas_heater"]

    # ── Heater reference ─────────────────────────────────────────────────
    heater_base = run_economics_heater(heater_sim, BASE_FULL_LOAD_HOURS, BASE_E1_C)
    heater_ref = {
        "c_P": heater_base["c_P"],
        "Z_sum": heater_base["Z_sum"],
        "COP": heater_sim["COP"],
        "epsilon": heater_sim["epsilon"],
        "E_F": heater_sim["E_F"],
        "E_P": heater_sim["E_P"],
        "E_D": heater_sim["E_D"],
    }
    print(f"  Heater (ref): COP={heater_ref['COP']:.3f}  "
          f"epsilon={heater_ref['epsilon']:.4f}  c_P={heater_ref['c_P']:.2f}")

    # ── Gas heater reference ─────────────────────────────────────────────
    gas_heater_base = run_economics_gas_heater(gas_heater_sim, BASE_FULL_LOAD_HOURS, BASE_GAS_C)
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

    # ── Heater sensitivity: c_P vs electricity price ─────────────────────
    heater_sens_e1c = []
    gas_heater_sens_e1c_val = run_economics_gas_heater(gas_heater_sim, BASE_FULL_LOAD_HOURS, BASE_GAS_C)["c_P"]
    gas_heater_sens_e1c = [gas_heater_sens_e1c_val] * len(E1_C_RANGE)
    for e1c in E1_C_RANGE:
        heater_sens_e1c.append(run_economics_heater(heater_sim, BASE_FULL_LOAD_HOURS, e1c)["c_P"])

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
        "heater_ref": heater_ref,
        "gas_heater_ref": gas_heater_ref,
        "base_results": base_results,
        "valid_combos": valid_combos,
        "sensitivity_e1c": sens_e1c,
        "heater_sens_e1c": heater_sens_e1c,
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
    out["heater_ref"] = data["heater_ref"]
    out["heater_sens_e1c"] = data["heater_sens_e1c"]
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
    data["heater_ref"] = raw["heater_ref"]
    data["heater_sens_e1c"] = raw["heater_sens_e1c"]
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
