"""
Stage 1 — Run TESPy simulations and cache results to disk.

This is the most expensive step of the pipeline.  Results are serialised
as JSON files (one folder per simulation) so that the analysis and plotting
stages can be re-run without repeating the TESPy solves.

Usage
-----
::

    python simulate.py                                    # all combos
    python simulate.py R290 R600a                         # single, defaults
    python simulate.py R290 R600a --lift-share 0.50       # custom lift share
    python simulate.py R290 R600a --T-source-in 40        # custom T_source_in
    python simulate.py R290 R600a --lift-share 0.50 --T-source-in 40
"""

import argparse
import json
import logging
import os

from config import (
    CACHE_DIR,
    FLUIDS_C1,
    FLUIDS_C2,
    GAS_HEATER_CACHE_FILE,
    HEATER_CACHE_FILE,
    LIFT_SHARE_DEFAULT,
    LIFT_SHARE_RANGE,
    NumpyEncoder,
    SIM_CACHE_DIR,
    SIM_INDEX_FILE,
    SOURCE_MASS_FLOW,
    SOURCE_MASS_FLOW_RANGE,
    T_SOURCE_IN_DEFAULT,
    T_SOURCE_IN_RANGE,
    lift_share_to_T34,
    sim_cache_folder,
    sim_key_to_str,
    str_to_sim_key,
)
from models import reconstruct_ean, simulate_gas_heater, simulate_heater, simulate_hthp

logging.basicConfig(level=logging.WARNING)
logging.disable(logging.CRITICAL)


def _fm_index_file(m_val):
    """Return the sim-index path for a given fixed mass flow value."""
    return os.path.join(CACHE_DIR, f"sim_index_fm_{int(m_val)}.json")


# ── Core helpers ─────────────────────────────────────────────────────────────

def _run_hthp_sweep(fluids_c1, fluids_c2, lift_share_values,
                     T_source_in_values, source_mode="fixed_delta_T",
                     m_source=None):
    """
    Run HTHP simulations for every (f1, f2, lift_share, T_source_in) combination
    with a given source constraint mode.

    Returns
    -------
    dict
        Mapping ``(f1, f2, lift_share, T_source_in) -> sim_dict | None``.
    """
    hthp = {}
    total = len(T_source_in_values) * len(lift_share_values) * len(fluids_c1) * len(fluids_c2)
    count = 0
    mode_tag = ""
    if source_mode != "fixed_delta_T":
        mode_tag = f" [{source_mode} m={m_source}]"
    for T_src_in in T_source_in_values:
        for ls in lift_share_values:
            T34 = lift_share_to_T34(ls, T_src_in)
            for f1 in fluids_c1:
                for f2 in fluids_c2:
                    count += 1
                    label = f"{f1}/{f2} LS={ls:.0%} T_source_in={T_src_in} (T34={T34:.1f}){mode_tag}"
                    print(f"  [{count}/{total}] Simulating: {label} ...")
                    hthp[(f1, f2, ls, T_src_in)] = simulate_hthp(
                        f1, f2,
                        T_evap_c2_override=T34,
                        T_source_in_override=T_src_in,
                        source_mode=source_mode,
                        m_source=m_source,
                    )
    return hthp


def run_all_simulations(
    fluids_c1=FLUIDS_C1,
    fluids_c2=FLUIDS_C2,
    lift_share_values=LIFT_SHARE_RANGE,
    T_source_in_values=T_SOURCE_IN_RANGE,
):
    """
    Run HTHP simulations for every (f1, f2, lift_share, T_source_in) combination
    in both source constraint modes, plus heater references.

    For fixed_mass_flow mode, simulations are run for each value in
    SOURCE_MASS_FLOW_RANGE.

    Returns
    -------
    dict
        ``"hthp"`` — fixed_delta_T results;
        ``"hthp_fm"`` — ``{m_val: {(f1, f2, ls, T_src): sim, ...}}``;
        ``"heater"`` / ``"gas_heater"`` — reference cases.
    """
    print("  Simulating: Electrical Heater (reference) ...")
    heater = simulate_heater()

    print("  Simulating: Gas Heater (reference) ...")
    gas_heater = simulate_gas_heater()

    print("\n--- Source mode: fixed_delta_T ---")
    hthp = _run_hthp_sweep(fluids_c1, fluids_c2, lift_share_values,
                            T_source_in_values, source_mode="fixed_delta_T")

    hthp_fm = {}
    for m_val in SOURCE_MASS_FLOW_RANGE:
        print(f"\n--- Source mode: fixed_mass_flow (m={m_val} kg/s) ---")
        hthp_fm[m_val] = _run_hthp_sweep(
            fluids_c1, fluids_c2, lift_share_values,
            T_source_in_values, source_mode="fixed_mass_flow",
            m_source=m_val,
        )

    return {"hthp": hthp, "hthp_fm": hthp_fm,
            "heater": heater, "gas_heater": gas_heater}


def run_single_simulation(f1, f2, lift_share=LIFT_SHARE_DEFAULT,
                          T_source_in=T_SOURCE_IN_DEFAULT,
                          source_mode="fixed_delta_T",
                          m_source=None):
    """
    Run a single HTHP simulation.

    Parameters
    ----------
    f1, f2 : str
        Cycle-1 and cycle-2 refrigerants.
    lift_share : float, optional
        Lower-cycle lift share fraction.
    T_source_in : float, optional
        Source water inlet temperature [deg C].
    source_mode : str, optional
        ``"fixed_delta_T"`` or ``"fixed_mass_flow"``.
    m_source : float, optional
        Source mass flow [kg/s] (only for fixed_mass_flow mode).

    Returns
    -------
    dict or None
        Simulation result (see ``simulate_hthp``), or ``None`` on failure.
    """
    T34 = lift_share_to_T34(lift_share, T_source_in)
    print(f"  Simulating: {f1}/{f2} LS={lift_share:.0%} T_source_in={T_source_in} "
          f"(T34={T34:.1f}) [{source_mode}] ...")
    return simulate_hthp(f1, f2, T_evap_c2_override=T34,
                         T_source_in_override=T_source_in,
                         source_mode=source_mode,
                         m_source=m_source)


# ── Persistence ──────────────────────────────────────────────────────────────

def _save_hthp_dict(hthp, index_file, source_mode="fixed_delta_T"):
    """Save one dict of HTHP simulation results (for a given source mode)."""
    index = {}
    for key, sim in hthp.items():
        f1, f2, ls, T_src = key
        folder = sim_cache_folder(f1, f2, ls, T_src, source_mode=source_mode)
        str_key = sim_key_to_str(key)

        if sim is None:
            index[str_key] = {"status": "failed", "folder": folder}
            continue

        os.makedirs(folder, exist_ok=True)

        exerpy_path = os.path.join(folder, "exerpy_data.json")
        with open(exerpy_path, "w") as f:
            json.dump(sim["exerpy_data"], f, cls=NumpyEncoder, indent=2)

        scalars = {k: v for k, v in sim.items()
                   if k not in ("ean", "exerpy_data")}
        scalars_path = os.path.join(folder, "scalars.json")
        with open(scalars_path, "w") as f:
            json.dump(scalars, f, cls=NumpyEncoder, indent=2)

        index[str_key] = {"status": "ok", "folder": folder}

    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)


def _load_hthp_dict(index_file):
    """Load one dict of HTHP simulation results from a given index file."""
    with open(index_file) as f:
        index = json.load(f)

    hthp = {}
    for str_key, meta in index.items():
        key = str_to_sim_key(str_key)

        if meta["status"] != "ok":
            hthp[key] = None
            continue

        folder = meta["folder"]
        scalars_path = os.path.join(folder, "scalars.json")
        exerpy_path = os.path.join(folder, "exerpy_data.json")

        with open(scalars_path) as f:
            sim = json.load(f)

        sim["ean"] = reconstruct_ean(exerpy_path, sim["T_source_in"])
        hthp[key] = sim

    return hthp


def save_simulations(data):
    """
    Save simulation results as JSON files (one folder per simulation).

    Parameters
    ----------
    data : dict
        Output of ``run_all_simulations``.
    """
    os.makedirs(SIM_CACHE_DIR, exist_ok=True)

    # fixed_delta_T simulations
    _save_hthp_dict(data["hthp"], SIM_INDEX_FILE, source_mode="fixed_delta_T")

    # fixed_mass_flow simulations (one index per mass flow value)
    for m_val, hthp_m in data["hthp_fm"].items():
        sm = f"fixed_mass_flow_m{int(m_val)}"
        _save_hthp_dict(hthp_m, _fm_index_file(m_val), source_mode=sm)

    # Heater
    with open(HEATER_CACHE_FILE, "w") as f:
        json.dump(data["heater"], f, cls=NumpyEncoder, indent=2)

    # Gas heater
    with open(GAS_HEATER_CACHE_FILE, "w") as f:
        json.dump(data["gas_heater"], f, cls=NumpyEncoder, indent=2)

    print(f"Simulations saved -> {SIM_CACHE_DIR}")


def load_simulations():
    """
    Load cached simulation results from JSON files.

    Returns
    -------
    dict
        Same structure as the output of ``run_all_simulations``.
    """
    hthp = _load_hthp_dict(SIM_INDEX_FILE)

    # Load fixed_mass_flow results for each m value
    hthp_fm = {}
    for m_val in SOURCE_MASS_FLOW_RANGE:
        idx = _fm_index_file(m_val)
        if os.path.exists(idx):
            hthp_fm[m_val] = _load_hthp_dict(idx)

    with open(HEATER_CACHE_FILE) as f:
        heater = json.load(f)

    with open(GAS_HEATER_CACHE_FILE) as f:
        gas_heater = json.load(f)

    print(f"Simulations loaded <- {SIM_CACHE_DIR}")
    return {"hthp": hthp, "hthp_fm": hthp_fm,
            "heater": heater, "gas_heater": gas_heater}


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run HTHP simulations")
    parser.add_argument("fluids", nargs="*", help="f1 f2 (e.g. R290 R600a)")
    parser.add_argument("--lift-share", type=float, default=None,
                        help="Lower-cycle lift share fraction (e.g. 0.50)")
    parser.add_argument("--T-source-in", type=float, default=None,
                        help="T_source_in override [°C]")
    parser.add_argument("--source-mode", choices=["fixed_delta_T", "fixed_mass_flow"],
                        default="fixed_delta_T",
                        help="Source water constraint mode")
    parser.add_argument("--m-source", type=float, default=None,
                        help="Source mass flow [kg/s] (fixed_mass_flow mode)")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.fluids:
        f1, f2 = args.fluids[0], args.fluids[1]
        ls = args.lift_share if args.lift_share is not None else LIFT_SHARE_DEFAULT
        T_src_in = args.T_source_in if args.T_source_in is not None else T_SOURCE_IN_DEFAULT
        sim = run_single_simulation(f1, f2, ls, T_src_in,
                                    source_mode=args.source_mode,
                                    m_source=args.m_source)
        if sim:
            print(f"  COP={sim['COP']:.3f}  epsilon={sim['epsilon']:.4f}"
                  f"  m_source={sim['m_source']:.3f} kg/s"
                  f"  T_source_out={sim['T_source_out']:.1f} °C")
        else:
            print("  Simulation failed or infeasible.")
    else:
        print("Running all simulations ...")
        data = run_all_simulations()
        save_simulations(data)
        n_ok = sum(1 for v in data["hthp"].values() if v is not None)
        n_total = len(data["hthp"])
        print(f"Done: fixed_delta_T {n_ok}/{n_total}")
        for m_val in sorted(data["hthp_fm"]):
            fm = data["hthp_fm"][m_val]
            n_ok_fm = sum(1 for v in fm.values() if v is not None)
            print(f"  fixed_mass_flow m={m_val} kg/s: {n_ok_fm}/{len(fm)}")


if __name__ == "__main__":
    main()
