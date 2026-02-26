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
    FLUIDS_C1,
    FLUIDS_C2,
    GAS_HEATER_CACHE_FILE,
    HEATER_CACHE_FILE,
    LIFT_SHARE_DEFAULT,
    LIFT_SHARE_RANGE,
    NumpyEncoder,
    SIM_CACHE_DIR,
    SIM_INDEX_FILE,
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


# ── Core helpers ─────────────────────────────────────────────────────────────

def run_all_simulations(
    fluids_c1=FLUIDS_C1,
    fluids_c2=FLUIDS_C2,
    lift_share_values=LIFT_SHARE_RANGE,
    T_source_in_values=T_SOURCE_IN_RANGE,
):
    """
    Run HTHP simulations for every (f1, f2, lift_share, T_source_in) combination.

    The evaporation temperature T34 is computed from the lift share and the
    source water inlet temperature:
    ``T34 = T_source_in + lift_share * (T_steam - T_source_in)``.

    Parameters
    ----------
    fluids_c1 : list of str, optional
        Refrigerants for cycle 1 (lower).
    fluids_c2 : list of str, optional
        Refrigerants for cycle 2 (upper).
    lift_share_values : list of float, optional
        Lower-cycle lift share fractions to sweep.
    T_source_in_values : list of float, optional
        Source water inlet temperatures to sweep [deg C].

    Returns
    -------
    dict
        ``"hthp"`` — mapping ``(f1, f2, lift_share, T_source_in) -> sim_dict | None``;
        ``"heater"`` — heater simulation dict.
    """
    print("  Simulating: Electrical Heater (reference) ...")
    heater = simulate_heater()

    print("  Simulating: Gas Heater (reference) ...")
    gas_heater = simulate_gas_heater()

    hthp = {}
    total = len(T_source_in_values) * len(lift_share_values) * len(fluids_c1) * len(fluids_c2)
    count = 0
    for T_src_in in T_source_in_values:
        for ls in lift_share_values:
            T34 = lift_share_to_T34(ls, T_src_in)
            for f1 in fluids_c1:
                for f2 in fluids_c2:
                    count += 1
                    label = f"{f1}/{f2} LS={ls:.0%} T_source_in={T_src_in} (T34={T34:.1f})"
                    print(f"  [{count}/{total}] Simulating: {label} ...")
                    hthp[(f1, f2, ls, T_src_in)] = simulate_hthp(
                        f1, f2,
                        T_evap_c2_override=T34,
                        T_source_in_override=T_src_in,
                    )

    return {"hthp": hthp, "heater": heater, "gas_heater": gas_heater}


def run_single_simulation(f1, f2, lift_share=LIFT_SHARE_DEFAULT,
                          T_source_in=T_SOURCE_IN_DEFAULT):
    """
    Run a single HTHP simulation.

    Parameters
    ----------
    f1 : str
        Cycle-1 refrigerant.
    f2 : str
        Cycle-2 refrigerant.
    lift_share : float, optional
        Lower-cycle lift share fraction.
    T_source_in : float, optional
        Source water inlet temperature [deg C].

    Returns
    -------
    dict or None
        Simulation result (see ``simulate_hthp``), or ``None`` on failure.
    """
    T34 = lift_share_to_T34(lift_share, T_source_in)
    print(f"  Simulating: {f1}/{f2} LS={lift_share:.0%} T_source_in={T_source_in} (T34={T34:.1f}) ...")
    return simulate_hthp(f1, f2, T_evap_c2_override=T34, T_source_in_override=T_source_in)


# ── Persistence ──────────────────────────────────────────────────────────────

def save_simulations(data):
    """
    Save simulation results as JSON files (one folder per simulation).

    For each successful HTHP simulation, writes:
    - ``exerpy_data.json`` — exerpy-format dict (for reconstructing the EAN)
    - ``scalars.json``     — all other serialisable results

    Also writes ``heater.json`` and ``sim_index.json``.

    Parameters
    ----------
    data : dict
        Output of ``run_all_simulations``.
    """
    os.makedirs(SIM_CACHE_DIR, exist_ok=True)

    index = {}
    for key, sim in data["hthp"].items():
        f1, f2, ls, T_src = key
        folder = sim_cache_folder(f1, f2, ls, T_src)
        str_key = sim_key_to_str(key)

        if sim is None:
            index[str_key] = {"status": "failed", "folder": folder}
            continue

        os.makedirs(folder, exist_ok=True)

        # Write exerpy data (for ean reconstruction)
        exerpy_path = os.path.join(folder, "exerpy_data.json")
        with open(exerpy_path, "w") as f:
            json.dump(sim["exerpy_data"], f, cls=NumpyEncoder, indent=2)

        # Write all other scalars
        scalars = {k: v for k, v in sim.items()
                   if k not in ("ean", "exerpy_data")}
        scalars_path = os.path.join(folder, "scalars.json")
        with open(scalars_path, "w") as f:
            json.dump(scalars, f, cls=NumpyEncoder, indent=2)

        index[str_key] = {"status": "ok", "folder": folder}

    # Sim index
    with open(SIM_INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)

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

    Reads the sim index, loads scalars for each simulation, and
    reconstructs the ``ExergyAnalysis`` object from the exerpy JSON.

    Returns
    -------
    dict
        Same structure as the output of ``run_all_simulations``.
    """
    with open(SIM_INDEX_FILE) as f:
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

        # Reconstruct the ExergyAnalysis from the exerpy JSON
        sim["ean"] = reconstruct_ean(exerpy_path, sim["T_source_in"])
        hthp[key] = sim

    with open(HEATER_CACHE_FILE) as f:
        heater = json.load(f)

    with open(GAS_HEATER_CACHE_FILE) as f:
        gas_heater = json.load(f)

    print(f"Simulations loaded <- {SIM_CACHE_DIR}")
    return {"hthp": hthp, "heater": heater, "gas_heater": gas_heater}


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run HTHP simulations")
    parser.add_argument("fluids", nargs="*", help="f1 f2 (e.g. R290 R600a)")
    parser.add_argument("--lift-share", type=float, default=None,
                        help="Lower-cycle lift share fraction (e.g. 0.50)")
    parser.add_argument("--T-source-in", type=float, default=None,
                        help="T_source_in override [°C]")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.fluids:
        f1, f2 = args.fluids[0], args.fluids[1]
        ls = args.lift_share if args.lift_share is not None else LIFT_SHARE_DEFAULT
        T_src_in = args.T_source_in if args.T_source_in is not None else T_SOURCE_IN_DEFAULT
        sim = run_single_simulation(f1, f2, ls, T_src_in)
        if sim:
            print(f"  COP={sim['COP']:.3f}  epsilon={sim['epsilon']:.4f}")
        else:
            print("  Simulation failed or infeasible.")
    else:
        print("Running all simulations ...")
        data = run_all_simulations()
        save_simulations(data)
        n_ok = sum(1 for v in data["hthp"].values() if v is not None)
        n_total = len(data["hthp"])
        print(f"Done: {n_ok}/{n_total} HTHP combinations succeeded.")


if __name__ == "__main__":
    main()
