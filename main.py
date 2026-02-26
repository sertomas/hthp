"""
Orchestrator — run the full pipeline: simulate -> analyze -> plot.

Flags allow skipping expensive stages when only downstream parameters
have changed (e.g. re-generate plots without re-running simulations).

Usage
-----
::

    python main.py                # full pipeline
    python main.py --skip-sim     # reuse cached simulations
    python main.py --only-plots   # only regenerate plots
"""

import argparse
import os

from analyze import load_analysis, run_all_analysis, save_analysis
from plot import generate_all_plots, print_summary_table
from simulate import load_simulations, run_all_simulations, save_simulations


def main():
    parser = argparse.ArgumentParser(description="HTHP comparison pipeline")
    parser.add_argument("--skip-sim", action="store_true",
                        help="Skip simulation, load from cache")
    parser.add_argument("--only-plots", action="store_true",
                        help="Only regenerate plots from cached analysis")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ── Stage 1 (Simulate) & Stage 2 (Analyze) ─────────────────────────
    if args.only_plots:
        print("Loading cached simulations ...")
        sims = load_simulations()
        print("Loading cached analysis ...")
        analysis = load_analysis()
    elif args.skip_sim:
        print("Loading cached simulations ...")
        sims = load_simulations()
        print("\n=== Stage 2: Analysis ===")
        analysis = run_all_analysis(sims)
        save_analysis(analysis)
    else:
        print("=== Stage 1: Simulations ===")
        sims = run_all_simulations()
        save_simulations(sims)
        print("\n=== Stage 2: Analysis ===")
        analysis = run_all_analysis(sims)
        save_analysis(analysis)

    # ── Stage 3: Plot ────────────────────────────────────────────────────
    print("\n=== Stage 3: Plots ===")
    print_summary_table(analysis)
    print("\nGenerating plots ...")
    generate_all_plots(analysis, sims)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
