"""One-shot migration: move legacy ``case_steam_<T>/economics/`` contents
into the new ``data/`` + ``gas_heater/`` + ``plots/`` layout.

Run once after the path refactor:

    python migrate_layout.py

For each ``results/case_steam_<T>/`` it:
  * moves ``economics/economics_*.csv`` → ``data/``
  * moves ``economics/gas_heater/*``    → ``gas_heater/``
  * **deletes** ``economics/*.png``     (they will be regenerated with the
    new title/no-title pair naming convention)
  * removes the now-empty ``economics/`` folder.

Idempotent: re-running after a successful migration is a no-op.
"""

from __future__ import annotations

import os
import shutil

from config import (
    T_STEAMS_TO_RUN, case_data_dir, case_gas_heater_dir,
    case_results_dir,
)


def migrate_one(T_steam: float) -> None:
    case_dir = case_results_dir(T_steam)
    legacy = os.path.join(case_dir, "economics")
    if not os.path.isdir(legacy):
        print(f"[skip] {case_dir}: no economics/ folder")
        return

    data_dir = case_data_dir(T_steam)
    gas_dir = case_gas_heater_dir(T_steam)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(gas_dir, exist_ok=True)

    for name in os.listdir(legacy):
        src = os.path.join(legacy, name)
        if name.endswith(".csv"):
            shutil.move(src, os.path.join(data_dir, name))
        elif name == "gas_heater" and os.path.isdir(src):
            for inner in os.listdir(src):
                shutil.move(os.path.join(src, inner),
                            os.path.join(gas_dir, inner))
            os.rmdir(src)
        elif name.endswith(".png"):
            os.remove(src)
        else:
            # Anything else (unexpected) → leave alone; surface a warning.
            print(f"  [warn] unexpected entry left in {legacy}: {name}")

    try:
        os.rmdir(legacy)
        print(f"[ok]  migrated {case_dir}/economics/  -> data/ + gas_heater/")
    except OSError as exc:
        print(f"[warn] {legacy} not empty after migration ({exc})")

    # plot_case_steam.py used to write the feasibility / cop / Tdisch / p_high
    # PNGs straight into the case-folder root. They now go to plots/ — wipe
    # any leftover root-level copies so the folder stays tidy.
    for name in os.listdir(case_dir):
        if not name.endswith(".png"):
            continue
        full = os.path.join(case_dir, name)
        if os.path.isfile(full):
            os.remove(full)
            print(f"       removed stale root-level PNG: {name}")


def main():
    for T in T_STEAMS_TO_RUN:
        migrate_one(T)


if __name__ == "__main__":
    main()
