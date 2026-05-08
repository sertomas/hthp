# Exergoeconomic Analysis of Cascaded High-Temperature Heat Pumps

Thermodynamic simulation, exergy analysis and exergoeconomic comparison of
cascaded two-stage high-temperature heat pumps (HTHP) for industrial steam
generation, benchmarked against a natural-gas heater reference. The gas
reference is priced fuel + EU ETS / BEHG carbon charge by default (set
`BASE_CO2_PRICE = 0` in `config.py` to reproduce the Ommen 2015 fuel-only
baseline).

## System description

The HTHP consists of two vapour-compression cycles coupled through an
internal heat exchanger (IHX):

- **Cycle 1 (lower)** absorbs heat from a source water stream and rejects it to cycle 2.
- **Cycle 2 (upper)** lifts the temperature further and generates saturated
  steam at the case-study sink temperature.

The total temperature lift is split between the two cycles using a **lift
share** parameter.  A lift share of 0.50 means the lower cycle handles 50 %
of the total lift and the upper cycle handles the remaining 50 %.  The
intermediate temperature T34 (evaporation temperature of cycle 2) is

```
T34 = T_source_in + lift_share * (T_steam - T_source_in)
```

This relative definition keeps the split physically meaningful as the
source water inlet and sink steam temperatures are varied.

Fluid combinations:

| Cycle 1 fluids | Cycle 2 fluids |
|----------------|----------------|
| R290, R1270, R717 | R600a, R600 |

(R717 in cycle 2 was dropped — see the comment block at the top of
`config.py` for the reasoning. R744 is intentionally excluded as well.)

Key boundary conditions:

- Source water: inlet 20–60 °C (sensitivity range), fixed mass flow
  `SOURCE_MASS_FLOW = 30 kg/s` (T_out is then free). A fixed-ΔT mode
  (`source_mode="fixed_delta_T"`) is still implemented in `simulate_hthp`
  but unused by the pipeline.
- Steam sink: saturated, 100 / 110 / 120 °C (case-study sweep), 1 MWth
  nominal heating capacity
- Pinch temperature difference: 5 K
- Compressor isentropic efficiency: 0.80 (Ommen 2015, Table 1)
- Pump isentropic efficiency: 0.80
- Motor electrical efficiency: 0.95 (Ommen 2015, Table 1)

## Project structure

```
.
├── main.py                            Pipeline orchestrator + single-design fast path
├── config.py                          Central configuration (fluids, economics, sweeps)
├── models.py                          TESPy + exergy: simulate_hthp, simulate_gas_heater
├── economics.py                       PEC correlations, run_economics, run_economics_gas_heater
│
├── case_steam.py                      Stage 1: 150-case TESPy feasibility screen (per T_steam)
├── reclassify_modern.py               Stage 2: relabel screen output under modern envelopes
├── plot_case_steam.py                 Stage 3: feasibility / COP / T_disch / p_high grids
│
├── case_steam_economics.py            Stage 4: exergoeconomic analysis on OK designs
├── plot_case_steam_economics.py       Stage 5: c_P heatmap, PEC vs Annex 58, sensitivities
│
├── export_design_details.py           Stage 6: per-design connections / components / Q-T / log(p)-h
├── plot_compare_T_steam.py            Stage 7: cross-T_steam comparison (after all 3 cases)
│
├── plot_common.py                     Shared 4-state classification + plotting helpers
├── screen_cascade.py                  Ommen 2015 envelope helpers (used by stages 1 + 2)
│
├── requirements.txt
└── results/                           Output (see "Output structure" below)
```

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `tespy`, `exerpy`, `numpy`, `pandas`, `matplotlib`, `CoolProp`,
`fluprodia`.

## Quick start

### Full pipeline

```powershell
python main.py
```

Runs the seven-stage pipeline once per `T_steam` ∈ {100, 110, 120} °C, then
writes the cross-T_steam comparison plots. Total runtime ≈ 24 min on a
typical laptop.

### Single steam temperature

```powershell
python main.py --t-steam 110     # only the 110 °C case (~8 min)
```

### Skipping cached stages

```powershell
python main.py --skip-screen        # reuse cached 150-case screen CSV
python main.py --skip-economics     # reuse cached economics CSVs
python main.py --skip-export        # skip per-design dumps (alias: --no-export)
python main.py --only-plots         # regenerate plots only, from cached CSVs
```

### Single-design fast path (~5 s)

```powershell
python main.py --design R290 R600 0.40 40
                      # f1   f2   LS  T_src(°C)   — defaults to T_steam = 110
```

Writes `connections.csv`, `components.csv`, `qt_diagram.png` and
`logph_diagram.png` under
`results/case_steam_110/designs/<F1>_<F2>/LS<pct>_Tsrc<T>/`, and prints
COP, ε, c_P, Z_sum and TCI/kW.

### List all OK designs

```powershell
python main.py --list-designs
```

Prints the table of all designs that survived the modern-envelope feasibility
filter (110 °C case), sorted by c_P. Requires the Stage 4 cache.

### Programmatic use

```python
from config import lift_share_to_T34
from models import simulate_hthp
from economics import run_economics

T34 = lift_share_to_T34(0.50, 40, T_steam=110)
sim = simulate_hthp(
    "R290", "R600",
    T_evap_c2_override=T34,
    T_source_in_override=40,
    T_steam_override=110,
    source_mode="fixed_mass_flow",  # mirrors the pipeline
)
eco = run_economics(sim, full_load_hours=7500, e1_c_ct_kwh=15.9)
print(f"COP = {sim['COP']:.3f}")
print(f"c_P = {eco['c_P']:.2f} EUR/GJ")
```

## Configuration

All tuneable parameters live in [config.py](config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FLUIDS_C1` | `["R290", "R1270", "R717"]` | Cycle-1 refrigerants |
| `FLUIDS_C2` | `["R600a", "R600"]` | Cycle-2 refrigerants |
| `LIFT_SHARE_RANGE` | `[0.30, 0.40, 0.50, 0.60, 0.70]` | Lower-cycle lift share fractions |
| `LIFT_SHARE_DEFAULT` | `0.50` | Base-case lift share (50/50) |
| `T_SOURCE_IN_RANGE` | `[20, 30, 40, 50, 60]` °C | Source water inlet sensitivity |
| `T_SOURCE_IN_DEFAULT` | `20` °C | Base-case source water inlet |
| `T_STEAMS_TO_RUN` | `[100, 110, 120]` °C | Per-case T_steam values for the full pipeline |
| `T_STEAM_CASE_DEFAULT` | `110` °C | Single-design fast-path default |
| `Q_H_NOMINAL_KW` | `1000` kW | Nominal heating output (1 MWth) |
| `BASE_FULL_LOAD_HOURS` | `7500` h/a | Operating hours |
| `BASE_E1_C` | `159` EUR/MWh | Industrial electricity (BDEW 2025, medium consumer) |
| `BASE_GAS_C` | `47` EUR/MWh | Natural gas wholesale (BDEW THE Day-Ahead, end-April 2026) |
| `BASE_CO2_PRICE` | `60` EUR/tCO2 | EU ETS / BEHG midpoint, valid from 2026-01-01 |
| `E1_C_RANGE` | 100–200 step 10 EUR/MWh | Electricity-price sweep |
| `PRICE_SENS_FRAC_RANGE` | −50 % … +50 % step 10 % | Joint e1/gas price sensitivity |

Changing a parameter and re-running the appropriate stage (with
`--skip-screen` / `--skip-economics` to reuse upstream caches) is all that
is needed.

## Module reference

### `models.py`

Two public functions:

- **`simulate_hthp(fluid_cycle1, fluid_cycle2, T_evap_c2_override=None, T_source_in_override=None, source_mode="fixed_delta_T", m_source=None, T_steam_override=None, skip_ommen_check=False)`**
  Builds a TESPy network, solves it (initial guess + pinch-based pass),
  runs an `exerpy` exergy analysis, and returns a dict with COP, ε, E_F,
  E_P, E_D, sizing data (V̇, W_shaft, HX areas), Q-T sections (for Q-T
  diagrams) and cycle state points (for log(p)-h diagrams). Returns
  `None` on infeasibility / non-convergence.

- **`simulate_gas_heater(eta_gas=0.90, T_steam_override=None)`**
  TESPy combustion-chamber + heat-exchanger model with Ahrendts chemical
  exergy accounting. Returns COP (= `eta_gas`), exergetic efficiency,
  exergy flows and the CO2 mass flow (used for the carbon charge).

### `economics.py`

PEC correlations and exergoeconomic analysis:

- **`pec_compressor`, `pec_motor`, `pec_plate_hx`, `pec_pump`** — power-law /
  polynomial scaling to reference-year EUR. Refrigerant-specific via an
  internal cost-type mapping; cost-index adjustment applied in
  `run_economics`.
- **`run_economics(sim, full_load_hours, e1_c_ct_kwh)`** — converts PEC into
  hourly cost rates `Z` via `EconomicAnalysis`, then solves the cost
  balance via `ExergoeconomicAnalysis`. Returns `c_P` [EUR/GJ] and
  `Z_sum` [EUR/h].
- **`run_economics_gas_heater(sim, full_load_hours, gas_c_ct_kwh, co2_price_eur_per_t=0.0)`** —
  cost balance for the gas reference. The pipeline calls it with
  `co2_price_eur_per_t = BASE_CO2_PRICE`, so the headline gas c_P includes
  the carbon charge; passing `0.0` (the function-level default) reproduces
  Ommen's fuel-only baseline.

### Stage 1 — `case_steam.py`

Cascaded HTHP feasibility screen at fixed `Q_H = 1 MWth` and a given
`T_steam`. Runs `simulate_hthp` for every (f1, f2, LS, T_src) combination,
applies Ommen 2015 Table 3 envelope checks (with +10 % pressure tolerance),
and writes `case_steam_<T>.csv`. ≈ 5 min per T_steam.

### Stage 2 — `reclassify_modern.py`

Re-applies the same 4-state classification (`OK` / `V_ONLY` / `HARD` /
`NOSOLVE`) using a *modern* compressor envelope (current commercial Vilter
/ GEA / Mayekawa specs) instead of the 2015 Ommen envelope. Writes
`case_steam_<T>_enriched.csv`. Instant.

### Stage 3 — `plot_case_steam.py`

Renders the feasibility, COP, T_disch and p_high grids. Run
twice automatically — once in *Ommen-strict* mode (no `_modern` suffix),
once in *modern* mode (`_modern` suffix). Instant.

### Stage 4 — `case_steam_economics.py`

Exergoeconomic analysis on every OK (modern) design at LS ∈ {0.30, 0.40,
0.50}. Scales each design to 1 MWth for direct Annex 58 (2023) capital-
cost comparison. Writes `economics/economics_base.csv`,
`economics_pec_breakdown.csv` and price/utilisation sensitivity CSVs.
≈ 3 min per T_steam.

### Stage 5 — `plot_case_steam_economics.py`

c_P heatmap, PEC-per-kW vs Annex 58 band, PEC component breakdown, c_P
vs e1 / gas / FLH, ±50 % 2-D price sensitivity, Tsatsaronis improvement-
priority quadrant, economic vs exergoeconomic ranking, best-design
dashboard. Instant.

### Stage 6 — `export_design_details.py`

Per-design dump of TESPy connection / component tables, Q-T diagrams,
log(p)-h diagrams (via `fluprodia`), and exergoeconomic component /
material / non-material CSVs. ≈ 5 min per T_steam.

### Stage 7 — `plot_compare_T_steam.py`

Cross-T_steam comparison plots (best-c_P trends, c_P heatmap across all
three temperatures). Triggered automatically once `economics_base.csv`
exists for every T_steam in `T_STEAMS_TO_RUN`.

### Helpers

- **`config.py`** — fluids, sensitivity ranges, economic constants,
  per-T_steam path helpers (`case_results_dir`, `t_steam_compare_dir`),
  `lift_share_to_T34`. Touch-and-rerun is the intended workflow.
- **`plot_common.py`** — `classify_status` (4-state precedence:
  `NOSOLVE` ≻ `HARD` ≻ `V_ONLY` ≻ `OK`), `slice_grid`, label shorteners.
- **`screen_cascade.py`** — Ommen 2015 envelope helpers
  (`COMPRESSOR_SPEC`, `OMMEN_P_TOL`, `T_DISCH_MAX`, `_envelope_label`,
  `_T_from_p_h`, `_pre_classify_failure`). Imported by stages 1 and 2;
  not runnable standalone.

## Output structure

After a full run, `results/` contains:

```
results/
├── case_steam_100/, case_steam_110/, case_steam_120/
│   ├── case_steam_<T>.csv               (Stage 1 — raw 150-case screen)
│   ├── case_steam_<T>_enriched.csv      (Stage 2 — with modern classification)
│   ├── case_steam_<T>_modern_OK.csv     (Stage 2 — OK subset)
│   ├── case_steam_<T>_per_pair.csv      (Stage 2 — best per fluid pair)
│   ├── case_steam_<T>_sorted.csv        (Stage 2 — sorted by c_P proxy)
│   ├── feasibility_grid[_modern].png    (Stage 3)
│   ├── cop_grid[_modern].png            (Stage 3)
│   ├── Tdisch_grid[_modern].png         (Stage 3)
│   ├── p_high_grid[_modern].png         (Stage 3)
│   ├── economics/
│   │   ├── economics_base.csv           (Stage 4 — one row per OK design)
│   │   ├── economics_pec_breakdown.csv  (Stage 4 — PEC by component)
│   │   ├── economics_sensitivity_e1.csv (Stage 4 — c_P vs e1)
│   │   ├── economics_sensitivity_FLH.csv(Stage 4 — c_P vs full-load hours)
│   │   ├── gas_heater/                  (Stage 4 — reference case)
│   │   ├── cP_grid.png, PEC_*.png,
│   │   │   cP_vs_*.png, price_sensitivity_2d*.png,
│   │   │   tsatsaronis_quadrant.png,
│   │   │   best_designs_per_T_src.png,
│   │   │   *_breakdown_*.png            (Stage 5)
│   └── designs/<F1>_<F2>/LS<pct>_Tsrc<T>/
│       ├── connections.csv              (Stage 6)
│       ├── components.csv               (Stage 6)
│       ├── qt_diagram.png               (Stage 6)
│       ├── logph_diagram.png            (Stage 6)
│       ├── exergoeco_components.csv          (Stage 6)
│       ├── exergoeco_connections_material.csv(Stage 6)
│       └── exergoeco_connections_nonmat.csv  (Stage 6)
└── case_steam_compare/                  (Stage 7 — cross-T_steam plots)
    ├── compare_cP_best_vs_T_steam.png
    └── compare_cP_heatmap.png
```

## Typical workflows

| What changed | Command |
|--------------|---------|
| Nothing yet, first run | `python main.py` |
| Fluid list, lift-share or T_source_in range | `python main.py` (full re-run) |
| Economic parameter (e.g. `BASE_E1_C`) | `python main.py --skip-screen` |
| Plot aesthetics only | `python main.py --only-plots` |
| Single combination check | `python main.py --design R290 R600 0.40 40` |
| Inspect ranked OK designs | `python main.py --list-designs` |
