# Exergoeconomic Analysis of Cascaded High-Temperature Heat Pumps

Thermodynamic simulation, exergy analysis and exergoeconomic comparison of
cascaded two-stage high-temperature heat pumps (HTHP) for industrial steam
generation, benchmarked against an electrical resistance heater and a gas
heater.

## System description

The HTHP consists of two vapour-compression cycles coupled through an
internal heat exchanger (IHX):

- **Cycle 1 (lower)** absorbs heat from a source water stream and rejects it to cycle 2.
- **Cycle 2 (upper)** lifts the temperature further and generates saturated
  steam at 2 bar via a steam generator.

The total temperature lift (from source water inlet to steam at ~120 °C) is
split between the two cycles using a **lift share** parameter.  A lift share
of 50/50 means the lower cycle handles 50 % of the total lift and the upper
cycle handles 50 %.  The intermediate temperature T34 (evaporation
temperature of cycle 2) is computed as:

```
T34 = T_source_in + lift_share * (T_steam - T_source_in)
```

This relative definition ensures the split remains physically meaningful
when the source water inlet temperature is varied.

The study evaluates all combinations of:

| Cycle 1 fluids | Cycle 2 fluids |
|----------------|----------------|
| R290, R1270, R717 | R600a, R600, R717 |

Key boundary conditions:

- Source water inlet: 20–60 °C (variable), 5–10 K temperature drop
- Steam: 2 bar, saturated liquid to saturated vapour, 1 kg/s
- Pinch temperature difference: 5 K
- Compressor isentropic efficiency: 0.74
- Pump isentropic efficiency: 0.8
- Motor efficiency: 0.985

## Project structure

```
.
├── config.py          Central configuration (fluids, economic params, ranges)
├── models.py          TESPy network builder + exergy analysis (simulate_hthp, simulate_heater, simulate_gas_heater)
├── economics.py       PEC cost correlations + exergoeconomic analysis (run_economics)
├── simulate.py        Stage 1 — batch simulation runner with JSON caching
├── analyze.py         Stage 2 — economics, sensitivities, CSV export
├── plot.py            Stage 3 — all figures (bars, heatmaps, sensitivities, Q-T, log(p)-h)
├── main.py            Orchestrator with CLI flags
├── requirements.txt   Python dependencies
└── results/
    ├── cache/
    │   ├── sims/              Per-simulation JSON folders
    │   ├── sim_index.json     Simulation index
    │   ├── heater.json        Cached heater reference
    │   ├── gas_heater.json    Cached gas heater reference
    │   └── analysis.json      Cached Stage 2 output
    ├── overview/              Comparison and sensitivity figures
    └── <F1>_<F2>/LS_<pct>_Tsrc_<T>/   Per-scenario results (CSV, Q-T, log(p)-h)
```

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `tespy`, `exerpy`, `numpy`, `pandas`, `matplotlib`, `CoolProp`,
`fluprodia`.

## Quick start

### Full pipeline

```bash
python main.py
```

This runs all three stages in sequence:

1. **Simulate** — solves the TESPy network for every fluid combination, lift
   share and T_source_in value, saves JSON files under `results/cache/sims/`.
2. **Analyze** — runs exergoeconomic cost balances and sensitivity sweeps,
   exports CSV tables, saves `results/cache/analysis.json`.
3. **Plot** — generates all figures under `results/`.

### Skipping expensive stages

```bash
# Changed an economic parameter in config.py? Re-run analysis + plots only:
python main.py --skip-sim

# Only tweaking a plot? Re-generate figures from cached data:
python main.py --only-plots
```

### Running individual stages

Each stage can be run independently:

```bash
python simulate.py       # Stage 1 only
python analyze.py        # Stage 2 only (requires cached simulations)
python plot.py           # Stage 3 only (requires both caches)
```

### Single simulation

Run a single fluid combination from the CLI:

```bash
python simulate.py R290 R600a                                    # defaults
python simulate.py R290 R600a --lift-share 0.50                   # custom lift share
python simulate.py R290 R600a --T-source-in 40                    # custom T_source_in
python simulate.py R290 R600a --lift-share 0.50 --T-source-in 40
```

Or programmatically:

```python
from config import lift_share_to_T34
from models import simulate_hthp
from economics import run_economics

T34 = lift_share_to_T34(0.50, 40)  # 50/50 split at 40 °C source water
sim = simulate_hthp("R290", "R600a", T_evap_c2_override=T34, T_source_in_override=40)
eco = run_economics(sim, full_load_hours=5500, e1_c_ct_kwh=18.0)

print(f"COP = {sim['COP']:.3f}")
print(f"c_P = {eco['c_P']:.2f} EUR/GJ")
```

## Configuration

All tuneable parameters live in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FLUIDS_C1` | `["R290", "R1270", "R717"]` | Cycle-1 refrigerants |
| `FLUIDS_C2` | `["R600a", "R600", "R717"]` | Cycle-2 refrigerants |
| `BASE_FULL_LOAD_HOURS` | 5500 h/a | Base-case operating hours |
| `BASE_E1_C` | 159 EUR/MWh | Industrial electricity, medium consumer (20–70 GWh/a), full end-customer price 2025 — BDEW |
| `BASE_GAS_C` | 47 EUR/MWh | Natural gas wholesale, Day-Ahead spot at German THE hub, end-April 2026 — BDEW |
| `BASE_CO2_PRICE` | 60 EUR/tCO2 | EU ETS / BEHG corridor 55–65 EUR/tCO2, midpoint, valid from 2026-01-01 — Destatis |
| `LIFT_SHARE_RANGE` | [0.30, 0.40, 0.50, 0.60, 0.70] | Lower-cycle lift share fractions |
| `LIFT_SHARE_DEFAULT` | 0.50 | Base-case lift share (50/50) |
| `T_SOURCE_IN_RANGE` | [20, 25, ..., 60] °C | Source water inlet temperature sensitivity values |
| `T_SOURCE_IN_DEFAULT` | 20 °C | Base-case source water inlet temperature |
| `FULL_LOAD_HOURS_RANGE` | 2000–8500, step 500 | Hours sweep range |
| `E1_C_RANGE` | 100–200, step 10 EUR/MWh | Electricity price sweep range (brackets `BASE_E1_C = 159`) |

Changing a parameter in `config.py` and re-running the appropriate stage is all
that is needed — no code modifications required.

## Module reference

### `models.py`

Thermodynamic simulation layer. Three public functions:

- **`simulate_hthp(fluid_cycle1, fluid_cycle2, T_evap_c2_override=None, T_source_in_override=None)`**
  Builds a TESPy network, solves it in two passes (initial guess then
  pinch-based), runs an exergy analysis via exerpy, and returns a dict with
  performance metrics (COP, epsilon, E_F, E_P, E_D), component sizing data
  (volumetric flows, shaft powers, HX areas), pre-computed Q-T section data
  (for Q-T diagrams) and cycle state points (for log(p)-h diagrams).
  Returns `None` if the combination is infeasible.

- **`simulate_heater()`**
  Analytical reference case: electrical resistance heater with the same
  water-side conditions. Returns COP (= 1), exergetic efficiency and
  exergy flows.

- **`simulate_gas_heater(eta_gas=0.95)`**
  TESPy-based gas heater reference: combustion chamber + heat exchanger
  with Ahrendts chemical exergy accounting. Returns COP (= eta_gas),
  exergetic efficiency and exergy flows.

### `economics.py`

Cost correlations and exergoeconomic analysis:

- **PEC functions** (`pec_compressor`, `pec_motor`, `pec_plate_hx`,
  `pec_pump`, `pec_air_cooler`):
  Power-law and polynomial scaling correlations returning
  purchased-equipment costs in reference-year EUR. Refrigerant-specific
  via an internal cost-type mapping. Cost index adjustment is applied in
  `run_economics`.

- **`run_economics(sim, full_load_hours, e1_c_ct_kwh)`**:
  Computes PEC for all components, converts to hourly cost rates Z via
  `EconomicAnalysis`, then solves the full cost balance via
  `ExergoeconomicAnalysis`. Returns c_P [EUR/GJ] and Z_sum [EUR/h].

- **`run_economics_heater(sim, full_load_hours, e1_c_ct_kwh)`**:
  Simplified cost balance for the electrical heater reference case.

- **`run_economics_gas_heater(sim, full_load_hours, gas_c_ct_kwh, co2_price_eur_per_t=0.0)`**:
  Simplified cost balance for the gas heater reference case. Optional
  `co2_price_eur_per_t` adds an EU ETS-style carbon charge on top of the
  fuel cost, using the CO2 mass flow returned by `simulate_gas_heater`
  (computed from the TESPy combustion-chamber CH4 input via stoichiometry).

### `simulate.py`

Batch runner for Stage 1. Iterates over all
(fluid_c1, fluid_c2, lift_share, T_source_in) combinations defined in
`config.py`, computing T34 from the lift share and T_source_in.  Results are
serialised as JSON files (one folder per simulation) under `results/cache/sims/`.

Key functions: `run_all_simulations`, `run_single_simulation`,
`save_simulations`, `load_simulations`.

### `analyze.py`

Economics runner for Stage 2. Loads cached simulations, computes:

- Base-case results for all fluid combinations at default lift share and T_source_in
- Heater and gas heater reference economics
- Sensitivity sweeps: c_P vs full-load hours and electricity price (base +
  alternative scenarios)
- Lift share sensitivity: COP, epsilon, c_P across all lift share values
- T_source_in sensitivity: COP, epsilon, c_P across all T_source_in values
- Cross-sensitivities: economic sweeps for every lift share and T_source_in value
- Exergoeconomic CSV tables (components, connections, non-material streams)

Results saved to `results/cache/analysis.json`.

Key functions: `run_all_analysis`, `save_analysis`, `load_analysis`.

### `plot.py`

Figure generator for Stage 3. Each `plot_*` function produces one figure
(or a family of per-scenario figures) and saves to `results/`:

| Function | Figure |
|----------|--------|
| `plot_comparison_bars` | Grouped bar chart (COP, epsilon, c_P, Z_sum) |
| `plot_comparison_heatmaps` | Heatmap matrix (C1 vs C2 fluids) |
| `plot_sensitivity_hours` | c_P vs full-load hours (base case) |
| `plot_sensitivity_e1c` | c_P vs electricity price (base case) |
| `plot_sensitivity_hours_alt` | c_P vs full-load hours (high elec. price) |
| `plot_sensitivity_e1c_alt` | c_P vs electricity price (high utilisation) |
| `plot_sensitivity_lift_share_COP` | COP vs lift share |
| `plot_sensitivity_lift_share_cP` | c_P vs lift share |
| `plot_sensitivity_lift_share_epsilon` | Exergetic efficiency vs lift share |
| `plot_sensitivity_lift_share_heatmap` | Heatmap of COP/epsilon/c_P across lift shares |
| `plot_sensitivity_hours_by_lift_share` | c_P vs hours, one subplot per lift share |
| `plot_sensitivity_e1c_by_lift_share` | c_P vs elec. price, one subplot per lift share |
| `plot_sensitivity_T_source_in_COP` | COP vs T_source_in |
| `plot_sensitivity_T_source_in_cP` | c_P vs T_source_in |
| `plot_sensitivity_T_source_in_epsilon` | Exergetic efficiency vs T_source_in |
| `plot_sensitivity_T_source_in_heatmap` | Heatmap of COP/epsilon/c_P across T_source_in |
| `plot_sensitivity_T_source_in_heatmap_by_lift_share` | Per-lift-share T_source_in heatmaps |
| `plot_sensitivity_hours_by_T_source_in` | c_P vs hours, one subplot per T_source_in |
| `plot_sensitivity_e1c_by_T_source_in` | c_P vs elec. price, one subplot per T_source_in |
| `plot_qt_diagrams` | Q-T diagrams for all HXs (per scenario) |
| `plot_logph_diagrams` | log(p)-h diagrams for both cycles (per scenario) |

`generate_all_plots(analysis, simulations)` calls all of the above.

### `main.py`

CLI orchestrator. Accepts `--skip-sim` and `--only-plots` flags to skip
upstream stages when only downstream parameters have changed.

## Output structure

After a full run, `results/` contains:

```
results/
├── cache/
│   ├── sims/                  Per-simulation JSON folders
│   ├── sim_index.json
│   ├── heater.json
│   ├── gas_heater.json
│   └── analysis.json
├── overview/
│   ├── comparison_bars.png
│   ├── comparison_heatmaps.png
│   ├── sensitivity_hours.png
│   ├── sensitivity_e1c.png
│   ├── sensitivity_hours_high_price.png
│   ├── sensitivity_e1c_high_util.png
│   ├── sensitivity_lift_share_COP.png
│   ├── sensitivity_lift_share_cP.png
│   ├── sensitivity_lift_share_epsilon.png
│   ├── sensitivity_lift_share_heatmap.png
│   ├── sensitivity_hours_by_lift_share.png
│   ├── sensitivity_e1c_by_lift_share.png
│   ├── sensitivity_T_source_in_COP.png
│   ├── sensitivity_T_source_in_cP.png
│   ├── sensitivity_T_source_in_epsilon.png
│   ├── sensitivity_T_source_in_heatmap.png
│   ├── sensitivity_T_source_in_heatmap_LS_<pct>.png
│   ├── sensitivity_hours_by_T_source_in.png
│   └── sensitivity_e1c_by_T_source_in.png
└── <F1>_<F2>/LS_<pct>_Tsrc_<T>/
    ├── components.csv            Exergoeconomic component results
    ├── connections_exergy.csv    Connection exergy values
    ├── connections_costs.csv     Connection cost rates
    ├── nonmaterial.csv           Non-material stream results
    ├── QT_diagram.png            Q-T diagram (3 heat exchangers)
    └── logph_diagram.png         log(p)-h diagram (both cycles)
```

## Typical workflows

| What changed | Command |
|--------------|---------|
| Nothing yet, first run | `python main.py` |
| Fluid list, lift share range or T_source_in range | `python main.py` (full re-run) |
| Economic parameter (e.g. electricity price) | `python main.py --skip-sim` |
| Plot aesthetics (colours, labels, layout) | `python main.py --only-plots` |
| Single combination check | `python simulate.py R290 R600a --lift-share 0.50 --T-source-in 40` |
