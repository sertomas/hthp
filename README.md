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

- Source water: inlet 20–60 °C (sensitivity range), fixed outlet temperature
  `T_out` with the mass flow free (`source_mode="fixed_T_out"`). `T_out` is the
  ambient dead state plus a 0.01 K margin (20.01 °C) so nearly all above-ambient
  thermal exergy is extracted while keeping the source-HX hot outlet just above
  ambient — cooling to *exactly* ambient drives exerpy's HX cost balance into a
  degenerate case that explodes the outlet specific cost. The exception is a
  source already at ambient (inlet = 20 °C), cooled to 10 °C so a finite duty
  remains. Fixed-mass-flow
  (`source_mode="fixed_mass_flow"`) and fixed-ΔT (`source_mode="fixed_delta_T"`)
  modes are still implemented in `simulate_hthp` but unused by the pipeline.
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
├── config.py                          Central configuration (fluids, economics, sweeps, path helpers)
├── models.py                          TESPy + exergy: simulate_hthp, simulate_gas_heater
├── economics.py                       PEC correlations, CELF levelization, run_economics(_gas_heater)
├── calculate_heatexchanger_area.py    Section-wise HX area sizing (per phase / per stream)
│
├── case_steam.py                      Stage 1: 150-case TESPy feasibility screen (per T_steam)
├── reclassify_modern.py               Stage 2: relabel screen output under modern envelopes
├── plot_case_steam.py                 Stage 3: feasibility / COP / T_disch / p_high grids
│
├── case_steam_economics.py            Stage 4: exergoeconomic analysis on OK designs
├── export_design_details.py           Stage 5: per-design connections / components / Q-T / log(p)-h
├── plot_case_steam_economics.py       Stage 6: c_P heatmap, PEC vs Annex 58, sensitivities
├── plot_compare_T_steam.py            Stage 7: cross-T_steam comparison (after all 3 cases)
│
├── plot_common.py                     Shared 4-state classification + plotting helpers
├── screen_cascade.py                  Ommen 2015 envelope helpers (used by stages 1 + 2)
│
├── plot_cP_Tsrc_Tsteam.py             Standalone: per-pair c_P heatmap over (T_src,in, T_steam)
├── plot_cdz_sensitivity_per_design.py Standalone: per-pair Z+C_D and c_P heatmaps inside designs/
├── migrate_layout.py                  One-shot: migrate legacy economics/ folder to data/+plots/+gas_heater/
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
    source_mode="fixed_T_out",  # mirrors the pipeline
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
| `I_EFF` | `0.10` 1/a | Effective discount rate (TRR / CELF) |
| `N_YEARS` | `20` a | Plant lifetime |
| `R_N_OM` | `0.02` 1/a | O&M nominal escalation (also drives CEPCI 2024 → 2026) |
| `R_N_EL` | `0.02` 1/a | Electricity nominal escalation (BDEW pre-2022 trend) |
| `R_N_GAS` | `0.03` 1/a | Natural-gas nominal escalation (BDEW pre-2022 trend) |
| `R_N_CO2` | `0.05` 1/a | CO2 emission-price nominal escalation (ETS/BEHG midpoint) |
| `E1_C_RANGE` | 100–200 step 10 EUR/MWh | Electricity-price sweep |
| `PRICE_SENS_FRAC_RANGE` | −50 % … +50 % step 10 % | Joint e1/gas price sensitivity |

Changing a parameter and re-running the appropriate stage (with
`--skip-screen` / `--skip-economics` to reuse upstream caches) is all that
is needed.

## Module reference

### `models.py`

Two public functions:

- **`simulate_hthp(fluid_cycle1, fluid_cycle2, T_evap_c2_override=None, T_source_in_override=None, source_mode="fixed_T_out", m_source=None, T_steam_override=None, skip_ommen_check=False)`**
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
- **`_celf(r_n, i_eff, n)`** — constant-escalation levelization factor
  (Bejan-Tsatsaronis-Moran textbook, end-of-year convention) used to
  convert first-year fuel cost rates into constant-equivalent annual cost
  rates over the plant lifetime.
- **`run_economics(sim, full_load_hours, e1_c_ct_kwh)`** — converts PEC into
  hourly cost rates `Z` via `EconomicAnalysis`, then solves the cost
  balance via `ExergoeconomicAnalysis`. Electricity cost is levelized via
  `_celf(R_N_EL)`. The OMC stream returned by `EconomicAnalysis` uses the
  begin-of-year convention and is divided by `(1 + I_EFF)` to align it
  with the end-of-year fuel basis. Returns `c_P` [EUR/GJ] and `Z_sum`
  [EUR/h].
- **`run_economics_gas_heater(sim, full_load_hours, gas_c_ct_kwh, co2_price_eur_per_t=0.0)`** —
  cost balance for the gas reference. Gas and CO2 prices are levelized
  independently via `_celf(R_N_GAS)` and `_celf(R_N_CO2)`. The pipeline
  calls it with `co2_price_eur_per_t = BASE_CO2_PRICE`, so the headline
  gas c_P includes the carbon charge; passing `0.0` (the function-level
  default) reproduces Ommen's fuel-only baseline.

PEC cost basis: bare correlation costs are at `COST_REF_YEAR = 2013`,
escalated to 2024 via the CEPCI ratio in `_CI_RATIO`, then to
`ANALYSIS_REF_YEAR = 2026` via `_CEPCI_TO_REF_YEAR = (1 + R_N_OM)^2`
(general inflation at the same rate as OMC). All Z and PEC values stored
in `economics_*.csv` are therefore in 2026 EUR, matching the year-1
basis of `BASE_E1_C`, `BASE_GAS_C` and `BASE_CO2_PRICE`.

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
cost comparison. Writes `data/economics_base.csv`,
`data/economics_pec_breakdown.csv` and price/utilisation sensitivity CSVs
into the per-T_steam `data/` folder, plus the gas-heater reference dump
under `gas_heater/`. ≈ 3 min per T_steam.

### Stage 5 — `export_design_details.py`

Per-design dump of TESPy connection / component tables, Q-T diagrams,
log(p)-h diagrams (via `fluprodia`), and exergoeconomic component /
material / non-material CSVs. Runs **before** the economics plots so
that the per-design `exergoeco_components.csv` files exist when Stage 6
reads them for the cost-breakdown, Tsatsaronis, ranking and best-design
plots — otherwise those plots silently fall back to a placeholder.
≈ 5 min per T_steam.

### Stage 6 — `plot_case_steam_economics.py`

c_P heatmap, PEC-per-kW vs Annex 58 band, PEC component breakdown, c_P
vs e1 / gas / FLH, ±50 % 2-D price sensitivity (also tiled across FLH ∈
{5000, 5500, 6000, 6500, 7000} h/a), best-LS-per-pair (τ, c_el) and
(τ, c_gas) sensitivities at T_src = 50 °C, Tsatsaronis improvement-
priority quadrant, economic vs exergoeconomic ranking, LS-choice
agreement / Pareto / regret bars, and the best-design dashboard. Every
plot is written twice: once paper-ready (no title) and once with a
descriptive title appended as `*_titled.pdf`. Instant.

### Stage 7 — `plot_compare_T_steam.py`

Cross-T_steam comparison plots — best-c_P trends, c_P heatmap across all
three temperatures, fleet-wide exergy-destruction breakdown, and an FLH
× c_el sensitivity for the best design. Triggered automatically once
`data/economics_base.csv` exists for every T_steam in `T_STEAMS_TO_RUN`.

### Standalone plotters (run manually)

- **`plot_cP_Tsrc_Tsteam.py`** — per-pair c_P heatmap over (T_src,in,
  T_steam) with the winning LS annotated in each cell. Writes
  `cP_heatmap_Tsrc_Tsteam.pdf` (2 × 3 grid) and
  `cP_heatmap_Tsrc_Tsteam_R717_R600.pdf` (single-panel for the
  cost-optimal pair) into `results/case_steam_compare/`.
- **`plot_cdz_sensitivity_per_design.py`** — per-pair Z + C_D breakdown
  bar charts and a (T_src × LS) c_P heatmap, written inside each pair's
  `designs/<pair>/` folder.
- **`migrate_layout.py`** — one-shot helper for the legacy `economics/`
  layout. Moves `economics/economics_*.csv` → `data/`,
  `economics/gas_heater/` → `gas_heater/`, deletes stale PNGs, then
  removes the empty `economics/` folder. Idempotent.

### Helpers

- **`config.py`** — fluids, sensitivity ranges, economic constants,
  per-T_steam path helpers (`case_results_dir`, `case_data_dir`,
  `case_plots_dir`, `case_gas_heater_dir`, `t_steam_compare_dir`),
  `lift_share_to_T34`. Touch-and-rerun is the intended workflow.
- **`plot_common.py`** — `classify_status` (4-state precedence:
  `NOSOLVE` ≻ `HARD` ≻ `V_ONLY` ≻ `OK`), `slice_grid`, label shorteners,
  `save_titled_and_paper` (writes every figure twice — paper-ready and
  `*_titled.pdf`).
- **`screen_cascade.py`** — Ommen 2015 envelope helpers
  (`COMPRESSOR_SPEC`, `OMMEN_P_TOL`, `T_DISCH_MAX`, `_envelope_label`,
  `_T_from_p_h`, `_pre_classify_failure`). Imported by stages 1 and 2;
  not runnable standalone.
- **`calculate_heatexchanger_area.py`** — section-wise heat exchanger
  area sizing. Walks each HX along an enthalpy grid, classifies each
  section by phase on both sides, and returns per-phase areas. Used by
  `models.simulate_hthp` to populate `sizing["A_HX"]` for the PEC
  calculation.

## Output structure

After a full run, `results/` contains the layout below. Plots are PDF
and each one is written twice — `<name>.pdf` (paper-ready, no title) and
`<name>_titled.pdf` (with a descriptive title). For brevity only the
paper-ready filename is listed below; the `_titled.pdf` companion is
always present.

```
results/
├── case_steam_100/, case_steam_110/, case_steam_120/
│   ├── case_steam_<T>.csv                   (Stage 1 — raw 150-case screen)
│   ├── case_steam_<T>_enriched.csv          (Stage 2 — with modern classification)
│   ├── case_steam_<T>_modern_OK.csv         (Stage 2 — OK subset)
│   ├── case_steam_<T>_per_pair.csv          (Stage 2 — best per fluid pair)
│   ├── case_steam_<T>_sorted.csv            (Stage 2 — sorted by c_P proxy)
│   │
│   ├── data/                                (Stage 4 — economics CSVs)
│   │   ├── economics_base.csv               (one row per OK design)
│   │   ├── economics_pec_breakdown.csv      (PEC by component)
│   │   ├── economics_sensitivity_e1.csv     (c_P vs electricity price)
│   │   └── economics_sensitivity_FLH.csv    (c_P vs full-load hours)
│   │
│   ├── gas_heater/                          (Stage 4 — reference case)
│   │   ├── gas_heater.json
│   │   ├── connections.csv, components.csv
│   │   ├── exergy_summary.csv
│   │   ├── exergoeco_components.csv
│   │   ├── exergoeco_connections_material.csv
│   │   └── exergoeco_connections_nonmat.csv
│   │
│   ├── designs/<F1>_<F2>/LS<pct>_Tsrc<T>/   (Stage 5 — per-design dumps)
│   │   ├── connections.csv
│   │   ├── components.csv
│   │   ├── qt_diagram.png
│   │   ├── logph_diagram.png
│   │   ├── exergoeco_components.csv
│   │   ├── exergoeco_connections_material.csv
│   │   └── exergoeco_connections_nonmat.csv
│   │
│   └── plots/                               (Stages 3 + 6 — every PDF figure)
│       ├── feasibility_grid[_modern].pdf            (Stage 3)
│       ├── cop_grid[_modern].pdf                    (Stage 3)
│       ├── Tdisch_grid[_modern].pdf                 (Stage 3)
│       ├── p_high_grid[_modern].pdf                 (Stage 3)
│       │
│       ├── cP_heatmap.pdf, LCOH_heatmap.pdf,        (Stage 6)
│       │   COP_heatmap.pdf, epsilon_heatmap.pdf,
│       │   T_lift_heatmap.pdf, T_min_lower_heatmap.pdf,
│       │   T_max_upper_heatmap.pdf,
│       │   p_max_COMP1_heatmap.pdf, p_max_COMP2_heatmap.pdf,
│       │   pr_lower_heatmap.pdf, pr_upper_heatmap.pdf,
│       │   feasibility_ommen_heatmap.pdf,
│       │   feasibility_thermo_heatmap.pdf
│       ├── PEC_breakdown.pdf, PEC_per_kW_heatmap.pdf, (Stage 6)
│       │   TCI_per_kW_heatmap.pdf
│       ├── cP_sorted_by_T_src.pdf,                  (Stage 6)
│       │   cP_vs_e1c.pdf, cP_vs_gas.pdf, cP_vs_FLH.pdf,
│       │   cP_vs_ED_EL_by_Tsrc.pdf
│       ├── price_sensitivity_2d.pdf,                (Stage 6)
│       │   price_sensitivity_2d_FLH{5000,…,7000}.pdf,
│       │   sensitivity_FLH_e1_bestpairs.pdf,
│       │   sensitivity_FLH_gas_bestpairs.pdf
│       ├── tsatsaronis_quadrant.pdf,                (Stage 6)
│       │   economic_vs_exergoeconomic_ranking.pdf,
│       │   best_designs_per_T_src.pdf,
│       │   lift_share_vs_T_src.pdf,
│       │   LS_choice_agreement.pdf, LS_pareto_trade_off.pdf,
│       │   LS_regret_bars.pdf
│       └── cost_breakdown_aggregated.pdf,           (Stage 6)
│           cost_breakdown_per_design.pdf,
│           z_breakdown_aggregated.pdf,
│           z_breakdown_per_design.pdf,
│           z_components_breakdown_per_design.pdf,
│           CDZ_components_breakdown_per_design.pdf,
│           ED_components_breakdown_per_design.pdf,
│           exergy_balance_breakdown_per_design.pdf
│
└── case_steam_compare/                      (Stage 7 — cross-T_steam plots)
    ├── compare_cP_best_vs_T_steam.pdf
    ├── compare_cP_heatmap.pdf
    ├── exergy_destruction_base.pdf
    ├── sensitivity_FLH_cel.pdf
    ├── cP_heatmap_Tsrc_Tsteam.pdf           (standalone: plot_cP_Tsrc_Tsteam.py)
    ├── cP_heatmap_Tsrc_Tsteam_R717_R600.pdf (standalone, cost-optimal pair only)
    ├── cP_vs_Tsrc_per_Tsteam.pdf
    └── cP_vs_Tsrc_per_Tsteam_R717_R600.pdf
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
