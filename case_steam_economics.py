"""
case_steam_economics.py — Exergoeconomic analysis for the OK designs of
the case study, for one steam temperature.

Pipeline:

  1. Load case_steam_<T>_enriched.csv produced by enrich_screen.py.
  2. Filter to status_4state != "NOSOLVE" AND ls ∈ {0.30, 0.40, 0.50}
     (every thermodynamically-feasible design; drop LS = 0.60, 0.70). The
     Ommen p / T_disch / V̇ limits are informational and do not exclude.
  3. For each design, re-run simulate_hthp to obtain a live `sim` dict
     with the ExergyAnalysis object (this is required by run_economics —
     the CSV alone is not enough).
  4. Run economics at the *native* simulation scale (m_steam =
     ``config.M_STEAM`` kg/s, Q_H = M_STEAM · Δh_sat,water(T_steam)).
     The c_P [EUR/GJ] from this is the intensive metric used for ranking
     designs. PEC values are reported at the same native scale (no virtual
     rescaling).
  5. Sensitivity sweeps on electricity price (E1_C_RANGE in config.py
     around BASE_E1_C) and full-load hours (FULL_LOAD_HOURS_RANGE defined
     locally below, around BASE_FULL_LOAD_HOURS).
  6. Write CSVs (economics_base, economics_pec_breakdown,
     economics_sensitivity_e1, economics_sensitivity_FLH) plus a
     gas-heater reference dump under
     results/case_steam_<int(T_steam)>/economics/.

The pipeline also passes ``co2_price_eur_per_t = BASE_CO2_PRICE`` to
``run_economics_gas_heater``, so the gas-reference c_P shown in CSVs and
plots includes the EU ETS / BEHG carbon charge by default (set
``BASE_CO2_PRICE = 0`` in config.py to reproduce the Ommen 2015 baseline).

Invoke through main.py or directly:
    python case_steam_economics.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings

import pandas as pd
from CoolProp.CoolProp import PropsSI

from config import (
    BASE_CO2_PRICE, BASE_E1_C, BASE_FULL_LOAD_HOURS, BASE_GAS_C,
    E1_C_RANGE, M_STEAM, T_STEAM_CASE_DEFAULT,
    lift_share_to_T34, m_steam_label, p_water_for_T_steam,
)
from economics import (
    F_INSTALL, _CI_RATIO, _CEPCI_TO_REF_YEAR, COST_REF_YEAR,
    pec_compressor, pec_motor, pec_plate_hx,
    run_economics,
)
from models import simulate_hthp


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


# Path globals set per case-study T_steam by ``_set_paths_for_T_steam``.
CASE_DIR = ""
DATA_DIR = ""
GAS_HEATER_DIR = ""
ENRICHED_CSV = ""


def _set_paths_for_T_steam(T_steam):
    """Rewrite the module-level path globals for a given T_steam."""
    global CASE_DIR, DATA_DIR, GAS_HEATER_DIR, ENRICHED_CSV
    from config import case_results_dir, case_data_dir, case_gas_heater_dir
    CASE_DIR = case_results_dir(T_steam)
    DATA_DIR = case_data_dir(T_steam)
    GAS_HEATER_DIR = case_gas_heater_dir(T_steam)
    ENRICHED_CSV = os.path.join(CASE_DIR, f"case_steam_{int(T_steam)}_enriched.csv")

# Full-load-hours sensitivity sweep (5000-7500 h/a, 500 h steps).
FULL_LOAD_HOURS_RANGE = list(range(5000, 7501, 500))

# E1_C_RANGE in config is in EUR/MWh; run_economics expects ct/kWh.
def _eur_mwh_to_ct_kwh(p_eur_mwh: float) -> float:
    return p_eur_mwh / 10.0  # 1 EUR/MWh = 0.1 ct/kWh


# ── Gas heater persistence ──────────────────────────────────────────────────
# Per-stream and per-component dumps from the TESPy + exerpy gas-burner
# simulation, written under economics/gas_heater/ as the economic reference.

def _gas_heater_connections_df(sim):
    """Build state-point CSV for the gas burner streams.

    Streams: g1 (air in), g2 (CH4 fuel in), g3 (combustion products),
    g4 (exhaust out), w1 (water in), w2 (saturated steam out).
    """
    label_to_role = {
        "g1": "air (in)", "g2": "CH4 fuel (in)",
        "g3": "combustion products", "g4": "flue gas (out)",
        "w1": "water (in)",          "w2": "steam (out)",
    }
    rows = []
    for label, c in sim["exerpy_data"]["connections"].items():
        if c.get("kind") != "material":
            continue
        T_K = c.get("T")
        p_Pa = c.get("p")
        h_J  = c.get("h")
        s_J  = c.get("s")
        v    = c.get("v")
        m    = c.get("m")
        rows.append({
            "label":         label,
            "stream":        label_to_role.get(label, ""),
            "from":          c.get("source_component", ""),
            "to":            c.get("target_component", ""),
            "m [kg/s]":      round(m, 6) if m is not None else "",
            "T [°C]":        round(T_K - 273.15, 2) if T_K is not None else "",
            "p [bar]":       round(p_Pa / 1e5, 4) if p_Pa is not None else "",
            "h [kJ/kg]":     round(h_J / 1e3, 3) if h_J is not None else "",
            "s [kJ/(kg·K)]": round(s_J / 1e3, 4) if s_J is not None else "",
            "v [m³/kg]":     round(v, 6) if v is not None else "",
            "e_T [J/kg]":    round(c.get("e_T", 0.0), 2),
            "e_M [J/kg]":    round(c.get("e_M", 0.0), 2),
            "e_PH [J/kg]":   round(c.get("e_PH", 0.0), 2),
        })
    order = ["g1", "g2", "g3", "g4", "w1", "w2"]
    df = pd.DataFrame(rows)
    df["_k"] = df["label"].map({l: i for i, l in enumerate(order)}).fillna(99)
    return df.sort_values("_k").drop(columns=["_k"])


def _gas_heater_components_df(sim):
    """Build component-parameter CSV (combustion chamber + heat exchanger)."""
    rows = []
    for ctype, units in sim["exerpy_data"]["components"].items():
        for name, comp in units.items():
            params = comp.get("parameters", {})
            row = {"name": name, "type": ctype}
            if ctype == "CombustionChamber":
                ti = params.get("ti")
                row["thermal input [kW]"] = round(ti / 1e3, 3) if ti else ""
                row["λ (excess air)"]    = round(params.get("lamb", 0), 3)
            elif ctype == "HeatExchanger":
                Q  = params.get("Q")
                kA = params.get("kA")
                row["Q [kW]"]      = round(Q / 1e3, 3) if Q else ""
                row["kA [kW/K]"]   = round(kA / 1e3, 3) if kA else ""
                row["ΔT_log [K]"]  = round(params.get("td_log", 0), 3)
                row["ΔT_upper [K]"] = round(params.get("ttd_u", 0), 3)
                row["ΔT_lower [K]"] = round(params.get("ttd_l", 0), 3)
                row["pr_hot"]       = round(params.get("pr1", 0), 4)
                row["pr_cold"]      = round(params.get("pr2", 0), 4)
            rows.append(row)
    return pd.DataFrame(rows)


def _write_gas_heater_export(sim, eco, out_dir):
    """Write CSVs + a full JSON snapshot for the gas-burner reference case.

    Outputs into ``out_dir``:
      * connections.csv                    — state points (g1..g4, w1, w2)
                                             with m, T, p, h, s, v, e_*
      * components.csv                     — combustion chamber + heat exchanger
                                             parameters (ti, λ, Q, kA, ttd, pr)
      * exergy_summary.csv                 — scalar exergy/cost results
                                             (E_F, E_P, E_D, ε, c_P, ṁ_CO2, …)
      * exergoeco_components.csv           — per-component F-P-D-Z table
                                             (C_F, C_P, C_D, Z, c_F, c_P, f, r)
      * exergoeco_connections_material.csv — per-stream cost rates
                                             (C^T, C^M, C^TOT, c^T, c^M, c^TOT)
      * exergoeco_connections_nonmat.csv   — power/heat connections
                                             (gas heater has none → empty)
      * gas_heater.json                    — everything in one reproducible
                                             payload (scalars + economics +
                                             exerpy_data + exergoeco tables)
    """
    os.makedirs(out_dir, exist_ok=True)

    _gas_heater_connections_df(sim).to_csv(
        os.path.join(out_dir, "connections.csv"), index=False)
    _gas_heater_components_df(sim).to_csv(
        os.path.join(out_dir, "components.csv"), index=False)

    summary_rows = [
        ("COP (= η_gas)",          round(sim["COP"], 4),       "-"),
        ("ε (exergy efficiency)",  round(sim["epsilon"], 4),   "-"),
        ("E_F (fuel chemical exergy)",  round(sim["E_F"]/1e3, 2),  "kW"),
        ("E_P (steam exergy gain)",     round(sim["E_P"]/1e3, 2),  "kW"),
        ("E_D (exergy destruction)",    round(sim["E_D"]/1e3, 2),  "kW"),
        ("Q_H (steam duty)",            round(sim["Q_H"]/1e3, 2),  "kW"),
        ("Q_gas (LHV thermal input)",   round(sim["Q_gas"]/1e3, 2),"kW"),
        ("ṁ_CO2 (combustion)",          round(sim["m_dot_CO2"]*3600, 4), "kg/h"),
        ("c_P (with CO2 charge)",  round(eco["c_P"], 3),       "EUR/GJ_ex"),
        ("Z_sum (retrofit ⇒ 0)",   round(eco["Z_sum"], 3),     "EUR/h"),
    ]
    pd.DataFrame(summary_rows, columns=["metric", "value", "unit"]).to_csv(
        os.path.join(out_dir, "exergy_summary.csv"), index=False)

    # Per-component / per-connection exergoeconomic tables. Same three CSVs
    # the HTHP per-design export writes (export_design_details.py:298), so a
    # consumer can diff HTHP vs gas-burner side by side.
    df_comp = df_mat = df_non_mat = None
    if "exergoeco" in eco:
        exergoeco = eco["exergoeco"]
        df_comp, df_mat1, df_mat2, df_non_mat = exergoeco.exergoeconomic_results(
            print_results=False)
        dup_cols = [c for c in df_mat2.columns
                    if c in df_mat1.columns and c != "Connection"]
        df_mat = df_mat1.merge(df_mat2.drop(columns=dup_cols),
                               on="Connection", how="left")
        df_comp.to_csv(os.path.join(out_dir, "exergoeco_components.csv"),
                       index=False, float_format="%.4g")
        df_mat.to_csv(os.path.join(out_dir, "exergoeco_connections_material.csv"),
                      index=False, float_format="%.4g")
        df_non_mat.to_csv(os.path.join(out_dir, "exergoeco_connections_nonmat.csv"),
                          index=False, float_format="%.4g")

    # Full JSON dump. Route numpy scalars through float()/int() here.
    def _json_safe(o):
        try:
            import numpy as _np
            if isinstance(o, _np.integer): return int(o)
            if isinstance(o, _np.floating): return float(o)
            if isinstance(o, _np.ndarray): return o.tolist()
        except Exception:
            pass
        if isinstance(o, set): return sorted(o)
        return str(o)

    # Drop the live ExergyAnalysis object from the JSON (not serialisable;
    # the exerpy_data dict already has the persisted form).
    payload = {
        "scalars": {k: v for k, v in sim.items()
                    if k not in ("exerpy_data", "ean")},
        "economics": {k: v for k, v in eco.items() if k != "exergoeco"},
        "exerpy_data": sim["exerpy_data"],
    }
    if df_comp is not None:
        payload["exergoeco"] = {
            "system_costs": exergoeco.system_costs,
            "components": df_comp.to_dict(orient="records"),
            "connections_material": df_mat.to_dict(orient="records"),
            "connections_nonmat": df_non_mat.to_dict(orient="records"),
        }
    with open(os.path.join(out_dir, "gas_heater.json"), "w") as f:
        json.dump(payload, f, indent=2, default=_json_safe)


# ── Native-scale PEC ────────────────────────────────────────────────────────

def _pec_native(sim):
    """Compute total PEC and per-component breakdown at the native simulation
    scale (m_steam = ``config.M_STEAM`` kg/s, Q_H = Δh_vap × M_STEAM).

    Returns a dict of component PEC values and total, all in reference-year
    EUR (with the same CI factor as run_economics applies). Pumps are
    excluded — brownfield-retrofit assumption: pre-existing source-loop
    circulator and feedwater pump remain in place."""
    sz = sim["sizing"]
    f1 = sim["fluid_cycle1"]
    f2 = sim["fluid_cycle2"]
    p_high_c1 = max(p["p"] for p in sim["cycle_states"]["cycle1"]["points"])
    p_high_c2 = max(p["p"] for p in sim["cycle_states"]["cycle2"]["points"])

    PEC_ref = {
        "COMP1":     (pec_compressor(sz["V_dot_comp1"], f1, p_high_c1,
                                    eta_vol=sz.get("eta_vol_comp1", 1.0)), COST_REF_YEAR),
        "COMP2":     (pec_compressor(sz["V_dot_comp2"], f2, p_high_c2,
                                    eta_vol=sz.get("eta_vol_comp2", 1.0)), COST_REF_YEAR),
        "SRC_HX":    (pec_plate_hx(sz["A_src_hx"], f1, p_high_c1), COST_REF_YEAR),
        "IHX":       (pec_plate_hx(sz["A_ihx"], f1, p_high_c1), COST_REF_YEAR),
        "SNK_HX":    (pec_plate_hx(sz["A_snk_hx"], f2, p_high_c2), COST_REF_YEAR),
        "MOT1":      (pec_motor(sz["W_comp1"], f1, p_high_c1), COST_REF_YEAR),
        "MOT2":      (pec_motor(sz["W_comp2"], f2, p_high_c2), COST_REF_YEAR),
    }
    # Apply CEPCI escalation to the analysis-reference year (e.g. 2024 → 2026
    # at r_n_OM = 2 %/a) so PEC_total in this breakdown is in the same
    # nominal-EUR basis as the Z values computed inside run_economics_hthp.
    pec = {k_: F_INSTALL * cost * _CI_RATIO[ref] * _CEPCI_TO_REF_YEAR
           for k_, (cost, ref) in PEC_ref.items()}
    pec["TOTAL"] = sum(pec.values())
    return pec


def _dh_steam_kJ_per_kg(T_steam_degC):
    p_pa = p_water_for_T_steam(T_steam_degC) * 1e5
    h_v = PropsSI("H", "P", p_pa, "Q", 1, "water")
    h_l = PropsSI("H", "P", p_pa, "Q", 0, "water")
    return (h_v - h_l) / 1e3


def main(T_steam=None):
    """Run exergoeconomic analysis for a given case-study T_steam."""
    if T_steam is None:
        T_steam = T_STEAM_CASE_DEFAULT
    _set_paths_for_T_steam(T_steam)
    if not os.path.exists(ENRICHED_CSV):
        print(f"Could not find {ENRICHED_CSV}. "
              f"Run `python main.py --t-steam {int(T_steam)}` (or stages 1+2 "
              f"individually: case_steam.main(T_steam={T_steam}) followed "
              f"by enrich_screen.main(T_steam={T_steam})) first.")
        sys.exit(1)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(GAS_HEATER_DIR, exist_ok=True)

    df = pd.read_csv(ENRICHED_CSV)
    # Consider every thermodynamically-feasible design (i.e. everything except
    # NOSOLVE — no convergence / over-critical). The Ommen compressor-envelope
    # limits (p / T_disch / V̇) are informational only and do NOT exclude a
    # design; designs that exceed the Ommen V̇ ceiling are still flagged in the
    # output via the ``V_envelope`` column.
    designs = df[(df["status_4state"] != "NOSOLVE")
                 & (df["ls"].isin([0.30, 0.40, 0.50]))].copy()
    designs = designs.sort_values(by=["f1", "f2", "ls", "T_src"]).reset_index(drop=True)
    n = len(designs)
    n_within = int((designs["status_4state"] == "OK").sum())
    print(f"Running economics on {n} thermodynamically-feasible designs at "
          f"LS ∈ {{0.30, 0.40, 0.50}}: {n_within} within the Ommen envelope, "
          f"{n - n_within} exceed one or more Ommen limits (informational)")
    print(f"  T_steam = {T_steam:.0f} °C")
    print(f"  Base electricity price: {BASE_E1_C} EUR/MWh "
          f"({_eur_mwh_to_ct_kwh(BASE_E1_C):.2f} ct/kWh)")
    print(f"  Full-load hours: {BASE_FULL_LOAD_HOURS} h/a")
    dh_steam = _dh_steam_kJ_per_kg(T_steam)
    Q_H_native_kW = dh_steam * M_STEAM
    print(f"  Native scale: {m_steam_label()} → Q_H = {Q_H_native_kW:.0f} kW "
          f"(no rescaling applied)")
    print()

    base_rows = []
    breakdown_rows = []
    sens_rows = []
    sens_flh_rows = []

    for i, design in designs.iterrows():
        f1, f2 = design["f1"], design["f2"]
        ls = float(design["ls"])
        T_src = float(design["T_src"])
        tag = f"[{i+1:>3d}/{n}] {f1}/{f2:<6s} LS={ls:.2f} T_src={T_src:.0f}"

        T34 = lift_share_to_T34(ls, T_src, T_steam=T_steam)
        sim = simulate_hthp(
            f1, f2,
            T_evap_c2_override=T34,
            T_source_in_override=T_src,
            T_steam_override=T_steam,
            source_mode="fixed_mass_flow",
            skip_ommen_check=True,
        )
        if sim is None:
            print(f"{tag}  SIM FAILED (skipped)")
            continue

        # Native-scale economics (Q_H = M_STEAM · Δh_vap of water at T_steam)
        eco = run_economics(sim, BASE_FULL_LOAD_HOURS,
                            _eur_mwh_to_ct_kwh(BASE_E1_C))
        if eco is None:
            print(f"{tag}  ECONOMICS FAILED (skipped)")
            continue

        # Native-scale PEC (the only scale we report — no virtual rescaling)
        pec = _pec_native(sim)
        pec_per_kW = pec["TOTAL"] / Q_H_native_kW

        # V_envelope flag: "V_OUT" if V̇ is outside the Ommen Table-3 range
        # for either cycle (informational — does not exclude the design),
        # "OK" otherwise. CSV booleans round-trip as "True"/"False" strings.
        def _is_true(v):
            return str(v).strip().lower() == "true"
        v_flag = "OK" if (_is_true(design["V_OK_c1"])
                          and _is_true(design["V_OK_c2"])) else "V_OUT"

        common = {
            "f1": f1, "f2": f2, "pair": f"{f1}/{f2}",
            "ls": ls, "T_src": T_src,
            "T_steam": T_steam,
            "COP": float(design["COP"]),
            "eta_Lorenz": float(design["eta_Lorenz"]),
            "V_envelope": v_flag,
        }

        # Base row
        base_rows.append({
            **common,
            "Q_H_kW": round(Q_H_native_kW, 1),
            "c_P [EUR/GJ]": round(eco["c_P"], 3),
            "Z_sum [EUR/h]": round(eco["Z_sum"], 3),
            "PEC_total [kEUR]": round(pec["TOTAL"] / 1000.0, 1),
            "PEC [EUR/kW]": round(pec_per_kW, 1),
            "V_dot_c1 [m3/h]": round(design["V_dot_c1_target [m3/h]"], 1),
            "V_dot_c2 [m3/h]": round(design["V_dot_c2_target [m3/h]"], 1),
            "E_F [kW]": round(sim["E_F"] / 1000.0, 2),
            "E_P [kW]": round(sim["E_P"] / 1000.0, 2),
            "E_D [kW]": round(sim["E_D"] / 1000.0, 2),
            "epsilon": round(sim["epsilon"], 3) if sim["epsilon"] else "",
        })

        # Component breakdown (native scale)
        for comp, val in pec.items():
            if comp == "TOTAL":
                continue
            breakdown_rows.append({
                **common,
                "component": comp,
                "PEC [EUR]": round(val, 1),
                "PEC [EUR/kW]": round(val / Q_H_native_kW, 2),
            })

        # Electricity price sensitivity (at BASE_FULL_LOAD_HOURS)
        for e1_c_eur_mwh in E1_C_RANGE:
            eco_s = run_economics(sim, BASE_FULL_LOAD_HOURS,
                                  _eur_mwh_to_ct_kwh(float(e1_c_eur_mwh)))
            if eco_s is None:
                continue
            sens_rows.append({
                **common,
                "e1_c [EUR/MWh]": float(e1_c_eur_mwh),
                "c_P [EUR/GJ]": round(eco_s["c_P"], 3),
                "Z_sum [EUR/h]": round(eco_s["Z_sum"], 3),
            })

        # Full-load-hours sensitivity (at BASE_E1_C electricity price)
        for hours in FULL_LOAD_HOURS_RANGE:
            eco_h = run_economics(sim, float(hours),
                                  _eur_mwh_to_ct_kwh(BASE_E1_C))
            if eco_h is None:
                continue
            sens_flh_rows.append({
                **common,
                "full_load_hours [h/a]": int(hours),
                "c_P [EUR/GJ]": round(eco_h["c_P"], 3),
                "Z_sum [EUR/h]": round(eco_h["Z_sum"], 3),
            })

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"{tag}  c_P={eco['c_P']:.2f} EUR/GJ  "
                  f"PEC/kW={pec_per_kW:.0f}  COP={common['COP']:.2f}")

    # Gas heater reference at base price (proper exergy basis to match HTHP c_P)
    from models import simulate_gas_heater
    from economics import run_economics_gas_heater
    gas_sim = simulate_gas_heater(eta_gas=0.90, T_steam_override=T_steam)
    gas_eco = run_economics_gas_heater(gas_sim, BASE_FULL_LOAD_HOURS,
                                        BASE_GAS_C / 10.0,  # EUR/MWh → ct/kWh
                                        co2_price_eur_per_t=BASE_CO2_PRICE)
    print()
    print(f"Gas heater reference (retrofit, no CAPEX): "
          f"c_P_gas = {gas_eco['c_P']:.2f} EUR/GJ_ex  "
          f"(at {BASE_GAS_C:.0f} EUR/MWh gas, η_boiler=0.90, "
          f"CO2 = {BASE_CO2_PRICE:.0f} EUR/t)")

    # Persist the full TESPy + exerpy state (state points, components,
    # exergy decomposition, CO2 mass flow, c_P) — same level of detail as
    # the per-design HTHP exports under designs/, but for the reference.
    _write_gas_heater_export(gas_sim, gas_eco, GAS_HEATER_DIR)
    print(f"Wrote {GAS_HEATER_DIR}/  "
          f"(connections.csv, components.csv, exergy_summary.csv, gas_heater.json)")

    # ── Write CSVs ────────────────────────────────────────────────────────
    base_df = pd.DataFrame(base_rows).sort_values(by=["c_P [EUR/GJ]"])
    base_df.to_csv(os.path.join(DATA_DIR, "economics_base.csv"), index=False)
    print(f"\nWrote {os.path.join(DATA_DIR, 'economics_base.csv')}  "
          f"({len(base_df)} rows, sorted by c_P)")

    bk_df = pd.DataFrame(breakdown_rows).sort_values(
        by=["pair", "ls", "T_src", "component"]
    )
    bk_df.to_csv(os.path.join(DATA_DIR, "economics_pec_breakdown.csv"), index=False)
    print(f"Wrote {os.path.join(DATA_DIR, 'economics_pec_breakdown.csv')}")

    sens_df = pd.DataFrame(sens_rows).sort_values(
        by=["pair", "ls", "T_src", "e1_c [EUR/MWh]"]
    )
    sens_df.to_csv(os.path.join(DATA_DIR, "economics_sensitivity_e1.csv"), index=False)
    print(f"Wrote {os.path.join(DATA_DIR, 'economics_sensitivity_e1.csv')}")

    sens_flh_df = pd.DataFrame(sens_flh_rows).sort_values(
        by=["pair", "ls", "T_src", "full_load_hours [h/a]"]
    )
    sens_flh_df.to_csv(os.path.join(DATA_DIR, "economics_sensitivity_FLH.csv"),
                        index=False)
    print(f"Wrote {os.path.join(DATA_DIR, 'economics_sensitivity_FLH.csv')}  "
          f"({FULL_LOAD_HOURS_RANGE[0]}-{FULL_LOAD_HOURS_RANGE[-1]} h/a, "
          f"{len(FULL_LOAD_HOURS_RANGE)} steps)")

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("Top 8 designs by c_P (EUR/GJ, lower = cheaper product exergy):")
    print("=" * 90)
    cols = ["pair", "ls", "T_src", "COP", "eta_Lorenz",
            "c_P [EUR/GJ]", "PEC [EUR/kW]", "V_envelope"]
    print(base_df[cols].head(8).to_string(index=False))

    print()
    print("=" * 90)
    print(f"Annex 58 EUR/kW band for 0.5-3 MWth, 110-150 degC: ~400-700 EUR/kW")
    print(f"Our PEC range (Q_H = {Q_H_native_kW:.0f} kW native): "
          f"{base_df['PEC [EUR/kW]'].min():.0f} to "
          f"{base_df['PEC [EUR/kW]'].max():.0f} EUR/kW "
          f"(median {base_df['PEC [EUR/kW]'].median():.0f})")
    in_band = ((base_df["PEC [EUR/kW]"] >= 400)
               & (base_df["PEC [EUR/kW]"] <= 700)).sum()
    print(f"  Within band: {in_band}/{len(base_df)} "
          f"({100*in_band/len(base_df):.0f} %)")
    print("=" * 90)


if __name__ == "__main__":
    main()
