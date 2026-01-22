import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from fair import FAIR
from fair.io import read_properties
from fair.interface import initialise, fill


# =========================================================
# Paths (repo-relative, Streamlit Cloud safe)
# =========================================================

DEFAULT_CALIB_PATH = Path(__file__).resolve().parents[1] / "data" / "calibrated_constrained_parameters.csv"
CALIB_PATH = Path(os.getenv("FAIR_CALIB_PATH", str(DEFAULT_CALIB_PATH)))

REPO_ROOT = Path(__file__).resolve().parents[1]
RCMIP_DIR = REPO_ROOT / "data" / "rcmip"
RAUX_DIR = REPO_ROOT / "data" / "raux"

RCMIP_EMMS_FILE = RCMIP_DIR / "rcmip-emissions-annual-means-v5-1-0.csv"
RCMIP_CONCS_FILE = RCMIP_DIR / "rcmip-concentrations-annual-means-v5-1-0.csv"
RCMIP_MAP_FILE = RAUX_DIR / "FaIRv2.0.0-alpha_RCMIP_inputmap.csv"


# =========================================================
# Helpers
# =========================================================

def _require_files_exist(paths) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s) in the GitHub repo:\n" + "\n".join(f"- {p}" for p in missing)
        )


def _numeric_year_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if str(c).isdigit()]


def _fill_climate_from_csv(f: FAIR, config: str = "default", calib_path: Path = CALIB_PATH) -> None:
    """
    Load calibrated climate parameters from CSV and apply to the FaIR instance.
    """
    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration CSV not found: {calib_path}")

    df = pd.read_csv(str(calib_path))
    if df.empty:
        raise ValueError(f"Calibration CSV is empty: {calib_path}")

    row = df.iloc[0]

    fill(f.climate_configs["ocean_heat_capacity"], [row["c1"], row["c2"], row["c3"]], config=config)
    fill(f.climate_configs["ocean_heat_transfer"], [row["kappa1"], row["kappa2"], row["kappa3"]], config=config)
    fill(f.climate_configs["deep_ocean_efficacy"], float(row["epsilon"]), config=config)
    fill(f.climate_configs["gamma_autocorrelation"], float(row["gamma"]), config=config)
    fill(f.climate_configs["stochastic_run"], False, config=config)
    fill(f.climate_configs["forcing_4co2"], 8.0, config=config)


def _clean_timeseries_df(df: pd.DataFrame, is_emissions: bool) -> pd.DataFrame:
    """
    FaIR cannot run with NaNs in driver arrays.
    RCMIP files sometimes contain NaNs for some late-century years.
    We fix it here:

    - emissions: interpolate and fill remaining NaNs with 0
    - concentrations: interpolate then forward/back fill (concs shouldn't go to 0)
    """
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")

    # Interpolate along time (index)
    df = df.interpolate(axis=0, limit_direction="both")

    if is_emissions:
        df = df.fillna(0.0)
    else:
        df = df.ffill().bfill()

    return df


# =========================================================
# Core init
# =========================================================

def _init_fair(scenario: str, start_year: int, end_year: int, fill_rcmip: bool = True) -> FAIR:
    """
    Initialise FaIR and optionally fill inputs from LOCAL RCMIP CSVs using the mapping file.

    - No internet downloads
    - Skips concentration variables not present in RCMIP concentrations (e.g., NOx/BC/OC/NH3/etc)
    - Fixes NaNs in RCMIP time series (interpolate/fill)
    """
    f = FAIR()
    f.define_time(start_year, end_year, 1)
    f.define_scenarios([scenario])
    f.define_configs(["default"])

    species, props = read_properties()
    f.define_species(species, props)

    f.allocate()
    f.fill_species_configs()

    # ---------------------------------------------------------
    # FIX #1: Initialize to 0 so no specie remains NaN by default
    # ---------------------------------------------------------
    f.emissions.loc[dict(scenario=scenario, config="default")] = 0.0
    f.concentration.loc[dict(scenario=scenario, config="default")] = 0.0
    f.forcing.loc[dict(scenario=scenario, config="default")] = 0.0

    if fill_rcmip:
        _require_files_exist([RCMIP_EMMS_FILE, RCMIP_CONCS_FILE, RCMIP_MAP_FILE])

        mapping = pd.read_csv(RCMIP_MAP_FILE, index_col=0)

        # Load RCMIP locally
        rcmip_emms = pd.read_csv(RCMIP_EMMS_FILE)
        rcmip_concs = pd.read_csv(RCMIP_CONCS_FILE)

        # Defensive: strip column names
        rcmip_emms.columns = [str(c).strip() for c in rcmip_emms.columns]
        rcmip_concs.columns = [str(c).strip() for c in rcmip_concs.columns]

        # Index (as used in FaIR helper scripts)
        rcmip_emms = rcmip_emms.set_index(["Region", "Scenario", "Variable"])
        rcmip_concs = rcmip_concs.set_index(["Region", "Scenario", "Variable"])

        years_needed = set(range(start_year, end_year + 1))

        # ---------- emissions mapping ----------
        def rcmip_to_fair_emms(ssp: str) -> pd.DataFrame:
            keys = mapping["RCMIP_emms_key"].dropna()
            ssp = str(ssp).strip()

            try:
                base = rcmip_emms.loc[("World", ssp)]
            except KeyError:
                scenarios = sorted(set(rcmip_emms.index.get_level_values("Scenario")))
                raise ValueError(
                    f"No RCMIP emissions rows for Region='World', Scenario='{ssp}'. "
                    f"Example scenarios: {scenarios[:20]}"
                )

            available_vars = set(base.index.get_level_values("Variable"))
            requested = list(keys.values)
            available = [v for v in requested if v in available_vars]
            missing = [v for v in requested if v not in available_vars]

            if not available:
                raise ValueError(
                    f"No mapped emission variables found for scenario '{ssp}'. "
                    f"First missing examples: {missing[:20]}"
                )
            if missing:
                print(f"[FaIR] Skipping {len(missing)} emission variables missing in RCMIP emissions. Example: {missing[:5]}")

            sub = rcmip_emms.loc[("World", ssp, available)].droplevel([0, 1])
            sub = sub[_numeric_year_columns(sub)]

            inv_map = {v: k for k, v in keys.to_dict().items()}
            sub.index = sub.index.map(inv_map)

            if "RCMIP_emms_scaling" in mapping.columns:
                scale = mapping.loc[sub.index, "RCMIP_emms_scaling"].astype(float)
                sub = sub.T.mul(scale, axis=1).T

            out = sub.T
            out.index = out.index.astype(int)
            out = out.loc[out.index.isin(years_needed)]
            return out

        # ---------- concentrations mapping (skip missing) ----------
        def rcmip_to_fair_concs(ssp: str) -> pd.DataFrame:
            keys = mapping["RCMIP_concs_key"].dropna()
            ssp = str(ssp).strip()

            try:
                base = rcmip_concs.loc[("World", ssp)]
            except KeyError:
                scenarios = sorted(set(rcmip_concs.index.get_level_values("Scenario")))
                raise ValueError(
                    f"No RCMIP concentrations rows for Region='World', Scenario='{ssp}'. "
                    f"Example scenarios: {scenarios[:20]}"
                )

            available_vars = set(base.index.get_level_values("Variable"))
            requested = list(keys.values)
            available = [v for v in requested if v in available_vars]
            missing = [v for v in requested if v not in available_vars]

            # KEY: Do not fail if some are missing (many are emissions-only)
            if not available:
                raise ValueError(
                    f"No mapped concentration variables were found in RCMIP concentrations for scenario '{ssp}'. "
                    f"First missing examples: {missing[:20]}"
                )
            if missing:
                print(f"[FaIR] Skipping {len(missing)} conc variables not in RCMIP concentrations. Example: {missing[:5]}")

            sub = rcmip_concs.loc[("World", ssp, available)].droplevel([0, 1])
            sub = sub[_numeric_year_columns(sub)]

            inv_map = {v: k for k, v in keys.to_dict().items()}
            sub.index = sub.index.map(inv_map)

            if "RCMIP_concs_scaling" in mapping.columns:
                scale = mapping.loc[sub.index, "RCMIP_concs_scaling"].astype(float)
                sub = sub.T.mul(scale, axis=1).T

            out = sub.T
            out.index = out.index.astype(int)
            out = out.loc[out.index.isin(years_needed)]
            return out

        emms_df = rcmip_to_fair_emms(scenario)
        conc_df = rcmip_to_fair_concs(scenario)

        # ---------------------------------------------------------
        # FIX #2: Interpolate/fill NaNs before writing to FaIR arrays
        # ---------------------------------------------------------
        emms_df = _clean_timeseries_df(emms_df, is_emissions=True)
        conc_df = _clean_timeseries_df(conc_df, is_emissions=False)

        # Write into FaIR arrays
        emms_species = set(f.emissions.specie.values)
        conc_species = set(f.concentration.specie.values)

        for sp in emms_df.columns:
            if sp in emms_species:
                series = emms_df[sp].dropna()
                if not series.empty:
                    f.emissions.loc[
                        dict(scenario=scenario, config="default", specie=sp, time=series.index)
                    ] = series.values

        for sp in conc_df.columns:
            if sp in conc_species:
                series = conc_df[sp].dropna()
                if not series.empty:
                    f.concentration.loc[
                        dict(scenario=scenario, config="default", specie=sp, time=series.index)
                    ] = series.values

    # Initialise state variables (FaIR requirement)
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0.0)
    initialise(f.temperature, 0.0)
    initialise(f.cumulative_emissions, 0.0)
    initialise(f.airborne_emissions, 0.0)
    initialise(f.ocean_heat_content_change, 0.0)

    # Apply calibrated climate parameters
    _fill_climate_from_csv(f, config="default", calib_path=CALIB_PATH)

    return f


# =========================================================
# Outputs for plotting
# =========================================================

def _extract_outputs(f: FAIR, scenario: str) -> dict:
    years = f.timepoints.values.astype(int)

    temp = np.asarray(f.temperature.sel(scenario=scenario, config="default").values).squeeze()
    total_forcing = np.asarray(
        f.forcing.sel(scenario=scenario, config="default").sum("specie").values
    ).squeeze()

    co2_conc = None
    if "CO2" in f.concentration.specie.values:
        co2_conc = np.asarray(
            f.concentration.sel(scenario=scenario, config="default", specie="CO2").values
        ).squeeze()

    return {
        "years": years,
        "temperature": temp,
        "total_forcing": total_forcing,
        "co2_concentration": co2_conc,
    }


# =========================================================
# Public API functions used by Streamlit app
# =========================================================

def run_fair_rcmip_scenario(scenario: str, start_year: int = 1850, end_year: int = 2100) -> dict:
    f = _init_fair(scenario, start_year, end_year, fill_rcmip=True)
    f.run()
    return _extract_outputs(f, scenario)


def run_fair_custom_single_gas_emissions(
    gas: str,
    df: pd.DataFrame,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> dict:
    gas = gas.upper()
    if "year" not in df.columns:
        raise ValueError("CSV must contain a 'year' column.")

    df = df.copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year")

    if start_year is None:
        start_year = int(df["year"].min())
    if end_year is None:
        end_year = int(df["year"].max()) + 1

    scenario = "custom"
    f = _init_fair(scenario, start_year, end_year, fill_rcmip=False)

    # helper to set an emissions specie from df series
    def set_emissions(specie_name: str, years_in: np.ndarray, values: np.ndarray):
        arr = np.asarray(f.emissions.sel(scenario=scenario, config="default", specie=specie_name).values).squeeze()
        for yr, val in zip(years_in, values):
            idx = yr - start_year
            if 0 <= idx < len(arr):
                arr[idx] = float(val)
        f.emissions.loc[dict(scenario=scenario, config="default", specie=specie_name)] = arr

    years_in = df["year"].values

    if gas == "CO2":
        if "co2_total" in df.columns:
            set_emissions("CO2 FFI", years_in, df["co2_total"].values)
            set_emissions("CO2 AFOLU", years_in, np.zeros_like(df["co2_total"].values, dtype=float))
        else:
            if "co2_fossil" not in df.columns and "co2_afolu" not in df.columns:
                raise ValueError("For CO2, provide 'co2_total' OR at least one of 'co2_fossil'/'co2_afolu'.")
            if "co2_fossil" in df.columns:
                set_emissions("CO2 FFI", years_in, df["co2_fossil"].values)
            if "co2_afolu" in df.columns:
                set_emissions("CO2 AFOLU", years_in, df["co2_afolu"].values)

    elif gas == "CH4":
        if "ch4" not in df.columns:
            raise ValueError("For CH4, provide a 'ch4' column.")
        set_emissions("CH4", years_in, df["ch4"].values)

    elif gas == "N2O":
        if "n2o" not in df.columns:
            raise ValueError("For N2O, provide an 'n2o' column.")
        set_emissions("N2O", years_in, df["n2o"].values)

    else:
        raise ValueError("gas must be one of: CO2, CH4, N2O")

    f.run()
    return _extract_outputs(f, scenario)


def run_fair_multi_reduction_scenario(
    base_scenario: str,
    reductions: Dict[str, float],
    reduction_start_year: int,
    reduction_end_year: int,
    start_year: int = 1850,
    end_year: int = 2100,
) -> dict:
    base = run_fair_rcmip_scenario(base_scenario, start_year, end_year)

    f = _init_fair(base_scenario, start_year, end_year, fill_rcmip=True)

    gas_to_species = {
        "CO2": ["CO2 FFI", "CO2 AFOLU"],
        "CH4": ["CH4"],
        "N2O": ["N2O"],
    }

    window = max(1, reduction_end_year - reduction_start_year)

    def ramp_factor(year: int, r: float) -> float:
        if year < reduction_start_year:
            return 1.0
        if year <= reduction_end_year:
            frac = (year - reduction_start_year) / window
            return 1.0 - r * frac
        return 1.0 - r

    for gas, r in reductions.items():
        r = float(r)
        if r <= 0:
            continue
        if gas not in gas_to_species:
            continue

        for specie in gas_to_species[gas]:
            arr = np.asarray(f.emissions.sel(scenario=base_scenario, config="default", specie=specie).values).squeeze().copy()

            for yr in range(start_year, end_year):
                idx = yr - start_year
                if 0 <= idx < len(arr):
                    arr[idx] *= ramp_factor(yr, r)

            f.emissions.loc[dict(scenario=base_scenario, config="default", specie=specie)] = arr

    f.run()
    reduced = _extract_outputs(f, base_scenario)

    return {
        "years": base["years"],
        "base": base,
        "reduced": reduced,
    }
