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

# Calibration CSV: allow override via env var, otherwise use repo-relative file
DEFAULT_CALIB_PATH = Path(__file__).resolve().parents[1] / "data" / "calibrated_constrained_parameters.csv"
CALIB_PATH = Path(os.getenv("FAIR_CALIB_PATH", str(DEFAULT_CALIB_PATH)))

# RCMIP and mapping files (you use "raux" not "aux")
REPO_ROOT = Path(__file__).resolve().parents[1]
RCMIP_DIR = REPO_ROOT / "data" / "rcmip"
RAUX_DIR = REPO_ROOT / "data" / "raux"

RCMIP_EMMS_FILE = RCMIP_DIR / "rcmip-emissions-annual-means-v5-1-0.csv"
RCMIP_CONCS_FILE = RCMIP_DIR / "rcmip-concentrations-annual-means-v5-1-0.csv"
RCMIP_MAP_FILE = RAUX_DIR / "FaIRv2.0.0-alpha_RCMIP_inputmap.csv"


# =========================================================
# Core helpers
# =========================================================

def _fill_climate_from_csv(f: FAIR, config: str = "default", calib_path: Path = CALIB_PATH) -> None:
    """
    Load calibrated climate parameters from CSV and apply to the FaIR instance.
    """
    if not Path(calib_path).exists():
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


def _require_files_exist(paths) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s) in the GitHub repo:\n" + "\n".join(f"- {p}" for p in missing)
        )


def _numeric_year_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if str(c).isdigit()]


def _init_fair(scenario: str, start_year: int, end_year: int, fill_rcmip: bool = True) -> FAIR:
    """
    Initialise FaIR and optionally fill inputs from LOCAL RCMIP CSVs using the mapping file.

    IMPORTANT:
    - No internet downloads
    - Skips concentration variables not present in RCMIP concentrations (e.g., NOx, BC, OC, NH3, etc.)
    """
    f = FAIR()
    f.define_time(start_year, end_year, 1)
    f.define_scenarios([scenario])
    f.define_configs(["default"])

    species, props = read_properties()
    f.define_species(species, props)

    f.allocate()
    f.fill_species_configs()

    if fill_rcmip:
        _require_files_exist([RCMIP_EMMS_FILE, RCMIP_CONCS_FILE, RCMIP_MAP_FILE])

        mapping = pd.read_csv(RCMIP_MAP_FILE, index_col=0)

        # Load RCMIP data locally
        rcmip_emms = pd.read_csv(RCMIP_EMMS_FILE)
        rcmip_concs = pd.read_csv(RCMIP_CONCS_FILE)

        # Defensive: strip column names
        rcmip_emms.columns = [str(c).strip() for c in rcmip_emms.columns]
        rcmip_concs.columns = [str(c).strip() for c in rcmip_concs.columns]

        # Set expected index
        rcmip_emms = rcmip_emms.set_index(["Region", "Scenario", "Variable"])
        rcmip_concs = rcmip_concs.set_index(["Region", "Scenario", "Variable"])

        years_needed = set(range(start_year, end_year + 1))

        # ---------- emissions mapping ----------
        def rcmip_to_fair_emms(ssp: str) -> pd.DataFrame:
            keys = mapping["RCMIP_emms_key"].dropna()
            ssp = str(ssp).strip()

            # Validate scenario exists
            try:
                base = rcmip_emms.loc[("World", ssp)]
            except KeyError:
                scenarios = sorted(set(rcmip_emms.index.get_level_values("Scenario")))
                raise ValueError(
                    f"No RCMIP emissions rows for Region='World', Scenario='{ssp}'. "
                    f"Example scenarios in file: {scenarios[:20]}"
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

            inv_map = {v: k for k, v in keys.to_dict().items()}  # map RCMIP var -> FaIR var
            sub.index = sub.index.map(inv_map)

            if "RCMIP_emms_scaling" in mapping.columns:
                scale = mapping.loc[sub.index, "RCMIP_emms_scaling"].astype(float)
                sub = sub.T.mul(scale, axis=1).T

            out = sub.T
            out.index = out.index.astype(int)
            out = out.loc[out.index.isin(years_needed)]
            return out.apply(pd.to_numeric, errors="coerce")

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
                    f"Example scenarios in file: {scenarios[:20]}"
                )

            available_vars = set(base.index.get_level_values("Variable"))
            requested = list(keys.values)
            available = [v for v in requested if v in available_vars]
            missing = [v for v in requested if v not in available_vars]

            # Key change: DO NOT FAIL if some are missing (NOx/BC/etc are emissions-only)
            if not available:
                raise ValueError(
                    f"No mapped concentration variables were found in RCMIP concentrations for scenario '{ssp}'. "
                    f"First missing examples: {missing[:20]}"
                )
            if missing:
                print(f"[FaIR] Skipping {len(missing)} concentration variables not in RCMIP concentrations. Example: {missing[:5]}")

            sub = rcmip_concs.loc[("World", ssp, available)].droplevel([0, 1])
            sub = sub[_numeric_year_columns(sub)]

            inv_map = {v: k for k, v in keys.to_dict().items()}  # map RCMIP var -> FaIR var
            sub.index = sub.index.map(inv_map)

            if "RCMIP_concs_scaling" in mapping.columns:
                scale = mapping.loc[sub.index, "RCMIP_concs_scaling"].astype(float)
                sub = sub.T.mul(scale, axis=1).T

            out = sub.T
            out.index = out.index.astype(int)
            out = out.loc[out.index.isin(years_needed)]
            return out.apply(pd.to_numeric, errors="coerce")

        emms_df = rcmip_to_fair_emms(scenario)
        conc_df = rcmip_to_fair_concs(scenario)

        # Write into FaIR arrays
        # Use the actual xarray coordinate lists to avoid mismatches
        emms_species = set(f.emissions.specie.values)
        conc_species = set(f.concentration.specie.values)

        for sp in emms_df.columns:
            if sp in emms_species:
                series = emms_df[sp].dropna()
                if not series.empty:
                    f.emissions.loc[dict(scenario=scenario, config="default", specie=sp, time=series.index)] = series.values

        for sp in conc_df.columns:
            if sp in conc_species:
                series = conc_df[sp].dropna()
                if not series.empty:
                    f.concentration.loc[dict(scenario=scenario, config="default", specie=sp, time=series.index)] = series.values

    else:
        # Custom mode (zeros; caller can overwrite selected species)
        f.emissions.loc[dict(scenario=scenario, config="default")] = 0.0
        f.concentration.loc[dict(scenario=scenario, config="default")] = 0.0
        f.forcing.loc[dict(scenario=scenario, config="default")] = 0.0

    # Initialise state variables
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0.0)
    initialise(f.temperature, 0.0)
    initialise(f.cumulative_emissions, 0.0)
    initialise(f.airborne_emissions, 0.0)
    initialise(f.ocean_heat_content_change, 0.0)

    # Apply calibrated climate parameters
    _fill_climate_from_csv(f, config="default", calib_path=CALIB_PATH)

    return f


def _extract_timeseries_aligned_on_timepoints(f: FAIR, scenario: str) -> dict:
    """
    Extract the outputs you plot in the app into plain numpy arrays.
    """
    years = f.timepoints.values.astype(int)

    # Temperature: usually shape (time,)
    temp = np.asarray(f.temperature.sel(scenario=scenario, config="default").values).squeeze()

    # Total forcing: usually shape (time,)
    forcing = np.asarray(f.forcing.sel(scenario=scenario, config="default").sum("specie").values).squeeze()

    # CO2 concentration:
    # Species naming in FaIR often uses "CO2" for concentration.
    co2_conc = None
    if "CO2" in f.concentration.specie.values:
        co2_conc = np.asarray(f.concentration.sel(scenario=scenario, config="default", specie="CO2").values).squeeze()

    return {
        "years": years,
        "temperature": temp,
        "total_forcing": forcing,
        "co2_concentration": co2_conc,
    }


# =========================================================
# Public API functions used by your Streamlit app
# =========================================================

def run_fair_rcmip_scenario(scenario: str, start_year: int = 1850, end_year: int = 2100) -> dict:
    f = _init_fair(scenario, start_year, end_year, fill_rcmip=True)
    f.run()
    return _extract_timeseries_aligned_on_timepoints(f, scenario)


def run_fair_custom_single_gas_emissions(
    gas: str,
    df: pd.DataFrame,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> dict:
    """
    gas in {"CO2", "CH4", "N2O"}

    Input df requirements:
    - year column
    - one emissions column:
        for CO2:
          either "co2_total" OR "co2_fossil" and/or "co2_afolu"
        for CH4:
          "ch4"
        for N2O:
          "n2o"
    """
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
        da = f.emissions.sel(scenario=scenario, config="default", specie=specie_name)
        arr = np.asarray(da.values).squeeze()
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
    return _extract_timeseries_aligned_on_timepoints(f, scenario)


def run_fair_multi_reduction_scenario(
    base_scenario: str,
    reductions: Dict[str, float],
    reduction_start_year: int,
    reduction_end_year: int,
    start_year: int = 1850,
    end_year: int = 2100,
) -> dict:
    """
    reductions example:
      {
        "CO2": 0.30,  # 30% reduction by end year
        "CH4": 0.10,
        "N2O": 0.00
      }

    Applies a LINEAR RAMP and holds after end year.
    """
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
            da = f.emissions.sel(scenario=base_scenario, config="default", specie=specie)
            arr = np.asarray(da.values).squeeze().copy()

            for yr in range(start_year, end_year):
                idx = yr - start_year
                if 0 <= idx < len(arr):
                    arr[idx] *= ramp_factor(yr, r)

            f.emissions.loc[dict(scenario=base_scenario, config="default", specie=specie)] = arr

    f.run()
    red = _extract_timeseries_aligned_on_timepoints(f, base_scenario)

    return {
        "years": base["years"],
        "base": base,
        "reduced": red,
    }
