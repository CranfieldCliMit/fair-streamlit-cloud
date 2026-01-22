import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from fair import FAIR
from fair.io import read_properties
from fair.interface import initialise, fill


# =========================================================
# Repo paths (Streamlit Cloud safe)
# =========================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

RCMIP_DIR = DATA_DIR / "rcmip"
RAUX_DIR = DATA_DIR / "raux"

RCMIP_EMMS_FILE = RCMIP_DIR / "rcmip-emissions-annual-means-v5-1-0.csv"
RCMIP_CONCS_FILE = RCMIP_DIR / "rcmip-concentrations-annual-means-v5-1-0.csv"  # optional output comparison
RCMIP_MAP_FILE = RAUX_DIR / "FaIRv2.0.0-alpha_RCMIP_inputmap.csv"

DEFAULT_CALIB_PATH = DATA_DIR / "calibrated_constrained_parameters.csv"
CALIB_PATH = Path(os.getenv("FAIR_CALIB_PATH", str(DEFAULT_CALIB_PATH)))


# =========================================================
# Small utilities
# =========================================================

def _require_files_exist(paths) -> None:
    missing = [Path(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s) in the GitHub repo:\n" + "\n".join(f"- {p}" for p in missing)
        )


def _to_year_array(timepoints) -> np.ndarray:
    """FaIR timepoints can be numpy or xarray-like."""
    years = np.asarray(getattr(timepoints, "values", timepoints)).astype(int)
    return np.ravel(years)


def _to_1d(a) -> np.ndarray:
    """Convert potentially multi-dim FaIR outputs to 1D (time)."""
    a = np.asarray(a)
    if a.ndim > 1:
        # average over any non-time axes (avoids ravel interleaving artifacts)
        a = np.nanmean(a, axis=tuple(range(1, a.ndim)))
    return np.ravel(a)


def _align_series_to_years(series_by_year: pd.Series, years: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Align a Series indexed by year to a 1D array of model years.
    Interpolates gaps; emissions fill_value=0 is safe for missing tails.
    """
    s = series_by_year.copy()
    s.index = s.index.astype(int)
    s = s.sort_index()

    # Reindex to model years
    s = s.reindex(years)

    # numeric + interpolate
    s = pd.to_numeric(s, errors="coerce").astype(float)
    s = s.interpolate(limit_direction="both")

    if fill_value is not None:
        s = s.fillna(fill_value)
    else:
        s = s.ffill().bfill()

    return s.to_numpy(dtype=float)


def _fill_climate_from_csv(f: FAIR, config: str = "default", calib_path: Path = CALIB_PATH) -> None:
    """Apply your calibrated parameters into FaIR climate configs."""
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


# =========================================================
# Core: Initialise + fill from RCMIP emissions (multi-species)
# =========================================================

def _init_fair(scenario: str, start_year: int, end_year: int, fill_rcmip: bool = True) -> FAIR:
    """
    Initialise FaIR.

    For SSP runs:
    - Fill *multi-species emissions* from local RCMIP emissions CSV using the mapping file.
    - DO NOT overwrite concentrations from RCMIP (FaIR computes concentrations/forcing/temperature).

    For custom runs:
    - Start from zeros and let caller set emissions.
    """
    f = FAIR()
    f.define_time(start_year, end_year, 1)
    f.define_scenarios([scenario])
    f.define_configs(["default"])

    species, props = read_properties()
    f.define_species(species, props)

    f.allocate()
    f.fill_species_configs()

    # Always start deterministic: no NaNs anywhere
    f.emissions.loc[dict(scenario=scenario, config="default")] = 0.0
    f.concentration.loc[dict(scenario=scenario, config="default")] = 0.0
    f.forcing.loc[dict(scenario=scenario, config="default")] = 0.0

    years = _to_year_array(f.timepoints)

    if fill_rcmip:
        _require_files_exist([RCMIP_EMMS_FILE, RCMIP_MAP_FILE, CALIB_PATH])

        mapping = pd.read_csv(RCMIP_MAP_FILE, index_col=0)

        # Load RCMIP emissions locally
        rcmip_emms = pd.read_csv(RCMIP_EMMS_FILE)
        rcmip_emms.columns = [str(c).strip() for c in rcmip_emms.columns]
        rcmip_emms = rcmip_emms.set_index(["Region", "Scenario", "Variable"])

        # Extract for World + scenario
        try:
            emms_world = rcmip_emms.loc[("World", str(scenario).strip())]
        except KeyError:
            scenarios = sorted(set(rcmip_emms.index.get_level_values("Scenario")))
            raise ValueError(
                f"Scenario '{scenario}' not found in RCMIP emissions for Region='World'. "
                f"Example scenarios in file: {scenarios[:20]}"
            )

        # Keep only numeric year columns
        year_cols = [c for c in emms_world.columns if str(c).isdigit()]
        if not year_cols:
            raise ValueError("No numeric year columns found in RCMIP emissions file.")
        emms_world = emms_world[year_cols]
        emms_world.columns = emms_world.columns.astype(int)

        # Mapping: FaIR specie (index) -> RCMIP variable key
        if "RCMIP_emms_key" not in mapping.columns:
            raise ValueError("Mapping file missing 'RCMIP_emms_key' column.")

        map_emms = mapping["RCMIP_emms_key"].dropna()

        fair_emms_species = set(f.emissions.specie.values)

        filled_count = 0
        skipped_not_in_fair = 0
        skipped_missing_in_rcmip = 0

        # Fill each mapped specie
        for fair_specie, rcmip_var in map_emms.items():
            fair_specie = str(fair_specie).strip()
            rcmip_var = str(rcmip_var).strip()

            if fair_specie not in fair_emms_species:
                skipped_not_in_fair += 1
                continue

            if rcmip_var not in emms_world.index:
                skipped_missing_in_rcmip += 1
                continue

            row = emms_world.loc[rcmip_var]
            row = pd.to_numeric(row, errors="coerce")

            # Scaling if present
            scale = 1.0
            if "RCMIP_emms_scaling" in mapping.columns and fair_specie in mapping.index:
                try:
                    scale = float(mapping.loc[fair_specie, "RCMIP_emms_scaling"])
                except Exception:
                    scale = 1.0

            series_by_year = pd.Series(row.values * scale, index=emms_world.columns.astype(int))
            aligned = _align_series_to_years(series_by_year, years, fill_value=0.0)

            # Write full vector (FaIR emission arrays are aligned to timepoints)
            f.emissions.loc[dict(scenario=scenario, config="default", specie=fair_specie)] = aligned
            filled_count += 1

        print(f"[FaIR] Filled emissions for {filled_count} species from RCMIP mapping.")
        print(f"[FaIR] Skipped (not in FaIR emissions): {skipped_not_in_fair}")
        print(f"[FaIR] Skipped (missing in RCMIP emissions): {skipped_missing_in_rcmip}")

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


# =========================================================
# Outputs (MATCHES app.py EXPECTATIONS)
# =========================================================

def _extract_outputs(f: FAIR, scenario: str) -> dict:
    years = _to_year_array(f.timepoints)

    # Temperature
    temp = _to_1d(f.temperature.sel(scenario=scenario, config="default").values)

    # Align lengths (FaIR can have N vs N+1 differences)
    n = min(len(years), len(temp))
    years = years[:n]
    temp = temp[:n]

    # Temperature anomaly baseline 1850–1900
    baseline_mask = (years >= 1850) & (years <= 1900)
    if np.any(baseline_mask):
        baseline = float(np.nanmean(temp[baseline_mask]))
    else:
        k = min(50, len(temp))
        baseline = float(np.nanmean(temp[:k])) if k > 0 else 0.0
    temp_anomaly = temp - baseline

    # Concentrations dict for app.py: out["conc"][gas]
    conc = {}
    for gas in ["CO2", "CH4", "N2O"]:
        if gas in f.concentration.specie.values:
            arr = _to_1d(f.concentration.sel(scenario=scenario, config="default").sel(specie=gas).values)
            conc[gas] = arr[:n] if len(arr) >= n else np.pad(arr, (0, n - len(arr)), constant_values=np.nan)
        else:
            conc[gas] = np.full(n, np.nan, dtype=float)

    # Forcing dict for app.py: out["forc"][gas]
    forc = {}
    forcing_species = set(f.forcing.specie.values)
    for gas in ["CO2", "CH4", "N2O"]:
        if gas in forcing_species:
            arr = _to_1d(f.forcing.sel(scenario=scenario, config="default").sel(specie=gas).values)
            forc[gas] = arr[:n] if len(arr) >= n else np.pad(arr, (0, n - len(arr)), constant_values=np.nan)
        else:
            forc[gas] = np.full(n, np.nan, dtype=float)

    # Total forcing for app.py: out["total_forcing"]
    total_forcing = _to_1d(f.forcing.sel(scenario=scenario, config="default").sum("specie").values)
    total_forcing = total_forcing[:n] if len(total_forcing) >= n else np.pad(total_forcing, (0, n - len(total_forcing)), constant_values=np.nan)

    return {
        "years": years,
        "temp_anomaly": temp_anomaly,
        "conc": conc,
        "forc": forc,
        "total_forcing": total_forcing,
    }


# =========================================================
# Public functions used by app.py
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
    """
    Custom run driven by a single gas emissions series.
    This keeps your app functionality; multi-species custom would need more inputs.
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

    years = _to_year_array(f.timepoints)

    # Build a series for the requested gas
    if gas == "CO2":
        if "co2_total" in df.columns:
            s = pd.Series(df["co2_total"].values, index=df["year"].values)
            values = _align_series_to_years(s, years, fill_value=0.0)
            # Try both common naming conventions
            if "CO2 FFI" in f.emissions.specie.values:
                f.emissions.loc[dict(scenario=scenario, config="default", specie="CO2 FFI")] = values
            if "CO2 AFOLU" in f.emissions.specie.values:
                f.emissions.loc[dict(scenario=scenario, config="default", specie="CO2 AFOLU")] = 0.0
            if "carbon_dioxide" in f.emissions.specie.values:
                f.emissions.loc[dict(scenario=scenario, config="default", specie="carbon_dioxide")] = values
        else:
            raise ValueError("For CO2 custom run, provide column 'co2_total'.")

    elif gas == "CH4":
        if "ch4" not in df.columns:
            raise ValueError("For CH4 custom run, provide column 'ch4'.")
        s = pd.Series(df["ch4"].values, index=df["year"].values)
        values = _align_series_to_years(s, years, fill_value=0.0)
        if "CH4" in f.emissions.specie.values:
            f.emissions.loc[dict(scenario=scenario, config="default", specie="CH4")] = values
        if "methane" in f.emissions.specie.values:
            f.emissions.loc[dict(scenario=scenario, config="default", specie="methane")] = values

    elif gas == "N2O":
        if "n2o" not in df.columns:
            raise ValueError("For N2O custom run, provide column 'n2o'.")
        s = pd.Series(df["n2o"].values, index=df["year"].values)
        values = _align_series_to_years(s, years, fill_value=0.0)
        if "N2O" in f.emissions.specie.values:
            f.emissions.loc[dict(scenario=scenario, config="default", specie="N2O")] = values
        if "nitrous_oxide" in f.emissions.specie.values:
            f.emissions.loc[dict(scenario=scenario, config="default", specie="nitrous_oxide")] = values

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
    """
    Applies linear emission reductions ONLY to selected gases on top of full multi-species SSP emissions.
    (Other species remain as in base SSP.)
    """
    f = _init_fair(base_scenario, start_year, end_year, fill_rcmip=True)

    years = _to_year_array(f.timepoints)
    n = len(years)

    window = max(1, reduction_end_year - reduction_start_year)

    def ramp_factor(year: int, r: float) -> float:
        if year < reduction_start_year:
            return 1.0
        if year <= reduction_end_year:
            frac = (year - reduction_start_year) / window
            return 1.0 - r * frac
        return 1.0 - r

    # Gas -> likely emissions specie names in FaIR
    gas_to_species = {
        "CO2": ["CO2 FFI", "CO2 AFOLU", "carbon_dioxide"],
        "CH4": ["CH4", "methane"],
        "N2O": ["N2O", "nitrous_oxide"],
    }

    for gas, r in reductions.items():
        gas = str(gas).upper()
        r = float(r)
        if r <= 0.0:
            continue
        if gas not in gas_to_species:
            continue

        for sp in gas_to_species[gas]:
            if sp not in f.emissions.specie.values:
                continue

            arr = np.asarray(f.emissions.sel(scenario=base_scenario, config="default", specie=sp).values).squeeze()
            arr = np.ravel(arr)
            m = min(len(arr), n)
            arr = arr[:m].copy()

            for i in range(m):
                yr = int(years[i])
                arr[i] *= ramp_factor(yr, r)

            # write back (pad if needed)
            out_arr = np.asarray(f.emissions.sel(scenario=base_scenario, config="default", specie=sp).values).squeeze()
            out_arr = np.ravel(out_arr)
            out_arr[:m] = arr
            f.emissions.loc[dict(scenario=base_scenario, config="default", specie=sp)] = out_arr

    f.run()
    return _extract_outputs(f, base_scenario)
