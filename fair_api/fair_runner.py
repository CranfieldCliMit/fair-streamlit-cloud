import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from fair import FAIR
from fair.io import read_properties
from fair.interface import initialise, fill


# -----------------------------
# Paths
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RCMIP_DIR = DATA_DIR / "rcmip"
DEFAULT_CALIB_PATH = DATA_DIR / "calibrated_constrained_parameters.csv"
CALIB_PATH = Path(os.getenv("FAIR_CALIB_PATH", str(DEFAULT_CALIB_PATH)))

# These filenames must match what FaIR requests (v5.1.0)
LOCAL_RCMIP_FILES = {
    "rcmip-emissions-annual-means-v5-1-0.csv": RCMIP_DIR / "rcmip-emissions-annual-means-v5-1-0.csv",
    "rcmip-concentrations-annual-means-v5-1-0.csv": RCMIP_DIR / "rcmip-concentrations-annual-means-v5-1-0.csv",
    "rcmip-radiative-forcing-annual-means-v5-1-0.csv": RCMIP_DIR / "rcmip-radiative-forcing-annual-means-v5-1-0.csv",
}


def _require_files_exist(paths) -> None:
    missing = [Path(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s) in the GitHub repo:\n" + "\n".join(f"- {p}" for p in missing)
        )


def _to_year_array(timepoints) -> np.ndarray:
    years = np.asarray(getattr(timepoints, "values", timepoints)).astype(int)
    return np.ravel(years)


def _to_1d(a) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim > 1:
        a = np.nanmean(a, axis=tuple(range(1, a.ndim)))
    return np.ravel(a)


def _fill_climate_from_csv(f: FAIR, config: str = "default", calib_path: Path = CALIB_PATH) -> None:
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


# -----------------------------
# Key trick: patch pooch.retrieve
# -----------------------------
def _patch_pooch_for_local_rcmip():
    """
    Monkey-patch pooch.retrieve so that when FaIR tries to download the RCMIP CSVs
    from S3, it instead copies our local repo files into the requested cache location.

    This lets f.fill_from_rcmip() work unchanged.
    """
    import pooch  # type: ignore

    original_retrieve = pooch.retrieve

    def local_retrieve(url, known_hash=None, fname=None, path=None, processor=None, downloader=None, progressbar=False):
        # Decide which filename FaIR is asking for
        requested_name = fname if fname is not None else str(url).split("/")[-1]
        requested_name = str(requested_name)

        if requested_name in LOCAL_RCMIP_FILES:
            src = LOCAL_RCMIP_FILES[requested_name]
            if not src.exists():
                raise FileNotFoundError(f"Local RCMIP file missing: {src}")

            # Compute destination path like pooch would
            if path is None:
                # fallback to default pooch cache path
                path = pooch.os_cache("pooch")
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            dst = path / requested_name

            # Copy local file into cache destination
            shutil.copyfile(src, dst)

            # If a processor is used, apply it
            if processor is not None:
                return processor(str(dst), action="fetch", pooch=None)
            return str(dst)

        # Otherwise behave normally
        return original_retrieve(
            url=url,
            known_hash=known_hash,
            fname=fname,
            path=path,
            processor=processor,
            downloader=downloader,
            progressbar=progressbar,
        )

    pooch.retrieve = local_retrieve


# -----------------------------
# Init FaIR and fill from RCMIP (native)
# -----------------------------
def _init_fair(scenario: str, start_year: int, end_year: int, fill_rcmip: bool = True) -> FAIR:
    f = FAIR()
    f.define_time(start_year, end_year, 1)
    f.define_scenarios([scenario])
    f.define_configs(["default"])

    species, props = read_properties()
    f.define_species(species, props)

    f.allocate()
    f.fill_species_configs()

    # init states
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0.0)
    initialise(f.temperature, 0.0)
    initialise(f.cumulative_emissions, 0.0)
    initialise(f.airborne_emissions, 0.0)
    initialise(f.ocean_heat_content_change, 0.0)

    # Apply climate calibration
    _fill_climate_from_csv(f, config="default", calib_path=CALIB_PATH)

    if fill_rcmip:
        # Ensure local RCMIP files exist
        _require_files_exist(list(LOCAL_RCMIP_FILES.values()))

        # Patch pooch so FaIR's internal downloader uses local CSVs
        _patch_pooch_for_local_rcmip()

        # Native FaIR pathway (handles all species + mapping correctly)
        f.fill_from_rcmip()

    return f


# -----------------------------
# Outputs expected by app.py
# -----------------------------
def _extract_outputs(f: FAIR, scenario: str) -> dict:
    years = np.asarray(f.timepoints, dtype=int)
    n = len(years)

    def to_1d(da):
        arr = np.asarray(da.values).squeeze()
        if arr.shape[0] > n:
            arr = arr[:n]
        elif arr.shape[0] < n:
            arr = np.pad(arr, (0, n - arr.shape[0]), constant_values=np.nan)
        return arr

    # ---- Temperature anomaly EXACTLY like local code ----
    ts = f.temperature.sel(scenario=scenario, config="default", layer=0)
    baseline_sel = ts.sel(timebounds=slice(1850, 1900))
    baseline_mean = float(baseline_sel.mean("timebounds").values)

    temp_anom = to_1d(ts) - baseline_mean

    # ---- Concentrations ----
    conc = {}
    for gas in ["CO2", "CH4", "N2O"]:
        conc[gas] = to_1d(
            f.concentration.sel(scenario=scenario, config="default", specie=gas)
        )

    # ---- Forcing ----
    forc = {}
    for gas in ["CO2", "CH4", "N2O"]:
        forc[gas] = to_1d(
            f.forcing.sel(scenario=scenario, config="default", specie=gas)
        )

    total_forcing = to_1d(
        f.forcing.sel(scenario=scenario, config="default").sum(dim="specie")
    )

    return {
        "years": years,
        "temp_anomaly": temp_anom,
        "conc": conc,
        "forc": forc,
        "total_forcing": total_forcing,
        "baseline_mean": baseline_mean,
    }


# -----------------------------
# Public API used by app.py
# -----------------------------
def run_fair_rcmip_scenario(scenario: str, start_year: int = 1850, end_year: int = 2100) -> dict:
    f = _init_fair(scenario, start_year, end_year, fill_rcmip=True)
    f.run()
    return _extract_outputs(f, scenario)


def run_fair_custom_single_gas_emissions(
    gas: str,
    df: pd.DataFrame,
    start_year: int | None = None,
    end_year: int | None = None,
) -> dict:
    gas = gas.upper().strip()
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

    # ---- CRITICAL: no NaNs anywhere in drivers ----
    f.emissions.loc[dict(scenario=scenario, config="default")] = 0.0
    f.forcing.loc[dict(scenario=scenario, config="default")] = 0.0

    years_model = np.asarray(f.timepoints, dtype=int)

    def build_series(col: str) -> np.ndarray:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' for gas {gas}.")
        s = pd.Series(df[col].values.astype(float), index=df["year"].values.astype(int))
        s = s.reindex(years_model).interpolate(limit_direction="both").fillna(0.0)
        return s.to_numpy(dtype=float)

    def set_emissions(specie_name: str, values: np.ndarray):
        if specie_name in f.emissions.specie.values:
            f.emissions.loc[dict(scenario=scenario, config="default", specie=specie_name)] = values

    if gas == "CO2":
        if "co2_total" in df.columns:
            v = build_series("co2_total")
            set_emissions("CO2 FFI", v)
            set_emissions("CO2 AFOLU", np.zeros_like(v))
            set_emissions("carbon_dioxide", v)
        else:
            # allow split columns too
            if "co2_fossil" not in df.columns and "co2_afolu" not in df.columns:
                raise ValueError("For CO2, provide 'co2_total' OR 'co2_fossil'/'co2_afolu'.")
            if "co2_fossil" in df.columns:
                set_emissions("CO2 FFI", build_series("co2_fossil"))
            if "co2_afolu" in df.columns:
                set_emissions("CO2 AFOLU", build_series("co2_afolu"))

    elif gas == "CH4":
        v = build_series("ch4")
        set_emissions("CH4", v)
        set_emissions("methane", v)

    elif gas == "N2O":
        v = build_series("n2o")
        set_emissions("N2O", v)
        set_emissions("nitrous_oxide", v)

    else:
        raise ValueError("gas must be one of: CO2, CH4, N2O")

    # Final hard safety check
    em = np.asarray(f.emissions.sel(scenario=scenario, config="default").values)
    forc = np.asarray(f.forcing.sel(scenario=scenario, config="default").values)
    if np.isnan(em).any():
        raise ValueError("Custom run: emissions still contain NaNs after filling.")
    if np.isnan(forc).any():
        raise ValueError("Custom run: forcing still contains NaNs after filling (Solar/Volcanic etc.).")

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
    Runs a baseline SSP (base) and a reduced-emissions scenario (reduced).
    Returns a dict shaped exactly as app.py expects:

    {
      "years": <np.ndarray>,
      "base": <output dict from _extract_outputs>,
      "reduced": <output dict from _extract_outputs>
    }
    """

    # -------------------------
    # 1) BASELINE RUN
    # -------------------------
    f_base = _init_fair(base_scenario, start_year, end_year, fill_rcmip=True)
    f_base.run()
    base_out = _extract_outputs(f_base, base_scenario)

    # -------------------------
    # 2) REDUCED RUN
    # -------------------------
    f_red = _init_fair(base_scenario, start_year, end_year, fill_rcmip=True)

    years = np.asarray(base_out["years"]).astype(int)
    n = len(years)
    window = max(1, reduction_end_year - reduction_start_year)

    def ramp_factor(year: int, r: float) -> float:
        """Linear ramp to (1-r) between start and end years, then hold."""
        if year < reduction_start_year:
            return 1.0
        if year <= reduction_end_year:
            frac = (year - reduction_start_year) / window
            return 1.0 - r * frac
        return 1.0 - r

    # Map user gas to likely FaIR emissions species names
    gas_to_species = {
        "CO2": ["CO2 FFI", "CO2 AFOLU", "carbon_dioxide"],
        "CH4": ["CH4", "methane"],
        "N2O": ["N2O", "nitrous_oxide"],
    }

    for gas, r in reductions.items():
        gas = str(gas).upper().strip()
        r = float(r)

        if r <= 0.0:
            continue
        if gas not in gas_to_species:
            continue

        for sp in gas_to_species[gas]:
            if sp not in f_red.emissions.specie.values:
                continue

            # Get emissions array (1D)
            arr = np.asarray(
                f_red.emissions.sel(
                    scenario=base_scenario, config="default", specie=sp
                ).values
            )
            arr = np.ravel(arr)

            m = min(len(arr), n)
            arr = arr[:m].copy()

            # Apply ramp
            for i in range(m):
                arr[i] *= ramp_factor(int(years[i]), r)

            # Write back
            full = np.asarray(
                f_red.emissions.sel(
                    scenario=base_scenario, config="default", specie=sp
                ).values
            )
            full = np.ravel(full)
            full[:m] = arr

            f_red.emissions.loc[dict(scenario=base_scenario, config="default", specie=sp)] = full

    # Run reduced simulation and extract
    f_red.run()
    red_out = _extract_outputs(f_red, base_scenario)

    # -------------------------
    # Return structure expected by app.py
    # -------------------------
    return {
        "years": base_out["years"],
        "base": base_out,
        "reduced": red_out,
    }

