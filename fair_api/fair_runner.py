import numpy as np
import pandas as pd

from fair import FAIR
from fair.io import read_properties
from fair.interface import initialise, fill

# Keep your calibration path as confirmed
from pathlib import Path
import os

# Allow override via environment variable, otherwise use repo-relative file
DEFAULT_CALIB_PATH = Path(__file__).resolve().parents[1] / "data" / "calibrated_constrained_parameters.csv"
CALIB_PATH = Path(os.getenv("FAIR_CALIB_PATH", str(DEFAULT_CALIB_PATH)))


# =========================================================
# Core helpers
# =========================================================
def _fill_climate_from_csv(f: FAIR, config: str = "default", calib_path: str = CALIB_PATH):
    df = pd.read_csv(str(calib_path))
    row = df.iloc[0]
    fill(f.climate_configs["ocean_heat_capacity"], [row["c1"], row["c2"], row["c3"]], config=config)
    fill(f.climate_configs["ocean_heat_transfer"], [row["kappa1"], row["kappa2"], row["kappa3"]], config=config)
    fill(f.climate_configs["deep_ocean_efficacy"], float(row["epsilon"]), config=config)
    fill(f.climate_configs["gamma_autocorrelation"], float(row["gamma"]), config=config)
    fill(f.climate_configs["stochastic_run"], False, config=config)
    fill(f.climate_configs["forcing_4co2"], 8.0, config=config)


from pathlib import Path
from fair import FAIR
from fair.interface import initialise
from fair.io import read_properties

# Assumes you already have:
# - CALIB_PATH defined (Path or str)
# - _fill_climate_from_csv(f, config, calib_path) defined elsewhere in this file


from pathlib import Path
import shutil
import pooch

from fair import FAIR
from fair.interface import initialise
from fair.io import read_properties


def _ensure_rcmip_cached() -> None:
    """
    Put the RCMIP CSVs (stored in our repo under data/rcmip/) into FaIR's pooch cache
    so that FAIR.fill_from_rcmip() finds them locally and does NOT download from Zenodo.
    """
    repo_rcmip_dir = Path(__file__).resolve().parents[1] / "data" / "rcmip"

    # These filenames must match exactly what FaIR expects.
    files = [
        "rcmip-emissions-annual-means-v5-1-0.csv",
        "rcmip-concentrations-annual-means-v5-1-0.csv",
        "rcmip-radiative-forcing-annual-means-v5-1-0.csv",
    ]

    missing = [f for f in files if not (repo_rcmip_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required RCMIP file(s) in data/rcmip/:\n"
            + "\n".join(f"- {repo_rcmip_dir / f}" for f in missing)
        )

    # FaIR uses pooch cache. This is typically ~/.cache/fair on linux.
    cache_dir = Path(pooch.os_cache("fair"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Copy repo files -> cache with the exact same names
    for fname in files:
        src = repo_rcmip_dir / fname
        dst = cache_dir / fname
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, dst)


from pathlib import Path
import pandas as pd

from fair import FAIR
from fair.interface import initialise
from fair.io import read_properties


def _init_fair(scenario: str, start_year: int, end_year: int, fill_rcmip: bool = True) -> FAIR:
    """
    Initialise FaIR and fill inputs from LOCAL RCMIP CSVs (no internet, no pooch).

    This follows the same logic as the FaIR GitHub helper script:
    - load RCMIP emissions + concentrations
    - use FaIRv2.0.0-alpha_RCMIP_inputmap.csv to map variables
    - populate f.emissions and f.concentration for the chosen SSP scenario
    """

    # ----------------------------
    # 1) Build FaIR object
    # ----------------------------
    f = FAIR()
    f.define_time(start_year, end_year, 1)
    f.define_scenarios([scenario])
    f.define_configs(["default"])

    species, props = read_properties()
    f.define_species(species, props)

    f.allocate()
    f.fill_species_configs()

    # ----------------------------
    # 2) Fill from local RCMIP
    # ----------------------------
    if fill_rcmip:
        root = Path(__file__).resolve().parents[1]
        rcmip_dir = root / "data" / "rcmip"
        aux_dir = root / "data" / "raux"

        emms_path = rcmip_dir / "rcmip-emissions-annual-means-v5-1-0.csv"
        conc_path = rcmip_dir / "rcmip-concentrations-annual-means-v5-1-0.csv"
        map_path = aux_dir / "FaIRv2.0.0-alpha_RCMIP_inputmap.csv"

        missing = [p for p in (emms_path, conc_path, map_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required input file(s):\n" + "\n".join(f"- {p}" for p in missing)
            )

        # Load mapping file (same one used in FaIR repo script)
        mapping = pd.read_csv(map_path, index_col=0)

        # Load RCMIP data locally (NO URL)
        rcmip_emms = pd.read_csv(emms_path).set_index(["Region", "Scenario", "Variable"])
        rcmip_concs = pd.read_csv(conc_path).set_index(["Region", "Scenario", "Variable"])

        # --- helper: build FaIR emissions dataframe (years x FAIR_species) ---
        def rcmip_to_fair_emms(ssp: str) -> pd.DataFrame:
            keys = mapping["RCMIP_emms_key"].dropna()
            # Select World + scenario + required variables
            sub = rcmip_emms.loc[("World", ssp, keys.values)]
            sub = sub.droplevel([0, 1])

            # Year columns in RCMIP start after meta columns; safer: keep only numeric year columns
            year_cols = [c for c in sub.columns if str(c).isdigit()]
            sub = sub[year_cols]

            # Map RCMIP variable -> FaIR variable name
            inv_map = {v: k for k, v in keys.to_dict().items()}
            sub.index = sub.index.map(inv_map)

            # Apply scaling if present
            if "RCMIP_emms_scaling" in mapping.columns:
                scale = mapping.loc[sub.index, "RCMIP_emms_scaling"].astype(float)
                sub = sub.T.mul(scale, axis=1).T

            out = sub.T
            out.index = out.index.astype(int)
            out = out.apply(pd.to_numeric)
            return out

        # --- helper: build FaIR concentrations dataframe (years x FAIR_species) ---
        def rcmip_to_fair_concs(ssp: str) -> pd.DataFrame:
            keys = mapping["RCMIP_concs_key"].dropna()
            sub = rcmip_concs.loc[("World", ssp, keys.values)]
            sub = sub.droplevel([0, 1])

            year_cols = [c for c in sub.columns if str(c).isdigit()]
            sub = sub[year_cols]

            inv_map = {v: k for k, v in keys.to_dict().items()}
            sub.index = sub.index.map(inv_map)

            if "RCMIP_concs_scaling" in mapping.columns:
                scale = mapping.loc[sub.index, "RCMIP_concs_scaling"].astype(float)
                sub = sub.T.mul(scale, axis=1).T

            out = sub.T
            out.index = out.index.astype(int)
            out = out.apply(pd.to_numeric)
            return out

        emms_df = rcmip_to_fair_emms(scenario)
        conc_df = rcmip_to_fair_concs(scenario)

        # ----------------------------
        # 3) Write into FaIR xarray arrays
        # ----------------------------
        # Ensure we only fill years that exist in our model time axis
        years = set(range(start_year, end_year + 1))

        # Fill emissions
        for sp in emms_df.columns:
            if sp in f.species:
                series = emms_df[sp]
                series = series[series.index.isin(years)]
                if not series.empty:
                    f.emissions.loc[dict(scenario=scenario, config="default", specie=sp, time=series.index)] = series.values

        # Fill concentrations
        for sp in conc_df.columns:
            if sp in f.species:
                series = conc_df[sp]
                series = series[series.index.isin(years)]
                if not series.empty:
                    f.concentration.loc[dict(scenario=scenario, config="default", specie=sp, time=series.index)] = series.values

    else:
        # Custom scenario mode (optional)
        f.emissions.loc[dict(scenario=scenario, config="default")] = 0.0
        f.concentration.loc[dict(scenario=scenario, config="default")] = 0.0
        f.forcing.loc[dict(scenario=scenario, config="default")] = 0.0

    # ----------------------------
    # 4) Initialise state variables + climate calibration
    # ----------------------------
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0.0)
    initialise(f.temperature, 0.0)
    initialise(f.cumulative_emissions, 0.0)
    initialise(f.airborne_emissions, 0.0)
    initialise(f.ocean_heat_content_change, 0.0)

    _fill_climate_from_csv(f, config="default", calib_path=CALIB_PATH)
    return f


def _extract_timeseries_aligned_on_timepoints(f: FAIR, scenario: str) -> dict:
    """
    Return everything aligned on timepoints (common axis for emissions/conc/forcing),
    and temperature aligned to same length (cropped/padded if needed).
    """
    years = np.asarray(f.timepoints, dtype=int)
    n = len(years)

    def to_1d(da):
        arr = np.asarray(da.values).squeeze()
        # align to n
        if arr.shape[0] > n:
            arr = arr[:n]
        elif arr.shape[0] < n:
            arr = np.pad(arr, (0, n - arr.shape[0]), constant_values=np.nan)
        return arr

    # Temperature anomaly vs 1850–1900 baseline (use timebounds for baseline selection)
    ts = f.temperature.sel(scenario=scenario, config="default", layer=0)
    baseline_sel = ts.sel(timebounds=slice(1850, 1900))
    baseline_mean = float(baseline_sel.mean("timebounds").values)
    temp_anom = to_1d(ts) - baseline_mean

    # Key emissions species
    emis = {}
    for sp in ["CO2 FFI", "CO2 AFOLU", "CH4", "N2O"]:
        if sp in f.emissions.specie.values:
            emis[sp] = to_1d(f.emissions.sel(scenario=scenario, config="default", specie=sp))
        else:
            emis[sp] = np.zeros(n, dtype=float)

    # Key concentrations/forcings
    conc = {}
    forc = {}
    gas_map = {"CO2": "CO2", "CH4": "CH4", "N2O": "N2O"}
    for gas in gas_map.values():
        conc[gas] = to_1d(f.concentration.sel(scenario=scenario, config="default", specie=gas))
        forc[gas] = to_1d(f.forcing.sel(scenario=scenario, config="default", specie=gas))

    # Total forcing (all agents)
    total_forcing = to_1d(f.forcing.sel(scenario=scenario, config="default").sum(dim="specie"))

    return dict(
        years=years,
        temp_anomaly=temp_anom,
        emis=emis,
        conc=conc,
        forc=forc,
        total_forcing=total_forcing,
        baseline_mean=baseline_mean,
    )


# =========================================================
# 1) Default SSP scenario (RCMIP)
# =========================================================
def run_fair_rcmip_scenario(scenario: str, start_year: int = 1850, end_year: int = 2100) -> dict:
    f = _init_fair(scenario, start_year, end_year, fill_rcmip=True)
    f.run()
    return _extract_timeseries_aligned_on_timepoints(f, scenario)


# =========================================================
# 2) Custom emissions: "single gas only" run
#    - set all emissions = 0
#    - set chosen gas emissions from CSV
#    - run FaIR and return outcomes for that gas only
# =========================================================
def run_fair_custom_single_gas_emissions(
    gas: str,
    df: pd.DataFrame,
    start_year: int | None = None,
    end_year: int | None = None,
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

    # IMPORTANT:
    # FaIR v2 ghg method typically expects CO2, CH4, N2O to be present (even zeros),
    # so we keep all emissions arrays defined (already zeroed), and only add the chosen gas.

    if gas == "CO2":
        # Accept either co2_total OR split fossil/afolu
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
    out = _extract_timeseries_aligned_on_timepoints(f, scenario)

    # For “single gas only”, total forcing is essentially that gas (since others emissions=0),
    # but we still return total_forcing (as calculated).
    return out


# =========================================================
# 3) Multi-gas gradual reduction (CO2, CH4, N2O together)
# =========================================================
def run_fair_multi_reduction_scenario(
    base_scenario: str,
    reductions: dict,
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

    Applies a LINEAR RAMP:
      factor(year) = 1 - r * (year-start)/(end-start)   within window
      then holds at (1-r) after end_year.
    No sudden drop.
    """

    # baseline
    base = run_fair_rcmip_scenario(base_scenario, start_year, end_year)

    # reduced model
    f = _init_fair(base_scenario, start_year, end_year, fill_rcmip=True)

    # map “gas” to emissions specie names
    # CO2 reduction applies to BOTH FFI and AFOLU (together), unless you later split
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

    # apply ramp scaling
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

    return dict(
        years=base["years"],
        base=base,
        reduced=red,
    )
