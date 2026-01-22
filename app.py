import io
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------------------------------------
# Ensure project root is on sys.path so "fair_api" imports work
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

from fair_api.fair_runner import (
    run_fair_rcmip_scenario,
    run_fair_custom_single_gas_emissions,
    run_fair_multi_reduction_scenario,
)

st.set_page_config(page_title="FCE – FaIR Climate Explorer", layout="wide")


# ============================================================
# Helpers
# ============================================================
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()


def conc_unit(gas: str) -> str:
    if gas == "CO2":
        return "ppm"
    if gas in ("CH4", "N2O"):
        return "ppb"
    return ""


def export_png_csv(fig, df, basename: str):
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=200, bbox_inches="tight")
    png_buf.seek(0)

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download PNG",
            data=png_buf,
            file_name=f"{basename}.png",
            mime="image/png",
        )
    with c2:
        st.download_button(
            "Download CSV",
            data=csv_buf.getvalue(),
            file_name=f"{basename}.csv",
            mime="text/csv",
        )


def find_first_existing(base_dir: str, stems: list[str], exts: list[str]) -> str | None:
    for stem in stems:
        for ext in exts:
            p = os.path.join(base_dir, f"{stem}{ext}")
            if os.path.exists(p):
                return p
    return None


# ============================================================
# Styling (Banner gradient BG + coloured tabs + footer box)
# ============================================================
st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* Banner: gradient BACKGROUND (not text) */
.banner {
    background: linear-gradient(90deg,
        #e60000 0%,
        #ff7a00 25%,
        #ffd400 50%,
        #00c2ff 75%,
        #00a650 100%
    );
    border-radius: 22px;
    padding: 54px 28px;
    text-align: center;
    margin-bottom: 18px;
    box-shadow: 0 10px 26px rgba(0,0,0,0.12);
}

.banner-title {
    font-size: 54px;
    font-weight: 950;
    margin-bottom: 10px;
    letter-spacing: 0.5px;
    color: #101010;
    text-shadow: 0 1px 0 rgba(255,255,255,0.35);
}

.banner-subtitle {
    font-size: 18px;
    color: #101010;
    opacity: 0.95;
}

/* Section blocks */
.section {
    border: 1px solid #e5e9f2;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 18px;
    background: #ffffff;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}

/* Tabs layout */
.stTabs [role="tablist"] {
    display: flex !important;
    gap: 12px;
    border-bottom: none !important;
    padding-bottom: 12px;
    margin-top: 6px;
    width: 100%;
}

.stTabs [role="tab"] {
    flex: 1 1 0% !important;
    justify-content: center !important;
    border-radius: 16px !important;
    padding: 18px 20px !important;
    border: 1px solid #cfd7e6 !important;
    box-shadow: 0 7px 0 rgba(0,0,0,0.12) !important;
    font-size: 22px !important;
    font-weight: 850 !important;
    color: #101828 !important;
    transition: 0.12s ease-in-out;
    text-align: center !important;
}

/* Different colours per tab (by order) */
.stTabs [role="tab"]:nth-child(1) { background: #E8F7FF !important; }  /* Home */
.stTabs [role="tab"]:nth-child(2) { background: #EAF7EA !important; }  /* FaIR Model */
.stTabs [role="tab"]:nth-child(3) { background: #FFF3DF !important; }  /* Scenario */
.stTabs [role="tab"]:nth-child(4) { background: #F1ECFF !important; }  /* Custom */
.stTabs [role="tab"]:nth-child(5) { background: #FFE9F2 !important; }  /* Reduction */

.stTabs [role="tab"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 9px 0 rgba(0,0,0,0.12) !important;
}

/* Selected tab: pressed look */
.stTabs [role="tab"][aria-selected="true"] {
    border: 1px solid #7aa7ff !important;
    box-shadow: 0 4px 0 rgba(0,0,0,0.18) !important;
    transform: translateY(2px);
}

/* Footer box */
.footer-box {
    margin-top: 28px;
    padding: 18px;
    border-radius: 16px;
    background: #f6f7fb;
    border: 1px solid #e1e5f0;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}
.footer-title {
    font-weight: 900;
    font-size: 14px;
    color: #1b2a4a;
    margin-bottom: 8px;
}
.footer-text {
    color: #4b5563;
    font-size: 13px;
    line-height: 1.45;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Banner
# ============================================================
st.markdown(
    """
<div class="banner">
  <div class="banner-title">FCE – FaIR Climate Explorer</div>
  <div class="banner-subtitle">
    Explore SSP scenarios, custom pathways, and gradual mitigation experiments using FaIR v2.2
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Tabs (NO Team tab)
# ============================================================
tab_home, tab_fair, tab_scen, tab_custom, tab_reduce = st.tabs(
    ["Home", "FaIR Model", "Scenario Exploration", "Custom Emissions", "Reduction Explorer"]
)

# ============================================================
# TAB 1: HOME (no "Home" word inside content)
# ============================================================
with tab_home:
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    st.markdown(
        """
Climate change and its impacts on society, the economy, human health, and livelihoods are unprecedented.
The primary driver of this change is the rise in anthropogenic greenhouse gas emissions.
The Intergovernmental Panel on Climate Change (IPCC), in all of its assessment reports, emphasizes the urgent need to reduce emissions to secure a sustainable future.

Over the past decades, many reduced-complexity climate models have been developed, and scientists continue to work on next-generation models to better understand cost-effective mitigation pathways.
However, these modelling tools are often accessible only to specialized research communities.

This dashboard integrates the **FaIR (Finite Amplitude Impulse Response)** model to make future climate scenario exploration more accessible.

### About the FaIR Model
This section guides users in understanding the FaIR model, including how it is constructed, its applications, and how it can be integrated into different workflows.
It also highlights the model development framework, installation process, and common use cases.

### Scenario Exploration
Users can simulate future temperature outcomes under the SSP1-1.9, SSP1-2.6, SSP2-4.5, SSP3-7.0, and SSP5-8.5 scenarios, considering all greenhouse gases as well as individual contributions from CO₂, CH₄, and N₂O.
The tool also enables simulation of atmospheric concentrations, radiative forcing, and temperature responses for these gases under each scenario, with options to download the generated data.

### Custom Emissions
This feature allows users to simulate concentrations, radiative forcing, and temperature outcomes based on their own emission pathways.
Guidelines are provided for data requirements and step-by-step instructions for running simulations specifically for CO₂, CH₄, and N₂O.

### Reduction Explorer
This section enables users to model gradual reductions in CO₂, CH₄, and N₂O emissions under different future scenarios and assess their effects on concentrations, forcing, and temperature.
"""
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 2: FAIR MODEL (no "FaIR model" word inside content)
# ============================================================
with tab_fair:
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    st.markdown(
        """
This platform allows users to simulate outcomes using the **FaIR model** under different scenarios without requiring prior knowledge of climate modelling.

For detailed information about the FaIR version 2.0.0 model, please refer to:
- **[Leach et al. (2021) – GMD paper](https://gmd.copernicus.org/articles/14/3007/2021/)**

The installation instructions and Python implementation are available at:
- **[FaIR installation & implementation (GitHub)](https://github.com/njleach/FAIR/tree/47c6eec031d2edcf09424394dbb86581a1b246ba)**

**FaIR Model infographic**:
"""
    )

    fair_img = find_first_existing(PROJECT_ROOT, stems=["FaIR", "fair", "FAIR"], exts=[".png", ".jpg", ".jpeg", ".webp"])
    if fair_img:
        # smaller than previous version
        st.image(fair_img, width=760, caption="Figure: Schematic overview of the FaIR model framework (generated using NotebookLM)")
    else:
        st.warning(
            "FaIR figure not found in project root. Expected FaIR.png/jpg/jpeg in: "
            f"{PROJECT_ROOT}"
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 3: SCENARIO EXPLORATION (stable + run summary + units)
# ============================================================
with tab_scen:
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    scenario = st.selectbox(
        "Select SSP scenario",
        ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"],
        index=1,
        key="scen_scenario",
    )

    run_clicked = st.button("Run SSP simulation", key="scen_run")

    if run_clicked:
        with st.spinner("Running FaIR..."):
            out = run_fair_rcmip_scenario(scenario)
        st.session_state["ssp_out"] = out
        st.session_state["ssp_meta"] = {"scenario": scenario}

    if "ssp_out" in st.session_state:
        out = st.session_state["ssp_out"]
        scenario_used = st.session_state.get("ssp_meta", {}).get("scenario", scenario)
        years = out["years"]

        st.info(
            f"Run settings summary:\n"
            f"- Scenario: **{scenario_used}**\n"
            f"- Dataset: **RCMIP (FaIR built-in)**\n"
            f"- Concentration units: CO₂ **ppm**, CH₄/N₂O **ppb**\n"
            f"- Forcing units: **W/m²**"
        )

        analysis_type = st.selectbox(
            "Select analysis type",
            ["Temperature", "Concentration", "Radiative Forcing"],
            key="scen_analysis",
        )

        gas = None
        if analysis_type != "Temperature":
            gas = st.selectbox("Select gas", ["CO2", "CH4", "N2O", "Total"], key="scen_gas")

        fig, ax = plt.subplots(figsize=(7.8, 4.0))

        if analysis_type == "Temperature":
            y = out["temp_anomaly"]
            ax.plot(years, y)
            ax.axhline(0, linewidth=1)
            ax.set_ylabel("°C (relative to 1850–1900)")
            title = "Temperature anomaly"
            df = pd.DataFrame({"year": years, "temp_anomaly_C": y})
            basename = f"temperature_{scenario_used}"

        elif analysis_type == "Concentration":
            if gas == "Total":
                st.warning("Total concentration is not defined. Choose CO2/CH4/N2O.")
                st.stop()

            y = out["conc"][gas]
            unit = conc_unit(gas)
            ax.plot(years, y)
            ax.set_ylabel(f"Concentration ({unit})")
            title = f"{gas} concentration"
            df = pd.DataFrame({"year": years, f"{gas}_concentration_{unit}": y})
            basename = f"concentration_{gas}_{scenario_used}"

        else:
            if gas == "Total":
                y = out["total_forcing"]
                label = "total_forcing_Wm2"
                title = "Total forcing"
            else:
                y = out["forc"][gas]
                label = f"{gas}_forcing_Wm2"
                title = f"{gas} forcing"

            ax.plot(years, y)
            ax.set_ylabel("W/m²")
            df = pd.DataFrame({"year": years, label: y})
            basename = f"forcing_{label}_{scenario_used}"

        ax.set_title(f"{title} – {scenario_used}")
        ax.set_xlabel("Year")
        st.pyplot(fig)
        export_png_csv(fig, df, basename)

        if st.button("Clear SSP results", key="ssp_clear"):
            st.session_state.pop("ssp_out", None)
            st.session_state.pop("ssp_meta", None)
            safe_rerun()
    else:
        st.info("Click **Run SSP simulation** to generate results.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 4: CUSTOM EMISSIONS
# ============================================================
with tab_custom:
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    gas = st.selectbox("Choose gas", ["CO2", "CH4", "N2O"], key="custom_gas")
    st.markdown("Upload a CSV with a **year** column and one emissions column.")
    st.caption("CO2: `co2_total` (or `co2_fossil`/`co2_afolu`) | CH4: `ch4` | N2O: `n2o`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="custom_upload")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state["custom_df"] = df

    if "custom_df" in st.session_state:
        df = st.session_state["custom_df"]
        st.dataframe(df.head())

        run_clicked = st.button("Run custom emissions", key="custom_run")

        if run_clicked:
            with st.spinner("Running FaIR..."):
                out = run_fair_custom_single_gas_emissions(gas, df)
            st.session_state["custom_out"] = out
            st.session_state["custom_meta"] = {
                "gas": gas,
                "year_min": int(df["year"].min()) if "year" in df.columns else None,
                "year_max": int(df["year"].max()) if "year" in df.columns else None,
            }

        if "custom_out" in st.session_state:
            out = st.session_state["custom_out"]
            meta = st.session_state.get("custom_meta", {})
            gas_used = meta.get("gas", gas)
            unit = conc_unit(gas_used)

            st.info(
                f"Run settings summary:\n"
                f"- Gas: **{gas_used}**\n"
                f"- Period: **{meta.get('year_min','?')}–{meta.get('year_max','?')}**\n"
                f"- Concentration units: **{unit}**\n"
                f"- Forcing units: **W/m²**"
            )

            years = out["years"]
            fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.0))

            axes[0].plot(years, out["temp_anomaly"])
            axes[0].set_title("Temperature anomaly")
            axes[0].set_xlabel("Year")
            axes[0].set_ylabel("°C")

            axes[1].plot(years, out["conc"][gas_used])
            axes[1].set_title(f"{gas_used} concentration")
            axes[1].set_xlabel("Year")
            axes[1].set_ylabel(unit)

            axes[2].plot(years, out["forc"][gas_used])
            axes[2].set_title(f"{gas_used} forcing")
            axes[2].set_xlabel("Year")
            axes[2].set_ylabel("W/m²")

            plt.tight_layout()
            st.pyplot(fig)

            df_out = pd.DataFrame(
                {
                    "year": years,
                    "temp_anomaly_C": out["temp_anomaly"],
                    f"{gas_used}_concentration_{unit}": out["conc"][gas_used],
                    f"{gas_used}_forcing_Wm2": out["forc"][gas_used],
                }
            )
            export_png_csv(fig, df_out, f"custom_{gas_used.lower()}")

            if st.button("Clear custom results", key="custom_clear"):
                st.session_state.pop("custom_out", None)
                st.session_state.pop("custom_meta", None)
                st.session_state.pop("custom_df", None)
                safe_rerun()
        else:
            st.info("Click **Run custom emissions** to generate results.")
    else:
        st.info("Upload a CSV to begin.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 5: REDUCTION EXPLORER
# ============================================================
with tab_reduce:
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    scenario = st.selectbox(
        "Baseline SSP",
        ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"],
        index=1,
        key="red_scenario",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        r_co2 = st.slider("CO₂ reduction by end (%)", 0, 100, 0, key="red_co2")
    with c2:
        r_ch4 = st.slider("CH₄ reduction by end (%)", 0, 100, 0, key="red_ch4")
    with c3:
        r_n2o = st.slider("N₂O reduction by end (%)", 0, 100, 0, key="red_n2o")

    y_start, y_end = st.slider(
        "Reduction window (gradual ramp)", 2000, 2100, (2030, 2050), key="red_window"
    )

    st.caption("Reductions are applied gradually (linear ramp) from start year to end year, then held constant.")
    run_clicked = st.button("Run reduction experiment", key="red_run")

    if run_clicked:
        reductions = {
            "CO2": r_co2 / 100.0,
            "CH4": r_ch4 / 100.0,
            "N2O": r_n2o / 100.0,
        }

        with st.spinner("Running baseline + reduced scenario..."):
            out = run_fair_multi_reduction_scenario(
                base_scenario=scenario,
                reductions=reductions,
                reduction_start_year=int(y_start),
                reduction_end_year=int(y_end),
            )

        st.session_state["reduction_result"] = out
        st.session_state["reduction_meta"] = {
            "scenario": scenario,
            "reductions": reductions,
            "y_start": int(y_start),
            "y_end": int(y_end),
        }

    if "reduction_result" in st.session_state:
        out = st.session_state["reduction_result"]
        meta = st.session_state.get("reduction_meta", {})
        scenario_used = meta.get("scenario", scenario)

        years = out["years"]
        base = out["base"]
        red = out["reduced"]

        st.info(
            f"Run settings summary:\n"
            f"- Scenario: **{scenario_used}**\n"
            f"- Ramp: **{meta.get('y_start','?')}–{meta.get('y_end','?')}**\n"
            f"- Reductions (end of ramp): CO2 **{int(meta.get('reductions',{}).get('CO2',0)*100)}%**, "
            f"CH4 **{int(meta.get('reductions',{}).get('CH4',0)*100)}%**, "
            f"N2O **{int(meta.get('reductions',{}).get('N2O',0)*100)}%**"
        )

        t_tab, c_tab, f_tab = st.tabs(["Temperature", "Concentration", "Forcing"])

        with t_tab:
            fig, ax = plt.subplots(figsize=(7.8, 4.0))
            ax.plot(years, base["temp_anomaly"], label="Baseline")
            ax.plot(years, red["temp_anomaly"], "--", label="Reduced")
            ax.set_title(f"Temperature anomaly – {scenario_used}")
            ax.set_xlabel("Year")
            ax.set_ylabel("°C (relative to 1850–1900)")
            ax.legend()
            st.pyplot(fig)

            df_temp = pd.DataFrame(
                {
                    "year": years,
                    "temp_anomaly_baseline_C": base["temp_anomaly"],
                    "temp_anomaly_reduced_C": red["temp_anomaly"],
                }
            )
            export_png_csv(fig, df_temp, f"reduction_temperature_{scenario_used}")

        with c_tab:
            gas_c = st.selectbox("Select gas", ["CO2", "CH4", "N2O"], key="red_conc_gas")
            unit = conc_unit(gas_c)

            fig, ax = plt.subplots(figsize=(7.8, 4.0))
            ax.plot(years, base["conc"][gas_c], label="Baseline")
            ax.plot(years, red["conc"][gas_c], "--", label="Reduced")
            ax.set_title(f"{gas_c} concentration – {scenario_used}")
            ax.set_xlabel("Year")
            ax.set_ylabel(f"Concentration ({unit})")
            ax.legend()
            st.pyplot(fig)

            df_conc = pd.DataFrame(
                {
                    "year": years,
                    f"{gas_c}_concentration_{unit}_baseline": base["conc"][gas_c],
                    f"{gas_c}_concentration_{unit}_reduced": red["conc"][gas_c],
                }
            )
            export_png_csv(fig, df_conc, f"reduction_concentration_{gas_c}_{scenario_used}")

        with f_tab:
            forcing_sel = st.selectbox(
                "Select forcing output",
                ["CO2", "CH4", "N2O", "Total forcing"],
                index=3,
                key="red_forcing_sel",
            )

            if forcing_sel == "Total forcing":
                b = base["total_forcing"]
                r = red["total_forcing"]
                series_name = "total_forcing_Wm2"
                title = "Total forcing"
            else:
                b = base["forc"][forcing_sel]
                r = red["forc"][forcing_sel]
                series_name = f"{forcing_sel}_forcing_Wm2"
                title = f"{forcing_sel} forcing"

            fig, ax = plt.subplots(figsize=(7.8, 4.0))
            ax.plot(years, b, label="Baseline")
            ax.plot(years, r, "--", label="Reduced")
            ax.set_title(f"{title} – {scenario_used}")
            ax.set_xlabel("Year")
            ax.set_ylabel("W/m²")
            ax.legend()
            st.pyplot(fig)

            df_forc = pd.DataFrame(
                {
                    "year": years,
                    f"{series_name}_baseline": b,
                    f"{series_name}_reduced": r,
                }
            )
            export_png_csv(fig, df_forc, f"reduction_forcing_{series_name}_{scenario_used}")

        if st.button("Clear results", key="red_clear"):
            st.session_state.pop("reduction_result", None)
            st.session_state.pop("reduction_meta", None)
            safe_rerun()

    else:
        st.info("Run an experiment to see results.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Footer (header-style light box; smaller logos; partners on one row)
# ============================================================
logo_dev = find_first_existing(
    PROJECT_ROOT,
    stems=["logo_v03", "logo-v03", "logo_v03 "],
    exts=[".png", ".jpg", ".jpeg", ".webp"],
)
logo_cu = find_first_existing(
    PROJECT_ROOT,
    stems=["Cranfield University", "cranfield_university", "cranfield"],
    exts=[".png", ".jpg", ".jpeg", ".webp"],
)
logo_ukri = find_first_existing(
    PROJECT_ROOT,
    stems=["UKRI", "ukri"],
    exts=[".png", ".jpg", ".jpeg", ".webp"],
)

# A header-style footer container (light colour, rounded, shadow)
st.markdown(
    """
<div class="footer-box" style="
    background: linear-gradient(90deg, #f7fbff 0%, #fbf7ff 50%, #f7fff9 100%);
    border-radius: 22px;
    padding: 22px 18px;
    box-shadow: 0 10px 26px rgba(0,0,0,0.08);
">
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1.25, 1.7, 1.25])

with c1:
    st.markdown("<div class='footer-title'>Developed by</div>", unsafe_allow_html=True)
    st.markdown(
        """
<div class="footer-text">
CranfieldCliMit Group<br/>
Email: <a href="mailto:cain.cmrg@gmail.com">cain.cmrg@gmail.com</a><br/>
Licence: <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">CC BY 4.0</a>
</div>
""",
        unsafe_allow_html=True,
    )
    if logo_dev:
        st.image(logo_dev, width=140)   # smaller logo
    else:
        st.caption("logo_v03 not found in project root.")

with c2:
    st.markdown("<div class='footer-title'>Address</div>", unsafe_allow_html=True)
    st.markdown(
        """
<div class="footer-text">
Cranfield Environmental Centre,<br/>
Faculty of Engineering and Applied Sciences,<br/>
Cranfield University, United Kingdom.<br/>
College Road, Cranfield,<br/>
Bedfordshire, MK43 0AL<br/>
Web: <a href="https://www.cranfield.ac.uk/" target="_blank">Cranfield University</a>
</div>
""",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown("<div class='footer-title'>Partner</div>", unsafe_allow_html=True)

    # Partners on ONE ROW
    p1, p2 = st.columns(2)
    with p1:
        if logo_cu:
            st.image(logo_cu, width=150)  # smaller partner logo
        else:
            st.caption("Cranfield logo not found.")
    with p2:
        if logo_ukri:
            st.image(logo_ukri, width=150)  # smaller partner logo
        else:
            st.caption("UKRI logo not found.")

st.markdown("</div>", unsafe_allow_html=True)