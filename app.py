# app.py
'''
References that were used for article
Townsend, T., &amp; Powers, L. (2011). pvlib.snow.loss_townsend [Function]. In pvlib
version 0.11.0 documentation. pvlib-python. https://pvlib-
python.readthedocs.io/en/v0.11.0/reference/generated/pvlib.snow.loss_townsend.html
Townsend, T., &amp; Powers, L. (2011). Photovoltaics and snow: An update from two winters
of measurements in the SIERRA. In 2011 37th IEEE Photovoltaic Specialists
Conference (pp. 3231–3236). IEEE. https://doi.org/10.1109/PVSC.2011.6186484
Coello, M., &amp; Boyle, L. (2019). Simple model for predicting time series soiling of
photovoltaic panels. IEEE Journal of Photovoltaics, 9(1), 242–248.
https://doi.org/10.1109/JPHOTOV.2018.2873451
'''

import io
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from typing import List
from pvlib.snow import loss_townsend

st.set_page_config(page_title="Solar PV Performance Loss", page_icon="❄️", layout="wide")

st.title("Solar PV Performance Loss Modelling Application: Snow & Soiling (Monthly)")

# --- Hero image (half size) just below the title ---
from pathlib import Path
try:
    from PIL import Image
except ImportError:
    Image = None

img_candidates = [Path("image_snow_loss.png"), Path("/mnt/data/image_snow_loss.png")]
img_path = next((p for p in img_candidates if p.exists()), None)
if img_path is not None:
    if Image is not None:
        im = Image.open(img_path)
        st.image(im, width=max(300, im.width // 2))  # half width with a sensible minimum
    else:
        st.image(str(img_path), width=600)  # fallback width

# --- Subtitle (combined) ---
st.markdown(
    "_Townsend snow loss (via pvlib) + precipitation-aware soiling model using monthly "
    "climate inputs (manual-clean option); month-by-month results to support EPC yield "
    "estimates and O&M cleaning plans._"
)

# --- Transition text ---
st.markdown(
    "Flat “2% soiling” assumptions hide seasonal risk and can distort both EPC yield models "
    "and O&M decisions. Soiling is dynamic and seasonal; losses during high-irradiance months "
    "hit hardest, and the right cleaning cadence balances cost against energy recovery. Use the "
    "tool below to convert those principles into numbers: it applies pvlib’s Townsend snow model and "
    "a simple precipitation-aware soiling progression (with a manual-clean toggle) to produce stacked "
    "monthly loss percentages one can use in EPC financials and O&M planning."
)

st.markdown(
    "Reference used Townsend, T., & Powers, L. (2011). Photovoltaics and snow: An update from" 
    "two winters of measurements in the Sierra In 37th IEEE Photovoltaic Specialists Conference (pp. 003231–003236). IEEE." 
    "https://doi.org/10.1109/PVSC.2011.6186627"
)
st.markdown("""
    <style>
      .attrib { font-size: 1.5rem; line-height: 1.8; opacity: 0.9; }
    </style>
""", unsafe_allow_html=True)
st.markdown('<p class="attrib">This webpage was created by <b>Sheth Kajal</b> 😊</p>',
            unsafe_allow_html=True)

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_ORDER = {m: i for i, m in enumerate(MONTHS, start=1)}
POA_UNIT = "kWh/m²-month"  # display unit for user entry

# --- helpers ---
def m_to_unit(m: float, unit: str) -> float:
    return m*100.0 if unit=="cm" else m/0.0254
def unit_to_m(x: float, unit: str) -> float:
    return x/100.0 if unit=="cm" else x*0.0254
def pad12(values, fill):
    lst = list(values)
    return (lst + [fill]*12)[:12]

# --- init geometry state ---
if "geo_slant_m" not in st.session_state: st.session_state["geo_slant_m"] = 1.70
if "geo_lower_m" not in st.session_state: st.session_state["geo_lower_m"] = 0.50
if "geom_unit"   not in st.session_state: st.session_state["geom_unit"]   = "cm"
if "slant_display" not in st.session_state:
    st.session_state["slant_display"] = m_to_unit(st.session_state["geo_slant_m"], st.session_state["geom_unit"])
if "lower_display" not in st.session_state:
    st.session_state["lower_display"] = m_to_unit(st.session_state["geo_lower_m"], st.session_state["geom_unit"])
if "slant_input" not in st.session_state:
    st.session_state["slant_input"] = float(st.session_state["slant_display"])
if "lower_input" not in st.session_state:
    st.session_state["lower_input"] = float(st.session_state["lower_display"])

# -----------------------------
# Sidebar: Presets / Units / CSV (includes geometry unit)
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings & Presets")

    preset = st.selectbox(
        "Preset",
        ["Custom (blank)","Mild winter (example)","Cold winter (example)"],
        help="Choose an example to prefill inputs or start from a blank sheet."
    )

    st.subheader("Units")
    snow_unit = st.radio("Snowfall unit", options=["cm","in"], index=0)
    temp_unit = st.radio("Temperature unit", options=["°C","°F"], index=0)
    # Geometry unit
    sel_index = 0 if st.session_state["geom_unit"]=="cm" else 1
    geom_unit_sidebar = st.radio("Geometry length unit (slant & lower edge)", ["cm","in"], index=sel_index)
    if geom_unit_sidebar != st.session_state["geom_unit"]:
        st.session_state["slant_display"] = m_to_unit(st.session_state["geo_slant_m"], geom_unit_sidebar)
        st.session_state["lower_display"] = m_to_unit(st.session_state["geo_lower_m"], geom_unit_sidebar)
        st.session_state["geom_unit"] = geom_unit_sidebar
        st.session_state["slant_input"] = float(st.session_state["slant_display"])
        st.session_state["lower_input"] = float(st.session_state["lower_display"])

    st.subheader("CSV")
    csv_help = """Upload CSV with columns. Any subset is okay:
- month (Jan..Dec) optional
- poa_global_kwhm2
- snow_total_cm or snow_total_in
- snow_events        (FLOAT allowed)
- relative_humidity_pct
- temp_air_c or temp_air_f
- rain_days
- cleaned   (true/false or 1/0)
"""
    up = st.file_uploader("Prefill from CSV (optional)", type=["csv"], help=csv_help)
    download_tmpl = st.checkbox("Show CSV template to download")

# -----------------------------
# Preset Data
# -----------------------------
def preset_frames(preset_name:str, snow_unit:str, temp_unit:str):
    if preset_name == "Mild winter (example)":
        poa_vals = [90,110,140,160,180,190,190,185,160,130,100,85]
        snow_cm =   [8, 10, 6,  2,  0,  0,  0,  0,  1,  3,  5,  7]
        temps_c =   [-2, 1,  5, 11, 16, 21, 24, 24, 20, 14, 7,  1]
        rh =        [70,68,65, 60, 60, 62, 65, 66, 65, 67, 69, 71]
        events =    [3.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0]
        rain_days = [6, 5, 4, 5, 6, 7, 7, 6, 5, 6, 6, 6]
    elif preset_name == "Cold winter (example)":
        poa_vals = [60,85,120,150,175,185,180,170,140,110,75,55]
        snow_cm =   [35,32,28, 15,  2,  0,  0,  0,  3, 12, 25,38]
        temps_c =   [-10,-6,-2,  5, 12, 18, 21, 20, 15,  8,  0,-7]
        rh =        [74,73,70, 65, 63, 64, 66, 67, 68, 70, 72, 74]
        events =    [6.0, 6.0, 5.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 5.0, 6.0]
        rain_days = [7, 6, 6, 7, 8, 9, 9, 7, 7, 8, 8, 7]
    else:
        poa_vals = [120.0]*12
        snow_cm  = [0.0]*12
        temps_c  = [0.0]*12
        rh       = [60.0]*12
        events   = [0.0]*12
        rain_days= [0.0]*12

    snow_disp = snow_cm if snow_unit=="cm" else [round(x/2.54,3) for x in snow_cm]
    temps_disp = temps_c if temp_unit=="°C" else [round(x*9/5+32,3) for x in temps_c]

    poa_df = pd.DataFrame({"month": MONTHS, "POA_global_kwhm2": poa_vals}).iloc[:12]
    weather = pd.DataFrame({
        "month": MONTHS,
        f"snow_total_{snow_unit}": snow_disp,
        "snow_events": events,
        "relative_humidity_pct": rh,
        f"temp_air_{'c' if temp_unit=='°C' else 'f'}": temps_disp,
        "rain_days": rain_days,
        "cleaned": [False]*12,
    }).iloc[:12]
    return poa_df, weather

poa_df, weather_df = preset_frames(preset, snow_unit, temp_unit)

# -----------------------------
# CSV Upload (prefill)
# -----------------------------
def _to_bool_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1","true","yes","y","t"])

def read_csv_upload(file, snow_unit:str, temp_unit:str):
    try:
        df = pd.read_csv(file)
        cols = {c.strip().lower(): c for c in df.columns}
        def has(name): return name in cols

        if has("month"):
            months_raw = df[cols["month"]].astype(str).str.slice(0,3).str.title().tolist()
            seen=set(); ordered=[]
            for m in months_raw:
                if m in MONTHS and m not in seen:
                    ordered.append(m); seen.add(m)
            months = ordered if len(ordered)==12 else MONTHS
        else:
            months = MONTHS

        if has("poa_global_kwhm2"):
            series = pd.to_numeric(df[cols["poa_global_kwhm2"]], errors="coerce").fillna(0.0).tolist()
            series = pad12(series, 0.0)
            poa_df_u = pd.DataFrame({"month": months, "POA_global_kwhm2": series}).set_index("month").reindex(MONTHS).reset_index().iloc[:12]
        else:
            poa_df_u, _ = preset_frames("Custom (blank)", snow_unit, temp_unit)

        w = pd.DataFrame({"month": MONTHS})
        if has("snow_total_cm"):
            v = pd.to_numeric(df[cols["snow_total_cm"]], errors="coerce").fillna(0.0).tolist()
            v = pad12(v, 0.0)
            w[f"snow_total_{snow_unit}"] = v if snow_unit=="cm" else [x/2.54 for x in v]
        elif has("snow_total_in"):
            v = pd.to_numeric(df[cols["snow_total_in"]], errors="coerce").fillna(0.0).tolist()
            v = pad12(v, 0.0)
            w[f"snow_total_{snow_unit}"] = v if snow_unit=="in" else [x*2.54 for x in v]
        else:
            w[f"snow_total_{snow_unit}"] = [0.0]*12

        if has("snow_events"):
            vals = pd.to_numeric(df[cols["snow_events"]], errors="coerce").fillna(0.0).tolist()
            w["snow_events"] = pad12(vals, 0.0)
        else:
            w["snow_events"] = [0.0]*12

        if has("relative_humidity_pct"):
            vals = pd.to_numeric(df[cols["relative_humidity_pct"]], errors="coerce").fillna(60.0).tolist()
            w["relative_humidity_pct"] = pad12(vals, 60.0)
        else:
            w["relative_humidity_pct"] = [60.0]*12

        if has("temp_air_c"):
            v = pd.to_numeric(df[cols["temp_air_c"]], errors="coerce").fillna(0.0).tolist()
            v = pad12(v, 0.0)
            w[f"temp_air_{'c' if temp_unit=='°C' else 'f'}"] = v if temp_unit=="°C" else [x*9/5+32 for x in v]
        elif has("temp_air_f"):
            v = pd.to_numeric(df[cols["temp_air_f"]], errors="coerce").fillna(32.0).tolist()
            v = pad12(v, 32.0)
            w[f"temp_air_{'c' if temp_unit=='°C' else 'f'}"] = [(x-32)*5/9 for x in v] if temp_unit=="°C" else v
        else:
            w[f"temp_air_{'c' if temp_unit=='°C' else 'f'}"] = [0.0]*12

        if has("rain_days"):
            vals = pd.to_numeric(df[cols["rain_days"]], errors="coerce").fillna(0.0).tolist()
            w["rain_days"] = pad12(vals, 0.0)
        else:
            w["rain_days"] = [0.0]*12

        if has("cleaned"):
            vals = _to_bool_series(df[cols["cleaned"]]).tolist()
            w["cleaned"] = pad12(vals, False)
        else:
            w["cleaned"] = [False]*12

        return poa_df_u.iloc[:12], w.iloc[:12]
    except Exception as e:
        st.warning(f"Could not parse CSV: {e}")
        return None, None

if up is not None:
    parsed_poa, parsed_weather = read_csv_upload(up, snow_unit, temp_unit)
    if parsed_poa is not None:
        poa_df = parsed_poa
    if parsed_weather is not None:
        weather_df = parsed_weather

# CSV template
if download_tmpl:
    tmpl = pd.DataFrame({
        "month": MONTHS,
        "poa_global_kwhm2": [120.0]*12,
        "snow_total_cm": [0.0]*12,
        "snow_events": [0.0]*12,
        "relative_humidity_pct": [60.0]*12,
        "temp_air_c": [0.0]*12,
        "rain_days": [0.0]*12,
        "cleaned": [False]*12,
    })
    buf = io.StringIO()
    tmpl.to_csv(buf, index=False)
    st.sidebar.download_button("Download CSV template", buf.getvalue(), file_name="snow_soil_template.csv", mime="text/csv")

# -----------------------------
# 1) Geometry & irradiance
# -----------------------------
st.subheader("1) Geometry & Irradiance")
geom_unit = st.session_state["geom_unit"]

c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    surface_tilt = st.number_input("Surface tilt (degrees)", min_value=0.0, max_value=90.0, value=30.0, step=1.0)
with c2:
    slant_height_disp = st.number_input(
        f"Slant height ({geom_unit})",
        min_value=0.0,
        value=float(st.session_state["slant_input"]),
        step=0.1 if geom_unit=="cm" else 0.05,
        key="slant_input"
    )
    st.session_state["geo_slant_m"] = unit_to_m(slant_height_disp, geom_unit)
    st.session_state["slant_display"] = slant_height_disp
with c3:
    lower_edge_height_disp = st.number_input(
        f"Lower edge height ({geom_unit})",
        min_value=0.0,
        value=float(st.session_state["lower_input"]),
        step=0.1 if geom_unit=="cm" else 0.05,
        key="lower_input"
    )
    st.session_state["geo_lower_m"] = unit_to_m(lower_edge_height_disp, geom_unit)
    st.session_state["lower_display"] = lower_edge_height_disp
with c4:
    string_factor = st.number_input("String factor", min_value=0.1, value=1.0, step=0.1)

# --- Vertical POA editor (12 rows, no extras) ---
st.markdown(f"**POA_global ({POA_UNIT}) — enter 12 monthly values**")
poa_df = poa_df.set_index("month").reindex(MONTHS).reset_index().iloc[:12]
poa_colcfg = {
    "month": st.column_config.TextColumn("month", disabled=True),
    "POA_global_kwhm2": st.column_config.NumberColumn("POA_global (kWh/m²-month)", min_value=0.0),
}
poa_editor = st.data_editor(
    poa_df, use_container_width=True, num_rows="fixed", hide_index=True, column_config=poa_colcfg
)
poa_editor_sorted = poa_editor.set_index("month").reindex(MONTHS).reset_index().iloc[:12]
poa_global_kwhm2 = poa_editor_sorted["POA_global_kwhm2"].astype(float).to_numpy()

# -----------------------------
# 2) Monthly weather & soiling inputs (12 rows, no extras)
# -----------------------------
st.subheader("2) Monthly Weather & Soiling Inputs")

st.info(
    "### Get monthly climate data\n"
    "Please retrieve monthly values (snowfall, temperature, RH, rain days, etc.) from:\n\n"
    "• **NOWData – NOAA Online Weather Data**: "
    "[sercc.com/noaa-online-weather](https://sercc.com/noaa-online-weather/)\n\n"
    "After you gather values for your station and year, paste them into the table below."
)

weather_df = weather_df.set_index("month").reindex(MONTHS).reset_index().iloc[:12]
colcfg = {
    "month": st.column_config.TextColumn("month", disabled=True),
    f"snow_total_{snow_unit}": st.column_config.NumberColumn(f"snow_total ({snow_unit})"),
    "snow_events": st.column_config.NumberColumn("snow_events (float)"),
    "relative_humidity_pct": st.column_config.NumberColumn("relative_humidity (%)", min_value=0.0, max_value=100.0),
    f"temp_air_{'c' if temp_unit=='°C' else 'f'}": st.column_config.NumberColumn(f"temp_air ({temp_unit})"),
    "rain_days": st.column_config.NumberColumn("rain_days", min_value=0.0, max_value=31.0),
    "cleaned": st.column_config.CheckboxColumn("manually cleaned?"),
}
weather_editor = st.data_editor(
    weather_df, use_container_width=True, num_rows="fixed", column_config=colcfg, hide_index=True
)

# -----------------------------
# Validation
# -----------------------------
def validate_inputs(poa, wdf, snow_unit, temp_unit) -> List[str]:
    errs = []
    if len(poa) != 12:
        errs.append("POA_global must have 12 monthly values.")
    for i, v in enumerate(poa):
        if v < 0:
            errs.append(f"POA value for {MONTHS[i]} is negative.")
    req_cols = ["month", "snow_events", "relative_humidity_pct",
                f"snow_total_{snow_unit}", f"temp_air_{'c' if temp_unit=='°C' else 'f'}",
                "rain_days", "cleaned"]
    for c in req_cols:
        if c not in wdf.columns:
            errs.append(f"Missing column: {c}")
    if "relative_humidity_pct" in wdf.columns:
        bad = wdf[(wdf["relative_humidity_pct"]<0) | (wdf["relative_humidity_pct"]>100)]
        if len(bad)>0: errs.append("Relative humidity must be 0–100%.")
    if (wdf[f"snow_total_{snow_unit}"]<0).any():
        errs.append("snow_total cannot be negative.")
    if "rain_days" in wdf.columns and (wdf["rain_days"]<0).any():
        errs.append("rain_days must be ≥ 0.")
    return errs

# -----------------------------
# Soiling helpers
# -----------------------------
def compute_soil_losses(rain_days, cleaned_flags):
    pre = np.zeros(12, dtype=float)
    post = np.zeros(12, dtype=float)
    prev_post = 0.75
    for i in range(12):
        if rain_days[i] >= 1:
            pre[i] = 0.75
        else:
            pre[i] = prev_post + 1.5
        post[i] = 0.75 if cleaned_flags[i] else pre[i]
        prev_post = post[i]
    return pre, post

# -----------------------------
# Run & persist results
# -----------------------------
st.divider()
run = st.button("▶️ Run model", help="Compute snow & soil losses")
if run:
    st.session_state["results_out_df"] = None
    st.session_state["last_inputs_df"] = None
    st.session_state["results_metrics"] = None

    errors = validate_inputs(poa_global_kwhm2, weather_editor, snow_unit, temp_unit)
    if errors:
        for e in errors:
            st.error(e)
    else:
        try:
            w_sorted = weather_editor.set_index("month").reindex(MONTHS).reset_index().iloc[:12]

            snow_vals_display = w_sorted[f"snow_total_{snow_unit}"].astype(float).to_numpy()
            snow_cm = snow_vals_display if snow_unit == "cm" else (snow_vals_display * 2.54)

            temp_disp = w_sorted[f"temp_air_{'c' if temp_unit=='°C' else 'f'}"].astype(float).to_numpy()
            temp_c = temp_disp if temp_unit == "°C" else (temp_disp - 32.0) * 5.0 / 9.0

            rh_pct = w_sorted["relative_humidity_pct"].astype(float).to_numpy()
            events = w_sorted["snow_events"].astype(float).to_numpy()

            slant_m = st.session_state["geo_slant_m"]
            lower_m = st.session_state["geo_lower_m"]

            poa_Wh_per_m2_month = np.array(poa_global_kwhm2, dtype=float) * 1000.0

            snow_frac = np.array(
                loss_townsend(
                    snow_total=snow_cm,
                    snow_events=events,
                    surface_tilt=surface_tilt,
                    relative_humidity=rh_pct,
                    temp_air=temp_c,
                    poa_global=poa_Wh_per_m2_month,
                    slant_height=slant_m,
                    lower_edge_height=lower_m,
                    string_factor=string_factor,
                    angle_of_repose=40,
                ),
                dtype=float
            )
            snow_pct = 100.0 * snow_frac

            rain_days = w_sorted["rain_days"].astype(float).to_numpy()
            cleaned_flags = w_sorted["cleaned"].astype(bool).to_numpy()
            _, soil_post_pct = compute_soil_losses(rain_days, cleaned_flags)

            final_pct = np.maximum(snow_pct, soil_post_pct)

            out = pd.DataFrame({
                "month": MONTHS,
                "Snow loss (%)": np.round(snow_pct, 2),
                "Soil loss (%)": np.round(soil_post_pct, 2),
                "Final loss (%)": np.round(final_pct, 2),
            }).iloc[:12]
            out["order"] = out["month"].map(MONTH_ORDER)
            out = out.sort_values("order").drop(columns=["order"]).iloc[:12]

            st.session_state["results_out_df"] = out
            st.session_state["last_inputs_df"] = pd.DataFrame({
                "month": MONTHS,
                "poa_Wh_per_m2_month": poa_Wh_per_m2_month,
                "snow_total_cm": snow_cm,
                "snow_events_float": events,
                "relative_humidity_pct": rh_pct,
                "temp_air_c": temp_c,
                "rain_days": rain_days,
                "cleaned": cleaned_flags,
                "slant_height_m": slant_m,
                "lower_edge_height_m": lower_m,
                "surface_tilt_deg": surface_tilt,
                "string_factor": string_factor,
            }).iloc[:12]

        except Exception as e:
            st.error(f"Error running model: {e}")

# -----------------------------
# Show results (percent)
# -----------------------------
if "results_out_df" in st.session_state and st.session_state["results_out_df"] is not None:
    out = st.session_state["results_out_df"]
    st.subheader("Results (percent)")
    st.dataframe(out, use_container_width=True)

    chart_df = out[["month","Snow loss (%)","Soil loss (%)"]].copy()
    chart_df["month"] = pd.Categorical(chart_df["month"], categories=MONTHS, ordered=True)
    chart_long = chart_df.melt("month", var_name="series", value_name="value")

    color_scale = alt.Scale(domain=["Snow loss (%)","Soil loss (%)"],
                            range=["#1f77b4", "#b08d57"])

    stacked = alt.Chart(chart_long).mark_bar().encode(
        x=alt.X('month:N', sort=MONTHS, title='Month'),
        y=alt.Y('value:Q', title='Loss (%)', stack='zero'),
        color=alt.Color('series:N', scale=color_scale, title=None),
        tooltip=['month','series','value']
    )
    st.altair_chart(stacked, use_container_width=True)

    st.download_button(
        "Download Results CSV",
        data=out.to_csv(index=False),
        file_name="pv_snow_soil_losses_results_percent.csv",
        mime="text/csv",
        key="dl_results"
    )

if "last_inputs_df" in st.session_state and st.session_state["last_inputs_df"] is not None:
    st.download_button(
        "Download Inputs Used (exact units)",
        data=st.session_state["last_inputs_df"].to_csv(index=False),
        file_name="pv_inputs_used_for_townsend.csv",
        mime="text/csv",
        key="dl_inputs"
    )
