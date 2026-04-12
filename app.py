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

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f6fbff 0%, #ffffff 100%);
    }

    section[data-testid="stSidebar"] {
        background-color: #edf5fb;
    }

    h1, h2, h3 {
        color: #103b5c;
    }

    div[data-testid="stMarkdownContainer"] p {
        color: #1f2937;
        font-size: 16px;
        line-height: 1.6;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Solar PV Performance Loss Modelling Application: Snow & Soiling Loss (Monthly)")

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
    "the Townsend precipitation-driven dust model (with seasonal ramp rates and optional wash optimization) to produce stacked "
    "monthly loss percentages one can use in EPC financials and O&M planning."
)
with st.sidebar:
    st.subheader("📘 How to Use")
    st.markdown("""
**1) Open the Tool**  
Use any modern browser.

**2) Enter Inputs**  
Choose units, then enter 12 monthly POA values and weather inputs (snow, events, RH, temp, precipitation).

**3) Enter Geometry**  
Tilt, slant height, lower edge height, and string factor.

**4) Optional CSV Upload**  
Upload a CSV to prefill tables.

**5) Run Model**  
Click **Run model** to compute snow, dust, and final loss.
**6) View & Download**  
Review the table and chart, then download Results and Inputs CSV files.
""")
st.markdown("""
    <style>
      .attrib { font-size: 1.5rem; line-height: 1.8; opacity: 0.9; }
      .attrib {
    font-size: 1.5rem;
    line-height: 1.8;
    opacity: 0.9;
}
    </style>
""", unsafe_allow_html=True)
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
- precip
Legacy columns still accepted (ignored by dust calculation):
- precip
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
        precip =    [2.8, 2.4, 2.1, 2.5, 3.2, 3.8, 4.1, 3.7, 3.1, 2.9, 2.7, 2.6]
    elif preset_name == "Cold winter (example)":
        poa_vals = [60,85,120,150,175,185,180,170,140,110,75,55]
        snow_cm =   [35,32,28, 15,  2,  0,  0,  0,  3, 12, 25,38]
        temps_c =   [-10,-6,-2,  5, 12, 18, 21, 20, 15,  8,  0,-7]
        rh =        [74,73,70, 65, 63, 64, 66, 67, 68, 70, 72, 74]
        events =    [6.0, 6.0, 5.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 5.0, 6.0]
        precip =    [1.8, 1.7, 1.9, 2.3, 2.8, 3.3, 3.6, 3.1, 2.5, 2.2, 2.0, 1.9]
    else:
        poa_vals = [120.0]*12
        snow_cm  = [0.0]*12
        temps_c  = [0.0]*12
        rh       = [60.0]*12
        events   = [0.0]*12
        precip   = [0.0]*12

    snow_disp = snow_cm if snow_unit=="cm" else [round(x/2.54,3) for x in snow_cm]
    temps_disp = temps_c if temp_unit=="°C" else [round(x*9/5+32,3) for x in temps_c]

    poa_df = pd.DataFrame({"month": MONTHS, "POA_global_kwhm2": poa_vals}).iloc[:12]
    weather = pd.DataFrame({
        "month": MONTHS,
        f"snow_total_{snow_unit}": snow_disp,
        "snow_events": events,
        "relative_humidity_pct": rh,
        f"temp_air_{'c' if temp_unit=='°C' else 'f'}": temps_disp,
        "precip": precip,
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

        if has("precip"):
            vals = pd.to_numeric(df[cols["precip"]], errors="coerce").fillna(0.0).tolist()
            w["precip"] = pad12(vals, 0.0)
        elif has("rain_days"):
            # Legacy fallback for older CSVs so they still load without breaking.
            vals = pd.to_numeric(df[cols["rain_days"]], errors="coerce").fillna(0.0).tolist()
            w["precip"] = pad12(vals, 0.0)
        else:
            w["precip"] = [0.0]*12

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
        "precip": [0.0]*12,
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
    string_factor = st.radio(
    "String factor",
    options=[1.0, 0.75],
    index=0,
    horizontal=True,
    help="Allowed values are 1.0 or 0.75."
)

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

c_d1, c_d2 = st.columns([1, 1])
with c_d1:
    precip_unit = st.radio("Precipitation unit", ["in", "mm"], horizontal=True, index=0)
with c_d2:
    manual_washes = st.radio("Manual washes per year", [0, 1, 2], horizontal=True, index=0)

RAMP_RATE_OPTIONS = {
    "Typical (0.10)": 0.10,
    "Ultra sandy (0.025)": 0.025,
    "Desert (0.05)": 0.05,
    "Humid/agri./sooty/birds (0.15)": 0.15,
}

c_r1, c_r2, c_r3, c_r4 = st.columns(4)
with c_r1:
    ramp_dec_feb = RAMP_RATE_OPTIONS[
        st.selectbox("Dec–Feb ramp (%/day)", list(RAMP_RATE_OPTIONS.keys()), index=0)
    ]
with c_r2:
    ramp_mar_may = RAMP_RATE_OPTIONS[
        st.selectbox("Mar–May ramp (%/day)", list(RAMP_RATE_OPTIONS.keys()), index=0)
    ]
with c_r3:
    ramp_jun_aug = RAMP_RATE_OPTIONS[
        st.selectbox("Jun–Aug ramp (%/day)", list(RAMP_RATE_OPTIONS.keys()), index=0)
    ]
with c_r4:
    ramp_sep_nov = RAMP_RATE_OPTIONS[
        st.selectbox("Sep–Nov ramp (%/day)", list(RAMP_RATE_OPTIONS.keys()), index=0)
    ]


st.info(
    "### Get monthly climate data\n"
    "Please retrieve monthly values (snowfall, temperature, RH, precipitation, etc.) from:\n\n"
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
    "precip": st.column_config.NumberColumn(f"precip ({precip_unit})", min_value=0.0),
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
                "precip"]
    for c in req_cols:
        if c not in wdf.columns:
            errs.append(f"Missing column: {c}")
    if "relative_humidity_pct" in wdf.columns:
        bad = wdf[(wdf["relative_humidity_pct"]<0) | (wdf["relative_humidity_pct"]>100)]
        if len(bad)>0: errs.append("Relative humidity must be 0–100%.")
    if (wdf[f"snow_total_{snow_unit}"]<0).any():
        errs.append("snow_total cannot be negative.")
    if "precip" in wdf.columns and (wdf["precip"]<0).any():
        errs.append("precip must be ≥ 0.")
    return errs

# -----------------------------
# Soiling helpers
# -----------------------------
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _seasonal_ramps(ramp_dec_feb, ramp_mar_may, ramp_jun_aug, ramp_sep_nov):
    r = [0.0] * 12
    for i in [11, 0, 1]:
        r[i] = float(ramp_dec_feb)
    for i in [2, 3, 4]:
        r[i] = float(ramp_mar_may)
    for i in [5, 6, 7]:
        r[i] = float(ramp_jun_aug)
    for i in [8, 9, 10]:
        r[i] = float(ramp_sep_nov)
    return r

def _precip_in_inches(precip, units):
    if units.lower() == "in":
        return [float(v) for v in precip]
    if units.lower() == "mm":
        return [float(v) / 25.4 for v in precip]
    raise ValueError("precip units must be 'in' or 'mm'.")

def compute_dust_baseline_pct_mono(precip_in, ramps, snow_loss_pct):
    max_p = max(precip_in)
    start_idx = precip_in.index(max_p)
    start_snow = float(snow_loss_pct[start_idx])

    if start_snow >= 3.0:
        start_soil = 0.0
    else:
        if max_p >= 4.0:
            start_soil = 0.0
        elif max_p >= 2.0:
            start_soil = 1.0
        else:
            start_soil = 2.0

    mtype = [""] * 12
    inc = [0.0] * 12
    fixed = [0.0] * 12

    for i in range(12):
        if i == start_idx:
            mtype[i] = "Start"
        else:
            mtype[i] = "Const." if precip_in[i] >= 2.0 else "Additive"

    for i in range(12):
        if mtype[i] in ("Const.", "Start"):
            inc[i] = 0.0
        else:
            if precip_in[i] < 1.0:
                inc[i] = DAYS_IN_MONTH[i] * ramps[i]
            else:
                prev = (i - 1) % 12
                if mtype[prev] == "Additive":
                    inc[i] = 15.0 * ramps[i]
                else:
                    inc[i] = 8.0 * ramps[i]

        if mtype[i] == "Const.":
            if precip_in[i] >= 4.0:
                fixed[i] = 0.0
            elif precip_in[i] >= 2.0:
                fixed[i] = 1.0
            else:
                fixed[i] = start_soil
        else:
            fixed[i] = start_soil

    base = [0.0] * 12
    for i in range(12):
        if snow_loss_pct[i] >= 3.0:
            soil = 0.0
        else:
            if mtype[i] == "Additive":
                prev = (i - 1) % 12
                soil = base[prev] + inc[i]
            elif mtype[i] == "Const.":
                soil = fixed[i]
            else:
                soil = start_soil
        base[i] = _clamp(soil, 0.0, 30.0)

    return base

def compute_month_only_soil_pct_mono(precip_in, ramps, snow_loss_pct):
    out = []
    for i in range(12):
        if float(snow_loss_pct[i]) >= 3.0:
            soil = 0.0
        else:
            p = float(precip_in[i])
            r = float(ramps[i])
            if p >= 4.0:
                soil = 0.0
            elif p >= 2.0:
                soil = 1.0
            elif p >= 1.0:
                soil = np.floor(DAYS_IN_MONTH[i] / 2.0) * r
            else:
                soil = (DAYS_IN_MONTH[i] / 2.0) * r
        out.append(_clamp(soil, 0.0, 30.0))
    return out

def compute_energy_weights_from_poa(poa_kwhm2):
    total = sum(max(0.0, float(v)) for v in poa_kwhm2)
    if total <= 0:
        return [1.0 / 12.0] * 12
    return [max(0.0, float(v)) / total for v in poa_kwhm2]

def optimize_washes_mono(baseline, month_only, energy_weights, washes):
    washes = int(max(0, min(2, washes)))

    def score(vec):
        return sum(float(vec[i]) * float(energy_weights[i]) for i in range(12))

    def cap_against(raw, cap_series):
        return [min(float(raw[i]), float(cap_series[i])) for i in range(12)]

    def build_1wash_raw(w1):
        raw = [float(v) for v in baseline]
        raw[w1] = float(month_only[w1])
        for m in range(w1 + 1, 12):
            delta = float(baseline[m]) - float(baseline[m - 1])
            raw[m] = max(0.0, raw[m - 1] + delta)
        return raw

    def build_2wash_raw(final1, w2):
        raw = [float(v) for v in final1]
        raw[w2] = float(month_only[w2])
        for m in range(w2 + 1, 12):
            delta = float(final1[m]) - float(final1[m - 1])
            raw[m] = max(0.0, raw[m - 1] + delta)
        return raw

    if washes == 0:
        return [float(v) for v in baseline], None, None

    raw_candidates_1 = [build_1wash_raw(w) for w in range(12)]
    final_candidates_1 = [cap_against(r, baseline) for r in raw_candidates_1]
    scores_1 = [score(r) for r in raw_candidates_1]

    min_s1 = min(scores_1)
    avg_s1 = sum(scores_1) / len(scores_1)

    if abs(min_s1 - avg_s1) < 1e-12:
        final1 = [float(v) for v in baseline]
        best1 = None
    else:
        best1_idx = scores_1.index(min_s1)
        final1 = final_candidates_1[best1_idx]
        best1 = best1_idx + 1

    if washes == 1:
        return final1, best1, None

    if best1 is None:
        return final1, None, None

    w1 = best1 - 1
    possible_w2 = list(range(w1 + 1, 12))
    if not possible_w2:
        return final1, best1, None

    raw_candidates_2 = []
    final_candidates_2 = []
    scores_2 = []

    for w2 in possible_w2:
        raw2 = build_2wash_raw(final1, w2)
        raw_candidates_2.append((w2, raw2))
        final2 = cap_against(raw2, final1)
        final_candidates_2.append(final2)
        scores_2.append(score(raw2))

    min_s2 = min(scores_2)
    avg_s2 = sum(scores_2) / len(scores_2)

    if abs(min_s2 - avg_s2) < 1e-12:
        return final1, best1, None

    best2_local_idx = scores_2.index(min_s2)
    best2_w2, _ = raw_candidates_2[best2_local_idx]
    final2 = final_candidates_2[best2_local_idx]
    best2 = best2_w2 + 1

    return final2, best1, best2

def compute_combined_loss_pct_mono(snow_loss_pct, dust_loss_pct):
    out = []
    for s, d in zip(snow_loss_pct, dust_loss_pct):
        if float(s) >= 3.0:
            out.append(float(s))
        else:
            sf = float(s) / 100.0
            df = float(d) / 100.0
            out.append(100.0 * (sf + df - sf * df))
    return out

# -----------------------------
# Run & persist results
# -----------------------------
st.divider()
run = st.button("▶️ Run model", help="Compute snow & soil losses")
st.caption(
    "Combining snow and dust losses: If monthly snow loss is ≥ 3%, snow governs and dust loss is ignored. "
    "If snow loss is 0%, dust loss is the only loss. If 0% < snow loss < 3%, losses are combined as "
    "A + B − (A × B), where A is fractional snow loss and B is fractional dust loss."
)
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

            precip = w_sorted["precip"].astype(float).tolist()
            precip_in = _precip_in_inches(precip, precip_unit)

            ramps = _seasonal_ramps(
                ramp_dec_feb=ramp_dec_feb,
                ramp_mar_may=ramp_mar_may,
                ramp_jun_aug=ramp_jun_aug,
                ramp_sep_nov=ramp_sep_nov,
            )

            baseline_dust_pct = compute_dust_baseline_pct_mono(
                precip_in=precip_in,
                ramps=ramps,
                snow_loss_pct=snow_pct.tolist(),
            )

            month_only_dust_pct = compute_month_only_soil_pct_mono(
                precip_in=precip_in,
                ramps=ramps,
                snow_loss_pct=snow_pct.tolist(),
            )

            energy_weights = compute_energy_weights_from_poa(poa_global_kwhm2.tolist())

            soil_post_pct, best_wash_1, best_wash_2 = optimize_washes_mono(
                baseline=baseline_dust_pct,
                month_only=month_only_dust_pct,
                energy_weights=energy_weights,
                washes=manual_washes,
            )

            final_pct = np.array(
                compute_combined_loss_pct_mono(
                    snow_loss_pct=snow_pct.tolist(),
                    dust_loss_pct=soil_post_pct,
                ),
                dtype=float
            )

            soil_post_pct = np.array(soil_post_pct, dtype=float)


            out = pd.DataFrame({
                "month": MONTHS,
                "Snow loss (%)": np.round(snow_pct, 2),
                "Soil loss (%)": np.round(soil_post_pct, 2),
                "Total loss (%)": np.round(final_pct, 2),
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
                "precip_input": precip,
                "precip_unit": [precip_unit] * 12,
                "manual_washes": [manual_washes] * 12,
                "ramp_dec_feb": [ramp_dec_feb] * 12,
                "ramp_mar_may": [ramp_mar_may] * 12,
                "ramp_jun_aug": [ramp_jun_aug] * 12,
                "ramp_sep_nov": [ramp_sep_nov] * 12,
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
    if manual_washes == 0:
        st.caption("Manual washes: None")
    else:
        wash1_label = "None" if best_wash_1 is None else f"{best_wash_1} ({MONTHS[best_wash_1-1]})"
        st.caption(f"Best wash month #1: {wash1_label}")
        if manual_washes == 2:
            wash2_label = "None" if best_wash_2 is None else f"{best_wash_2} ({MONTHS[best_wash_2-1]})"
            st.caption(f"Best wash month #2: {wash2_label}")
    st.dataframe(out, use_container_width=True)

    # chart_df = out[["month","Snow loss (%)","Soil loss (%)"]].copy()
    # chart_df["month"] = pd.Categorical(chart_df["month"], categories=MONTHS, ordered=True)
    # chart_long = chart_df.melt("month", var_name="series", value_name="value")

    # color_scale = alt.Scale(domain=["Snow loss (%)","Soil loss (%)"],
    #                         range=["#1f77b4", "#b08d57"])

    # stacked = alt.Chart(chart_long).mark_bar().encode(
    #     x=alt.X('month:N', sort=MONTHS, title='Month'),
    #     y=alt.Y('value:Q', title='Loss (%)', stack='zero'),
    #     color=alt.Color('series:N', scale=color_scale, title=None),
    #     tooltip=['month','series','value']
    # )
    # st.altair_chart(stacked, use_container_width=True)

    chart_df = out[["month", "Total loss (%)"]].copy()
    chart_df["month"] = pd.Categorical(chart_df["month"], categories=MONTHS, ordered=True)
    
    total_bar = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("month:N", sort=MONTHS, title="Month"),
        y=alt.Y("Total loss (%):Q", title="Total loss (%)"),
        tooltip=["month", "Total loss (%)"]
    )
    
    st.altair_chart(total_bar, use_container_width=True)
    
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

from pathlib import Path

st.divider()
st.subheader("📄 Technical Documentation")

st.markdown(
    """
Download the technical documentation for Townsend’s monthly snow loss model,
including equations, ground-interference term, tracking guidance, bifacial guidance,
and the string-factor discussion.
"""
)

pdf_path = Path(__file__).parent / "SnowModelTheory.pdf"

if pdf_path.exists():
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Download Snow Model Theory (PDF)",
            data=f,
            file_name="SnowModelTheory.pdf",
            mime="application/pdf"
        )
else:
    st.warning("SnowModelTheory.pdf was not found in the deployed app folder.")
st.markdown(
    "<p style='font-size:18px; font-weight:bold;'>Advanced Bifacial PV Snow & Soiling Loss Calculator (Townsend Model)</p>",
    unsafe_allow_html=True
)

st.caption(
    "An engineering-based tool for estimating photovoltaic performance losses due to snow cover and environmental soiling, "
    "designed with support for bifacial PV systems and detailed monthly inputs."
)

st.caption("Reference: https://townsendsnowdustmodel.streamlit.app/")

st.markdown("### References")

st.markdown("""
1. Townsend, Tim & Powers, Loren. (2011). *Photovoltaics and snow: An update from two winters of measurements in the SIERRA.* 37th IEEE Photovoltaic Specialists Conference, Seattle, WA, USA. DOI: 10.1109/PVSC.2011.6186627

2. Townsend, T. and Previtali, J. (2023). *A Fresh Dusting: Current Uses of the Townsend Snow Model.* In “Photovoltaic Reliability Workshop (PVRW) 2023 Proceedings: Posters.”, ed. Silverman, T. J. Dec. 2023. NREL/CP-5900-87918. Available at: https://docs.nlr.gov/docs/fy24osti/87918.pdf

3. Townsend, T. (2013). *Predicting PV Energy Loss Caused by Snow.* Solar Power International, Chicago IL. DOI: 10.13140/RG.2.2.14299.68647
""")
st.markdown('<p class="attrib">This webpage was created by <b>Sheth Kajal</b> 😊</p>',
            unsafe_allow_html=True)
