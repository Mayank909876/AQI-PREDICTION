import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np

# -----------------------------------------------
# PAGE CONFIG
# -----------------------------------------------
st.set_page_config(
    page_title="AQI Intelligence Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------
# GLOBAL STYLES
# -----------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #080c14; color: #e8eaf0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1400px; }
.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 60%, #0a1628 100%);
    border: 1px solid #1e3a5f; border-radius: 20px;
    padding: 2.5rem 3rem; margin-bottom: 2.5rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,200,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem; font-weight: 700; letter-spacing: -0.03em;
    background: linear-gradient(90deg, #e8eaf0 0%, #64b8ff 60%, #00e5ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero-sub { color: #8fa8c8; font-size: 1rem; font-weight: 400; letter-spacing: 0.04em; margin: 0; }
.hero-badge {
    display: inline-block; background: rgba(0,180,255,0.12);
    border: 1px solid rgba(0,180,255,0.3); color: #00c8ff;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    padding: 4px 12px; border-radius: 20px; margin-bottom: 1rem;
    letter-spacing: 0.12em; text-transform: uppercase;
}
.section-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.2em; text-transform: uppercase; color: #4a7fa5;
    margin-bottom: 0.8rem; display: flex; align-items: center; gap: 8px;
}
.section-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #1e3a5f, transparent);
}
.input-panel {
    background: #0d1b2a; border: 1px solid #1a2e45;
    border-radius: 16px; padding: 1.8rem 2rem; margin-bottom: 1.5rem;
}
div[data-testid="stNumberInput"] label {
    color: #8fa8c8 !important; font-size: 0.82rem !important;
    font-weight: 500 !important; letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
div[data-testid="stNumberInput"] input {
    background: #101e30 !important; border: 1px solid #1e3a5f !important;
    border-radius: 8px !important; color: #e8eaf0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #00c8ff !important;
    box-shadow: 0 0 0 2px rgba(0,200,255,0.15) !important;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0066cc, #0099ff) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important; font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important; font-size: 1rem !important;
    letter-spacing: 0.04em !important; width: 100% !important;
    transition: all 0.2s ease !important; box-shadow: 0 4px 20px rgba(0,100,255,0.3) !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(0,150,255,0.45) !important;
}
.result-card { border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; border: 1px solid; }
.aqi-number { font-family: 'JetBrains Mono', monospace; font-size: 3.2rem; font-weight: 700; line-height: 1; }
.aqi-label { font-size: 0.8rem; letter-spacing: 0.15em; text-transform: uppercase; opacity: 0.7; margin-top: 0.3rem; }
.category-pill { display: inline-block; padding: 6px 18px; border-radius: 30px; font-weight: 600; font-size: 1rem; letter-spacing: 0.04em; }
div[data-testid="stDownloadButton"] > button {
    background: transparent !important; border: 1px solid #1e3a5f !important;
    color: #8fa8c8 !important; border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important; font-weight: 500 !important;
    width: 100% !important; transition: all 0.2s ease !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    border-color: #00c8ff !important; color: #00c8ff !important;
    background: rgba(0,200,255,0.06) !important;
}
.chart-container {
    background: #0d1b2a; border: 1px solid #1a2e45;
    border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;
}
.advisory-box {
    border-radius: 14px; padding: 1.2rem 1.6rem;
    margin-bottom: 1.5rem; border: 1px solid;
    display: flex; align-items: flex-start; gap: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------
# LOAD MODEL
# -----------------------------------------------
@st.cache_resource
def load_model():
    model    = joblib.load("aqi_model.pkl")
    features = joblib.load("features.pkl")
    return model, features

model, feature_columns = load_model()

# -----------------------------------------------
# HELPERS
# -----------------------------------------------
def get_category(aqi):
    if aqi <= 50:    return "Good",        "#00c48c", "#0a2e1e", 50
    elif aqi <= 100: return "Satisfactory", "#a8e063", "#1a2a0a", 100
    elif aqi <= 200: return "Moderate",     "#f9ca24", "#2a2000", 200
    elif aqi <= 300: return "Poor",         "#f0932b", "#2a1500", 300
    elif aqi <= 400: return "Very Poor",    "#eb4d4b", "#2a0a0a", 400
    else:            return "Severe",       "#6c5ce7", "#120a2a", 500

# Safe limits for each pollutant (India NAAQS / AQI standards)
SAFE_LIMITS = {
    "PM2.5": 60.0,
    "PM10":  100.0,
    "NO":    80.0,
    "NO2":   80.0,
    "CO":    2.0,
    "SO2":   80.0,
    "O3":    100.0,
}

POLLUTANT_ICONS = {
    "PM2.5": "●", "PM10": "●", "NO": "◆",
    "NO2": "◆", "CO": "▲", "SO2": "▲", "O3": "■",
}

HEALTH_ADVISORIES = {
    "Good": {
        "icon": "✅",
        "title": "Air quality is excellent — no precautions needed.",
        "groups": [
            ("General Population", "Safe to enjoy outdoor activities freely."),
            ("Sensitive Groups",    "No restrictions. Great day to exercise outside."),
            ("Children & Elderly",  "Ideal conditions. Outdoor play is fully encouraged."),
        ],
        "tip": "Enjoy the clean air! Consider opening windows for natural ventilation.",
    },
    "Satisfactory": {
        "icon": "🟢",
        "title": "Air quality is acceptable with minor risk for sensitive individuals.",
        "groups": [
            ("General Population", "Outdoor activities are fine."),
            ("Sensitive Groups",    "Limit prolonged, heavy exertion outdoors."),
            ("Children & Elderly",  "Short outdoor exposure is safe; watch for symptoms."),
        ],
        "tip": "Keep hydrated during outdoor activities. Air purifiers optional indoors.",
    },
    "Moderate": {
        "icon": "🟡",
        "title": "Sensitive groups may experience health effects.",
        "groups": [
            ("General Population", "Unusually sensitive people should reduce prolonged outdoor exertion."),
            ("Sensitive Groups",    "Reduce prolonged exertion. Wear a mask if needed."),
            ("Children & Elderly",  "Limit time outdoors. Keep windows closed during peak hours."),
        ],
        "tip": "Use an N95 mask outdoors. Run indoor air purifiers if available.",
    },
    "Poor": {
        "icon": "🟠",
        "title": "Everyone may begin experiencing health effects.",
        "groups": [
            ("General Population", "Reduce prolonged or heavy outdoor exertion."),
            ("Sensitive Groups",    "Avoid all outdoor physical activity."),
            ("Children & Elderly",  "Stay indoors. Keep windows shut."),
        ],
        "tip": "Wear an N95 mask outdoors. Avoid rush-hour traffic exposure. Drink plenty of water.",
    },
    "Very Poor": {
        "icon": "🔴",
        "title": "Health alert — serious effects possible for everyone.",
        "groups": [
            ("General Population", "Avoid any prolonged outdoor activity."),
            ("Sensitive Groups",    "Stay indoors. Seek medical advice if symptomatic."),
            ("Children & Elderly",  "Do not go outside. Medical-grade mask if unavoidable."),
        ],
        "tip": "Keep all doors and windows sealed. Use air purifiers. Avoid cooking with high smoke.",
    },
    "Severe": {
        "icon": "🚨",
        "title": "Emergency conditions — entire population is at risk.",
        "groups": [
            ("General Population", "Avoid all outdoor exposure. Emergency conditions."),
            ("Sensitive Groups",    "Stay indoors. Contact health services if unwell."),
            ("Children & Elderly",  "Absolute indoor confinement. Seek medical help immediately."),
        ],
        "tip": "Seal all gaps in doors/windows. N95 masks mandatory if outdoors. Check local emergency alerts.",
    },
}

# -----------------------------------------------
# MATPLOTLIB GLOBAL STYLE
# -----------------------------------------------
DARK_BG    = "#0d1b2a"
PANEL_BG   = "#101e30"
GRID_COLOR = "#1a2e45"
TEXT_COLOR = "#8fa8c8"
ACCENT     = "#00c8ff"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   "#e8eaf0",
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.6,
    "text.color":        "#e8eaf0",
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# -----------------------------------------------
# HERO HEADER
# -----------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-badge">🌫️ Environmental Intelligence</div>
    <h1 class="hero-title">AQI Prediction Dashboard</h1>
    <p class="hero-sub">Real-time air quality forecasting powered by machine learning</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------
# INPUT SECTION
# -----------------------------------------------
st.markdown('<div class="section-label">Pollutant Parameters</div>', unsafe_allow_html=True)
st.markdown('<div class="input-panel">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    pm25 = st.number_input("PM 2.5  (µg/m³)", value=50.0,  min_value=0.0, step=0.5)
    no   = st.number_input("NO  (µg/m³)",      value=20.0,  min_value=0.0, step=0.5)
with col2:
    pm10 = st.number_input("PM 10  (µg/m³)",   value=100.0, min_value=0.0, step=0.5)
    no2  = st.number_input("NO₂  (µg/m³)",     value=30.0,  min_value=0.0, step=0.5)
with col3:
    co   = st.number_input("CO  (mg/m³)",       value=1.0,   min_value=0.0, step=0.1)
    so2  = st.number_input("SO₂  (µg/m³)",     value=10.0,  min_value=0.0, step=0.5)
with col4:
    o3   = st.number_input("O₃  (µg/m³)",      value=20.0,  min_value=0.0, step=0.5)

month = 6
st.markdown('</div>', unsafe_allow_html=True)

_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    predict = st.button("⚡  Run Prediction")

# -----------------------------------------------
# PREDICTION LOGIC
# -----------------------------------------------
if predict:
    input_dict = {c: 0 for c in feature_columns}
    input_dict.update({
        'PM2.5': pm25, 'PM10': pm10,
        'NO': no,       'NO2': no2,
        'CO': co,       'SO2': so2,
        'O3': o3,       'Month': month,
        'Year': 2020,   'Day': 15,
        'NOx': no + no2,
        'NH3': 0, 'Benzene': 0, 'Toluene': 0, 'Xylene': 0,
        'PM_ratio':  pm25 / (pm10 + 1),
        'NOx_ratio': no   / (no2  + 1),
        'is_winter':  int(month in [11,12,1,2]),
        'is_monsoon': int(month in [6,7,8]),
    })
    input_df   = pd.DataFrame([input_dict])[feature_columns]
    prediction = model.predict(input_df)[0]
    category, cat_color, cat_bg, max_val = get_category(prediction)

    # Compute per-pollutant exceedance ratios
    pollutant_vals_raw = {
        "PM2.5": pm25, "PM10": pm10,
        "NO": no, "NO2": no2,
        "CO": co, "SO2": so2, "O3": o3,
    }
    exceedance = {k: v / SAFE_LIMITS[k] for k, v in pollutant_vals_raw.items()}
    dominant_pollutant = max(exceedance, key=exceedance.get)
    dominant_ratio     = exceedance[dominant_pollutant]

    # ── RESULT CARDS ───────────────────────────────────────────
    st.markdown("")
    st.markdown('<div class="section-label">Prediction Results</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(f"""
        <div class="result-card" style="background:{cat_bg}; border-color:{cat_color}44;">
            <div class="aqi-label">Predicted AQI</div>
            <div class="aqi-number" style="color:{cat_color};">{round(prediction, 1)}</div>
        </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="result-card" style="background:#0d1b2a; border-color:#1a2e45;">
            <div class="aqi-label">Air Quality Category</div>
            <div style="margin-top:0.6rem;">
                <span class="category-pill"
                      style="background:{cat_color}22; color:{cat_color}; border:1px solid {cat_color}55;">
                    {category}
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

    with r3:
        dom_ratio_pct = round(dominant_ratio * 100, 1)
        dom_status    = "⚠️ Exceeding" if dominant_ratio > 1 else "✓ Within"
        dom_status_color = "#eb4d4b" if dominant_ratio > 1 else "#00c48c"
        st.markdown(f"""
        <div class="result-card" style="background:#0d1b2a; border-color:#1a2e45;">
            <div class="aqi-label">Dominant Pollutant</div>
            <div style="font-size:1.5rem; font-weight:700; margin-top:0.4rem;
                        color:#e8eaf0; font-family:'JetBrains Mono',monospace;">
                {dominant_pollutant}
            </div>
            <div style="font-size:0.78rem; margin-top:0.3rem; color:{dom_status_color};">
                {dom_status} safe limit &nbsp;·&nbsp; {dom_ratio_pct}%
            </div>
        </div>""", unsafe_allow_html=True)

    # ── HEALTH ADVISORY ────────────────────────────────────────
    adv = HEALTH_ADVISORIES[category]
    st.markdown('<div class="section-label">Health Advisory</div>', unsafe_allow_html=True)

    # Build group cards HTML separately to avoid nested f-string issues
    group_cards_html = ""
    for g, d in adv["groups"]:
        group_cards_html += (
            f'<div style="background:rgba(255,255,255,0.04); border-radius:8px;'
            f' padding:0.6rem 0.8rem; border:1px solid {cat_color}22;">'
            f'<div style="font-size:0.68rem; color:{cat_color}; font-weight:600;'
            f' letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.3rem;">{g}</div>'
            f'<div style="font-size:0.8rem; color:#c8d8e8; line-height:1.4;">{d}</div>'
            f'</div>'
        )

    advisory_html = (
        f'<div class="advisory-box" style="background:{cat_bg}; border-color:{cat_color}44;">'
        f'<div style="font-size:2rem; line-height:1;">{adv["icon"]}</div>'
        f'<div style="flex:1;">'
        f'<div style="font-weight:600; color:{cat_color}; margin-bottom:0.7rem; font-size:0.97rem;">'
        f'{adv["title"]}</div>'
        f'<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:0.8rem; margin-bottom:0.8rem;">'
        f'{group_cards_html}</div>'
        f'<div style="font-size:0.8rem; color:#8fa8c8; border-top:1px solid {cat_color}22;'
        f' padding-top:0.6rem; margin-top:0.2rem;">'
        f'💡 <em>{adv["tip"]}</em></div>'
        f'</div></div>'
    )
    st.markdown(advisory_html, unsafe_allow_html=True)

    # ── AQI GAUGE ──────────────────────────────────────────────
    st.markdown('<div class="section-label">AQI Gauge</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)

    fig_gauge, ax_g = plt.subplots(figsize=(7, 3.5), subplot_kw={"projection": "polar"})
    fig_gauge.patch.set_facecolor(DARK_BG)
    ax_g.set_facecolor(DARK_BG)

    segments = [
        (0,   50,  "#00c48c", "Good"),
        (50,  100, "#a8e063", "Satisfactory"),
        (100, 200, "#f9ca24", "Moderate"),
        (200, 300, "#f0932b", "Poor"),
        (300, 400, "#eb4d4b", "Very Poor"),
        (400, 500, "#6c5ce7", "Severe"),
    ]
    for lo, hi, color, _ in segments:
        theta = np.linspace(np.pi*(1-lo/500), np.pi*(1-hi/500), 60)
        ax_g.fill_between(theta, 0.6, 1.0, color=color, alpha=0.85)

    needle_angle = np.pi * (1 - min(prediction, 500) / 500)
    ax_g.annotate("", xy=(needle_angle, 0.85), xytext=(needle_angle, 0.1),
                  arrowprops=dict(arrowstyle="-|>", color="white", lw=2.0))
    ax_g.plot(needle_angle, 0.1, "o", color="white", markersize=8, zorder=5)
    ax_g.set_ylim(0, 1.1)
    ax_g.set_xlim(0, np.pi)
    ax_g.set_theta_zero_location("W")
    ax_g.set_theta_direction(1)
    ax_g.axis("off")

    for lo, hi, color, label in segments:
        angle = np.pi * (1 - (lo+hi)/1000)
        ax_g.text(angle, 1.18, label, ha="center", va="center",
                  fontsize=6.5, color=color, fontweight="bold")

    ax_g.text(np.pi/2, 0.28, f"{round(prediction,1)}", ha="center", va="center",
              fontsize=22, color="white", fontweight="bold", transform=ax_g.transData)
    ax_g.text(np.pi/2, 0.08, "AQI", ha="center", va="center",
              fontsize=9, color=TEXT_COLOR, transform=ax_g.transData)
    plt.tight_layout()
    st.pyplot(fig_gauge, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── POLLUTANT BAR CHART ─────────────────────────────────────
    st.markdown('<div class="section-label">Pollutant Breakdown</div>', unsafe_allow_html=True)

    pollutants_display = {
        "PM2.5": pm25, "PM10": pm10,
        "NO":    no,   "NO2": no2,
        "CO (×10)": co * 10,
        "SO₂":  so2,  "O₃": o3,
    }
    values  = list(pollutants_display.values())
    max_v   = max(values) if max(values) > 0 else 1
    colors  = [plt.cm.cool(v / max_v) for v in values]

    fig_bar, ax_b = plt.subplots(figsize=(11, 4))
    bars = ax_b.bar(pollutants_display.keys(), values, color=colors, width=0.55,
                    edgecolor="#1a2e45", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_v*0.015,
                  f"{val:.1f}", ha="center", va="bottom", fontsize=9,
                  color="#e8eaf0", fontweight="500")

    ax_b.set_xlabel("Pollutant", labelpad=8, fontsize=10)
    ax_b.set_ylabel("Concentration", labelpad=8, fontsize=10)
    ax_b.set_title("Input Pollutant Levels", fontsize=13, fontweight="600",
                   color="#e8eaf0", pad=14)
    ax_b.grid(axis="y", alpha=0.4)
    ax_b.set_axisbelow(True)
    plt.tight_layout()

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.pyplot(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── POLLUTANT SUB-INDEX  +  RADAR ──────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-label">Pollutant Sub-Index vs Safe Limit</div>',
                    unsafe_allow_html=True)

        # Build sub-index: how many % of safe limit is each pollutant at
        sub_names  = list(SAFE_LIMITS.keys())
        sub_raw    = [pollutant_vals_raw[k] for k in sub_names]
        sub_limits = [SAFE_LIMITS[k] for k in sub_names]
        sub_pct    = [min(v / s * 100, 300) for v, s in zip(sub_raw, sub_limits)]

        # Bar colour: green ≤ 80 %, amber ≤ 150 %, red > 150 %
        bar_colors = []
        for p in sub_pct:
            if p <= 80:    bar_colors.append("#00c48c")
            elif p <= 150: bar_colors.append("#f9ca24")
            else:          bar_colors.append("#eb4d4b")

        fig_si, ax_si = plt.subplots(figsize=(6, 4))
        y_pos = np.arange(len(sub_names))

        # Background 100 % line
        ax_si.axvline(100, color="#f9ca24", lw=1.2, linestyle="--", alpha=0.5, zorder=1)
        ax_si.barh(y_pos, sub_pct, color=bar_colors, height=0.55,
                   edgecolor=GRID_COLOR, linewidth=0.6, zorder=2)

        for i, (p, v) in enumerate(zip(sub_pct, sub_raw)):
            label_x = p + 3
            ax_si.text(label_x, i, f"{v:.1f}", va="center", ha="left",
                       fontsize=8.5, color="#e8eaf0", fontweight="500")
            pct_str = f"{p:.0f}%"
            ax_si.text(max(p - 4, 2), i, pct_str, va="center", ha="right",
                       fontsize=7.5, color="#080c14", fontweight="700")

        ax_si.set_yticks(y_pos)
        ax_si.set_yticklabels(sub_names, fontsize=9.5)
        ax_si.set_xlabel("% of Safe Limit", labelpad=8, fontsize=9)
        ax_si.set_title("Pollutant Load vs Safe Threshold", fontsize=11,
                        fontweight="600", color="#e8eaf0", pad=12)
        ax_si.set_xlim(0, max(max(sub_pct) * 1.18, 130))
        ax_si.grid(axis="x", alpha=0.3)
        ax_si.set_axisbelow(True)

        # Legend patches
        patches = [
            mpatches.Patch(color="#00c48c", label="≤ 80 % of limit"),
            mpatches.Patch(color="#f9ca24", label="80–150 %"),
            mpatches.Patch(color="#eb4d4b", label="> 150 % (exceeded)"),
        ]
        ax_si.legend(handles=patches, fontsize=7.5, loc="lower right",
                     framealpha=0.2, edgecolor=GRID_COLOR, labelcolor="#e8eaf0")
        plt.tight_layout()

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.pyplot(fig_si, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-label">Pollutant Radar</div>', unsafe_allow_html=True)

        radar_labels = ["PM2.5", "PM10", "NO", "NO2", "SO₂", "O₃"]
        radar_vals   = [pm25, pm10, no, no2, so2, o3]
        safe_limits  = [60, 100, 80, 80, 80, 100]
        radar_norm   = [min(v/s, 2.0) for v, s in zip(radar_vals, safe_limits)]

        N      = len(radar_labels)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        vals_plot = radar_norm + radar_norm[:1]

        fig_r, ax_r = plt.subplots(figsize=(5, 4), subplot_kw={"projection": "polar"})
        fig_r.patch.set_facecolor(DARK_BG)
        ax_r.set_facecolor(PANEL_BG)

        for ring in [0.5, 1.0, 1.5, 2.0]:
            ax_r.plot(angles, [ring] * len(angles),
                      color=GRID_COLOR, lw=0.8, linestyle="--")

        # Safe-limit ring highlighted
        ax_r.plot(angles, [1.0] * len(angles),
                  color=cat_color, lw=1.5, linestyle="-", alpha=0.4)

        ax_r.fill(angles, vals_plot, color=cat_color, alpha=0.25)
        ax_r.plot(angles, vals_plot, color=cat_color, lw=2.0)
        ax_r.scatter(angles[:-1], radar_norm, color=cat_color, s=60,
                     zorder=5, edgecolors=DARK_BG, linewidths=1.5)

        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(radar_labels, fontsize=9, color="#e8eaf0")
        ax_r.set_yticks([0.5, 1.0, 1.5, 2.0])
        ax_r.set_yticklabels(["0.5×", "1× limit", "1.5×", "2×"],
                              fontsize=6.5, color=TEXT_COLOR)
        ax_r.set_ylim(0, 2.1)
        ax_r.spines["polar"].set_color(GRID_COLOR)
        ax_r.set_title("Pollution vs Safe Limits", fontsize=11,
                        fontweight="600", color="#e8eaf0", pad=20)
        plt.tight_layout()

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.pyplot(fig_r, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── AQI SCALE LEGEND ───────────────────────────────────────
    st.markdown('<div class="section-label">AQI Reference Scale</div>', unsafe_allow_html=True)

    scale_data = [
        ("0 – 50",    "Good",         "#00c48c", "Air quality is satisfactory. No health risk."),
        ("51 – 100",  "Satisfactory", "#a8e063", "Acceptable quality. Minor risk for sensitive groups."),
        ("101 – 200", "Moderate",     "#f9ca24", "Sensitive individuals may experience effects."),
        ("201 – 300", "Poor",         "#f0932b", "Everyone may begin to experience health effects."),
        ("301 – 400", "Very Poor",    "#eb4d4b", "Health alert: serious effects for everyone."),
        ("401 +",     "Severe",       "#6c5ce7", "Emergency conditions. Entire population affected."),
    ]
    scale_cols = st.columns(6)
    for sc, (rng, lbl, color, desc) in zip(scale_cols, scale_data):
        active_border = "3px" if lbl == category else "3px"
        active_glow   = f"box-shadow:0 0 12px {color}55;" if lbl == category else ""
        sc.markdown(f"""
        <div style="background:#0d1b2a; border:1px solid {color}44;
                    border-top:{active_border} solid {color}; border-radius:10px;
                    padding:0.9rem 0.8rem; text-align:center; {active_glow}">
            <div style="color:{color}; font-weight:700; font-size:0.85rem;">{lbl}</div>
            <div style="color:#4a7fa5; font-size:0.72rem; margin:3px 0;">{rng}</div>
            <div style="color:#8fa8c8; font-size:0.68rem; line-height:1.4;">{desc}</div>
            {"<div style='margin-top:0.4rem; font-size:0.65rem; color:" + color + "; font-weight:700;'>◀ CURRENT</div>" if lbl == category else ""}
        </div>""", unsafe_allow_html=True)

    # ── DOWNLOAD REPORT ────────────────────────────────────────
    st.markdown("")
    st.markdown('<div class="section-label">Export Data</div>', unsafe_allow_html=True)

    report = pd.DataFrame({
        "Parameter": list(pollutants_display.keys()) + ["AQI", "Category", "Dominant Pollutant"],
        "Value":     list(pollutants_display.values()) + [round(prediction, 2), category, dominant_pollutant],
    })
    csv = report.to_csv(index=False).encode("utf-8")

    _, dl_col, _ = st.columns([2, 1, 2])
    with dl_col:
        st.download_button(
            label="⬇  Download AQI Report (.csv)",
            data=csv,
            file_name="aqi_report.csv",
            mime="text/csv",
        )

else:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#4a7fa5;">
        <div style="font-size:3rem; margin-bottom:1rem;">🌫️</div>
        <div style="font-size:1.1rem; font-weight:500; color:#8fa8c8;">
            Enter pollutant values above and click
            <strong style="color:#00c8ff;">Run Prediction</strong>
        </div>
        <div style="font-size:0.85rem; margin-top:0.5rem;">
            Charts, gauge, health advisory, and insights will appear here
        </div>
    </div>""", unsafe_allow_html=True)