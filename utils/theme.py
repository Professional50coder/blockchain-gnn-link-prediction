"""
utils/theme.py — Colour palette, theme helpers, and CSS injection.
"""
import streamlit as st

# ── Palette constants ──────────────────────────────────────────────────
DARK = {
    "bg":           "#0d1117",
    "sidebar":      "#090c10",
    "card":         "rgba(22,27,34,0.9)",
    "border":       "#21262d",
    "border2":      "#30363d",
    "primary":      "#58a6ff",
    "primary2":     "#79c0ff",
    "accent":       "#a371f7",
    "success":      "#3fb950",
    "danger":       "#f85149",
    "warning":      "#d29922",
    "info":         "#79c0ff",
    "text":         "#e6edf3",
    "text_muted":   "#8b949e",
    "text_subtle":  "#484f58",
    "plotly":       "plotly_dark",
}

LIGHT = {
    "bg":           "#f6f8fa",
    "sidebar":      "#ffffff",
    "card":         "#ffffff",
    "border":       "#d0d7de",
    "border2":      "#afb8c1",
    "primary":      "#0969da",
    "primary2":     "#1158c7",
    "accent":       "#8250df",
    "success":      "#1a7f37",
    "danger":       "#cf222e",
    "warning":      "#9a6700",
    "info":         "#0550ae",
    "text":         "#1f2328",
    "text_muted":   "#636c76",
    "text_subtle":  "#8c959f",
    "plotly":       "plotly_white",
}


def get_palette() -> dict:
    return DARK if st.session_state.get("dark_mode", True) else LIGHT


def inject_css(dark: bool) -> None:
    p = DARK if dark else LIGHT

    if dark:
        extra = f"""
    .stApp{{background-color:{p['bg']}!important;}}
    section[data-testid="stSidebar"]{{background-color:{p['sidebar']}!important;border-right:1px solid {p['border']}!important;}}
    [data-testid="stMetricValue"]{{color:{p['text']}!important;}}
    [data-testid="stMetricLabel"]{{color:{p['text_muted']}!important;}}
    .stApp .stTextInput input{{background:{p['card']}!important;color:{p['text']}!important;border-color:{p['border2']}!important;}}
    .stApp .stSelectbox div[data-baseweb]{{background:{p['card']}!important;}}
"""
    else:
        extra = f"""
    .stApp{{background-color:{p['bg']}!important;}}
    section[data-testid="stSidebar"]{{background-color:{p['sidebar']}!important;border-right:1px solid {p['border']}!important;}}
    section[data-testid="stSidebar"] *{{color:{p['text']}!important;}}
    .stApp p,.stApp li{{color:{p['text']};}}
    .stApp h1,.stApp h2,.stApp h3,.stApp h4{{color:{p['text']};}}
    [data-testid="stMetricValue"]{{color:{p['text']}!important;}}
    [data-testid="stMetricLabel"]{{color:{p['text_muted']}!important;}}
    .stApp .stTextInput input{{background:#ffffff!important;color:{p['text']}!important;border-color:{p['border']}!important;}}
"""

    css = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700;800&display=swap');
    html,body,[class*="css"]{{font-family:'Inter',sans-serif;}}
    #MainMenu,footer{{visibility:hidden;}}
    {extra}

    /* ── Headers ─────────────────────────────── */
    .main-header{{
        font-family:'Inter',sans-serif;font-size:2rem;font-weight:800;
        background:linear-gradient(135deg,{p['primary']} 0%,{p['primary2']} 40%,{p['accent']} 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;text-align:center;margin-bottom:0.35rem;letter-spacing:-0.6px;}}
    .sub-header{{
        text-align:center;color:{p['text_muted']};font-size:0.86rem;font-weight:400;
        margin-bottom:1.6rem;letter-spacing:0.15px;}}
    .section-label{{
        font-size:0.7rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
        color:{p['text_subtle']};margin-bottom:0.5rem;}}

    /* ── Cards / boxes ─────────────────────────── */
    .metric-card{{
        background:{'linear-gradient(145deg,rgba(88,166,255,0.07),rgba(163,113,247,0.04))' if dark else p['card']};
        padding:1rem 1.2rem;border-radius:10px;
        border:1px solid {'rgba(88,166,255,0.16)' if dark else p['border']};
        box-shadow:{'0 2px 10px rgba(0,0,0,0.35)' if dark else '0 1px 3px rgba(31,35,40,0.07)'};
        transition:transform 0.15s,box-shadow 0.15s;}}
    .metric-card:hover{{
        transform:translateY(-2px);
        box-shadow:{'0 6px 20px rgba(88,166,255,0.1),0 2px 6px rgba(0,0,0,0.4)' if dark else '0 4px 12px rgba(9,105,218,0.1)'};}}

    .fraud-alert{{background:{'rgba(248,81,73,0.07)' if dark else '#fff8f8'};padding:1rem 1.2rem;border-radius:10px;border:1px solid {'rgba(248,81,73,0.22)' if dark else '#ffb8b8'};}}
    .success-box{{background:{'rgba(63,185,80,0.07)' if dark else '#f0fff4'};padding:1rem 1.2rem;border-radius:10px;border:1px solid {'rgba(63,185,80,0.22)' if dark else '#aceebb'};}}
    .warning-box{{background:{'rgba(210,153,34,0.08)' if dark else '#fffbf0'};padding:1rem 1.2rem;border-radius:10px;border:1px solid {'rgba(210,153,34,0.24)' if dark else '#f0c070'};}}
    .info-box{{background:{'rgba(163,113,247,0.07)' if dark else '#f5f0ff'};padding:1rem 1.2rem;border-radius:10px;border:1px solid {'rgba(163,113,247,0.22)' if dark else '#cbb7f7'};}}

    /* ── Badges / pills ─────────────────────────── */
    .contract-badge{{background:{'rgba(163,113,247,0.12)' if dark else '#f5f0ff'};border:1px solid {'rgba(163,113,247,0.3)' if dark else '#cbb7f7'};border-radius:6px;padding:0.3rem 0.8rem;color:{'#c9a9ff' if dark else '#6e40c9'};font-size:0.77rem;font-weight:600;display:inline-block;margin:2px 0;}}
    .eoa-badge{{background:{'rgba(63,185,80,0.1)' if dark else '#f0fff4'};border:1px solid {'rgba(63,185,80,0.28)' if dark else '#aceebb'};border-radius:6px;padding:0.3rem 0.8rem;color:{'#56d364' if dark else '#1a7f37'};font-size:0.77rem;font-weight:600;display:inline-block;margin:2px 0;}}
    .protocol-tag{{background:{'rgba(88,166,255,0.09)' if dark else '#ddf4ff'};color:{'#79c0ff' if dark else '#0550ae'};border-radius:999px;padding:2px 10px;font-size:0.73rem;font-family:'JetBrains Mono',monospace;margin:2px;border:1px solid {'rgba(88,166,255,0.18)' if dark else '#aecbf5'};display:inline-block;}}
    .event-tag{{background:{'rgba(63,185,80,0.09)' if dark else '#dafbe1'};color:{'#3fb950' if dark else '#1a7f37'};border-radius:999px;padding:2px 10px;font-size:0.73rem;font-family:'JetBrains Mono',monospace;margin:2px;border:1px solid {'rgba(63,185,80,0.2)' if dark else '#aceebb'};display:inline-block;}}

    /* ── Risk pills ─────────────────────────── */
    .risk-critical{{background:{'rgba(248,81,73,0.15)' if dark else '#ffe8e7'};color:{'#ffa198' if dark else '#a0111f'};padding:3px 11px;border-radius:999px;font-size:0.73rem;font-weight:700;border:1px solid {'rgba(248,81,73,0.35)' if dark else '#ffb8b8'};}}
    .risk-high{{background:{'rgba(248,81,73,0.1)' if dark else '#fff0ee'};color:{'#ff7b72' if dark else '#cf222e'};padding:3px 11px;border-radius:999px;font-size:0.73rem;font-weight:700;border:1px solid {'rgba(248,81,73,0.25)' if dark else '#ffc9c5'};}}
    .risk-medium{{background:{'rgba(210,153,34,0.12)' if dark else '#fff8c5'};color:{'#e3b341' if dark else '#7d4e00'};padding:3px 11px;border-radius:999px;font-size:0.73rem;font-weight:700;border:1px solid {'rgba(210,153,34,0.3)' if dark else '#f0c070'};}}
    .risk-low{{background:{'rgba(63,185,80,0.1)' if dark else '#dafbe1'};color:{'#56d364' if dark else '#116329'};padding:3px 11px;border-radius:999px;font-size:0.73rem;font-weight:700;border:1px solid {'rgba(63,185,80,0.25)' if dark else '#aceebb'};}}

    /* ── Wallet address chip ─────────────────────────── */
    .wallet-address-full{{
        font-family:'JetBrains Mono',monospace;font-size:0.79rem;
        background:{'rgba(33,38,45,0.9)' if dark else '#f6f8fa'};
        border:1px solid {p['border2']};border-radius:8px;
        padding:0.45rem 0.8rem;color:{'#79c0ff' if dark else '#0550ae'};
        word-break:break-all;letter-spacing:0.15px;display:block;margin:4px 0;}}
    .valid-address{{color:{p['success']};font-size:0.79rem;font-weight:600;}}
    .invalid-address{{color:{p['danger']};font-size:0.79rem;font-weight:600;}}
    .history-pill{{display:inline-block;background:{'rgba(33,38,45,0.9)' if dark else '#f6f8fa'};color:{p['text_muted']};border-radius:999px;padding:2px 10px;font-size:0.71rem;font-family:'JetBrains Mono',monospace;margin:2px;border:1px solid {p['border2']};}}

    /* ── Formula blocks ─────────────────────────── */
    .formula-block{{
        font-family:'JetBrains Mono',monospace;font-size:0.82rem;
        background:{'rgba(22,27,34,0.95)' if dark else '#f6f8fa'};
        border:1px solid {p['border']};border-left:3px solid {p['accent']};
        border-radius:0 8px 8px 0;padding:0.7rem 1rem;
        color:{'#c9a9ff' if dark else '#8250df'};margin:0.4rem 0;
        line-height:1.6;}}

    /* ── Layer info card (architecture section) ─────────────────────────── */
    .layer-card{{
        background:{'rgba(22,27,34,0.7)' if dark else '#ffffff'};
        border:1px solid {p['border']};border-radius:10px;
        padding:0.9rem 1.1rem;margin:0.3rem 0;}}
    .layer-title{{font-weight:700;font-size:0.9rem;color:{p['primary']};margin-bottom:0.2rem;}}
    .layer-detail{{font-size:0.78rem;color:{p['text_muted']};line-height:1.5;}}

    /* ── Misc ─────────────────────────── */
    .footer-text{{text-align:center;color:{p['text_subtle']};font-size:0.77rem;padding:2rem 0 1rem;letter-spacing:0.3px;}}
    .footer-text strong{{color:{p['text_muted']};}}
    .stDataFrame td{{white-space:pre-wrap!important;word-break:break-all!important;}}
    .stDataFrame th{{font-family:'Inter',sans-serif!important;font-weight:600!important;}}
    [data-testid="metric-container"] [data-testid="stMetricValue"]{{font-family:'Inter',sans-serif;font-weight:700;}}
    .stTabs [data-baseweb="tab"]{{font-family:'Inter',sans-serif;font-weight:600;font-size:0.84rem;}}
    [data-testid="stSidebar"] hr{{border-color:{p['border']}!important;}}
    .stApp hr{{border-color:{p['border']}!important;opacity:0.5;}}
</style>"""
    st.markdown(css, unsafe_allow_html=True)
