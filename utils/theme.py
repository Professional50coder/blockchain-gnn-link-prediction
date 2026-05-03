"""
utils/theme.py — iOS 16-inspired design system.
Light: Premium bright SaaS (Apple-grade). Dark: Deep-space analytics platform.
"""
import streamlit as st

# ── iOS 16 System Palette ─────────────────────────────────────────────────

DARK = {
    # Backgrounds — layered depth
    "bg":           "#07080F",   # True deep space
    "bg2":          "#0D0E1A",   # Slightly elevated
    "sidebar":      "rgba(12,13,22,0.96)",
    "card":         "#111220",   # Card surface
    "card2":        "#171828",   # Elevated card
    "glass":        "rgba(17,18,32,0.75)",

    # iOS system colors (dark variant)
    "primary":      "#0A84FF",   # iOS Blue Dark
    "primary2":     "#409CFF",
    "accent":       "#5E5CE6",   # iOS Indigo Dark
    "teal":         "#5AC8FA",   # iOS Teal Dark
    "success":      "#30D158",   # iOS Green Dark
    "danger":       "#FF453A",   # iOS Red Dark
    "warning":      "#FF9F0A",   # iOS Orange Dark
    "pink":         "#FF375F",   # iOS Pink Dark
    "purple":       "#BF5AF2",   # iOS Purple Dark
    "yellow":       "#FFD60A",   # iOS Yellow Dark

    # Typography
    "text":         "#F5F5F7",
    "text_muted":   "#8E8E93",
    "text_subtle":  "#3A3A3C",

    # Structural
    "border":       "rgba(255,255,255,0.08)",
    "border2":      "rgba(255,255,255,0.13)",
    "separator":    "rgba(84,84,88,0.36)",

    # Semantic
    "info":         "#64D2FF",

    # Plotly template
    "plotly":       "plotly_dark",

    # Gradient stops
    "grad_start":   "#0A84FF",
    "grad_mid":     "#5E5CE6",
    "grad_end":     "#BF5AF2",
}

LIGHT = {
    # Backgrounds
    "bg":           "#F0F4FF",   # Subtle cool white (iOS tinted)
    "bg2":          "#E8EEFF",
    "sidebar":      "rgba(255,255,255,0.92)",
    "card":         "#FFFFFF",
    "card2":        "#F7F8FF",
    "glass":        "rgba(255,255,255,0.78)",

    # iOS system colors (light variant)
    "primary":      "#007AFF",   # iOS Blue
    "primary2":     "#0A84FF",
    "accent":       "#5856D6",   # iOS Indigo
    "teal":         "#32ADE6",   # iOS Teal
    "success":      "#34C759",   # iOS Green
    "danger":       "#FF3B30",   # iOS Red
    "warning":      "#FF9500",   # iOS Orange
    "pink":         "#FF2D55",   # iOS Pink
    "purple":       "#AF52DE",   # iOS Purple
    "yellow":       "#FFCC00",   # iOS Yellow

    # Typography
    "text":         "#1C1C1E",   # iOS Label
    "text_muted":   "#636366",   # iOS Secondary Label
    "text_subtle":  "#AEAEB2",   # iOS Tertiary Label

    # Structural
    "border":       "rgba(60,60,67,0.12)",
    "border2":      "rgba(60,60,67,0.20)",
    "separator":    "rgba(60,60,67,0.18)",

    # Semantic
    "info":         "#32ADE6",

    # Plotly template
    "plotly":       "plotly_white",

    # Gradient stops
    "grad_start":   "#007AFF",
    "grad_mid":     "#5856D6",
    "grad_end":     "#AF52DE",
}


def get_palette() -> dict:
    return DARK if st.session_state.get("dark_mode", True) else LIGHT


def inject_css(dark: bool) -> None:
    p = DARK if dark else LIGHT

    # ── Base app overrides ───────────────────────────────────────────────
    if dark:
        base = f"""
    .stApp{{
        background:linear-gradient(160deg,{p['bg']} 0%,{p['bg2']} 100%)!important;
        min-height:100vh;
    }}
    section[data-testid="stSidebar"]{{
        background:{p['sidebar']}!important;
        backdrop-filter:saturate(180%) blur(24px);
        -webkit-backdrop-filter:saturate(180%) blur(24px);
        border-right:1px solid {p['border2']}!important;
    }}
    .stApp p,.stApp li,.stApp label{{color:{p['text_muted']};}}
    .stApp h1,.stApp h2,.stApp h3,.stApp h4{{color:{p['text']};}}
    [data-testid="stMetricValue"]{{color:{p['text']}!important;font-weight:700!important;}}
    [data-testid="stMetricLabel"]{{color:{p['text_muted']}!important;font-size:0.78rem!important;}}
    [data-testid="stMetricDelta"]{{font-size:0.75rem!important;}}
    .stApp .stTextInput input{{
        background:rgba(255,255,255,0.06)!important;color:{p['text']}!important;
        border:1px solid {p['border2']}!important;border-radius:10px!important;
        transition:border-color 0.15s;
    }}
    .stApp .stTextInput input:focus{{border-color:{p['primary']}!important;box-shadow:0 0 0 3px rgba(10,132,255,0.15)!important;}}
    .stApp .stSelectbox [data-baseweb="select"]>div{{background:rgba(255,255,255,0.06)!important;border:1px solid {p['border2']}!important;border-radius:10px!important;}}
    .stApp .stNumberInput input{{background:rgba(255,255,255,0.06)!important;color:{p['text']}!important;border:1px solid {p['border2']}!important;border-radius:10px!important;}}
    .stApp .stSlider [data-testid="stSliderThumb"]{{background:{p['primary']}!important;}}
    .stApp .stRadio label{{color:{p['text_muted']}!important;}}
    .stApp .stRadio [data-baseweb="radio"] div{{border-color:{p['primary']}!important;}}
    [data-testid="stSidebar"] .stRadio label{{color:{p['text']}!important;font-size:0.84rem!important;font-weight:500!important;}}
    .stApp [data-testid="stExpander"]{{border:1px solid {p['border']}!important;border-radius:12px!important;background:{p['card']}!important;}}
    .stApp [data-testid="stExpander"] summary{{color:{p['text_muted']}!important;}}
    .stTabs [data-baseweb="tab-list"]{{background:rgba(255,255,255,0.04)!important;border-radius:12px!important;padding:3px!important;gap:2px!important;}}
    .stTabs [data-baseweb="tab"]{{border-radius:10px!important;color:{p['text_muted']}!important;font-weight:600!important;}}
    .stTabs [aria-selected="true"]{{background:{p['primary']}!important;color:#fff!important;}}
    .stDownloadButton>button,.stButton>button{{
        border-radius:10px!important;font-weight:600!important;font-size:0.84rem!important;
        transition:all 0.18s cubic-bezier(0.34,1.56,0.64,1)!important;
    }}
    .stButton>button[kind="primary"]{{
        background:linear-gradient(135deg,{p['primary']},{p['accent']})!important;
        border:none!important;color:white!important;
        box-shadow:0 4px 16px rgba(10,132,255,0.30)!important;
    }}
    .stButton>button[kind="primary"]:hover{{transform:translateY(-1px);box-shadow:0 6px 20px rgba(10,132,255,0.40)!important;}}
    .stDataFrame{{background:{p['card']}!important;border-radius:12px!important;overflow:hidden;}}
    .stDataFrame th{{background:rgba(255,255,255,0.06)!important;color:{p['text']}!important;font-weight:700!important;}}
    .stDataFrame td{{color:{p['text_muted']}!important;}}
"""
    else:
        base = f"""
    .stApp{{
        background:linear-gradient(160deg,#EBF3FF 0%,#F2F0FF 45%,#FDF0FF 100%)!important;
        min-height:100vh;
    }}
    section[data-testid="stSidebar"]{{
        background:{p['sidebar']}!important;
        backdrop-filter:saturate(180%) blur(24px);
        -webkit-backdrop-filter:saturate(180%) blur(24px);
        border-right:1px solid {p['border']}!important;
        box-shadow:4px 0 24px rgba(0,0,0,0.06)!important;
    }}
    .stApp p,.stApp li{{color:{p['text_muted']};}}
    .stApp h1,.stApp h2,.stApp h3,.stApp h4{{color:{p['text']};}}
    [data-testid="stMetricValue"]{{color:{p['text']}!important;font-weight:700!important;}}
    [data-testid="stMetricLabel"]{{color:{p['text_muted']}!important;font-size:0.78rem!important;}}
    .stApp .stTextInput input{{
        background:#fff!important;color:{p['text']}!important;
        border:1.5px solid {p['border2']}!important;border-radius:10px!important;
        box-shadow:0 1px 4px rgba(0,0,0,0.04)!important;transition:border-color 0.15s;
    }}
    .stApp .stTextInput input:focus{{border-color:{p['primary']}!important;box-shadow:0 0 0 4px rgba(0,122,255,0.12)!important;}}
    .stApp .stSelectbox [data-baseweb="select"]>div{{background:#fff!important;border:1.5px solid {p['border2']}!important;border-radius:10px!important;}}
    .stApp .stNumberInput input{{background:#fff!important;border:1.5px solid {p['border2']}!important;border-radius:10px!important;}}
    .stApp .stRadio label{{color:{p['text']}!important;}}
    [data-testid="stSidebar"] .stRadio label{{font-size:0.84rem!important;font-weight:500!important;}}
    .stApp [data-testid="stExpander"]{{border:1.5px solid {p['border']}!important;border-radius:12px!important;background:{p['card']}!important;box-shadow:0 2px 8px rgba(0,0,0,0.04)!important;}}
    .stTabs [data-baseweb="tab-list"]{{background:rgba(0,0,0,0.04)!important;border-radius:12px!important;padding:3px!important;gap:2px!important;}}
    .stTabs [data-baseweb="tab"]{{border-radius:10px!important;color:{p['text_muted']}!important;font-weight:600!important;}}
    .stTabs [aria-selected="true"]{{background:{p['primary']}!important;color:#fff!important;}}
    .stDownloadButton>button,.stButton>button{{border-radius:10px!important;font-weight:600!important;font-size:0.84rem!important;transition:all 0.18s cubic-bezier(0.34,1.56,0.64,1)!important;}}
    .stButton>button[kind="primary"]{{
        background:linear-gradient(135deg,{p['primary']},{p['accent']})!important;
        border:none!important;color:white!important;
        box-shadow:0 4px 16px rgba(0,122,255,0.28)!important;
    }}
    .stButton>button[kind="primary"]:hover{{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(0,122,255,0.36)!important;}}
    .stDataFrame{{background:{p['card']}!important;border-radius:12px!important;overflow:hidden;border:1px solid {p['border']}!important;box-shadow:0 2px 8px rgba(0,0,0,0.04)!important;}}
    .stDataFrame th{{background:linear-gradient(90deg,rgba(0,122,255,0.06),rgba(88,86,214,0.04))!important;color:{p['text']}!important;font-weight:700!important;}}
    .stDataFrame td{{color:{p['text']}!important;}}
    section[data-testid="stSidebar"] *{{color:{p['text']}!important;}}
"""

    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;0,14..32,800;0,14..32,900;1,14..32,400&display=swap');

html,body,[class*="css"]{{font-family:'Inter',system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important;-webkit-font-smoothing:antialiased;}}
#MainMenu,footer,[data-testid="stToolbar"]{{visibility:hidden;}}

{base}

/* ══════════════════════════════════════════
   PRODUCT BRANDING
══════════════════════════════════════════ */
.product-badge{{
    display:inline-block;
    background:linear-gradient(135deg,{p['primary']},{p['accent']});
    color:#fff;border-radius:999px;padding:3px 12px;
    font-size:0.62rem;font-weight:800;letter-spacing:1.5px;
    text-transform:uppercase;vertical-align:middle;
    box-shadow:{'0 2px 12px rgba(10,132,255,0.35)' if dark else '0 2px 12px rgba(0,122,255,0.30)'};
}}

/* ══════════════════════════════════════════
   PAGE HEADERS
══════════════════════════════════════════ */
.main-header{{
    font-family:'Inter',sans-serif;font-size:2.1rem;font-weight:900;letter-spacing:-1px;
    background:linear-gradient(135deg,{p['grad_start']} 0%,{p['grad_mid']} 50%,{p['grad_end']} 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    text-align:center;margin-bottom:0.3rem;line-height:1.15;}}
.sub-header{{
    text-align:center;color:{p['text_muted']};font-size:0.88rem;font-weight:400;
    margin-bottom:1.8rem;letter-spacing:0.1px;}}
.section-label{{
    font-size:0.68rem;font-weight:800;letter-spacing:2px;text-transform:uppercase;
    color:{p['text_subtle']};margin-bottom:0.6rem;}}
.page-eyebrow{{
    font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
    color:{p['primary']};margin-bottom:0.3rem;}}

/* ══════════════════════════════════════════
   HERO SECTION (B2B overview)
══════════════════════════════════════════ */
.hero-wrap{{
    background:{'linear-gradient(145deg,rgba(10,132,255,0.08),rgba(94,92,230,0.06),rgba(191,90,242,0.05))' if dark else 'linear-gradient(145deg,rgba(0,122,255,0.07),rgba(88,86,214,0.05),rgba(175,82,222,0.04))'};
    border:1px solid {'rgba(10,132,255,0.18)' if dark else 'rgba(0,122,255,0.14)'};
    border-radius:24px;padding:2.2rem 2.4rem;margin-bottom:1.2rem;
    backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
    position:relative;overflow:hidden;}}
.hero-title{{
    font-size:2.4rem;font-weight:900;letter-spacing:-1.2px;line-height:1.1;
    background:linear-gradient(135deg,{p['grad_start']},{p['grad_mid']},{p['grad_end']});
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    margin-bottom:0.6rem;}}
.hero-sub{{
    font-size:0.95rem;color:{p['text_muted']};font-weight:400;line-height:1.6;max-width:560px;}}
.hero-stat-row{{display:flex;gap:2rem;margin-top:1.4rem;flex-wrap:wrap;}}
.hero-stat{{}}
.hero-stat-num{{
    font-size:1.8rem;font-weight:900;letter-spacing:-1px;color:{p['text']};
    font-variant-numeric:tabular-nums;}}
.hero-stat-label{{font-size:0.72rem;color:{p['text_muted']};font-weight:500;letter-spacing:0.3px;text-transform:uppercase;margin-top:1px;}}

/* ══════════════════════════════════════════
   METRIC / KPI CARDS
══════════════════════════════════════════ */
.metric-card{{
    background:{'rgba(255,255,255,0.05)' if dark else p['card']};
    border-radius:20px;padding:1.1rem 1.3rem;
    border:1px solid {p['border']};
    box-shadow:{'0 2px 20px rgba(0,0,0,0.4),0 0 0 1px rgba(255,255,255,0.04)' if dark else '0 2px 12px rgba(0,0,0,0.06),0 8px 32px rgba(0,122,255,0.05)'};
    backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
    transition:all 0.22s cubic-bezier(0.34,1.56,0.64,1);}}
.metric-card:hover{{
    transform:translateY(-3px);
    border-color:{'rgba(10,132,255,0.35)' if dark else 'rgba(0,122,255,0.25)'};
    box-shadow:{'0 8px 32px rgba(0,0,0,0.5),0 0 0 1px rgba(10,132,255,0.20)' if dark else '0 8px 32px rgba(0,122,255,0.12),0 2px 8px rgba(0,0,0,0.08)'};}}

.fraud-alert{{
    background:{'rgba(255,69,58,0.07)' if dark else '#FFF5F5'};
    padding:1rem 1.2rem;border-radius:16px;
    border:1px solid {'rgba(255,69,58,0.22)' if dark else 'rgba(255,59,48,0.20)'};
    box-shadow:{'0 2px 12px rgba(255,69,58,0.08)' if dark else '0 2px 12px rgba(255,59,48,0.07)'};}}
.success-box{{
    background:{'rgba(48,209,88,0.06)' if dark else '#F0FFF4'};
    padding:1rem 1.2rem;border-radius:16px;
    border:1px solid {'rgba(48,209,88,0.20)' if dark else 'rgba(52,199,89,0.22)'};}}
.warning-box{{
    background:{'rgba(255,159,10,0.07)' if dark else '#FFFBF0'};
    padding:1rem 1.2rem;border-radius:16px;
    border:1px solid {'rgba(255,159,10,0.22)' if dark else 'rgba(255,149,0,0.22)'};}}
.info-box{{
    background:{'rgba(94,92,230,0.06)' if dark else '#F5F0FF'};
    padding:1rem 1.2rem;border-radius:16px;
    border:1px solid {'rgba(94,92,230,0.20)' if dark else 'rgba(88,86,214,0.18)'};}}

/* ══════════════════════════════════════════
   VALUE PROPOSITION (B2B)
══════════════════════════════════════════ */
.value-card{{
    background:{'rgba(255,255,255,0.04)' if dark else p['card']};
    border:1px solid {p['border']};border-radius:20px;
    padding:1.3rem 1.4rem;
    transition:all 0.22s cubic-bezier(0.34,1.56,0.64,1);
    box-shadow:{'0 2px 16px rgba(0,0,0,0.3)' if dark else '0 2px 12px rgba(0,0,0,0.05)'};}}
.value-card:hover{{
    transform:translateY(-4px);
    border-color:{'rgba(10,132,255,0.30)' if dark else 'rgba(0,122,255,0.22)'};
    box-shadow:{'0 12px 40px rgba(0,0,0,0.45),0 0 0 1px rgba(10,132,255,0.15)' if dark else '0 12px 40px rgba(0,122,255,0.10)'};}}
.value-icon{{
    width:44px;height:44px;border-radius:12px;
    display:flex;align-items:center;justify-content:center;
    font-size:1.3rem;margin-bottom:0.7rem;}}
.value-title{{font-size:0.92rem;font-weight:700;color:{p['text']};margin-bottom:0.3rem;}}
.value-desc{{font-size:0.78rem;color:{p['text_muted']};line-height:1.55;}}

/* ══════════════════════════════════════════
   INTEGRATION / TECH BADGES
══════════════════════════════════════════ */
.integration-badge{{
    display:inline-flex;align-items:center;gap:5px;
    background:{'rgba(255,255,255,0.06)' if dark else p['card']};
    border:1px solid {p['border2']};border-radius:10px;
    padding:5px 12px;font-size:0.76rem;font-weight:600;color:{p['text_muted']};
    margin:3px;transition:all 0.15s;}}
.integration-badge:hover{{
    border-color:{p['primary']};color:{p['primary']};
    background:{'rgba(10,132,255,0.08)' if dark else 'rgba(0,122,255,0.06)'};}}

.protocol-tag{{
    display:inline-block;
    background:{'rgba(10,132,255,0.10)' if dark else 'rgba(0,122,255,0.08)'};
    color:{p['primary']};border-radius:999px;
    padding:3px 11px;font-size:0.72rem;font-family:'JetBrains Mono',monospace;font-weight:500;
    margin:2px;border:1px solid {'rgba(10,132,255,0.20)' if dark else 'rgba(0,122,255,0.18)'};}}
.event-tag{{
    display:inline-block;
    background:{'rgba(48,209,88,0.08)' if dark else 'rgba(52,199,89,0.08)'};
    color:{p['success']};border-radius:999px;
    padding:3px 11px;font-size:0.72rem;font-family:'JetBrains Mono',monospace;font-weight:500;
    margin:2px;border:1px solid {'rgba(48,209,88,0.20)' if dark else 'rgba(52,199,89,0.20)'};}}

/* ══════════════════════════════════════════
   ADDRESS / WALLET CHIPS
══════════════════════════════════════════ */
.contract-badge{{
    background:{'rgba(94,92,230,0.10)' if dark else 'rgba(88,86,214,0.08)'};
    border:1px solid {'rgba(94,92,230,0.28)' if dark else 'rgba(88,86,214,0.22)'};
    border-radius:8px;padding:0.3rem 0.9rem;
    color:{'#A0A0FF' if dark else p['accent']};
    font-size:0.76rem;font-weight:600;display:inline-block;margin:2px 0;}}
.eoa-badge{{
    background:{'rgba(48,209,88,0.08)' if dark else 'rgba(52,199,89,0.07)'};
    border:1px solid {'rgba(48,209,88,0.25)' if dark else 'rgba(52,199,89,0.22)'};
    border-radius:8px;padding:0.3rem 0.9rem;
    color:{p['success']};
    font-size:0.76rem;font-weight:600;display:inline-block;margin:2px 0;}}
.wallet-address-full{{
    font-family:'JetBrains Mono',monospace;font-size:0.79rem;
    background:{'rgba(255,255,255,0.04)' if dark else 'rgba(0,122,255,0.04)'};
    border:1px solid {p['border2']};border-radius:10px;
    padding:0.5rem 0.9rem;color:{p['primary']};
    word-break:break-all;letter-spacing:0.2px;display:block;margin:4px 0;}}
.valid-address{{color:{p['success']};font-size:0.79rem;font-weight:600;}}
.invalid-address{{color:{p['danger']};font-size:0.79rem;font-weight:600;}}
.history-pill{{
    display:inline-block;
    background:{'rgba(255,255,255,0.06)' if dark else 'rgba(0,0,0,0.04)'};
    color:{p['text_muted']};border-radius:999px;
    padding:3px 11px;font-size:0.70rem;font-family:'JetBrains Mono',monospace;
    margin:2px;border:1px solid {p['border2']};}}

/* ══════════════════════════════════════════
   RISK BADGES
══════════════════════════════════════════ */
.risk-critical{{
    background:{'rgba(255,69,58,0.14)' if dark else '#FFE5E3'};
    color:{'#FF6B63' if dark else '#C0392B'};
    padding:3px 12px;border-radius:999px;font-size:0.72rem;font-weight:700;
    border:1px solid {'rgba(255,69,58,0.30)' if dark else 'rgba(255,59,48,0.30)'};}}
.risk-high{{
    background:{'rgba(255,159,10,0.12)' if dark else '#FFF3E0'};
    color:{'#FFAA30' if dark else '#D4710A'};
    padding:3px 12px;border-radius:999px;font-size:0.72rem;font-weight:700;
    border:1px solid {'rgba(255,159,10,0.28)' if dark else 'rgba(255,149,0,0.28)'};}}
.risk-medium{{
    background:{'rgba(255,214,10,0.10)' if dark else '#FFFDE7'};
    color:{'#FFD60A' if dark else '#8B6900'};
    padding:3px 12px;border-radius:999px;font-size:0.72rem;font-weight:700;
    border:1px solid {'rgba(255,214,10,0.22)' if dark else 'rgba(255,204,0,0.28)'};}}
.risk-low{{
    background:{'rgba(48,209,88,0.09)' if dark else '#E8F9EC'};
    color:{'#34D163' if dark else '#1A7F37'};
    padding:3px 12px;border-radius:999px;font-size:0.72rem;font-weight:700;
    border:1px solid {'rgba(48,209,88,0.22)' if dark else 'rgba(52,199,89,0.25)'};}}

/* ══════════════════════════════════════════
   FORMULA / CODE BLOCKS
══════════════════════════════════════════ */
.formula-block{{
    font-family:'JetBrains Mono',monospace;font-size:0.81rem;
    background:{'rgba(0,0,0,0.30)' if dark else 'rgba(0,122,255,0.04)'};
    border:1px solid {p['border']};
    border-left:3px solid {p['accent']};
    border-radius:0 10px 10px 0;padding:0.7rem 1.1rem;
    color:{'#9D9BFF' if dark else p['accent']};
    margin:0.4rem 0;line-height:1.65;}}

/* ══════════════════════════════════════════
   LAYER / ARCHITECTURE CARDS
══════════════════════════════════════════ */
.layer-card{{
    background:{'rgba(255,255,255,0.03)' if dark else p['card']};
    border:1px solid {p['border']};border-radius:16px;
    padding:1rem 1.2rem;margin:0.35rem 0;
    box-shadow:{'0 2px 12px rgba(0,0,0,0.25)' if dark else '0 2px 8px rgba(0,0,0,0.04)'};
    transition:border-color 0.15s;}}
.layer-card:hover{{border-color:{'rgba(10,132,255,0.30)' if dark else 'rgba(0,122,255,0.22)'};}}
.layer-title{{font-weight:700;font-size:0.88rem;margin-bottom:0.2rem;}}
.layer-detail{{font-size:0.77rem;color:{p['text_muted']};line-height:1.55;}}

/* ══════════════════════════════════════════
   STATUS INDICATORS
══════════════════════════════════════════ */
.status-dot-green{{
    display:inline-block;width:8px;height:8px;border-radius:999px;
    background:{p['success']};
    box-shadow:0 0 6px {'rgba(48,209,88,0.6)' if dark else 'rgba(52,199,89,0.5)'};
    animation:pulse-green 2s infinite;}}
.status-dot-blue{{
    display:inline-block;width:8px;height:8px;border-radius:999px;
    background:{p['primary']};
    box-shadow:0 0 6px {'rgba(10,132,255,0.6)' if dark else 'rgba(0,122,255,0.5)'};
    animation:pulse-blue 2s infinite;}}
.status-dot-red{{
    display:inline-block;width:8px;height:8px;border-radius:999px;
    background:{p['danger']};opacity:0.8;}}
@keyframes pulse-green{{0%,100%{{box-shadow:0 0 6px {'rgba(48,209,88,0.6)' if dark else 'rgba(52,199,89,0.5)'};}}50%{{box-shadow:0 0 12px {'rgba(48,209,88,0.9)' if dark else 'rgba(52,199,89,0.7)'};}}}}
@keyframes pulse-blue{{0%,100%{{box-shadow:0 0 6px {'rgba(10,132,255,0.6)' if dark else 'rgba(0,122,255,0.5)'};}}50%{{box-shadow:0 0 12px {'rgba(10,132,255,0.9)' if dark else 'rgba(0,122,255,0.7)'};}}}}

/* ══════════════════════════════════════════
   SIDEBAR PRODUCT LOGO
══════════════════════════════════════════ */
.sidebar-logo{{
    text-align:center;padding:1.2rem 0 0.8rem;
}}
.sidebar-logo-icon{{
    width:56px;height:56px;border-radius:16px;margin:0 auto 0.6rem;
    background:linear-gradient(135deg,{p['primary']},{p['accent']});
    display:flex;align-items:center;justify-content:center;
    font-size:1.8rem;
    box-shadow:{'0 8px 24px rgba(10,132,255,0.35)' if dark else '0 8px 24px rgba(0,122,255,0.30)'};
}}
.sidebar-product-name{{
    font-size:0.98rem;font-weight:800;letter-spacing:-0.3px;
    background:linear-gradient(135deg,{p['primary']},{p['accent']});
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.sidebar-tagline{{
    font-size:0.62rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
    color:{p['text_subtle']};margin-top:2px;
}}

/* ══════════════════════════════════════════
   CONNECTION STATUS PILLS
══════════════════════════════════════════ */
.conn-pill{{
    display:flex;align-items:center;gap:6px;
    background:{'rgba(255,255,255,0.05)' if dark else 'rgba(0,0,0,0.04)'};
    border:1px solid {p['border']};border-radius:999px;
    padding:5px 12px;font-size:0.73rem;font-weight:600;color:{p['text_muted']};
    margin:3px 0;}}
.conn-pill.connected{{
    border-color:{'rgba(48,209,88,0.30)' if dark else 'rgba(52,199,89,0.28)'};
    color:{p['success']};
    background:{'rgba(48,209,88,0.06)' if dark else 'rgba(52,199,89,0.06)'};}}
.conn-pill.disconnected{{
    border-color:{'rgba(255,69,58,0.25)' if dark else 'rgba(255,59,48,0.22)'};
    color:{p['danger']};
    background:{'rgba(255,69,58,0.05)' if dark else 'rgba(255,59,48,0.05)'};}}

/* ══════════════════════════════════════════
   MISC
══════════════════════════════════════════ */
.footer-text{{
    text-align:center;color:{p['text_subtle']};font-size:0.74rem;
    padding:2rem 0 1rem;letter-spacing:0.3px;}}
.footer-text strong{{color:{p['text_muted']};}}
.stDataFrame td{{white-space:pre-wrap!important;word-break:break-all!important;}}
.stDataFrame th{{font-family:'Inter',sans-serif!important;font-weight:700!important;letter-spacing:0.2px!important;}}
[data-testid="metric-container"] [data-testid="stMetricValue"]{{font-family:'Inter',sans-serif;font-weight:800;font-size:1.7rem!important;letter-spacing:-0.5px;}}
.stTabs [data-baseweb="tab"]{{font-family:'Inter',sans-serif;font-weight:600;font-size:0.83rem;padding:6px 14px!important;}}
[data-testid="stSidebar"] hr{{border-color:{p['separator']}!important;margin:0.6rem 0!important;}}
.stApp hr{{border-color:{p['separator']}!important;opacity:0.6;}}
.stApp [data-testid="stMarkdownContainer"] a{{color:{p['primary']}!important;text-decoration:none!important;}}
.stApp [data-testid="stMarkdownContainer"] a:hover{{text-decoration:underline!important;}}
</style>"""

    st.markdown(css, unsafe_allow_html=True)
