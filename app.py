import os
import re
import pickle
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from sklearn.metrics import roc_curve, auc

# -----------------------------------------------
# Logging setup
# -----------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------
# Page configuration
# -----------------------------------------------
st.set_page_config(
    page_title="Blockchain GNN Analytics",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------
# Session-state initialisation
# -----------------------------------------------
for key, default in [
    ("search_history", []),
    ("dark_mode", True),
    ("live_lookup_addr", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -----------------------------------------------
# Theme variables
# -----------------------------------------------
def get_theme():
    if st.session_state.dark_mode:
        return {
            "plotly_template": "plotly_dark",
            "primary":    "#58a6ff",
            "danger":     "#f85149",
            "success":    "#3fb950",
            "accent":     "#a371f7",
            "warning":    "#d29922",
            "bg_main":    "rgba(10,12,16,0.0)",
            "bg_plot":    "rgba(13,17,23,0.0)",
            "text_muted": "#8b949e",
        }
    return {
        "plotly_template": "plotly_white",
        "primary":    "#0969da",
        "danger":     "#cf222e",
        "success":    "#1a7f37",
        "accent":     "#8250df",
        "warning":    "#9a6700",
        "bg_main":    "rgba(255,255,255,0.0)",
        "bg_plot":    "rgba(246,248,250,0.0)",
        "text_muted": "#636c76",
    }

THEME           = get_theme()
PLOTLY_TEMPLATE = THEME["plotly_template"]
PRIMARY_COLOR   = THEME["primary"]
DANGER_COLOR    = THEME["danger"]
SUCCESS_COLOR   = THEME["success"]
ACCENT_COLOR    = THEME["accent"]

# -----------------------------------------------
# Custom CSS
# -----------------------------------------------
def inject_css(dark: bool):
    if dark:
        css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700;800&display=swap');
    html,body,[class*="css"]{font-family:'Inter',sans-serif;}
    #MainMenu,footer{visibility:hidden;}

    /* ── Backgrounds ── */
    .stApp{background-color:#0d1117!important;}
    section[data-testid="stSidebar"]{
        background-color:#090c10!important;
        border-right:1px solid #21262d!important;}

    /* ── Main header gradient: steel-blue → violet ── */
    .main-header{
        font-family:'Inter',sans-serif;font-size:2.05rem;font-weight:800;
        background:linear-gradient(135deg,#58a6ff 0%,#79c0ff 40%,#a371f7 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;text-align:center;margin-bottom:0.4rem;letter-spacing:-0.6px;}
    .sub-header{
        text-align:center;color:#8b949e;font-size:0.88rem;font-weight:400;
        margin-bottom:1.8rem;letter-spacing:0.2px;}

    /* ── Cards ── */
    .metric-card{
        background:linear-gradient(145deg,rgba(88,166,255,0.07),rgba(163,113,247,0.04));
        padding:1.1rem 1.3rem;border-radius:10px;
        border:1px solid rgba(88,166,255,0.16);
        box-shadow:0 2px 12px rgba(0,0,0,0.35),inset 0 1px 0 rgba(255,255,255,0.04);
        transition:transform 0.18s,box-shadow 0.18s;}
    .metric-card:hover{
        transform:translateY(-2px);
        box-shadow:0 6px 24px rgba(88,166,255,0.1),0 2px 8px rgba(0,0,0,0.4);}

    .fraud-alert{
        background:rgba(248,81,73,0.07);
        padding:1.1rem 1.3rem;border-radius:10px;
        border:1px solid rgba(248,81,73,0.22);
        box-shadow:0 2px 12px rgba(0,0,0,0.3);}
    .success-box{
        background:rgba(63,185,80,0.07);
        padding:1.1rem 1.3rem;border-radius:10px;
        border:1px solid rgba(63,185,80,0.22);
        box-shadow:0 2px 12px rgba(0,0,0,0.3);}
    .warning-box{
        background:rgba(210,153,34,0.08);
        padding:1.1rem 1.3rem;border-radius:10px;
        border:1px solid rgba(210,153,34,0.24);
        box-shadow:0 2px 12px rgba(0,0,0,0.3);}
    .info-box{
        background:rgba(163,113,247,0.07);
        padding:1.1rem 1.3rem;border-radius:10px;
        border:1px solid rgba(163,113,247,0.22);
        box-shadow:0 2px 12px rgba(0,0,0,0.3);}
    .live-box{
        background:linear-gradient(145deg,rgba(88,166,255,0.06),rgba(163,113,247,0.04));
        padding:1.2rem 1.4rem;border-radius:12px;
        border:1px solid #21262d;
        box-shadow:0 2px 16px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.03);
        margin-bottom:1rem;}

    /* ── Badges ── */
    .contract-badge{
        background:rgba(163,113,247,0.12);border:1px solid rgba(163,113,247,0.3);
        border-radius:6px;padding:0.35rem 0.85rem;color:#c9a9ff;
        font-size:0.78rem;font-weight:600;display:inline-block;margin:2px 0;}
    .eoa-badge{
        background:rgba(63,185,80,0.1);border:1px solid rgba(63,185,80,0.28);
        border-radius:6px;padding:0.35rem 0.85rem;color:#56d364;
        font-size:0.78rem;font-weight:600;display:inline-block;margin:2px 0;}
    .protocol-tag{
        background:rgba(88,166,255,0.09);color:#79c0ff;border-radius:999px;
        padding:3px 11px;font-size:0.74rem;font-family:'JetBrains Mono',monospace;
        margin:2px;border:1px solid rgba(88,166,255,0.18);display:inline-block;}

    /* ── Risk level pills ── */
    .risk-critical{
        background:rgba(248,81,73,0.15);color:#ffa198;
        padding:3px 12px;border-radius:999px;font-size:0.74rem;font-weight:700;
        letter-spacing:0.4px;border:1px solid rgba(248,81,73,0.35);}
    .risk-high{
        background:rgba(248,81,73,0.1);color:#ff7b72;
        padding:3px 12px;border-radius:999px;font-size:0.74rem;font-weight:700;
        letter-spacing:0.4px;border:1px solid rgba(248,81,73,0.25);}
    .risk-medium{
        background:rgba(210,153,34,0.12);color:#e3b341;
        padding:3px 12px;border-radius:999px;font-size:0.74rem;font-weight:700;
        letter-spacing:0.4px;border:1px solid rgba(210,153,34,0.3);}
    .risk-low{
        background:rgba(63,185,80,0.1);color:#56d364;
        padding:3px 12px;border-radius:999px;font-size:0.74rem;font-weight:700;
        letter-spacing:0.4px;border:1px solid rgba(63,185,80,0.25);}

    /* ── Wallet address chip ── */
    .wallet-address-full{
        font-family:'JetBrains Mono',monospace;font-size:0.80rem;
        background:rgba(33,38,45,0.9);border:1px solid #30363d;
        border-radius:8px;padding:0.5rem 0.85rem;color:#79c0ff;
        word-break:break-all;letter-spacing:0.2px;display:block;margin:4px 0;}
    .valid-address{color:#3fb950;font-size:0.80rem;font-weight:600;}
    .invalid-address{color:#f85149;font-size:0.80rem;font-weight:600;}
    .history-pill{
        display:inline-block;background:rgba(33,38,45,0.9);color:#8b949e;
        border-radius:999px;padding:3px 11px;font-size:0.72rem;
        font-family:'JetBrains Mono',monospace;margin:2px;border:1px solid #30363d;}

    /* ── Misc ── */
    .section-divider{height:1px;background:linear-gradient(90deg,transparent,#21262d,transparent);margin:1.5rem 0;border:none;}
    .footer-text{text-align:center;color:#484f58;font-size:0.78rem;padding:2rem 0 1rem;letter-spacing:0.3px;}
    .footer-text strong{color:#6e7681;}
    .stDataFrame td{white-space:pre-wrap!important;word-break:break-all!important;}
    .stDataFrame th{font-family:'Inter',sans-serif!important;font-weight:600!important;}
    [data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:'Inter',sans-serif;font-weight:700;}
    .stTabs [data-baseweb="tab"]{font-family:'Inter',sans-serif;font-weight:600;font-size:0.86rem;}
    [data-testid="stSidebar"] hr{border-color:#21262d!important;}
</style>"""
    else:
        css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700;800&display=swap');
    html,body,[class*="css"]{font-family:'Inter',sans-serif;}
    #MainMenu,footer{visibility:hidden;}

    /* ── Backgrounds ── */
    .stApp{background-color:#f6f8fa!important;}
    section[data-testid="stSidebar"]{
        background-color:#ffffff!important;
        border-right:1px solid #d0d7de!important;}
    section[data-testid="stSidebar"] *{color:#1f2328!important;}
    .stApp p,.stApp li{color:#1f2328;}
    .stApp h1,.stApp h2,.stApp h3,.stApp h4{color:#1f2328;}
    [data-testid="stMetricValue"]{color:#1f2328!important;}
    [data-testid="stMetricLabel"]{color:#636c76!important;}
    .stApp .stTextInput input{background:#ffffff!important;color:#1f2328!important;border-color:#d0d7de!important;}

    /* ── Main header ── */
    .main-header{
        font-family:'Inter',sans-serif;font-size:2.05rem;font-weight:800;
        background:linear-gradient(135deg,#0969da 0%,#1158c7 40%,#8250df 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;text-align:center;margin-bottom:0.4rem;letter-spacing:-0.6px;}
    .sub-header{
        text-align:center;color:#636c76;font-size:0.88rem;font-weight:400;
        margin-bottom:1.8rem;letter-spacing:0.2px;}

    /* ── Cards ── */
    .metric-card{
        background:#ffffff;
        padding:1.1rem 1.3rem;border-radius:10px;
        border:1px solid #d0d7de;
        box-shadow:0 1px 4px rgba(31,35,40,0.06),inset 0 1px 0 rgba(255,255,255,0.9);
        transition:transform 0.18s,box-shadow 0.18s;}
    .metric-card:hover{
        transform:translateY(-2px);
        box-shadow:0 4px 16px rgba(9,105,218,0.1),0 1px 4px rgba(31,35,40,0.06);}

    .fraud-alert{
        background:#fff8f8;
        padding:1.1rem 1.3rem;border-radius:10px;
        border:1px solid #ffb8b8;
        box-shadow:0 1px 4px rgba(31,35,40,0.06);}
    .success-box{
        background:#f8fff9;
        padding:1.1rem 1.3rem;border-radius:10px;
        border:1px solid #aceebb;
        box-shadow:0 1px 4px rgba(31,35,40,0.06);}
    .warning-box{
        background:#fffbf0;
        padding:1.1rem 1.3rem;border-radius:10px;
        border:1px solid #f0c070;
        box-shadow:0 1px 4px rgba(31,35,40,0.06);}
    .info-box{
        background:#f5f0ff;
        padding:1.1rem 1.3rem;border-radius:10px;
        border:1px solid #cbb7f7;
        box-shadow:0 1px 4px rgba(31,35,40,0.06);}
    .live-box{
        background:#ffffff;
        padding:1.2rem 1.4rem;border-radius:12px;
        border:1px solid #d0d7de;
        box-shadow:0 1px 6px rgba(31,35,40,0.07);
        margin-bottom:1rem;}

    /* ── Badges ── */
    .contract-badge{
        background:#f5f0ff;border:1px solid #cbb7f7;
        border-radius:6px;padding:0.35rem 0.85rem;color:#6e40c9;
        font-size:0.78rem;font-weight:600;display:inline-block;margin:2px 0;}
    .eoa-badge{
        background:#f0fff4;border:1px solid #aceebb;
        border-radius:6px;padding:0.35rem 0.85rem;color:#1a7f37;
        font-size:0.78rem;font-weight:600;display:inline-block;margin:2px 0;}
    .protocol-tag{
        background:#ddf4ff;color:#0550ae;border-radius:999px;
        padding:3px 11px;font-size:0.74rem;font-family:'JetBrains Mono',monospace;
        margin:2px;border:1px solid #aecbf5;display:inline-block;}

    /* ── Risk level pills ── */
    .risk-critical{
        background:#ffe8e7;color:#a0111f;
        padding:3px 12px;border-radius:999px;font-size:0.74rem;font-weight:700;
        letter-spacing:0.4px;border:1px solid #ffb8b8;}
    .risk-high{
        background:#fff0ee;color:#cf222e;
        padding:3px 12px;border-radius:999px;font-size:0.74rem;font-weight:700;
        letter-spacing:0.4px;border:1px solid #ffc9c5;}
    .risk-medium{
        background:#fff8c5;color:#7d4e00;
        padding:3px 12px;border-radius:999px;font-size:0.74rem;font-weight:700;
        letter-spacing:0.4px;border:1px solid #f0c070;}
    .risk-low{
        background:#dafbe1;color:#116329;
        padding:3px 12px;border-radius:999px;font-size:0.74rem;font-weight:700;
        letter-spacing:0.4px;border:1px solid #aceebb;}

    /* ── Wallet address chip ── */
    .wallet-address-full{
        font-family:'JetBrains Mono',monospace;font-size:0.80rem;
        background:#f6f8fa;border:1px solid #d0d7de;
        border-radius:8px;padding:0.5rem 0.85rem;color:#0550ae;
        word-break:break-all;letter-spacing:0.2px;display:block;margin:4px 0;}
    .valid-address{color:#1a7f37;font-size:0.80rem;font-weight:600;}
    .invalid-address{color:#cf222e;font-size:0.80rem;font-weight:600;}
    .history-pill{
        display:inline-block;background:#f6f8fa;color:#636c76;
        border-radius:999px;padding:3px 11px;font-size:0.72rem;
        font-family:'JetBrains Mono',monospace;margin:2px;border:1px solid #d0d7de;}

    /* ── Misc ── */
    .footer-text{text-align:center;color:#8c959f;font-size:0.78rem;padding:2rem 0 1rem;letter-spacing:0.3px;}
    .footer-text strong{color:#636c76;}
    .stDataFrame td{white-space:pre-wrap!important;word-break:break-all!important;}
    .stDataFrame th{font-family:'Inter',sans-serif!important;font-weight:600!important;}
    [data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:'Inter',sans-serif;font-weight:700;}
    .stTabs [data-baseweb="tab"]{font-family:'Inter',sans-serif;font-weight:600;font-size:0.86rem;}
    [data-testid="stSidebar"] hr{border-color:#d0d7de!important;}
</style>"""
    st.markdown(css, unsafe_allow_html=True)


inject_css(st.session_state.dark_mode)

# -----------------------------------------------
# Known DeFi / Protocol Addresses
# -----------------------------------------------
DEFI_PROTOCOLS = {
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": "Uniswap V2 Router",
    "0xe592427a0aece92de3edee1f18e0157c05861564": "Uniswap V3 Router",
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": "Uniswap Universal Router",
    "0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b": "Uniswap Universal Router 2",
    "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9": "Aave V2 Lending Pool",
    "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2": "Aave V3 Pool",
    "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b": "Compound Comptroller",
    "0x7be8076f4ea4a4ad08075c2508e481d6c946d12b": "OpenSea Wyvern V1",
    "0x7f268357a8c2552623316e2562d90e642bb538e5": "OpenSea Wyvern V2",
    "0x00000000006c3852cbef3e08e8df289169ede581": "OpenSea Seaport 1.1",
    "0x00000000000000adc04c56bf30ac9d3c0aaf14dc": "OpenSea Seaport 1.5",
    "0x1111111254fb6c44bac0bed2854e76f90643097d": "1inch V4",
    "0x1111111254eeb25477b68fb85ed929f73a960582": "1inch V5",
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f": "SushiSwap Router",
    "0xd51a44d3fae010294c616388b506acda1bfaae46": "Curve Tricrypto2",
    "0xa5407eae9ba41422680e2e00537571bcc53efbfd": "Curve sUSD Pool",
    "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7": "Curve 3Pool",
    "0x881d40237659c251811cec9c364ef91dc08d300c": "MetaMask Swap Router",
    "0xba12222222228d8ba445958a75a0704d566bf2c8": "Balancer Vault",
    "0xdef1c0ded9bec7f1a1670819833240f027b25eff": "0x Exchange Proxy",
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "Wrapped ETH (WETH)",
    "0x6b175474e89094c44da98b954eedeac495271d0f": "DAI Stablecoin",
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USD Coin (USDC)",
    "0xdac17f958d2ee523a2206206994597c13d831ec7": "Tether (USDT)",
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": "Wrapped Bitcoin (WBTC)",
    "0x514910771af9ca656af840dff83e8264ecf986ca": "Chainlink (LINK)",
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": "Uniswap Token (UNI)",
    "0xc18360217d8f7ab5e7c516566761ea12ce7f9d72": "ENS Token",
    "0x57f1887a8bf19b14fc0df6fd9b2acc9af147ea85": "ENS Registrar",
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84": "Lido stETH",
    "0x5a98fcbea516cf06857215779fd812ca3bef1b32": "Lido (LDO)",
    "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2": "MakerDAO (MKR)",
    "0x0bc529c00c6401aef6d220be8c6ea1667f6ad93e": "Yearn Finance (YFI)",
    "0x408e41876cccdc0f92210600ef50372656052a38": "Ren (REN)",
}

# -----------------------------------------------
# Helper — address validation
# -----------------------------------------------
def is_valid_eth_address(addr: str) -> bool:
    return bool(re.fullmatch(r"0x[0-9a-fA-F]{40}", addr.strip()))


def normalise_fraud_score(score: float, score_min: float, score_max: float) -> float:
    if score_max == score_min:
        return 50.0
    return float(np.clip((score_max - score) / (score_max - score_min) * 100, 0, 100))


def risk_label(risk_score: float) -> str:
    if risk_score >= 80:   return "Critical"
    if risk_score >= 60:   return "High"
    if risk_score >= 35:   return "Medium"
    return "Low"


def risk_badge(risk_score: float) -> str:
    label = risk_label(risk_score)
    css = {"Critical": "risk-critical", "High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}
    return f'<span class="{css[label]}">{label}</span>'


# -----------------------------------------------
# Helper — link probability (multi-signal)
# -----------------------------------------------
def compute_link_probability(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    dot = float(np.clip(np.dot(emb_a, emb_b), -500.0, 500.0))
    norm_a = np.linalg.norm(emb_a) + 1e-9
    norm_b = np.linalg.norm(emb_b) + 1e-9
    cosine = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
    l2_sim = 1.0 / (1.0 + float(np.linalg.norm(emb_a - emb_b)))
    combined = np.clip(0.5 * dot + 0.3 * cosine * abs(dot) + 0.2 * l2_sim * abs(dot), -500.0, 500.0)
    return float(1.0 / (1.0 + np.exp(-combined)))


# -----------------------------------------------
# Helper — graph statistics
# -----------------------------------------------
@st.cache_data
def create_graph_statistics(edges_df: pd.DataFrame) -> dict:
    return {
        "total_transactions": len(edges_df),
        "unique_senders":     edges_df["from_id"].nunique(),
        "unique_receivers":   edges_df["to_id"].nunique(),
        "avg_out_degree":     edges_df.groupby("from_id").size().mean(),
        "avg_in_degree":      edges_df.groupby("to_id").size().mean(),
        "max_out_degree":     int(edges_df.groupby("from_id").size().max()),
        "max_in_degree":      int(edges_df.groupby("to_id").size().max()),
    }


# -----------------------------------------------
# Data loading
# -----------------------------------------------
@st.cache_data
def load_data():
    required = {
        "node_embeddings.npy": "GNN node embeddings",
        "edges.csv":           "Transaction edge list",
        "fraudulent_wallets.csv": "Fraud detection results",
        "label_encoder.pkl":   "LabelEncoder (address ↔ ID)",
    }
    missing = [f"• **{f}** — {d}" for f, d in required.items() if not os.path.exists(f)]
    if missing:
        st.error("### ❌ Required data files are missing")
        st.markdown("\n".join(missing))
        st.stop()

    try:
        embeddings = np.load("node_embeddings.npy")
        edges      = pd.read_csv("edges.csv")
        fraud_df   = pd.read_csv("fraudulent_wallets.csv")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open("label_encoder.pkl", "rb") as f:
                le = pickle.load(f)

        loss_history = None
        try:
            loss_history = np.load("loss_history.npy")
        except FileNotFoundError:
            pass

        fpr = tpr = roc_auc_val = None
        try:
            roc = np.load("roc_data.npz")
            fpr, tpr, roc_auc_val = roc["fpr"], roc["tpr"], float(roc["auc"])
        except FileNotFoundError:
            pass

        s_min = float(fraud_df["fraud_score"].min())
        s_max = float(fraud_df["fraud_score"].max())
        fraud_df["risk_score"] = fraud_df["fraud_score"].apply(
            lambda s: round(normalise_fraud_score(s, s_min, s_max), 1)
        )
        fraud_df["risk_level"] = fraud_df["risk_score"].apply(risk_label)
        return embeddings, edges, fraud_df, le, loss_history, (fpr, tpr, roc_auc_val)

    except Exception as exc:
        logger.exception("Fatal error loading data")
        st.error(f"Error loading data: {exc}")
        st.stop()


# -----------------------------------------------
# Etherscan API helpers
# -----------------------------------------------
ETHERSCAN_BASE = "https://api.etherscan.io/api"

def _etherscan_get(params: dict) -> dict | None:
    api_key = os.environ.get("ETHERSCAN_API_KEY", "")
    if api_key:
        params["apikey"] = api_key
    try:
        resp = requests.get(ETHERSCAN_BASE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "1" or data.get("message") == "OK":
            return data
        return data  # return anyway so caller can inspect message
    except Exception as exc:
        logger.warning(f"Etherscan API error: {exc}")
        return None


@st.cache_data(ttl=300)
def fetch_eth_balance(address: str) -> float | None:
    data = _etherscan_get({"module": "account", "action": "balance",
                           "address": address, "tag": "latest"})
    if data and data.get("status") == "1":
        try:
            return int(data["result"]) / 1e18
        except Exception:
            return None
    return None


@st.cache_data(ttl=300)
def fetch_transactions(address: str, limit: int = 25) -> pd.DataFrame:
    data = _etherscan_get({
        "module": "account", "action": "txlist",
        "address": address,
        "startblock": 0, "endblock": 99999999,
        "page": 1, "offset": limit,
        "sort": "desc",
    })
    if data and data.get("status") == "1" and isinstance(data.get("result"), list):
        txs = data["result"]
        rows = []
        for tx in txs:
            val_eth = int(tx.get("value", 0)) / 1e18
            gas_price_gwei = int(tx.get("gasPrice", 0)) / 1e9
            ts = int(tx.get("timeStamp", 0))
            dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "—"
            protocol = DEFI_PROTOCOLS.get(tx.get("to", "").lower(), "")
            rows.append({
                "Time":          dt,
                "Hash":          tx.get("hash", "")[:18] + "…",
                "Full Hash":     tx.get("hash", ""),
                "From":          tx.get("from", ""),
                "To":            tx.get("to", ""),
                "Value (ETH)":   round(val_eth, 6),
                "Gas Price (Gwei)": round(gas_price_gwei, 2),
                "Status":        "✅" if tx.get("isError", "0") == "0" else "❌",
                "Protocol":      protocol if protocol else "—",
                "Block":         tx.get("blockNumber", ""),
            })
        return pd.DataFrame(rows)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_token_transfers(address: str, limit: int = 15) -> pd.DataFrame:
    data = _etherscan_get({
        "module": "account", "action": "tokentx",
        "address": address,
        "page": 1, "offset": limit,
        "sort": "desc",
    })
    if data and data.get("status") == "1" and isinstance(data.get("result"), list):
        rows = []
        for tx in data["result"]:
            decimals = int(tx.get("tokenDecimal", 18) or 18)
            val = int(tx.get("value", 0)) / (10 ** decimals)
            ts = int(tx.get("timeStamp", 0))
            dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "—"
            rows.append({
                "Time":      dt,
                "Token":     tx.get("tokenSymbol", "?"),
                "Name":      tx.get("tokenName", ""),
                "Value":     round(val, 6),
                "From":      tx.get("from", ""),
                "To":        tx.get("to", ""),
                "Contract":  tx.get("contractAddress", ""),
                "Hash":      tx.get("hash", "")[:18] + "…",
            })
        return pd.DataFrame(rows)
    return pd.DataFrame()


@st.cache_data(ttl=600)
def check_is_contract(address: str) -> tuple[bool, str]:
    """Returns (is_contract, contract_name_or_empty)."""
    data = _etherscan_get({"module": "proxy", "action": "eth_getCode",
                           "address": address, "tag": "latest"})
    if data:
        code = data.get("result", "0x")
        is_contract = len(code) > 4
    else:
        is_contract = False

    contract_name = ""
    if is_contract:
        src_data = _etherscan_get({"module": "contract", "action": "getsourcecode",
                                   "address": address})
        if src_data and isinstance(src_data.get("result"), list) and src_data["result"]:
            contract_name = src_data["result"][0].get("ContractName", "")
    return is_contract, contract_name


@st.cache_data(ttl=300)
def fetch_tx_count(address: str) -> int | None:
    data = _etherscan_get({"module": "proxy", "action": "eth_getTransactionCount",
                           "address": address, "tag": "latest"})
    if data and "result" in data:
        try:
            return int(data["result"], 16)
        except Exception:
            return None
    return None


def detect_protocols_from_txs(tx_df: pd.DataFrame) -> list[str]:
    if tx_df.empty:
        return []
    protos = set()
    for addr in tx_df["To"].str.lower():
        if addr in DEFI_PROTOCOLS:
            protos.add(DEFI_PROTOCOLS[addr])
    return sorted(protos)


# -----------------------------------------------
# Visualisation helpers
# -----------------------------------------------
def plot_loss_curve(loss_history: np.ndarray) -> go.Figure:
    epochs = list(range(1, len(loss_history) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=loss_history, mode="lines", name="Training Loss",
        line=dict(color=PRIMARY_COLOR, width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({'0,210,255' if st.session_state.dark_mode else '3,105,161'},0.07)",
    ))
    fig.add_annotation(x=epochs[-1], y=float(loss_history[-1]),
                       text=f"Final: {loss_history[-1]:.4f}", showarrow=True,
                       arrowhead=2, ax=-60, ay=-30, font=dict(color=PRIMARY_COLOR, size=11))
    fig.update_layout(title="GNN Training Loss Curve", xaxis_title="Epoch",
                      yaxis_title="Loss", hovermode="x unified",
                      template=PLOTLY_TEMPLATE, height=400)
    return fig


def plot_roc_curve(fpr, tpr, auc_score: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"ROC Curve (AUC = {auc_score:.3f})",
        line=dict(color=PRIMARY_COLOR, width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({'0,210,255' if st.session_state.dark_mode else '3,105,161'},0.08)",
    ))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Random Classifier",
                             line=dict(color="gray", width=1.5, dash="dash")))
    fig.update_layout(title="ROC Curve — Transaction Link Prediction",
                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                      hovermode="closest", template=PLOTLY_TEMPLATE, height=400)
    return fig


def plot_fraud_distribution(fraud_df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(fraud_df, x="risk_score", nbins=50,
                       title="Risk Score Distribution (0 = safe · 100 = critical)",
                       labels={"risk_score": "Risk Score (0–100)"},
                       color_discrete_sequence=[DANGER_COLOR])
    for thresh, label, color in [(35, "Medium", "#f59e0b"), (60, "High", "#ef4444"), (80, "Critical", "#dc2626")]:
        fig.add_vline(x=thresh, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_position="top right",
                      annotation_font_color=color)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=380)
    return fig


def create_network_subgraph(
    edges_df: pd.DataFrame, wallet_id: int, le, fraud_df: pd.DataFrame,
    max_connections: int = 25, layout: str = "spring", show_labels: bool = True,
) -> go.Figure | None:
    related = edges_df[
        (edges_df["from_id"] == wallet_id) | (edges_df["to_id"] == wallet_id)
    ].head(max_connections)
    if len(related) == 0:
        return None

    fraud_risk  = fraud_df.set_index("wallet_id")["risk_score"].to_dict() if "wallet_id" in fraud_df.columns else {}
    fraud_level = fraud_df.set_index("wallet_id")["risk_level"].to_dict() if "wallet_id" in fraud_df.columns else {}
    fraudulent_ids = set(fraud_risk.keys())

    G = nx.DiGraph()
    for _, row in related.iterrows():
        G.add_edge(int(row["from_id"]), int(row["to_id"]))

    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    pos = {
        "spring":   lambda: nx.spring_layout(G, seed=42, k=2.2),
        "kamada":   lambda: nx.kamada_kawai_layout(G),
        "circular": lambda: nx.circular_layout(G),
        "shell":    lambda: nx.shell_layout(G),
    }.get(layout, lambda: nx.spring_layout(G, seed=42))()

    out_ex, out_ey, in_ex, in_ey, int_ex, int_ey = [], [], [], [], [], []
    arrows = []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        if u == wallet_id:
            out_ex += [x0, x1, None]; out_ey += [y0, y1, None]
        elif v == wallet_id:
            in_ex  += [x0, x1, None]; in_ey  += [y0, y1, None]
        else:
            int_ex += [x0, x1, None]; int_ey += [y0, y1, None]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        arrows.append(dict(x=x1, y=y1, ax=mx, ay=my,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.2,
                           arrowcolor=(PRIMARY_COLOR if u == wallet_id else "#a78bfa" if v == wallet_id else "rgba(148,163,184,0.4)")))

    def edge_trace(ex, ey, color, name):
        return go.Scatter(x=ex, y=ey, mode="lines", name=name,
                          line=dict(width=1.4, color=color), hoverinfo="none", showlegend=bool(ex))

    traces = [
        edge_trace(out_ex, out_ey, f"rgba({'0,210,255' if st.session_state.dark_mode else '3,105,161'},0.55)", "Outgoing"),
        edge_trace(in_ex,  in_ey,  "rgba(167,139,250,0.55)", "Incoming"),
        edge_trace(int_ex, int_ey, "rgba(148,163,184,0.25)", "Internal"),
    ]

    categories = {
        "center_fraud": {"color": "#fbbf24", "border": "#f59e0b", "size_base": 34, "label": "⭐ Centre (Flagged)"},
        "center":       {"color": "#3b82f6", "border": "#60a5fa", "size_base": 32, "label": "🔵 Centre Wallet"},
        "fraud":        {"color": "#ef4444", "border": "#dc2626", "size_base": 22, "label": "🔴 Flagged Wallet"},
        "normal":       {"color": PRIMARY_COLOR, "border": "#38bdf8", "size_base": 16, "label": "⚪ Normal Wallet"},
    }
    cat_nodes: dict[str, list] = {k: [] for k in categories}
    for node in G.nodes():
        is_fraud  = node in fraudulent_ids
        is_center = node == wallet_id
        if   is_center and is_fraud: cat_nodes["center_fraud"].append(node)
        elif is_center:              cat_nodes["center"].append(node)
        elif is_fraud:               cat_nodes["fraud"].append(node)
        else:                        cat_nodes["normal"].append(node)

    for cat, cfg in categories.items():
        nids = cat_nodes[cat]
        if not nids:
            continue
        nx_list, ny_list, hover_list, size_list, sym_list = [], [], [], [], []
        for node in nids:
            x, y = pos[node]
            nx_list.append(x); ny_list.append(y)
            try:    addr_str = le.inverse_transform([node])[0]
            except: addr_str = f"ID {node}"
            rs_val  = fraud_risk.get(node)
            rl_val  = fraud_level.get(node, "Clean")
            id_deg  = in_deg.get(node, 0)
            od_deg  = out_deg.get(node, 0)
            hover_list.append(
                f"<b>{addr_str}</b><br>Wallet ID: {node}<br>Risk: {rl_val}"
                + (f" ({rs_val:.1f}/100)" if rs_val is not None else "")
                + f"<br>Out-degree: {od_deg}  In-degree: {id_deg}"
                + (" <b>⭐ CENTRE</b>" if node == wallet_id else ""))
            size_list.append(cfg["size_base"] + min(int((id_deg + od_deg) ** 0.5) * 2, 14))
            sym_list.append("star" if node == wallet_id else "circle")

        traces.append(go.Scatter(
            x=nx_list, y=ny_list,
            mode="markers+text" if show_labels else "markers",
            name=cfg["label"], hoverinfo="text", hovertext=hover_list,
            text=[(le.inverse_transform([n])[0][:8] + "…" if show_labels else "") for n in nids],
            textposition="top center",
            textfont=dict(size=8, color=THEME["text_muted"], family="JetBrains Mono"),
            marker=dict(size=size_list, color=cfg["color"], symbol=sym_list,
                        line=dict(width=2, color=cfg["border"]), opacity=0.92),
            showlegend=True,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        annotations=arrows,
        title=dict(text=f"Transaction Network — Wallet {wallet_id}",
                   font=dict(size=13, color=THEME["text_muted"], family="Sora")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11, color=THEME["text_muted"]),
                    bgcolor="rgba(255,255,255,0.05)" if st.session_state.dark_mode else "rgba(255,255,255,0.85)",
                    bordercolor="rgba(255,255,255,0.08)" if st.session_state.dark_mode else "rgba(0,0,0,0.1)",
                    borderwidth=1),
        hovermode="closest", template=PLOTLY_TEMPLATE,
        paper_bgcolor=THEME["bg_main"], plot_bgcolor=THEME["bg_plot"],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600, margin=dict(t=80, b=20, l=20, r=20),
    )
    return fig


# -----------------------------------------------
# Load data + precompute sets
# -----------------------------------------------
embeddings, edges, fraud_df, le, loss_history, (fpr, tpr, roc_auc_val) = load_data()
stats      = create_graph_statistics(edges)
WALLET_SET = set(le.classes_)
FRAUD_IDS: set = set(fraud_df["wallet_id"].tolist()) if "wallet_id" in fraud_df.columns else set()

# -----------------------------------------------
# Sidebar
# -----------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1rem;">
        <div style="font-size:2.8rem;">🔗</div>
        <div style="font-family:'Sora',sans-serif;font-size:1.05rem;font-weight:700;
                    background:linear-gradient(135deg,#00d2ff,#8b5cf6);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;margin-top:4px;">GNN Analytics</div>
        <div style="color:#475569;font-size:0.72rem;margin-top:2px;letter-spacing:1px;">
            BLOCKCHAIN INTELLIGENCE
        </div>
    </div>""", unsafe_allow_html=True)

    mode_icon  = "☀️" if st.session_state.dark_mode else "🌙"
    mode_label = "Switch to Light Mode" if st.session_state.dark_mode else "Switch to Dark Mode"
    if st.button(f"{mode_icon}  {mode_label}", use_container_width=True, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown('<hr style="border-color:rgba(128,128,128,0.2);margin:0.5rem 0 1rem;">', unsafe_allow_html=True)

    section = st.radio("Navigate", [
        "🏠 Overview",
        "📊 Graph Analytics",
        "🔮 Link Prediction",
        "🚨 Fraud Detection",
        "📈 Model Performance",
        "🌐 Network Visualization",
        "🔌 Live Blockchain Explorer",
    ])

    st.markdown('<hr style="border-color:rgba(128,128,128,0.2);margin:1rem 0;">', unsafe_allow_html=True)
    st.markdown("**📌 Quick Stats**")
    st.metric("Total Wallets",      f"{embeddings.shape[0]:,}")
    st.metric("Total Transactions", f"{stats['total_transactions']:,}")
    st.metric("Suspicious Wallets", f"{len(fraud_df):,}")
    if roc_auc_val is not None:
        st.metric("ROC-AUC Score", f"{roc_auc_val:.4f}")

    st.markdown('<hr style="border-color:rgba(128,128,128,0.2);margin:1rem 0;">', unsafe_allow_html=True)

    etherscan_key = os.environ.get("ETHERSCAN_API_KEY", "")
    if etherscan_key:
        st.success("🔗 Etherscan API connected")
    else:
        st.warning("⚠️ No Etherscan API key\nLive Explorer requires `ETHERSCAN_API_KEY` secret.")

    st.info("**Model:** GraphSAGE\n\n**Dataset:** Ethereum Mainnet\n\n**Decoder:** Multi-signal (dot + cosine + L2)")


# ═══════════════════════════════════════════════
# SECTION: Overview
# ═══════════════════════════════════════════════
if "🏠 Overview" in section:
    st.markdown('<p class="main-header">🔗 Blockchain GNN Transaction Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">GraphSAGE-powered link prediction & anomaly detection on Ethereum Mainnet</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, label, val, cls in [
        (col1, "Wallet Addresses",  f"{embeddings.shape[0]:,}", "metric-card"),
        (col2, "Transactions",       f"{stats['total_transactions']:,}", "metric-card"),
        (col3, "Avg Out-Degree",     f"{stats['avg_out_degree']:.2f}", "metric-card"),
        (col4, "Suspicious Wallets", f"{len(fraud_df):,}", "fraud-alert"),
    ]:
        with col:
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            st.metric(label, val)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📖 Project Overview")
        st.markdown("""
This dashboard presents a **Graph Neural Network (GNN)** system for analysing Ethereum blockchain
transactions. It combines a **GraphSAGE** model for link prediction with **Isolation Forest**
anomaly detection to identify suspicious wallet behaviour.

- **Transaction Link Prediction** — predict the probability of a future transaction between any two wallet addresses
- **Fraud Detection** — identify anomalous wallets using unsupervised learning on GNN embeddings
- **Network Visualisation** — explore the transaction graph around any wallet
- **Live Blockchain Explorer** — query any Ethereum address in real time via Etherscan
- **Smart Contract Detection** — identify contracts, DeFi protocol interactions & token transfers
""")
    with col2:
        st.markdown("### 🎯 Key Features")
        st.markdown("""
✅ Real Ethereum mainnet data  
✅ GraphSAGE GNN architecture  
✅ Multi-signal link predictor  
✅ Risk score 0–100 scale  
✅ Input validation  
✅ Session search history  
✅ **Live Etherscan API queries**  
✅ **Smart contract detection**  
✅ **DeFi protocol recognition**  
✅ Dark / Light mode  
""")

    st.markdown("---")
    with st.expander("📚 Methodology Details"):
        st.markdown("""
#### 1. Data Collection
- Source: Google BigQuery Ethereum Public Dataset
- Transactions: sender, receiver, value, gas, timestamp

#### 2. Graph Construction
- **Nodes:** wallet addresses
- **Edges:** directed transaction relationships
- **Features:** in-degree / out-degree statistics

#### 3. Model Architecture
- **GraphSAGE** — 2-layer Graph Convolutional Network
- **Embedding size:** 64 dimensions
- **Improved decoder:** weighted combination of dot-product, cosine similarity, and L2 distance

#### 4. Training
- Train / Val / Test: 70 / 15 / 15
- Loss: Binary Cross-Entropy · Optimiser: Adam (lr=0.01)
- Negative sampling: dynamic

#### 5. Fraud Detection
- **Algorithm:** Isolation Forest (unsupervised)
- **Input:** 64-dim node embeddings
- **Output:** Risk score **0–100** (higher = more suspicious)
- **Thresholds:** Low < 35 · Medium 35–60 · High 60–80 · Critical ≥ 80

#### 6. Live Blockchain Integration
- **API:** Etherscan v1
- **Capabilities:** ETH balance, tx history, contract detection, token transfers, DeFi protocol tagging
""")


# ═══════════════════════════════════════════════
# SECTION: Graph Analytics
# ═══════════════════════════════════════════════
elif "📊 Graph Analytics" in section:
    st.markdown('<p class="main-header">📊 Graph Analytics & Statistics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Degree distribution, transaction explorer, and network topology metrics</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Network Overview")
        st.metric("Total Nodes (Wallets)",      f"{embeddings.shape[0]:,}")
        st.metric("Total Edges (Transactions)", f"{stats['total_transactions']:,}")
        st.metric("Graph Density", f"{stats['total_transactions'] / (embeddings.shape[0] ** 2):.2e}")
    with col2:
        st.markdown("### Degree Statistics")
        st.metric("Avg Out-Degree", f"{stats['avg_out_degree']:.2f}")
        st.metric("Avg In-Degree",  f"{stats['avg_in_degree']:.2f}")
        st.metric("Max Out-Degree", f"{stats['max_out_degree']:,}")
    with col3:
        st.markdown("### Node Info")
        st.metric("Unique Senders",    f"{stats['unique_senders']:,}")
        st.metric("Unique Receivers",  f"{stats['unique_receivers']:,}")
        st.metric("Embedding Dimensions", f"{embeddings.shape[1]}")

    st.markdown("---")
    st.markdown("### 📋 Transaction Explorer")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_wallet = st.text_input("🔍 Search by Wallet ID (numeric)", "")
    with col2:
        num_rows = st.selectbox("Rows", [10, 25, 50, 100], index=0)
    with col3:
        sort_col = st.selectbox("Sort by", edges.columns.tolist(), index=0)

    display_edges = edges.sort_values(sort_col)
    if search_wallet:
        try:
            wid = int(search_wallet)
            display_edges = display_edges[(display_edges["from_id"] == wid) | (display_edges["to_id"] == wid)]
            st.info(f"Found **{len(display_edges):,}** transactions for wallet `{wid}`")
        except ValueError:
            st.error("Please enter a numeric wallet ID.")

    show = display_edges.head(num_rows).copy()
    try:
        show["from_address"] = le.inverse_transform(show["from_id"].astype(int))
        show["to_address"]   = le.inverse_transform(show["to_id"].astype(int))
    except Exception:
        pass
    st.dataframe(show, use_container_width=True,
                 column_config={
                     "from_address": st.column_config.TextColumn("From Address", width="large"),
                     "to_address":   st.column_config.TextColumn("To Address",   width="large"),
                 })

    st.markdown("---")
    st.markdown("### 📊 Degree Distribution")
    col1, col2 = st.columns(2)
    with col1:
        out_deg = edges.groupby("from_id").size()
        fig = px.histogram(out_deg, nbins=50, title="Out-Degree Distribution",
                           labels={"value": "Out-Degree"}, color_discrete_sequence=[PRIMARY_COLOR])
        fig.update_layout(template=PLOTLY_TEMPLATE, height=340)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        in_deg = edges.groupby("to_id").size()
        fig = px.histogram(in_deg, nbins=50, title="In-Degree Distribution",
                           labels={"value": "In-Degree"}, color_discrete_sequence=["#ff7f0e"])
        fig.update_layout(template=PLOTLY_TEMPLATE, height=340)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# SECTION: Link Prediction
# ═══════════════════════════════════════════════
elif "🔮 Link Prediction" in section:
    st.markdown('<p class="main-header">🔮 Transaction Link Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict future transaction probability · Multi-signal decoder: dot-product + cosine + L2</p>', unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.search_history:
        st.markdown("**Recent lookups:**")
        history_html = " ".join(
            f'<span class="history-pill" title="{a}">{a[:10]}…{a[-6:]}</span>'
            for a in st.session_state.search_history[-6:]
        )
        st.markdown(history_html, unsafe_allow_html=True)
        st.markdown("")

    tab1, tab2 = st.tabs(["🔤 Manual Input", "🎲 Random Prediction"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            wallet_a = st.text_input("Sender Wallet Address",
                                     placeholder="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
                                     help="Enter a valid Ethereum wallet address (0x + 40 hex chars)")
            if wallet_a:
                if not is_valid_eth_address(wallet_a):
                    st.markdown('<p class="invalid-address">⚠ Invalid format — must be 0x followed by 40 hex characters</p>', unsafe_allow_html=True)
                elif wallet_a not in WALLET_SET:
                    st.markdown('<p class="invalid-address">⚠ Address not found in this dataset</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="valid-address">✔ Valid address found in dataset</p>', unsafe_allow_html=True)
        with col2:
            wallet_b = st.text_input("Receiver Wallet Address",
                                     placeholder="0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",
                                     help="Enter a valid Ethereum wallet address (0x + 40 hex chars)")
            if wallet_b:
                if not is_valid_eth_address(wallet_b):
                    st.markdown('<p class="invalid-address">⚠ Invalid format — must be 0x followed by 40 hex characters</p>', unsafe_allow_html=True)
                elif wallet_b not in WALLET_SET:
                    st.markdown('<p class="invalid-address">⚠ Address not found in this dataset</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="valid-address">✔ Valid address found in dataset</p>', unsafe_allow_html=True)

        if st.button("🔮 Predict Transaction Probability", type="primary", use_container_width=True):
            errors = []
            if not wallet_a: errors.append("Sender address is empty.")
            elif not is_valid_eth_address(wallet_a): errors.append("Sender address has invalid format.")
            elif wallet_a not in WALLET_SET: errors.append("Sender address not found in dataset.")
            if not wallet_b: errors.append("Receiver address is empty.")
            elif not is_valid_eth_address(wallet_b): errors.append("Receiver address has invalid format.")
            elif wallet_b not in WALLET_SET: errors.append("Receiver address not found in dataset.")
            if wallet_a and wallet_b and wallet_a == wallet_b: errors.append("Sender and receiver must be different.")

            if errors:
                for err in errors:
                    st.error(f"❌ {err}")
            else:
                id_a = int(le.transform([wallet_a])[0])
                id_b = int(le.transform([wallet_b])[0])
                probability = compute_link_probability(embeddings[id_a], embeddings[id_b])

                for addr in [wallet_a, wallet_b]:
                    if addr not in st.session_state.search_history:
                        st.session_state.search_history.append(addr)

                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("### 📊 Prediction Result")
                    st.metric("Transaction Probability", f"{probability:.4f}", delta=f"{probability*100:.1f}%")
                    if probability > 0.7:
                        st.success("🟢 **High likelihood** of future transaction")
                    elif probability > 0.4:
                        st.warning("🟡 **Moderate likelihood** of future transaction")
                    else:
                        st.info("🔵 **Low likelihood** of future transaction")
                    st.markdown("**Sender:**")
                    st.markdown(f'<span class="wallet-address-full">{wallet_a}</span>', unsafe_allow_html=True)
                    st.markdown("**Receiver:**")
                    st.markdown(f'<span class="wallet-address-full">{wallet_b}</span>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")
                with st.expander("📈 Detailed Wallet Analysis"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Sender Wallet**")
                        sent = edges[edges["from_id"] == id_a]
                        st.write(f"- Transactions sent: **{len(sent)}**")
                        st.write(f"- Unique receivers: **{sent['to_id'].nunique()}**")
                        st.write(f"- Fraud flag: {'🚨 Flagged' if id_a in FRAUD_IDS else '✅ Clean'}")
                    with col2:
                        st.markdown("**Receiver Wallet**")
                        recv = edges[edges["to_id"] == id_b]
                        st.write(f"- Transactions received: **{len(recv)}**")
                        st.write(f"- Unique senders: **{recv['from_id'].nunique()}**")
                        st.write(f"- Fraud flag: {'🚨 Flagged' if id_b in FRAUD_IDS else '✅ Clean'}")
                    existing = edges[(edges["from_id"] == id_a) & (edges["to_id"] == id_b)]
                    if len(existing) > 0:
                        st.warning(f"⚠️ **{len(existing)} historical transaction(s)** already exist between these wallets")
                    else:
                        st.info("ℹ️ No prior transactions between these wallets")

                result_df = pd.DataFrame([{
                    "Sender": wallet_a, "Receiver": wallet_b,
                    "Probability": round(probability, 6),
                    "Likelihood": "High" if probability > 0.7 else ("Medium" if probability > 0.4 else "Low"),
                    "Timestamp": datetime.now().isoformat(),
                }])
                st.download_button("⬇️ Export Result (CSV)", result_df.to_csv(index=False),
                                   file_name="link_prediction_result.csv", mime="text/csv")

    with tab2:
        st.markdown("### 🎲 Random Wallet Pair Prediction")
        num_predictions = st.slider("Number of random predictions", 5, 30, 10)
        if st.button("🎲 Generate Random Predictions", use_container_width=True):
            rows = []
            for _ in range(num_predictions):
                ia, ib = np.random.choice(embeddings.shape[0], 2, replace=False)
                prob  = compute_link_probability(embeddings[ia], embeddings[ib])
                addr_a = le.inverse_transform([ia])[0]
                addr_b = le.inverse_transform([ib])[0]
                rows.append({
                    "Sender":   addr_a, "Receiver": addr_b,
                    "Probability": round(prob, 4),
                    "Likelihood":  "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low"),
                    "Sender Fraud":   "🚨" if ia in FRAUD_IDS else "✅",
                    "Receiver Fraud": "🚨" if ib in FRAUD_IDS else "✅",
                })
            df_pred = pd.DataFrame(rows).sort_values("Probability", ascending=False)
            st.dataframe(df_pred, use_container_width=True, height=420,
                         column_config={
                             "Sender":      st.column_config.TextColumn("Sender Address",   width="large"),
                             "Receiver":    st.column_config.TextColumn("Receiver Address", width="large"),
                             "Probability": st.column_config.NumberColumn("Probability", format="%.4f"),
                         })
            st.download_button("⬇️ Export All Predictions (CSV)", df_pred.to_csv(index=False),
                               file_name="random_predictions.csv", mime="text/csv")


# ═══════════════════════════════════════════════
# SECTION: Fraud Detection
# ═══════════════════════════════════════════════
elif "🚨 Fraud Detection" in section:
    st.markdown('<p class="main-header">🚨 Fraud Detection Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Wallets scored 0–100 via Isolation Forest on 64-dim GNN embeddings · Higher = more suspicious</p>', unsafe_allow_html=True)
    st.markdown("---")

    critical_count = len(fraud_df[fraud_df["risk_level"] == "Critical"])
    high_count     = len(fraud_df[fraud_df["risk_level"] == "High"])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
        st.metric("Critical Risk Wallets", f"{critical_count:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.metric("High Risk Wallets", f"{high_count:,}")
    with col3:
        st.metric("Total Flagged", f"{len(fraud_df):,}")
    with col4:
        st.metric("Detection Rate", f"{len(fraud_df)/embeddings.shape[0]*100:.2f}%")

    st.markdown("---")
    st.markdown("### 📊 Risk Score Distribution")
    st.plotly_chart(plot_fraud_distribution(fraud_df), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Suspicious Wallet Explorer")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        risk_filter = st.select_slider("Minimum Risk Score",
                                       options=[0, 35, 60, 80], value=35,
                                       format_func=lambda v: {0:"All (0+)",35:"Medium+(35+)",60:"High+(60+)",80:"Critical(80+)"}[v])
    with col2:
        top_n = st.selectbox("Show top N", [10, 25, 50, 100], index=1)
    with col3:
        sort_fraud_by = st.selectbox("Sort by", ["risk_score", "fraud_score"], index=0)

    filtered = (fraud_df[fraud_df["risk_score"] >= risk_filter]
                .sort_values(sort_fraud_by, ascending=(sort_fraud_by == "fraud_score"))
                .head(top_n).copy())

    if "wallet_id" in filtered.columns:
        tx_rows = [{"Sent Txs": len(edges[edges["from_id"] == w]),
                    "Recv Txs": len(edges[edges["to_id"]   == w]),
                    "Total Txs": len(edges[edges["from_id"] == w]) + len(edges[edges["to_id"] == w])}
                   for w in filtered["wallet_id"]]
        filtered = pd.concat([filtered.reset_index(drop=True), pd.DataFrame(tx_rows)], axis=1)

    display_cols = {}
    if "wallet_address" in filtered.columns: display_cols["wallet_address"] = "Wallet Address"
    if "wallet_id"      in filtered.columns: display_cols["wallet_id"]      = "Wallet ID"
    display_cols.update({"risk_score": "Risk Score (0–100)", "risk_level": "Risk Level",
                          "fraud_score": "Raw IF Score"})
    for extra in ["Total Txs", "Sent Txs", "Recv Txs"]:
        if extra in filtered.columns: display_cols[extra] = extra

    disp = filtered[[c for c in display_cols if c in filtered.columns]].rename(columns=display_cols)
    col_cfg = {}
    if "Wallet Address"    in disp.columns: col_cfg["Wallet Address"]    = st.column_config.TextColumn("Wallet Address", width="large")
    if "Risk Score (0–100)" in disp.columns: col_cfg["Risk Score (0–100)"] = st.column_config.NumberColumn("Risk Score", format="%.1f")
    st.dataframe(disp, use_container_width=True, height=420, column_config=col_cfg)
    st.download_button("⬇️ Export Fraud Report (CSV)", disp.to_csv(index=False),
                       file_name="fraud_report.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("### 🕵️ Investigate Specific Wallet")

    inv_col1, inv_col2, inv_col3 = st.columns([3, 1, 1])
    with inv_col1:
        wallet_id_input = st.number_input("Enter Wallet ID (numeric)", min_value=0,
                                          max_value=int(embeddings.shape[0]-1),
                                          value=int(st.session_state.get("_inv_wallet", 0)),
                                          key="inv_wallet_id")
    with inv_col2:
        if st.button("🚨 Load Top Suspicious", use_container_width=True):
            st.session_state["_inv_wallet"] = int(fraud_df.sort_values("risk_score", ascending=False).iloc[0]["wallet_id"])
            st.rerun()
    with inv_col3:
        if st.button("🎲 Random Wallet", use_container_width=True):
            st.session_state["_inv_wallet"] = int(np.random.randint(0, embeddings.shape[0]))
            st.rerun()

    if st.button("🔍 Investigate Wallet", type="primary", use_container_width=True):
        wid = int(wallet_id_input)
        try:   addr = le.inverse_transform([wid])[0]
        except: addr = None

        info     = fraud_df[fraud_df["wallet_id"] == wid]
        sent_txs = edges[edges["from_id"] == wid]
        recv_txs = edges[edges["to_id"]   == wid]
        all_txs  = pd.concat([sent_txs, recv_txs]).drop_duplicates()

        sent_count = len(sent_txs); recv_count = len(recv_txs)
        total_txs  = sent_count + recv_count
        unique_counterparties = len(set(sent_txs["to_id"].tolist()) | set(recv_txs["from_id"].tolist()))
        all_counterparty_ids  = set(sent_txs["to_id"].tolist()) | set(recv_txs["from_id"].tolist())
        flagged_counterparties = all_counterparty_ids & FRAUD_IDS
        fraud_exposure_pct = round(len(flagged_counterparties) / max(len(all_counterparty_ids), 1) * 100, 1)
        tx_ratio = sent_count / max(recv_count, 1)
        is_flagged = len(info) > 0
        rs  = float(info.iloc[0]["risk_score"])  if is_flagged else 0.0
        rl  = info.iloc[0]["risk_level"]          if is_flagged else "Clean"
        raw = float(info.iloc[0]["fraud_score"])  if is_flagged else None
        emb = embeddings[wid]

        st.markdown("---")
        if addr:
            st.markdown(f'<span class="wallet-address-full">📍 {addr}</span>', unsafe_allow_html=True)

        if is_flagged:
            badge_color = {"Critical":"#dc2626","High":"#ef4444","Medium":"#d97706","Low":"#16a34a"}.get(rl,"#6366f1")
            rank_among_flagged = int((fraud_df["risk_score"] > rs).sum() + 1)
            pct_rank = round((1 - (rank_among_flagged-1) / max(len(fraud_df),1)) * 100, 1)
            st.markdown(f"""<div style="background:{badge_color}22;border:1px solid {badge_color}55;border-radius:12px;padding:0.9rem 1.2rem;margin:0.8rem 0;">
                <span style="font-size:1.15rem;font-weight:700;color:{badge_color};">
                ⚠️ FLAGGED — {rl} Risk &nbsp;
                <span style="font-size:0.85rem;font-weight:400;color:#94a3b8;">
                    Risk Score: {rs:.1f}/100 &nbsp;·&nbsp; Top #{rank_among_flagged} of {len(fraud_df):,} flagged wallets
                </span></span></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="background:#16a34a22;border:1px solid #16a34a44;border-radius:12px;padding:0.9rem 1.2rem;margin:0.8rem 0;">
                <span style="font-size:1.1rem;font-weight:700;color:#4ade80;">✅ CLEAN — No fraud flags detected</span></div>""",
                        unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Wallet ID",             f"#{wid:,}")
        m2.metric("Txs Sent",              f"{sent_count:,}")
        m3.metric("Txs Received",          f"{recv_count:,}")
        m4.metric("Unique Counterparties", f"{unique_counterparties:,}")
        m5.metric("Fraud Exposure",        f"{fraud_exposure_pct}%",
                  delta="⚠️ High" if fraud_exposure_pct > 30 else ("🟡 Moderate" if fraud_exposure_pct > 10 else "✅ Low"),
                  delta_color="off")

        tab_risk, tab_txn, tab_network, tab_embedding = st.tabs([
            "🔴 Risk Profile", "📋 Transaction Details", "🌐 Network Context", "🧬 Embedding Analysis"])

        with tab_risk:
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("#### Risk Breakdown")
                if is_flagged:
                    gauge_color = {"Critical":"#dc2626","High":"#ef4444","Medium":"#f59e0b","Low":"#22c55e"}.get(rl,"#6366f1")
                    st.markdown(f"""<div style="margin:0.5rem 0 1rem;">
                        <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#94a3b8;margin-bottom:4px;">
                            <span>Risk Score</span><span>{rs:.1f}/100</span></div>
                        <div style="background:rgba(255,255,255,0.06);border-radius:999px;height:14px;overflow:hidden;border:1px solid rgba(255,255,255,0.08);">
                            <div style="width:{rs}%;height:100%;background:linear-gradient(90deg,{gauge_color}99,{gauge_color});border-radius:999px;"></div>
                        </div></div>""", unsafe_allow_html=True)
                    st.metric("Raw Isolation Forest Score", f"{raw:.6f}" if raw else "N/A")
                else:
                    st.success("This wallet has **not** been flagged by the Isolation Forest model.")
            with rc2:
                st.markdown("#### Behavioural Signals")
                signals = []
                if tx_ratio > 5:    signals.append(("🚩","High send/receive ratio", f"{tx_ratio:.1f}x — predominantly outgoing"))
                elif tx_ratio < 0.2: signals.append(("🚩","High receive/send ratio", f"{1/max(tx_ratio,0.01):.1f}x — predominantly incoming"))
                else:                signals.append(("✅","Balanced send/receive",    f"Ratio: {tx_ratio:.2f}"))
                if fraud_exposure_pct > 30:   signals.append(("🚩","High fraud-network exposure",     f"{fraud_exposure_pct}% of counterparties are flagged"))
                elif fraud_exposure_pct > 10: signals.append(("⚠️","Moderate fraud-network exposure", f"{fraud_exposure_pct}% of counterparties are flagged"))
                else:                         signals.append(("✅","Low fraud-network exposure",       f"Only {fraud_exposure_pct}%"))
                for icon, label, detail in signals:
                    st.markdown(f"""<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.55rem 0.8rem;margin:5px 0;
                            border-left:3px solid {'#ef4444' if icon=='🚩' else '#f59e0b' if icon=='⚠️' else '#22c55e'};">
                        <span style="font-size:0.88rem;font-weight:600;">{icon} {label}</span><br>
                        <span style="font-size:0.78rem;color:#94a3b8;">{detail}</span></div>""", unsafe_allow_html=True)

        with tab_txn:
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown(f"#### 📤 Sent ({sent_count:,})")
                if sent_count > 0:
                    sd = sent_txs.copy()
                    try: sd["to_address"] = le.inverse_transform(sd["to_id"].astype(int))
                    except: pass
                    sd["counterparty_fraud"] = sd["to_id"].apply(lambda x: "🚨" if x in FRAUD_IDS else "✅")
                    st.dataframe(sd, use_container_width=True, height=300,
                                 column_config={"to_address": st.column_config.TextColumn("To Address", width="large"),
                                                "counterparty_fraud": st.column_config.TextColumn("Flag", width="small")})
                else:
                    st.info("No outgoing transactions.")
            with tc2:
                st.markdown(f"#### 📥 Received ({recv_count:,})")
                if recv_count > 0:
                    rd = recv_txs.copy()
                    try: rd["from_address"] = le.inverse_transform(rd["from_id"].astype(int))
                    except: pass
                    rd["counterparty_fraud"] = rd["from_id"].apply(lambda x: "🚨" if x in FRAUD_IDS else "✅")
                    st.dataframe(rd, use_container_width=True, height=300,
                                 column_config={"from_address": st.column_config.TextColumn("From Address", width="large"),
                                                "counterparty_fraud": st.column_config.TextColumn("Flag", width="small")})
                else:
                    st.info("No incoming transactions.")
            if total_txs > 0:
                st.markdown("---")
                exp = all_txs.copy()
                try:
                    exp["from_address"] = le.inverse_transform(exp["from_id"].astype(int))
                    exp["to_address"]   = le.inverse_transform(exp["to_id"].astype(int))
                except: pass
                st.download_button(f"⬇️ Export All {total_txs:,} Transactions (CSV)",
                                   exp.to_csv(index=False),
                                   file_name=f"wallet_{wid}_transactions.csv", mime="text/csv",
                                   use_container_width=True)

        with tab_network:
            nc1, nc2 = st.columns([2, 1])
            with nc1:
                st.markdown("#### Flagged Counterparties")
                if flagged_counterparties:
                    fp_rows = []
                    for fid in flagged_counterparties:
                        fi = fraud_df[fraud_df["wallet_id"] == fid]
                        try:    f_addr = le.inverse_transform([int(fid)])[0]
                        except: f_addr = f"ID {fid}"
                        dirs = []
                        if fid in set(sent_txs["to_id"]):   dirs.append("Sent To")
                        if fid in set(recv_txs["from_id"]): dirs.append("Received From")
                        fp_rows.append({"Wallet Address": f_addr, "Wallet ID": int(fid),
                                        "Risk Level": fi.iloc[0]["risk_level"] if len(fi)>0 else "Unknown",
                                        "Risk Score":  round(float(fi.iloc[0]["risk_score"]),1) if len(fi)>0 else 0,
                                        "Relationship": " & ".join(dirs)})
                    fp_df = pd.DataFrame(fp_rows).sort_values("Risk Score", ascending=False)
                    st.dataframe(fp_df, use_container_width=True, height=280,
                                 column_config={"Wallet Address": st.column_config.TextColumn("Wallet Address", width="large"),
                                                "Risk Score":     st.column_config.NumberColumn("Risk Score", format="%.1f")})
                else:
                    st.success("✅ No flagged wallets found among direct counterparties.")
            with nc2:
                st.markdown("#### Network Summary")
                st.metric("Total Counterparties",  f"{len(all_counterparty_ids):,}")
                st.metric("Flagged Counterparties", f"{len(flagged_counterparties):,}")
                st.metric("Unique Receivers",       f"{sent_txs['to_id'].nunique():,}")
                st.metric("Unique Senders",         f"{recv_txs['from_id'].nunique():,}")

        with tab_embedding:
            ec1, ec2 = st.columns(2)
            with ec1:
                emb_norm = float(np.linalg.norm(emb))
                st.markdown("#### GNN Embedding Statistics")
                st.metric("Embedding Dimensions", f"{len(emb)}")
                st.metric("L2 Norm",    f"{emb_norm:.4f}")
                st.metric("Mean Value", f"{float(np.mean(emb)):.4f}")
                st.metric("Std Dev",    f"{float(np.std(emb)):.4f}")

                st.markdown("#### 🔗 Top 5 Similar Wallets (Cosine)")
                sample_ids  = np.random.choice(embeddings.shape[0], min(2000, embeddings.shape[0]), replace=False)
                sample_embs = embeddings[sample_ids]
                cosines     = sample_embs @ emb / (np.linalg.norm(sample_embs, axis=1) * (emb_norm + 1e-9) + 1e-9)
                top5_local  = np.argsort(cosines)[::-1][1:6]
                top5_ids    = sample_ids[top5_local]
                sim_rows = []
                for sid, li in zip(top5_ids, top5_local):
                    si = fraud_df[fraud_df["wallet_id"] == sid]
                    try:    sa = le.inverse_transform([int(sid)])[0]
                    except: sa = f"ID {sid}"
                    sim_rows.append({"Wallet Address": sa, "Wallet ID": int(sid),
                                     "Cosine Similarity": round(float(cosines[li]),4),
                                     "Risk Level": si.iloc[0]["risk_level"] if len(si)>0 else "Clean"})
                st.dataframe(pd.DataFrame(sim_rows), use_container_width=True,
                             column_config={"Wallet Address": st.column_config.TextColumn("Wallet Address", width="large"),
                                            "Cosine Similarity": st.column_config.NumberColumn("Cosine Sim", format="%.4f")},
                             hide_index=True)
            with ec2:
                st.markdown("#### Embedding Heatmap (first 64 dims)")
                heatmap_vals = emb[:64].reshape(8, 8)
                fig_emb = go.Figure(go.Heatmap(z=heatmap_vals, colorscale="RdBu", zmid=0, showscale=True))
                fig_emb.update_layout(template=PLOTLY_TEMPLATE, height=320,
                                      margin=dict(t=10, b=10, l=10, r=10),
                                      xaxis=dict(showticklabels=False),
                                      yaxis=dict(showticklabels=False))
                st.plotly_chart(fig_emb, use_container_width=True)

        st.markdown("---")
        report_data = {"wallet_id":[wid],"wallet_address":[addr or "N/A"],"is_flagged":[is_flagged],
                       "risk_level":[rl],"risk_score":[rs],"raw_if_score":[raw],
                       "txs_sent":[sent_count],"txs_received":[recv_count],"total_txs":[total_txs],
                       "unique_counterparties":[unique_counterparties],"flagged_counterparties":[len(flagged_counterparties)],
                       "fraud_exposure_pct":[fraud_exposure_pct],"send_recv_ratio":[round(tx_ratio,4)]}
        st.download_button("⬇️ Export Full Investigation Report (CSV)",
                           pd.DataFrame(report_data).to_csv(index=False),
                           file_name=f"investigation_wallet_{wid}.csv", mime="text/csv",
                           use_container_width=True)


# ═══════════════════════════════════════════════
# SECTION: Model Performance
# ═══════════════════════════════════════════════
elif "📈 Model Performance" in section:
    st.markdown('<p class="main-header">📈 Model Performance Metrics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive GNN evaluation — ROC-AUC, training loss curve & architecture details</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📉 Training Loss Curve")
        if loss_history is not None:
            st.plotly_chart(plot_loss_curve(loss_history), use_container_width=True)
            with st.expander("Loss Statistics"):
                reduction = (loss_history[0] - loss_history[-1]) / loss_history[0] * 100
                c1, c2, c3 = st.columns(3)
                c1.metric("Initial Loss", f"{loss_history[0]:.4f}")
                c2.metric("Final Loss",   f"{loss_history[-1]:.4f}")
                c3.metric("Reduction",    f"{reduction:.1f}%")
                st.write(f"Total epochs: **{len(loss_history)}**")
        else:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.info("Loss history not available. Save `loss_history.npy` during training.")
            st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("### 📊 ROC Curve")
        if all(x is not None for x in [fpr, tpr, roc_auc_val]):
            st.plotly_chart(plot_roc_curve(fpr, tpr, roc_auc_val), use_container_width=True)
            with st.expander("ROC Statistics"):
                st.metric("AUC Score", f"{roc_auc_val:.4f}")
                if   roc_auc_val > 0.9: st.success("🟢 Excellent — AUC > 0.9")
                elif roc_auc_val > 0.7: st.success("🟢 Good — AUC > 0.7")
                elif roc_auc_val > 0.6: st.warning("🟡 Moderate — AUC > 0.6")
                else:                   st.error("🔴 Poor — AUC ≤ 0.6")
                st.write(f"Distinguishes linked wallet pairs with **{roc_auc_val*100:.1f}%** accuracy.")
        else:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.info("ROC data not available. Save `roc_data.npz` after evaluation.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏗️ Model Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### GraphSAGE Configuration")
        st.code("Model          : GraphSAGE\nLayers         : 2\nHidden channels: 64\n"
                "Input features : 2  (in/out degree)\nOutput embedding: 64 dimensions\n"
                "Decoder        : Multi-signal (dot + cosine + L2)", language="text")
    with col2:
        st.markdown("#### Training Configuration")
        st.code("Optimiser      : Adam\nLearning rate  : 0.01\nLoss function  : Binary Cross-Entropy\n"
                "Epochs         : 100\nTrain/Val/Test : 70 / 15 / 15\nNegative sampling: Dynamic", language="text")

    st.markdown("---")
    st.markdown("### 📊 Evaluation Summary")
    metrics_df = pd.DataFrame({
        "Metric": ["ROC-AUC","Model Type","Embedding Dim","Total Nodes","Total Edges","Decoder"],
        "Value":  [f"{roc_auc_val:.4f}" if roc_auc_val else "N/A", "GraphSAGE", "64",
                   f"{embeddings.shape[0]:,}", f"{len(edges):,}", "Multi-signal (dot + cosine + L2)"],
        "Status": ["✅","✅","✅","✅","✅","✅"],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# SECTION: Network Visualization
# ═══════════════════════════════════════════════
elif "🌐 Network Visualization" in section:
    st.markdown('<p class="main-header">🌐 Network Visualization</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore the live transaction graph · Directional edges · Fraud-aware node colouring</p>', unsafe_allow_html=True)

    vc1, vc2, vc3 = st.columns([1, 1, 1])
    with vc1:
        wallet_id_viz = st.number_input("Centre Wallet ID", min_value=0,
                                        max_value=int(embeddings.shape[0]-1),
                                        value=int(st.session_state.get("_viz_wallet", 0)),
                                        key="viz_wallet_id")
    with vc2:
        if st.button("🚨 Jump to Top Fraud Wallet", use_container_width=True):
            st.session_state["_viz_wallet"] = int(fraud_df.sort_values("risk_score", ascending=False).iloc[0]["wallet_id"])
            st.rerun()
    with vc3:
        if st.button("🎲 Random Wallet", use_container_width=True, key="viz_rand"):
            st.session_state["_viz_wallet"] = int(np.random.randint(0, embeddings.shape[0]))
            st.rerun()

    with st.expander("⚙️ Graph Settings", expanded=True):
        gs1, gs2, gs3, gs4 = st.columns(4)
        with gs1: max_connections = st.slider("Max edges shown", 10, 100, 40, step=5)
        with gs2: depth = st.selectbox("Hop depth", ["1-hop", "2-hop"], index=0)
        with gs3: layout_algo = st.selectbox("Layout", ["spring","kamada","circular","shell"], index=0)
        with gs4: show_labels = st.toggle("Show labels", value=True)

    if st.button("🌐 Generate Network Graph", type="primary", use_container_width=True):
        wid_viz = int(wallet_id_viz)
        with st.spinner("Building graph …"):
            if depth == "2-hop":
                hop1 = edges[(edges["from_id"] == wid_viz) | (edges["to_id"] == wid_viz)]
                hop1_nodes = set(hop1["from_id"].tolist() + hop1["to_id"].tolist())
                graph_edges = pd.concat([hop1, edges[edges["from_id"].isin(hop1_nodes) | edges["to_id"].isin(hop1_nodes)]]).drop_duplicates()
            else:
                graph_edges = edges[(edges["from_id"] == wid_viz) | (edges["to_id"] == wid_viz)]

            fig = create_network_subgraph(graph_edges, wid_viz, le, fraud_df,
                                          max_connections=max_connections, layout=layout_algo, show_labels=show_labels)
        if not fig:
            st.warning(f"No transactions found for wallet ID {wid_viz}.")
            st.stop()

        try:    centre_addr = le.inverse_transform([wid_viz])[0]
        except: centre_addr = None
        if centre_addr:
            st.markdown(f'<span class="wallet-address-full">📍 {centre_addr}</span>', unsafe_allow_html=True)

        fraud_info = fraud_df[fraud_df["wallet_id"] == wid_viz]
        if wid_viz in FRAUD_IDS and len(fraud_info) > 0:
            rs_c = float(fraud_info.iloc[0]["risk_score"])
            rl_c = fraud_info.iloc[0]["risk_level"]
            badge_color = {"Critical":"#dc2626","High":"#ef4444","Medium":"#d97706","Low":"#16a34a"}.get(rl_c,"#6366f1")
            st.markdown(f'<div style="background:{badge_color}22;border:1px solid {badge_color}55;border-radius:10px;padding:0.6rem 1rem;margin:0.5rem 0;"><b style="color:{badge_color};">⚠️ {rl_c} Risk — Score {rs_c:.1f}/100</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#16a34a22;border:1px solid #16a34a44;border-radius:10px;padding:0.6rem 1rem;margin:0.5rem 0;"><b style="color:#4ade80;">✅ Clean Wallet</b></div>', unsafe_allow_html=True)

        st.plotly_chart(fig, use_container_width=True)

        out_txs = edges[edges["from_id"] == wid_viz]
        in_txs  = edges[edges["to_id"]   == wid_viz]
        all_cp  = set(out_txs["to_id"].tolist()) | set(in_txs["from_id"].tolist())
        fraud_cp = all_cp & FRAUD_IDS
        fraud_pct = round(len(fraud_cp) / max(len(all_cp), 1) * 100, 1)
        km1, km2, km3, km4, km5, km6 = st.columns(6)
        km1.metric("Outgoing Txs",          f"{len(out_txs):,}")
        km2.metric("Incoming Txs",          f"{len(in_txs):,}")
        km3.metric("Total Txs",             f"{len(out_txs)+len(in_txs):,}")
        km4.metric("Unique Counterparties", f"{len(all_cp):,}")
        km5.metric("Flagged Counterparties",f"{len(fraud_cp):,}")
        km6.metric("Fraud Exposure",        f"{fraud_pct}%",
                   delta="⚠️ High" if fraud_pct > 30 else ("🟡 Moderate" if fraud_pct > 10 else "✅ Low"),
                   delta_color="off")

    st.markdown("---")
    st.markdown("### 🚨 Top Suspicious Wallet Networks")
    sp1, sp2, sp3 = st.columns([2, 1, 1])
    with sp1: top_n_fraud  = st.selectbox("Show top N suspicious", [3, 5, 10], index=1)
    with sp2: susp_max_conn = st.slider("Max edges/graph", 10, 40, 15, key="susp_conn")
    with sp3: susp_layout   = st.selectbox("Layout", ["spring","kamada","circular"], key="susp_layout")

    if st.button("🔍 Render Suspicious Networks", use_container_width=True):
        for _, row in fraud_df.sort_values("risk_score", ascending=False).head(top_n_fraud).iterrows():
            fw_id = int(row["wallet_id"])
            try:    label_str = f"{le.inverse_transform([fw_id])[0][:20]}… — {row['risk_level']} Risk ({row['risk_score']:.1f}/100)"
            except: label_str = f"Wallet {fw_id} — {row['risk_level']} Risk ({row['risk_score']:.1f}/100)"
            with st.expander(label_str, expanded=False):
                sub_fig = create_network_subgraph(edges, fw_id, le, fraud_df,
                                                  max_connections=susp_max_conn, layout=susp_layout, show_labels=True)
                if sub_fig:
                    st.plotly_chart(sub_fig, use_container_width=True)
                else:
                    st.info("No transactions found.")


# ═══════════════════════════════════════════════
# SECTION: Live Blockchain Explorer  (NEW)
# ═══════════════════════════════════════════════
elif "🔌 Live Blockchain Explorer" in section:
    st.markdown('<p class="main-header">🔌 Live Blockchain Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Query any Ethereum address in real time · Smart contract detection · DeFi protocol recognition · GNN cross-reference</p>', unsafe_allow_html=True)

    etherscan_key = os.environ.get("ETHERSCAN_API_KEY", "")

    if not etherscan_key:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("""
### ⚠️ Etherscan API Key Not Configured

To enable live blockchain lookups, add your free Etherscan API key:

1. Go to [etherscan.io/register](https://etherscan.io/register) and create a free account
2. Navigate to **My Profile → API Keys** and generate a new key
3. In Replit, open the **Secrets** tab (🔒) and add:
   - **Key:** `ETHERSCAN_API_KEY`
   - **Value:** your API key
4. The app will reload automatically

**Free tier:** 5 calls/sec · 100,000 calls/day — more than enough for this dashboard.
""")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Address Input ────────────────────────────────────────────────────
    col1, col2 = st.columns([4, 1])
    with col1:
        lookup_addr = st.text_input(
            "🔍 Enter any Ethereum Address",
            value=st.session_state.get("live_lookup_addr", ""),
            placeholder="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            help="Works for both wallet addresses (EOA) and smart contract addresses",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        sample_btn = st.button("📋 Try Sample", use_container_width=True)
        if sample_btn:
            lookup_addr = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"  # vitalik.eth

    if lookup_addr:
        if not is_valid_eth_address(lookup_addr.strip()):
            st.markdown('<p class="invalid-address">⚠ Invalid format — must be 0x followed by 40 hex characters</p>', unsafe_allow_html=True)
        else:
            addr_lower = lookup_addr.strip().lower()
            addr_display = lookup_addr.strip()

            st.markdown(f'<span class="wallet-address-full">📍 {addr_display}</span>', unsafe_allow_html=True)

            # ── Etherscan fetch (only if key exists) ────────────────────
            if etherscan_key:
                with st.spinner("Querying Ethereum network …"):
                    balance    = fetch_eth_balance(addr_display)
                    is_contract, contract_name = check_is_contract(addr_display)
                    tx_df      = fetch_transactions(addr_display, limit=25)
                    token_df   = fetch_token_transfers(addr_display, limit=15)
                    tx_count   = fetch_tx_count(addr_display)

                # ── Address type badge ───────────────────────────────────
                if is_contract:
                    badge_label = f"Smart Contract" + (f" — {contract_name}" if contract_name else "")
                    st.markdown(f'<span class="contract-badge">📄 {badge_label}</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="eoa-badge">👤 Externally Owned Account (EOA)</span>', unsafe_allow_html=True)

                # ── Top metrics ──────────────────────────────────────────
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("ETH Balance",
                          f"{balance:.6f} ETH" if balance is not None else "N/A",
                          delta=f"≈ ${balance * 2500:,.0f}" if balance else None,
                          delta_color="off")
                m2.metric("Total Nonce (Sent Txs)", f"{tx_count:,}" if tx_count is not None else "N/A")
                m3.metric("Address Type",   "Contract 📄" if is_contract else "EOA 👤")
                m4.metric("Recent Txs Fetched", f"{len(tx_df):,}")

                if not tx_df.empty:
                    protocols = detect_protocols_from_txs(tx_df)
                    if protocols:
                        st.markdown("**🏦 DeFi / Protocol Interactions Detected:**")
                        proto_html = " ".join(f'<span class="protocol-tag">{p}</span>' for p in protocols)
                        st.markdown(proto_html, unsafe_allow_html=True)
                        st.markdown("")

                # ── Tabs for different data ──────────────────────────────
                tab_tx, tab_tokens, tab_gnn, tab_contract = st.tabs([
                    "📋 Transactions", "🪙 Token Transfers", "🤖 GNN Cross-Reference", "📄 Contract Info"])

                with tab_tx:
                    st.markdown(f"#### Recent Transactions ({len(tx_df):,})")
                    if not tx_df.empty:
                        # Highlight if sent to known protocol
                        display_tx = tx_df.drop(columns=["Full Hash"], errors="ignore")
                        st.dataframe(display_tx, use_container_width=True, height=420,
                                     column_config={
                                         "From":     st.column_config.TextColumn("From",     width="large"),
                                         "To":       st.column_config.TextColumn("To",       width="large"),
                                         "Value (ETH)":     st.column_config.NumberColumn("Value (ETH)", format="%.6f"),
                                         "Gas Price (Gwei)": st.column_config.NumberColumn("Gas (Gwei)", format="%.2f"),
                                         "Protocol": st.column_config.TextColumn("Protocol", width="medium"),
                                         "Status":   st.column_config.TextColumn("Status",   width="small"),
                                     })

                        # Transaction value chart
                        if tx_df["Value (ETH)"].sum() > 0:
                            non_zero = tx_df[tx_df["Value (ETH)"] > 0].head(20)
                            if not non_zero.empty:
                                fig_tx = px.bar(non_zero, x="Time", y="Value (ETH)",
                                                color="Status", title="ETH Value per Transaction",
                                                color_discrete_map={"✅": SUCCESS_COLOR, "❌": DANGER_COLOR})
                                fig_tx.update_layout(template=PLOTLY_TEMPLATE, height=320)
                                st.plotly_chart(fig_tx, use_container_width=True)

                        st.download_button("⬇️ Export Transactions (CSV)",
                                           tx_df.to_csv(index=False),
                                           file_name=f"live_txs_{addr_display[:10]}.csv",
                                           mime="text/csv")
                    else:
                        st.info("No recent transactions found, or API limit reached.")

                with tab_tokens:
                    st.markdown(f"#### ERC-20 Token Transfers ({len(token_df):,})")
                    if not token_df.empty:
                        # Token breakdown pie
                        token_counts = token_df.groupby("Token").size().reset_index(name="Count")
                        col_a, col_b = st.columns([2, 1])
                        with col_a:
                            st.dataframe(token_df, use_container_width=True, height=380,
                                         column_config={
                                             "From":     st.column_config.TextColumn("From",     width="large"),
                                             "To":       st.column_config.TextColumn("To",       width="large"),
                                             "Contract": st.column_config.TextColumn("Contract", width="large"),
                                             "Value":    st.column_config.NumberColumn("Value",  format="%.4f"),
                                         })
                        with col_b:
                            if len(token_counts) > 0:
                                fig_pie = px.pie(token_counts, names="Token", values="Count",
                                                 title="Token Distribution",
                                                 color_discrete_sequence=px.colors.qualitative.Set3)
                                fig_pie.update_layout(template=PLOTLY_TEMPLATE, height=320)
                                st.plotly_chart(fig_pie, use_container_width=True)
                        st.download_button("⬇️ Export Token Transfers (CSV)",
                                           token_df.to_csv(index=False),
                                           file_name=f"live_tokens_{addr_display[:10]}.csv",
                                           mime="text/csv")
                    else:
                        st.info("No ERC-20 token transfers found.")

                with tab_gnn:
                    st.markdown("#### 🤖 GNN Model Cross-Reference")
                    addr_normalised = addr_display.lower()
                    # Try to find the address in the dataset (case-insensitive)
                    matching = [a for a in WALLET_SET if a.lower() == addr_normalised]

                    if matching:
                        matched_addr = matching[0]
                        wallet_id_found = int(le.transform([matched_addr])[0])
                        wallet_fraud_info = fraud_df[fraud_df["wallet_id"] == wallet_id_found]

                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success(f"✅ Address found in GNN dataset! Wallet ID: **{wallet_id_found}**")
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("")

                        gnn_col1, gnn_col2 = st.columns(2)
                        with gnn_col1:
                            st.markdown("**GNN Dataset Stats**")
                            sent_in_dataset = edges[edges["from_id"] == wallet_id_found]
                            recv_in_dataset = edges[edges["to_id"]   == wallet_id_found]
                            st.metric("Txs in GNN Dataset (sent)",     f"{len(sent_in_dataset):,}")
                            st.metric("Txs in GNN Dataset (received)", f"{len(recv_in_dataset):,}")
                            st.metric("Unique Counterparties",         f"{len(set(sent_in_dataset['to_id'].tolist()) | set(recv_in_dataset['from_id'].tolist())):,}")

                        with gnn_col2:
                            st.markdown("**Fraud Risk Assessment**")
                            if len(wallet_fraud_info) > 0:
                                rs_gnn = float(wallet_fraud_info.iloc[0]["risk_score"])
                                rl_gnn = wallet_fraud_info.iloc[0]["risk_level"]
                                gauge_color = {"Critical":"#dc2626","High":"#ef4444","Medium":"#d97706","Low":"#16a34a"}.get(rl_gnn,"#6366f1")
                                st.markdown(f'<div class="fraud-alert">', unsafe_allow_html=True)
                                st.metric("Risk Score", f"{rs_gnn:.1f} / 100")
                                st.metric("Risk Level", rl_gnn)
                                st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                                st.metric("Risk Score", "0 / 100")
                                st.metric("Risk Level", "Clean ✅")
                                st.markdown("</div>", unsafe_allow_html=True)

                        # Link prediction against random partners from dataset
                        st.markdown("---")
                        st.markdown("**🔮 Quick Link Prediction (vs. random dataset wallets)**")
                        if st.button("Run Quick Predictions", key="gnn_quick_pred"):
                            sample_ids = np.random.choice(embeddings.shape[0], 8, replace=False)
                            qp_rows = []
                            for sid in sample_ids:
                                prob = compute_link_probability(embeddings[wallet_id_found], embeddings[sid])
                                try:    s_addr = le.inverse_transform([int(sid)])[0]
                                except: s_addr = f"ID {sid}"
                                si = fraud_df[fraud_df["wallet_id"] == sid]
                                qp_rows.append({
                                    "Target Wallet": s_addr,
                                    "Link Probability": round(prob, 4),
                                    "Target Risk":  si.iloc[0]["risk_level"] if len(si)>0 else "Clean",
                                })
                            qp_df = pd.DataFrame(qp_rows).sort_values("Link Probability", ascending=False)
                            st.dataframe(qp_df, use_container_width=True,
                                         column_config={
                                             "Target Wallet":      st.column_config.TextColumn("Target Wallet", width="large"),
                                             "Link Probability":   st.column_config.NumberColumn("Probability", format="%.4f"),
                                         }, hide_index=True)

                        # Embedding heatmap
                        st.markdown("---")
                        st.markdown("**🧬 GNN Node Embedding (first 64 dims)**")
                        emb_live = embeddings[wallet_id_found]
                        fig_emb_live = go.Figure(go.Heatmap(z=emb_live[:64].reshape(8,8),
                                                             colorscale="RdBu", zmid=0, showscale=True))
                        fig_emb_live.update_layout(template=PLOTLY_TEMPLATE, height=300,
                                                    margin=dict(t=10,b=10,l=10,r=10),
                                                    xaxis=dict(showticklabels=False),
                                                    yaxis=dict(showticklabels=False))
                        st.plotly_chart(fig_emb_live, use_container_width=True)

                    else:
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.info("""
**Address not found in the GNN training dataset.**

This is expected — the GNN model was trained on a specific historical snapshot of ~25,542 Ethereum wallets.
New addresses, recently active wallets, or addresses not in the original BigQuery export won't appear here.

You can still see real-time on-chain data in the Transactions and Token Transfers tabs above.
""")
                        st.markdown("</div>", unsafe_allow_html=True)

                with tab_contract:
                    st.markdown("#### 📄 Contract / Address Details")
                    if is_contract:
                        st.markdown(f'<span class="contract-badge">📄 Smart Contract{(" — " + contract_name) if contract_name else ""}</span>', unsafe_allow_html=True)
                        st.markdown("")

                        known_proto = DEFI_PROTOCOLS.get(addr_lower)
                        if known_proto:
                            st.success(f"🏦 **Known Protocol:** {known_proto}")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("**What is a Smart Contract?**")
                            st.markdown("""
A smart contract is a self-executing program stored on the Ethereum blockchain.
Unlike EOA wallets controlled by private keys, contracts run automatically when
triggered by transactions.

**Common contract types:**
- 🔄 DEX/AMM (e.g., Uniswap pools)
- 💰 Lending protocols (e.g., Aave, Compound)
- 🎨 NFT contracts (ERC-721/ERC-1155)
- 🏦 Multisig wallets (e.g., Gnosis Safe)
- 🪙 Token contracts (ERC-20)
""")
                        with col_b:
                            if known_proto:
                                st.markdown("**Protocol Classification:**")
                                category = "DEX / AMM" if any(x in known_proto for x in ["Uniswap","SushiSwap","Curve","Balancer"]) \
                                    else "Lending" if any(x in known_proto for x in ["Aave","Compound","Maker"]) \
                                    else "NFT Marketplace" if "OpenSea" in known_proto \
                                    else "Aggregator" if any(x in known_proto for x in ["1inch","0x","MetaMask"]) \
                                    else "Token / Asset"
                                st.metric("Category", category)
                                st.metric("Protocol", known_proto)
                            else:
                                st.markdown("**Etherscan Links:**")
                                etherscan_url = f"https://etherscan.io/address/{addr_display}"
                                st.markdown(f"[🔗 View on Etherscan]({etherscan_url})")
                    else:
                        st.markdown('<span class="eoa-badge">👤 Externally Owned Account (EOA)</span>', unsafe_allow_html=True)
                        st.markdown("")
                        st.markdown("""
**About this address:**

This is an Externally Owned Account — a wallet controlled by a private key (e.g., MetaMask, Ledger).
EOAs can:
- Hold ETH and tokens
- Send and receive transactions
- Interact with smart contracts

Unlike contracts, EOAs have no code and are fully controlled by whoever holds the private key.
""")
                        etherscan_url = f"https://etherscan.io/address/{addr_display}"
                        st.markdown(f"[🔗 View full history on Etherscan]({etherscan_url})")

            else:
                # No API key — show what we can from GNN dataset
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.info("Live blockchain data requires an Etherscan API key. Showing GNN dataset data only.")
                st.markdown("</div>", unsafe_allow_html=True)

                addr_normalised = lookup_addr.strip().lower()
                matching = [a for a in WALLET_SET if a.lower() == addr_normalised]
                if matching:
                    matched_addr = matching[0]
                    wallet_id_found = int(le.transform([matched_addr])[0])
                    wallet_fraud_info = fraud_df[fraud_df["wallet_id"] == wallet_id_found]
                    st.success(f"✅ Found in GNN dataset — Wallet ID: **{wallet_id_found}**")

                    gnn_col1, gnn_col2 = st.columns(2)
                    with gnn_col1:
                        sent_in_ds = edges[edges["from_id"] == wallet_id_found]
                        recv_in_ds = edges[edges["to_id"]   == wallet_id_found]
                        st.metric("Dataset Txs (sent)",     f"{len(sent_in_ds):,}")
                        st.metric("Dataset Txs (received)", f"{len(recv_in_ds):,}")
                    with gnn_col2:
                        if len(wallet_fraud_info) > 0:
                            st.metric("Risk Score", f"{wallet_fraud_info.iloc[0]['risk_score']:.1f}/100")
                            st.metric("Risk Level", wallet_fraud_info.iloc[0]["risk_level"])
                        else:
                            st.metric("Risk Score", "0/100")
                            st.metric("Risk Level", "Clean ✅")
                else:
                    st.info("Address not found in the GNN dataset either. Add an Etherscan API key to query live blockchain data.")


# ═══════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════
st.markdown("---")
st.markdown(f"""<div class="footer-text">
    <strong>Transaction Link Prediction in Blockchain using Graph Neural Networks</strong><br>
    Powered by GraphSAGE &nbsp;·&nbsp; Ethereum Mainnet &nbsp;·&nbsp; Built with Streamlit<br>
    <span style="color:#334155;font-size:0.75rem;">
        Dashboard v3.0 &nbsp;·&nbsp; {datetime.now().strftime("%Y-%m-%d")} &nbsp;·&nbsp;
        Live Explorer via Etherscan API &nbsp;·&nbsp;
        {'🌙 Dark Mode' if st.session_state.dark_mode else '☀️ Light Mode'}
    </span>
</div>""", unsafe_allow_html=True)
