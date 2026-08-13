# ChainIntel Pro — Blockchain Intelligence Platform v5.0

## Project Overview
An enterprise-grade Blockchain Transaction Link Prediction and Fraud Detection Dashboard. Uses GraphSAGE Graph Neural Networks to analyse Ethereum blockchain transactions, with live on-chain data via Etherscan API and Web3.py. iOS 16-inspired premium design system — bright light mode + deep-space dark mode. B2B/enterprise-ready.

## Architecture

### Application Structure
```
app.py                   — Main Streamlit dashboard (~1,630 lines, modular)
utils/
  theme.py               — iOS 16 design system: colour palettes + full CSS injection
  ml_utils.py            — ML helpers: PCA, link probability, fraud scoring
  blockchain.py          — Etherscan REST API + Web3.py event log helpers + API key resolver
  viz.py                 — All Plotly visualisations (charts, graphs, diagrams)
  __init__.py            — Package init
```

### Data Files
- `node_embeddings.npy`   — Pre-trained 64-dim GNN node embeddings (25,542 × 64)
- `edges.csv`             — Directed transaction edge list (from_id, to_id)
- `fraudulent_wallets.csv`— Isolation Forest anomaly scores + risk levels
- `label_encoder.pkl`     — LabelEncoder: wallet address ↔ numeric ID
- `loss_history.npy`      — (optional) GNN training loss per epoch
- `roc_data.npz`          — (optional) ROC curve data (fpr, tpr, auc)

## Design System (v5.0 — iOS 16 Inspired)

### Light Mode — "Premium Bright"
- Background: `#F0F4FF` warm-cool tinted white with subtle blue-purple gradient
- Cards: `#FFFFFF` pure white, 20px border-radius, layered iOS-style shadows
- Primary: `#007AFF` (iOS System Blue)
- Accent: `#5856D6` (iOS Indigo)
- Success: `#34C759` · Danger: `#FF3B30` · Warning: `#FF9500`

### Dark Mode — "Deep Space"
- Background: `#07080F` true deep-space black with subtle navy gradient
- Cards: `#111220` elevated surfaces
- Primary: `#0A84FF` · Accent: `#5E5CE6`
- Success: `#30D158` · Danger: `#FF453A` · Warning: `#FF9F0A`

### Key CSS Features
- Glassmorphism: `backdrop-filter: saturate(180%) blur(24px)` on sidebar + hero
- iOS-style animated status dots with glow pulse
- `cubic-bezier(0.34, 1.56, 0.64, 1)` spring animations on card hover
- Gradient text for all headers: `#007AFF → #5856D6 → #AF52DE`
- Premium metric cards with hover lift and glow shadow

## Technology Stack
- **Framework**: Streamlit (port 5000)
- **ML**: scikit-learn (PCA, Isolation Forest)
- **Graph**: NetworkX, PyTorch Geometric (training only)
- **Visualisation**: Plotly, NetworkX
- **Live Blockchain**: Etherscan REST API + Web3.py (eth_getLogs RPC)
- **Model**: GraphSAGE 2-layer, 64-dim embeddings, multi-signal decoder
- **Fonts**: Inter (variable) + JetBrains Mono via Google Fonts

## Dashboard Sections (9 total)
1. 🏠 Overview — B2B hero banner, value propositions, connection status, methodology
2. 📊 Graph Analytics — degree distributions, transaction explorer
3. 🔮 Link Prediction — predict future transaction probability (manual + random)
4. 🚨 Fraud Detection — risk table, wallet investigation, embedding analysis
5. 📈 Model Performance — ROC-AUC, training loss, evaluation summary
6. 🏗️ Architecture & ML — live interactive GNN diagram, formulas, decoder analysis
7. 🧬 Embedding Space — 2D/3D PCA scatter of all 25K wallets, nearest-neighbour search
8. 🌐 Network Visualization — directed transaction subgraph around any wallet
9. 🔌 Live Blockchain Explorer — Etherscan balance/txs + Web3.py event logs + GNN cross-ref

## API Key Configuration
Etherscan API key resolved in this priority order (utils/blockchain.py):
1. `ETHERSCAN_API_KEY` Replit Secret (highest priority — override in Secrets tab)
2. `ETHERSCAN_API_KEY_DEFAULT` env var (set via Replit env vars)
3. No key — Etherscan's stricter anonymous rate limit applies

> ⚠️ A hard-coded fallback key was previously documented here. It has been removed.
> Set `ETHERSCAN_API_KEY` yourself; get a free one at https://etherscan.io/myapikey

## Secrets & Environment Variables
- `ETHERSCAN_API_KEY` — override API key (optional, fallback pre-configured)
- `ETHERSCAN_API_KEY_DEFAULT` — secondary fallback (set to built-in key)
- `WEB3_PROVIDER_URL` — override Web3 RPC (default: public BlastAPI endpoint)

## Configuration
- Streamlit config: `.streamlit/config.toml` — port 5000, host 0.0.0.0
- CORS and XSRF protection disabled for Replit proxy

## Running
- Workflow "Start application": `streamlit run app.py`
- Port: 5000

## Key Design Decisions
- iOS 16 colour system with full light + dark variants across all 20 palette keys
- Plotly colours always use `rgba()` format (never 8-digit hex — Plotly validation)
- `@st.cache_data` on all expensive calls (data loading, PCA, Etherscan API)
- Web3.py decodes raw `eth_getLogs` Transfer events — no external ABI needed
- Multi-signal decoder: dot(0.5) + cosine(0.3) + L2(0.2) for link probability
- Risk scores normalised 0–100 from Isolation Forest anomaly scores
- B2B overview: hero banner + 4-column value props + connection status grid
- Sidebar: glassmorphism frosted glass, product logo, animated connection pills
