# Blockchain GNN Analytics Dashboard — v4.0

## Project Overview
An interactive Blockchain Transaction Link Prediction and Fraud Detection Dashboard built with Streamlit. Uses GraphSAGE Graph Neural Networks to analyse Ethereum blockchain transactions, with live on-chain data via Etherscan API and Web3.py.

## Architecture

### Application Structure
```
app.py                   — Main Streamlit dashboard (~550 lines, modular)
utils/
  theme.py               — CSS injection + colour palettes (DARK/LIGHT)
  ml_utils.py            — ML helpers: PCA, link probability, fraud scoring
  blockchain.py          — Etherscan REST API + Web3.py event log helpers
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

## Technology Stack
- **Framework**: Streamlit
- **ML**: scikit-learn (PCA, Isolation Forest)
- **Graph**: NetworkX, PyTorch Geometric (training only)
- **Visualisation**: Plotly, NetworkX
- **Live Blockchain**: Etherscan REST API + Web3.py (eth_getLogs RPC)
- **Model**: GraphSAGE 2-layer, 64-dim embeddings, multi-signal decoder

## Dashboard Sections (9 total)
1. 🏠 Overview — project summary, methodology, tech stack
2. 📊 Graph Analytics — degree distributions, transaction explorer
3. 🔮 Link Prediction — predict future transaction probability (manual + random)
4. 🚨 Fraud Detection — risk table, wallet investigation, embedding analysis
5. 📈 Model Performance — ROC-AUC, training loss, evaluation summary
6. 🏗️ Architecture & ML — live interactive GNN diagram, formulas, decoder analysis
7. 🧬 Embedding Space — 2D/3D PCA scatter of all 25K wallets, nearest-neighbour search
8. 🌐 Network Visualization — directed transaction subgraph around any wallet
9. 🔌 Live Blockchain Explorer — Etherscan balance/txs + Web3.py event logs + GNN cross-ref

## Configuration
- Streamlit config: `.streamlit/config.toml` — port 5000, host 0.0.0.0
- CORS and XSRF protection disabled for Replit proxy
- `allowedHosts = "all"` for Replit iframe preview

## Secrets
- `ETHERSCAN_API_KEY` — required for live blockchain data (Etherscan REST API)
- `WEB3_PROVIDER_URL` — optional override for Web3 RPC (default: public BlastAPI)

## Running
- Workflow "Start application": `streamlit run app.py`
- Port: 5000

## Key Design Decisions
- All CSS in `utils/theme.py` — supports dark (GitHub Obsidian) and light (GitHub Pearl) themes
- `@st.cache_data` on all expensive calls (data loading, PCA, Etherscan API)
- Web3.py decodes raw `eth_getLogs` Transfer events — no external ABI needed
- Multi-signal decoder: dot(0.5) + cosine(0.3) + L2(0.2) for link probability
- Risk scores normalised 0–100 from Isolation Forest anomaly scores
