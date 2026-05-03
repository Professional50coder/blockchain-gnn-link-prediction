# Blockchain GNN Analytics Dashboard

## Project Overview
An interactive Blockchain Transaction Link Prediction and Fraud Detection Dashboard built with Streamlit. Uses Graph Neural Networks (GraphSAGE) to analyze Ethereum blockchain transactions.

## Architecture
- **Framework**: Streamlit (Python)
- **Visualization**: Plotly, NetworkX
- **ML**: scikit-learn (Isolation Forest, Label Encoding)
- **Data**: Pre-trained GNN embeddings and transaction data

## Key Files
- `app.py` — Main Streamlit dashboard application
- `app_fixed.py` — Revised version with bug fixes
- `node_embeddings.npy` — Pre-trained 64-dim GNN node embeddings
- `label_encoder.pkl` — Wallet address to ID mapping
- `edges.csv` — Transaction edge list
- `fraudulent_wallets.csv` — Anomaly detection results
- `loss_history.npy` — Training loss metrics
- `roc_data.npz` — ROC-AUC evaluation data

## Configuration
- Streamlit config: `.streamlit/config.toml`
  - Port: 5000, Host: 0.0.0.0
  - CORS and XSRF protection disabled for Replit proxy

## Running the App
- Workflow: "Start application" — `streamlit run app.py`
- Port: 5000

## Deployment
- Target: autoscale
- Run: `streamlit run app.py`
