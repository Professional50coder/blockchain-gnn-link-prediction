# 🔗 Blockchain Transaction Link Prediction Dashboard

An interactive Streamlit dashboard for analyzing Ethereum blockchain transactions using Graph Neural Networks (GNN).

## 📋 Features

- **📊 Graph Analytics**: Comprehensive network statistics and visualizations
- **🔮 Link Prediction**: Predict future transaction probabilities between wallet addresses
- **🚨 Fraud Detection**: Identify suspicious wallets using unsupervised anomaly detection
- **📈 Model Performance**: ROC curves, loss curves, and evaluation metrics
- **🌐 Network Visualization**: Interactive transaction network graphs
- **🎨 Modern UI**: Clean, professional interface with Plotly visualizations

## 🏗️ Architecture

```
Ethereum Transaction Data
        ↓
Graph Construction (Wallets as Nodes, Transactions as Edges)
        ↓
GraphSAGE GNN Model
        ↓
Node Embeddings (64-dim)
        ↓
├─ Link Prediction (Dot Product Decoder)
└─ Fraud Detection (Isolation Forest)
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone/Download the Project

```bash
# Download all files to a directory
mkdir blockchain-gnn-dashboard
cd blockchain-gnn-dashboard
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Prepare Data Files

You need the following files in the same directory as `app.py`:

1. **node_embeddings.npy** - Learned node embeddings from GNN (required)
2. **label_encoder.pkl** - Wallet address encoder (required)
3. **edges.csv** - Transaction edge list (required)
4. **fraudulent_wallets.csv** - Fraud detection results (required)
5. **loss_history.npy** - Training loss history (optional)
6. **roc_data.npz** - ROC curve data (optional)

#### Generate Data Files from Your Notebook

In your Colab notebook, run the `save_dashboard_data.py` script:

```python
# At the end of your notebook, after training the model
!python save_dashboard_data.py
```

This will create all necessary files. Then download them:

```python
from google.colab import files
import zipfile

# Create zip with all files
files_list = [
    "node_embeddings.npy",
    "label_encoder.pkl", 
    "edges.csv",
    "fraudulent_wallets.csv",
    "loss_history.npy",
    "roc_data.npz"
]

with zipfile.ZipFile('dashboard_files.zip', 'w') as zipf:
    for filename in files_list:
        zipf.write(filename)

files.download('dashboard_files.zip')
```

Extract the zip file in your dashboard directory.

## 🚀 Running the Dashboard

### Local Development

```bash
streamlit run app.py
```

The dashboard will automatically open in your default browser at `http://localhost:8501`

### Deploy to Streamlit Cloud (Optional)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Upload data files to the repository
5. Deploy!

## 📊 Dashboard Sections

### 1. 🏠 Overview
- Project summary and key metrics
- Quick statistics (wallets, transactions, fraud alerts)
- Methodology overview

### 2. 📊 Graph Analytics
- Network statistics (nodes, edges, density)
- Degree distributions (in-degree, out-degree)
- Transaction search and filtering
- Sample data exploration

### 3. 🔮 Link Prediction
- **Manual Input**: Enter two wallet addresses to predict transaction probability
- **Random Predictions**: Generate multiple random predictions
- Detailed wallet analysis
- Historical transaction lookup

### 4. 🚨 Fraud Detection
- Suspicious wallet identification
- Fraud score distribution
- Top N fraudulent wallets
- Individual wallet investigation
- Transaction-level fraud analysis

### 5. 📈 Model Performance
- Training loss curve
- ROC-AUC curve with evaluation
- Model architecture details
- Training configuration
- Performance metrics summary

### 6. 🌐 Network Visualization
- Interactive transaction network graphs
- Wallet-centric network views
- Fraud network analysis
- Connected wallet exploration

## 🎯 Key Metrics Explained

### Link Prediction Probability
- **High (>0.7)**: Strong likelihood of future transaction
- **Medium (0.4-0.7)**: Moderate likelihood
- **Low (<0.4)**: Weak likelihood

### Fraud Score
- Lower scores indicate **higher suspicion**
- Based on Isolation Forest anomaly detection
- Scores typically range from -0.5 to 0.5

### ROC-AUC Score
- **>0.9**: Excellent model performance
- **0.7-0.9**: Good performance
- **0.6-0.7**: Moderate performance
- **<0.6**: Poor performance

## 🛠️ Customization

### Modify Color Scheme

Edit the custom CSS in `app.py`:

```python
st.markdown("""
<style>
    .main-header {
        color: #YOUR_COLOR;
    }
</style>
""", unsafe_allow_html=True)
```

### Adjust Fraud Detection Threshold

In `save_dashboard_data.py`, modify the contamination parameter:

```python
iso_forest = IsolationForest(
    contamination=0.02,  # Change this (0.02 = 2% of wallets flagged)
    ...
)
```

### Change Number of Network Connections

In the Network Visualization section, adjust the slider range:

```python
max_connections = st.slider("Max Connections", 10, 100, 20)  # (min, max, default)
```

## 📁 File Structure

```
blockchain-gnn-dashboard/
├── app.py                          # Main Streamlit application
├── save_dashboard_data.py          # Data export script for Colab
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── node_embeddings.npy            # GNN learned embeddings
├── label_encoder.pkl              # Address encoder
├── edges.csv                      # Transaction data
├── fraudulent_wallets.csv         # Fraud detection results
├── loss_history.npy               # Training metrics
└── roc_data.npz                   # Evaluation metrics
```

## 🔧 Troubleshooting

### "No module named 'streamlit'"
```bash
pip install streamlit
```

### "Error loading data: [Errno 2] No such file or directory"
- Ensure all required `.npy`, `.pkl`, and `.csv` files are in the same directory as `app.py`
- Check file names match exactly (case-sensitive)

### Dashboard is slow
- Reduce the number of transactions in `edges.csv` (sample first 10k-50k rows)
- Decrease `max_connections` in network visualization
- Use `@st.cache_data` decorators (already implemented)

### Network visualization not showing
- Check that the wallet ID exists in the dataset
- Ensure the wallet has at least one transaction
- Try increasing `max_connections` parameter

## 📚 Technical Details

### Model Architecture
- **Type**: GraphSAGE (Graph Sample and Aggregate)
- **Layers**: 2-layer GCN
- **Hidden Dimensions**: 64
- **Activation**: ReLU
- **Decoder**: Dot Product

### Training Details
- **Optimizer**: Adam (lr=0.01)
- **Loss**: Binary Cross-Entropy
- **Epochs**: 100
- **Split**: 70/15/15 (train/val/test)
- **Negative Sampling**: Dynamic during training

### Fraud Detection
- **Algorithm**: Isolation Forest
- **Input**: 64-dimensional node embeddings
- **Type**: Unsupervised anomaly detection
- **Contamination**: 2% (default)

## 📖 Usage Examples

### Example 1: Check Transaction Probability

1. Navigate to "🔮 Link Prediction"
2. Enter sender wallet: `0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb`
3. Enter receiver wallet: `0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE`
4. Click "Predict Transaction Probability"
5. View probability score and interpretation

### Example 2: Investigate Suspicious Wallet

1. Navigate to "🚨 Fraud Detection"
2. Scroll to "Investigate Specific Wallet"
3. Enter a wallet ID (try one from the top suspicious list)
4. Click "Investigate Wallet"
5. View fraud score, transaction history, and network

### Example 3: Visualize Wallet Network

1. Navigate to "🌐 Network Visualization"
2. Enter a wallet ID
3. Adjust "Max Connections" slider
4. Click "Generate Network Graph"
5. Explore the interactive network visualization

## 🎓 For Academic Use

### Citing This Work

If you use this dashboard in your research or project, please cite:

```
Transaction Link Prediction in Blockchain using Graph Neural Networks
Model: GraphSAGE
Dataset: Ethereum Mainnet (Google BigQuery)
Implementation: PyTorch Geometric
```

### Report Sections

This dashboard supports the following report sections:
- ✅ Introduction & Background
- ✅ Methodology (Graph Construction, GNN Architecture)
- ✅ Implementation Details
- ✅ Results & Evaluation (ROC-AUC, Loss Curves)
- ✅ Fraud Detection Extension
- ✅ Visualization & Analysis

## 🤝 Contributing

Suggestions for improvements:

1. Add more GNN architectures (GAT, GIN)
2. Temporal analysis features
3. Real-time data updates
4. Advanced fraud patterns
5. DeFi protocol analysis
6. Token transfer analysis

## 📄 License

This project is for educational and research purposes.

## 🆘 Support

For issues or questions:

1. Check the Troubleshooting section above
2. Review your Colab notebook for missing variables
3. Verify all data files are properly generated
4. Check Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)

## 🎉 Acknowledgments

- **Dataset**: Google BigQuery Ethereum Public Dataset
- **Framework**: PyTorch Geometric
- **Visualization**: Streamlit, Plotly
- **Model**: GraphSAGE (Hamilton et al., 2017)

---

**Version**: 1.0  
**Last Updated**: February 2024  
**Status**: Production Ready ✅
