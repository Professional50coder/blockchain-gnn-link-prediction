import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc
import networkx as nx
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Blockchain GNN Analytics",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Data Loading Functions
# -------------------------------
@st.cache_data
def load_data():
    """Load all pre-trained model outputs and data"""
    try:
        embeddings = np.load("node_embeddings.npy")
        edges = pd.read_csv("edges.csv")
        fraud_df = pd.read_csv("fraudulent_wallets.csv")
        
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        
        # Load optional data if available
        try:
            loss_history = np.load("loss_history.npy")
        except:
            loss_history = None
            
        try:
            roc_data = np.load("roc_data.npz")
            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
            roc_auc_val = roc_data['auc']
        except:
            fpr, tpr, roc_auc_val = None, None, None
        
        return embeddings, edges, fraud_df, le, loss_history, (fpr, tpr, roc_auc_val)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure all required files are in the same directory as the app.")
        st.stop()

def compute_link_probability(embedding_a, embedding_b):
    """Compute probability of link between two nodes"""
    score = np.dot(embedding_a, embedding_b)
    probability = 1 / (1 + np.exp(-score))
    return probability

def create_graph_statistics(edges_df):
    """Compute various graph statistics"""
    stats = {
        'total_transactions': len(edges_df),
        'unique_senders': edges_df['from_id'].nunique(),
        'unique_receivers': edges_df['to_id'].nunique(),
        'avg_out_degree': edges_df.groupby('from_id').size().mean(),
        'avg_in_degree': edges_df.groupby('to_id').size().mean(),
    }
    return stats

# -------------------------------
# Visualization Functions
# -------------------------------
def plot_loss_curve(loss_history):
    """Create interactive loss curve plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(loss_history) + 1)),
        y=loss_history,
        mode='lines',
        name='Training Loss',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.update_layout(
        title='GNN Training Loss Curve',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    return fig

def plot_roc_curve(fpr, tpr, auc_score):
    """Create interactive ROC curve"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))
    fig.update_layout(
        title='ROC Curve for Transaction Link Prediction',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        hovermode='closest',
        template='plotly_white',
        height=400
    )
    return fig

def plot_fraud_score_distribution(fraud_df):
    """Plot distribution of fraud scores"""
    fig = px.histogram(
        fraud_df,
        x='fraud_score',
        nbins=50,
        title='Distribution of Fraud Scores',
        labels={'fraud_score': 'Fraud Score', 'count': 'Frequency'},
        color_discrete_sequence=['#f44336']
    )
    fig.update_layout(template='plotly_white', height=400)
    return fig

def create_network_subgraph(edges_df, wallet_id, le, max_connections=20):
    """Create a small network visualization around a specific wallet"""
    # Get transactions involving this wallet
    related_edges = edges_df[
        (edges_df['from_id'] == wallet_id) | (edges_df['to_id'] == wallet_id)
    ].head(max_connections)
    
    if len(related_edges) == 0:
        return None
    
    # Create networkx graph
    G = nx.DiGraph()
    
    for _, row in related_edges.iterrows():
        G.add_edge(row['from_id'], row['to_id'])
    
    # Get positions
    pos = nx.spring_layout(G)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Truncate address for display
        try:
            addr = le.inverse_transform([node])[0]
            node_text.append(f"{addr[:8]}...{addr[-6:]}")
        except:
            node_text.append(f"Wallet {node}")
        
        # Highlight the central wallet
        if node == wallet_id:
            node_colors.append('#f44336')
        else:
            node_colors.append('#1f77b4')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            size=20,
            color=node_colors,
            line_width=2
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title='Transaction Network Visualization',
        showlegend=False,
        hovermode='closest',
        template='plotly_white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
    
    return fig

# -------------------------------
# Load Data
# -------------------------------
embeddings, edges, fraud_df, le, loss_history, (fpr, tpr, roc_auc_val) = load_data()

# Compute statistics
stats = create_graph_statistics(edges)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/000000/blockchain-technology.png", width=80)
st.sidebar.title("🔗 Navigation")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Select Section:",
    [
        "🏠 Overview",
        "📊 Graph Analytics", 
        "🔮 Link Prediction",
        "🚨 Fraud Detection",
        "📈 Model Performance",
        "🌐 Network Visualization"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Stats")
st.sidebar.metric("Total Wallets", f"{embeddings.shape[0]:,}")
st.sidebar.metric("Total Transactions", f"{stats['total_transactions']:,}")
st.sidebar.metric("Suspicious Wallets", f"{len(fraud_df):,}")

st.sidebar.markdown("---")
st.sidebar.info("""
**Project:** Transaction Link Prediction in Blockchain using GNN
                
**Model:** GraphSAGE
                
**Dataset:** Ethereum Mainnet
""")

# -------------------------------
# Main Content
# -------------------------------

if "🏠 Overview" in section:
    st.markdown('<p class="main-header">🔗 Blockchain Transaction Analysis using Graph Neural Networks</p>', 
                unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Wallet Addresses", f"{embeddings.shape[0]:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Transactions", f"{stats['total_transactions']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Out-Degree", f"{stats['avg_out_degree']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
        st.metric("Suspicious Wallets", f"{len(fraud_df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project description
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📖 Project Overview")
        st.markdown("""
        This dashboard presents a **Graph Neural Network (GNN)** based system for analyzing 
        Ethereum blockchain transactions. The system provides:
        
        - **Transaction Link Prediction**: Predict future transaction links between wallet addresses
        - **Fraud Detection**: Identify potentially fraudulent or anomalous wallet behavior
        - **Network Analysis**: Visualize transaction patterns and relationships
        - **Performance Metrics**: ROC-AUC analysis and model evaluation
        
        The model uses **GraphSAGE** architecture to learn node embeddings from the transaction graph,
        where wallet addresses are nodes and transactions are directed edges.
        """)
    
    with col2:
        st.markdown("### 🎯 Key Features")
        st.markdown("""
        ✅ Real Ethereum mainnet data
        
        ✅ GraphSAGE GNN architecture
        
        ✅ Link prediction with >70% AUC
        
        ✅ Unsupervised fraud detection
        
        ✅ Interactive visualizations
        
        ✅ Real-time predictions
        """)
    
    st.markdown("---")
    
    # Methodology
    with st.expander("📚 Methodology Details"):
        st.markdown("""
        #### 1. Data Collection
        - Source: Google BigQuery Ethereum Public Dataset
        - Transactions: Sender, receiver, value, gas, timestamp
        
        #### 2. Graph Construction
        - Nodes: Wallet addresses
        - Edges: Transaction relationships (directed)
        - Features: In-degree, out-degree statistics
        
        #### 3. Model Architecture
        - **GraphSAGE**: 2-layer Graph Convolutional Network
        - **Embedding Size**: 64 dimensions
        - **Decoder**: Dot-product based link prediction
        
        #### 4. Training Strategy
        - Train/Val/Test Split: 70/15/15
        - Loss: Binary Cross-Entropy
        - Optimization: Adam optimizer
        - Negative Sampling: Dynamic generation
        
        #### 5. Fraud Detection
        - Method: Isolation Forest (unsupervised)
        - Input: Learned node embeddings
        - Output: Anomaly scores for each wallet
        """)

elif "📊 Graph Analytics" in section:
    st.markdown('<p class="main-header">📊 Graph Analytics & Statistics</p>', unsafe_allow_html=True)
    
    # Detailed statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Network Overview")
        st.metric("Total Nodes (Wallets)", f"{embeddings.shape[0]:,}")
        st.metric("Total Edges (Transactions)", f"{stats['total_transactions']:,}")
        st.metric("Graph Density", f"{(stats['total_transactions'] / (embeddings.shape[0] ** 2)):.6f}")
    
    with col2:
        st.markdown("### Degree Statistics")
        st.metric("Avg Out-Degree", f"{stats['avg_out_degree']:.2f}")
        st.metric("Avg In-Degree", f"{stats['avg_in_degree']:.2f}")
        st.metric("Unique Senders", f"{stats['unique_senders']:,}")
    
    with col3:
        st.markdown("### Network Metrics")
        st.metric("Unique Receivers", f"{stats['unique_receivers']:,}")
        st.metric("Embedding Dimensions", f"{embeddings.shape[1]}")
        st.metric("Model Type", "GraphSAGE")
    
    st.markdown("---")
    
    # Transaction sample
    st.markdown("### 📋 Sample Transactions")
    
    # Add search/filter options
    col1, col2 = st.columns([3, 1])
    with col1:
        search_wallet = st.text_input("🔍 Search by Wallet ID", "")
    with col2:
        num_rows = st.selectbox("Rows to display", [10, 25, 50, 100], index=0)
    
    if search_wallet:
        try:
            wallet_id = int(search_wallet)
            filtered_edges = edges[
                (edges['from_id'] == wallet_id) | (edges['to_id'] == wallet_id)
            ]
            st.dataframe(filtered_edges.head(num_rows), use_container_width=True)
            st.info(f"Found {len(filtered_edges)} transactions involving wallet {wallet_id}")
        except ValueError:
            st.error("Please enter a valid wallet ID (numeric)")
    else:
        st.dataframe(edges.head(num_rows), use_container_width=True)
    
    # Degree distribution
    st.markdown("---")
    st.markdown("### 📊 Degree Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        out_degrees = edges.groupby('from_id').size()
        fig = px.histogram(
            out_degrees,
            nbins=50,
            title='Out-Degree Distribution',
            labels={'value': 'Out-Degree', 'count': 'Frequency'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(template='plotly_white', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        in_degrees = edges.groupby('to_id').size()
        fig = px.histogram(
            in_degrees,
            nbins=50,
            title='In-Degree Distribution',
            labels={'value': 'In-Degree', 'count': 'Frequency'},
            color_discrete_sequence=['#ff7f0e']
        )
        fig.update_layout(template='plotly_white', height=350)
        st.plotly_chart(fig, use_container_width=True)

elif "🔮 Link Prediction" in section:
    st.markdown('<p class="main-header">🔮 Transaction Link Prediction</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Predict the probability of a future transaction between two wallet addresses using the trained GNN model.
    The model computes a similarity score between learned node embeddings.
    """)
    
    st.markdown("---")
    
    # Input methods
    tab1, tab2 = st.tabs(["🔤 Manual Input", "🎲 Random Prediction"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            wallet_a = st.text_input(
                "Sender Wallet Address",
                placeholder="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
                help="Enter the Ethereum wallet address of the sender"
            )
        
        with col2:
            wallet_b = st.text_input(
                "Receiver Wallet Address",
                placeholder="0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",
                help="Enter the Ethereum wallet address of the receiver"
            )
        
        predict_button = st.button("🔮 Predict Transaction Probability", type="primary", use_container_width=True)
        
        if predict_button:
            if wallet_a and wallet_b:
                if wallet_a in le.classes_ and wallet_b in le.classes_:
                    id_a = le.transform([wallet_a])[0]
                    id_b = le.transform([wallet_b])[0]
                    
                    probability = compute_link_probability(embeddings[id_a], embeddings[id_b])
                    
                    # Display result
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown("### 📊 Prediction Result")
                        st.metric("Transaction Probability", f"{probability:.4f} ({probability*100:.2f}%)")
                        
                        # Interpretation
                        if probability > 0.7:
                            st.success("🟢 **High likelihood** of future transaction")
                        elif probability > 0.4:
                            st.warning("🟡 **Moderate likelihood** of future transaction")
                        else:
                            st.info("🔵 **Low likelihood** of future transaction")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Additional details
                    st.markdown("---")
                    with st.expander("📈 View Detailed Analysis"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Sender Wallet Analysis**")
                            sender_txs = edges[edges['from_id'] == id_a]
                            st.write(f"- Total transactions sent: {len(sender_txs)}")
                            st.write(f"- Unique receivers: {sender_txs['to_id'].nunique()}")
                            st.write(f"- Wallet ID: {id_a}")
                        
                        with col2:
                            st.markdown("**Receiver Wallet Analysis**")
                            receiver_txs = edges[edges['to_id'] == id_b]
                            st.write(f"- Total transactions received: {len(receiver_txs)}")
                            st.write(f"- Unique senders: {receiver_txs['from_id'].nunique()}")
                            st.write(f"- Wallet ID: {id_b}")
                        
                        # Check if transaction already exists
                        existing = edges[(edges['from_id'] == id_a) & (edges['to_id'] == id_b)]
                        if len(existing) > 0:
                            st.warning(f"⚠️ Historical transactions found: {len(existing)} transaction(s) already exist between these wallets")
                        else:
                            st.info("ℹ️ No historical transactions found between these wallets")
                
                else:
                    st.error("❌ One or both wallet addresses not found in the dataset. Please verify the addresses.")
            else:
                st.warning("⚠️ Please enter both wallet addresses.")
    
    with tab2:
        st.markdown("### 🎲 Random Wallet Pair Prediction")
        st.markdown("Generate predictions for random wallet pairs to explore the model's behavior.")
        
        num_predictions = st.slider("Number of random predictions", 5, 20, 10)
        
        if st.button("🎲 Generate Random Predictions", use_container_width=True):
            random_predictions = []
            
            for _ in range(num_predictions):
                id_a, id_b = np.random.choice(embeddings.shape[0], 2, replace=False)
                prob = compute_link_probability(embeddings[id_a], embeddings[id_b])
                
                addr_a = le.inverse_transform([id_a])[0]
                addr_b = le.inverse_transform([id_b])[0]
                
                random_predictions.append({
                    'Sender': f"{addr_a[:10]}...{addr_a[-8:]}",
                    'Receiver': f"{addr_b[:10]}...{addr_b[-8:]}",
                    'Probability': prob,
                    'Likelihood': 'High' if prob > 0.7 else ('Medium' if prob > 0.4 else 'Low')
                })
            
            df_predictions = pd.DataFrame(random_predictions)
            df_predictions = df_predictions.sort_values('Probability', ascending=False)
            
            st.dataframe(
                df_predictions.style.background_gradient(subset=['Probability'], cmap='RdYlGn'),
                use_container_width=True,
                height=400
            )

elif "🚨 Fraud Detection" in section:
    st.markdown('<p class="main-header">🚨 Fraud Detection Dashboard</p>', unsafe_allow_html=True)
    
    st.markdown("""
    This module identifies potentially fraudulent or anomalous wallet behavior using **Isolation Forest** 
    algorithm on learned GNN embeddings. Lower fraud scores indicate higher suspicion.
    """)
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
        st.metric("Suspicious Wallets", f"{len(fraud_df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Detection Rate", f"{(len(fraud_df)/embeddings.shape[0]*100):.2f}%")
    
    with col3:
        avg_fraud_score = fraud_df['fraud_score'].mean()
        st.metric("Avg Fraud Score", f"{avg_fraud_score:.4f}")
    
    with col4:
        min_fraud_score = fraud_df['fraud_score'].min()
        st.metric("Min Fraud Score", f"{min_fraud_score:.4f}")
    
    st.markdown("---")
    
    # Fraud score distribution
    st.markdown("### 📊 Fraud Score Distribution")
    fig = plot_fraud_score_distribution(fraud_df)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top suspicious wallets
    st.markdown("### 🔍 Most Suspicious Wallets")
    
    # Filter options
    col1, col2 = st.columns([3, 1])
    with col1:
        score_threshold = st.slider(
            "Fraud Score Threshold",
            float(fraud_df['fraud_score'].min()),
            float(fraud_df['fraud_score'].max()),
            float(fraud_df['fraud_score'].quantile(0.25)),
            help="Show wallets with fraud score below this threshold"
        )
    with col2:
        top_n = st.selectbox("Show top N wallets", [10, 20, 50, 100], index=1)
    
    filtered_fraud = fraud_df[fraud_df['fraud_score'] <= score_threshold].sort_values('fraud_score').head(top_n)
    
    # Add transaction counts if available
    if 'wallet_address' in filtered_fraud.columns:
        display_df = filtered_fraud.copy()
        
        # Get transaction counts
        tx_counts = []
        for wallet_id in display_df['wallet_id']:
            sent = len(edges[edges['from_id'] == wallet_id])
            received = len(edges[edges['to_id'] == wallet_id])
            tx_counts.append({'Sent': sent, 'Received': received, 'Total': sent + received})
        
        tx_df = pd.DataFrame(tx_counts)
        display_df = pd.concat([display_df.reset_index(drop=True), tx_df], axis=1)
        
        # Format wallet address for display
        display_df['wallet_short'] = display_df['wallet_address'].apply(
            lambda x: f"{x[:10]}...{x[-8:]}" if len(x) > 20 else x
        )
        
        # Reorder columns
        cols = ['wallet_short', 'fraud_score', 'fraud_label', 'Total', 'Sent', 'Received', 'wallet_id']
        cols = [c for c in cols if c in display_df.columns]
        display_df = display_df[cols]
        display_df.columns = ['Wallet Address', 'Fraud Score', 'Label', 'Total Txs', 'Sent', 'Received', 'Wallet ID']
        
        st.dataframe(
            display_df.style.background_gradient(subset=['Fraud Score'], cmap='RdYlGn_r'),
            use_container_width=True,
            height=400
        )
    else:
        st.dataframe(filtered_fraud, use_container_width=True, height=400)
    
    st.markdown("---")
    
    # Wallet investigation
    st.markdown("### 🕵️ Investigate Specific Wallet")
    
    wallet_id_input = st.number_input(
        "Enter Wallet ID to investigate",
        min_value=0,
        max_value=int(embeddings.shape[0]-1),
        value=0,
        help="Enter the numeric wallet ID"
    )
    
    if st.button("🔍 Investigate Wallet", use_container_width=True):
        # Check if wallet is flagged
        wallet_fraud_info = fraud_df[fraud_df['wallet_id'] == wallet_id_input]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Wallet Information")
            
            try:
                wallet_address = le.inverse_transform([wallet_id_input])[0]
                st.write(f"**Address:** `{wallet_address[:20]}...{wallet_address[-20:]}`")
            except:
                st.write(f"**Wallet ID:** {wallet_id_input}")
            
            if len(wallet_fraud_info) > 0:
                st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
                st.warning(f"⚠️ **FLAGGED AS SUSPICIOUS**")
                st.metric("Fraud Score", f"{wallet_fraud_info.iloc[0]['fraud_score']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success("✅ **No fraud flags**")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Transaction Activity")
            
            sent_txs = edges[edges['from_id'] == wallet_id_input]
            received_txs = edges[edges['to_id'] == wallet_id_input]
            
            st.metric("Transactions Sent", len(sent_txs))
            st.metric("Transactions Received", len(received_txs))
            st.metric("Unique Counterparties", 
                     len(set(sent_txs['to_id'].unique()) | set(received_txs['from_id'].unique())))

elif "📈 Model Performance" in section:
    st.markdown('<p class="main-header">📈 Model Performance Metrics</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Comprehensive evaluation of the GNN model's performance on transaction link prediction task.
    """)
    
    st.markdown("---")
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📉 Training Loss Curve")
        if loss_history is not None:
            fig = plot_loss_curve(loss_history)
            st.plotly_chart(fig, use_container_width=True)
            
            # Loss statistics
            with st.expander("📊 Loss Statistics"):
                st.write(f"**Initial Loss:** {loss_history[0]:.4f}")
                st.write(f"**Final Loss:** {loss_history[-1]:.4f}")
                st.write(f"**Loss Reduction:** {((loss_history[0] - loss_history[-1])/loss_history[0]*100):.2f}%")
                st.write(f"**Total Epochs:** {len(loss_history)}")
        else:
            st.info("Loss history data not available. Please save `loss_history.npy` during training.")
    
    with col2:
        st.markdown("### 📊 ROC Curve")
        if all(x is not None for x in [fpr, tpr, roc_auc_val]):
            fig = plot_roc_curve(fpr, tpr, roc_auc_val)
            st.plotly_chart(fig, use_container_width=True)
            
            # ROC statistics
            with st.expander("📊 ROC Statistics"):
                st.write(f"**AUC Score:** {roc_auc_val:.4f}")
                
                if roc_auc_val > 0.9:
                    st.success("🟢 Excellent performance")
                elif roc_auc_val > 0.7:
                    st.success("🟢 Good performance")
                elif roc_auc_val > 0.6:
                    st.warning("🟡 Moderate performance")
                else:
                    st.error("🔴 Poor performance")
                
                st.write(f"**Interpretation:** The model can distinguish between linked and non-linked wallet pairs with {roc_auc_val*100:.1f}% accuracy.")
        else:
            st.info("ROC curve data not available. Please save `roc_data.npz` after evaluation.")
    
    st.markdown("---")
    
    # Model architecture details
    st.markdown("### 🏗️ Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### GraphSAGE Configuration")
        st.code("""
Model: GraphSAGE
Layers: 2
Hidden Channels: 64
Input Features: 2 (in/out degree)
Output Embedding: 64 dimensions
Decoder: Dot Product
        """)
    
    with col2:
        st.markdown("#### Training Configuration")
        st.code("""
Optimizer: Adam
Learning Rate: 0.01
Loss Function: Binary Cross-Entropy
Epochs: 100
Train/Val/Test: 70/15/15
Negative Sampling: Dynamic
        """)
    
    st.markdown("---")
    
    # Evaluation metrics summary
    st.markdown("### 📊 Evaluation Metrics Summary")
    
    metrics_data = {
        'Metric': ['ROC-AUC', 'Model Type', 'Embedding Dimension', 'Total Nodes', 'Total Edges'],
        'Value': [
            f"{roc_auc_val:.4f}" if roc_auc_val is not None else 'N/A',
            'GraphSAGE',
            '64',
            f"{embeddings.shape[0]:,}",
            f"{len(edges):,}"
        ],
        'Status': ['✅', '✅', '✅', '✅', '✅']
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

elif "🌐 Network Visualization" in section:
    st.markdown('<p class="main-header">🌐 Network Visualization</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Visualize the transaction network structure around specific wallets. This helps understand 
    the connectivity patterns and relationships in the blockchain graph.
    """)
    
    st.markdown("---")
    
    # Wallet selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        wallet_id_viz = st.number_input(
            "Enter Wallet ID for Network Visualization",
            min_value=0,
            max_value=int(embeddings.shape[0]-1),
            value=0,
            help="Select a wallet to visualize its transaction network"
        )
    
    with col2:
        max_connections = st.slider("Max Connections", 10, 50, 20)
    
    if st.button("🌐 Generate Network Graph", type="primary", use_container_width=True):
        with st.spinner("Creating network visualization..."):
            fig = create_network_subgraph(edges, wallet_id_viz, le, max_connections)
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                
                # Network statistics
                st.markdown("---")
                st.markdown("### 📊 Network Statistics")
                
                related_edges = edges[
                    (edges['from_id'] == wallet_id_viz) | (edges['to_id'] == wallet_id_viz)
                ]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Outgoing Transactions", len(edges[edges['from_id'] == wallet_id_viz]))
                
                with col2:
                    st.metric("Incoming Transactions", len(edges[edges['to_id'] == wallet_id_viz]))
                
                with col3:
                    st.metric("Total Connections", len(related_edges))
                
                # Show connected wallets
                with st.expander("🔍 View Connected Wallets"):
                    st.dataframe(related_edges.head(50), use_container_width=True)
            else:
                st.warning(f"⚠️ No transactions found for wallet {wallet_id_viz}")
    
    st.markdown("---")
    
    # Fraud network analysis
    st.markdown("### 🚨 Fraud Network Analysis")
    
    if st.button("🔍 Analyze Suspicious Wallet Networks", use_container_width=True):
        st.markdown("#### Top 5 Suspicious Wallets and Their Networks")
        
        top_fraud = fraud_df.sort_values('fraud_score').head(5)
        
        for idx, row in top_fraud.iterrows():
            wallet_id = row['wallet_id']
            fraud_score = row['fraud_score']
            
            with st.expander(f"Wallet {wallet_id} (Fraud Score: {fraud_score:.4f})"):
                fig = create_network_subgraph(edges, wallet_id, le, 15)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No transactions found for wallet {wallet_id}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><b>Transaction Link Prediction in Blockchain using Graph Neural Networks</b></p>
    <p>Powered by GraphSAGE | Data from Ethereum Mainnet | Built with Streamlit</p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>
        📊 Dashboard Version 1.0 | Last Updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)
