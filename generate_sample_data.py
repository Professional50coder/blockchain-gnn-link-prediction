"""
Sample Data Generator for Dashboard Testing
============================================
Use this if you want to test the dashboard before having real data.

WARNING: This creates FAKE data for demonstration purposes only.
For your actual project, use real data from your trained model!
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("GENERATING SAMPLE DATA FOR DASHBOARD TESTING")
print("=" * 60)
print("\n⚠️  WARNING: This is FAKE data for testing only!")
print("For your real project, use data from your trained GNN model.\n")

# Configuration
NUM_WALLETS = 1000
NUM_TRANSACTIONS = 5000
EMBEDDING_DIM = 64
FRAUD_RATE = 0.02

print(f"Configuration:")
print(f"  - Wallets: {NUM_WALLETS}")
print(f"  - Transactions: {NUM_TRANSACTIONS}")
print(f"  - Embedding dimension: {EMBEDDING_DIM}")
print(f"  - Fraud rate: {FRAUD_RATE*100}%\n")

# -------------------------------
# 1. Generate fake wallet addresses
# -------------------------------
print("[1/7] Generating wallet addresses...")
fake_addresses = [
    f"0x{''.join(np.random.choice(list('0123456789abcdef'), 40))}"
    for _ in range(NUM_WALLETS)
]

le = LabelEncoder()
le.fit(fake_addresses)
print(f"✅ Created {len(fake_addresses)} fake wallet addresses")

# -------------------------------
# 2. Generate node embeddings
# -------------------------------
print("\n[2/7] Generating node embeddings...")
embeddings = np.random.randn(NUM_WALLETS, EMBEDDING_DIM).astype(np.float32)
# Normalize
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
np.save("node_embeddings.npy", embeddings)
print(f"✅ Saved node_embeddings.npy - Shape: {embeddings.shape}")

# -------------------------------
# 3. Save label encoder
# -------------------------------
print("\n[3/7] Saving label encoder...")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print(f"✅ Saved label_encoder.pkl")

# -------------------------------
# 4. Generate transaction edges
# -------------------------------
print("\n[4/7] Generating transaction edges...")
from_ids = np.random.randint(0, NUM_WALLETS, NUM_TRANSACTIONS)
to_ids = np.random.randint(0, NUM_WALLETS, NUM_TRANSACTIONS)

# Remove self-loops
mask = from_ids != to_ids
from_ids = from_ids[mask]
to_ids = to_ids[mask]

edges_df = pd.DataFrame({
    'from_id': from_ids,
    'to_id': to_ids
})

edges_df.to_csv("edges.csv", index=False)
print(f"✅ Saved edges.csv - {len(edges_df)} transactions")

# -------------------------------
# 5. Generate fraud detection results
# -------------------------------
print("\n[5/7] Generating fraud detection results...")
num_fraud = int(NUM_WALLETS * FRAUD_RATE)

# Generate fraud scores (lower = more suspicious)
fraud_scores = np.random.randn(NUM_WALLETS) * 0.2
fraud_scores[:num_fraud] = np.random.uniform(-0.5, -0.1, num_fraud)  # Make some suspicious

# Sort by score to get top fraudulent
fraud_indices = np.argsort(fraud_scores)[:num_fraud]

fraud_df = pd.DataFrame({
    'wallet_id': fraud_indices,
    'fraud_label': -1,
    'fraud_score': fraud_scores[fraud_indices],
    'wallet_address': [le.inverse_transform([i])[0] for i in fraud_indices]
})

fraud_df.to_csv("fraudulent_wallets.csv", index=False)
print(f"✅ Saved fraudulent_wallets.csv - {len(fraud_df)} suspicious wallets")

# -------------------------------
# 6. Generate loss history
# -------------------------------
print("\n[6/7] Generating loss history...")
# Simulate decreasing loss
epochs = 100
initial_loss = 0.8
final_loss = 0.1
loss_history = initial_loss * np.exp(-np.linspace(0, 3, epochs)) + final_loss

np.save("loss_history.npy", loss_history)
print(f"✅ Saved loss_history.npy - {len(loss_history)} epochs")

# -------------------------------
# 7. Generate ROC data
# -------------------------------
print("\n[7/7] Generating ROC curve data...")
# Simulate a good ROC curve
fpr = np.linspace(0, 1, 100)
# Create a curved line (better than random)
tpr = np.minimum(1.0, fpr * 1.5 + 0.2)
# Add some smoothing
tpr = np.clip(tpr + np.random.randn(100) * 0.05, 0, 1)
tpr = np.sort(tpr)  # Ensure monotonic

# Calculate AUC
from sklearn.metrics import auc
auc_value = auc(fpr, tpr)

np.savez("roc_data.npz", fpr=fpr, tpr=tpr, auc=auc_value)
print(f"✅ Saved roc_data.npz - AUC: {auc_value:.4f}")

# -------------------------------
# Summary
# -------------------------------
print("\n" + "=" * 60)
print("SAMPLE DATA GENERATION COMPLETE!")
print("=" * 60)

import os
files_created = [
    "node_embeddings.npy",
    "label_encoder.pkl",
    "edges.csv",
    "fraudulent_wallets.csv",
    "loss_history.npy",
    "roc_data.npz"
]

print("\nFiles created:")
for filename in files_created:
    size = os.path.getsize(filename)
    print(f"  ✅ {filename} ({size:,} bytes)")

print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("""
You can now test the dashboard:

    streamlit run app.py

Remember: This is FAKE data for testing only!

For your actual project:
1. Run your Colab notebook completely
2. Use the real save_dashboard_data.py script
3. Download the actual trained model data
4. Replace these sample files with real data

The sample data will help you:
- Test that the dashboard works
- Understand the expected file formats
- Debug any installation issues
- Practice your presentation
""")

print("=" * 60)
