"""
Save Required Data Files for Streamlit Dashboard
=================================================
Run this script at the END of your Colab notebook to save all necessary files
for the Streamlit dashboard.

Make sure you have already:
1. Trained the GNN model
2. Generated embeddings
3. Run fraud detection
4. Collected loss history and ROC data
"""

import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.metrics import roc_curve, auc

print("=" * 60)
print("SAVING DATA FILES FOR STREAMLIT DASHBOARD")
print("=" * 60)

# -------------------------------
# 1. Save Node Embeddings
# -------------------------------
print("\n[1/7] Saving node embeddings...")
try:
    # Assuming you have 'embeddings' variable from: embeddings = z.detach().cpu().numpy()
    np.save("node_embeddings.npy", embeddings)
    print(f"✅ Saved node_embeddings.npy - Shape: {embeddings.shape}")
except Exception as e:
    print(f"❌ Error saving embeddings: {e}")
    print("   Make sure 'embeddings' variable exists")

# -------------------------------
# 2. Save Label Encoder
# -------------------------------
print("\n[2/7] Saving label encoder...")
try:
    # Assuming you have 'le' (LabelEncoder) variable
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print(f"✅ Saved label_encoder.pkl - Classes: {len(le.classes_)}")
except Exception as e:
    print(f"❌ Error saving label encoder: {e}")
    print("   Make sure 'le' variable exists")

# -------------------------------
# 3. Save Edge List
# -------------------------------
print("\n[3/7] Saving edge list...")
try:
    # Assuming you have 'df' with from_id and to_id columns
    df[["from_id", "to_id"]].to_csv("edges.csv", index=False)
    print(f"✅ Saved edges.csv - Rows: {len(df)}")
except Exception as e:
    print(f"❌ Error saving edges: {e}")
    print("   Make sure 'df' DataFrame exists with from_id and to_id columns")

# -------------------------------
# 4. Save Fraud Detection Results
# -------------------------------
print("\n[4/7] Saving fraud detection results...")
try:
    # Assuming you have 'fraudulent_wallets' DataFrame
    fraudulent_wallets.to_csv("fraudulent_wallets.csv", index=False)
    print(f"✅ Saved fraudulent_wallets.csv - Rows: {len(fraudulent_wallets)}")
except Exception as e:
    print(f"❌ Error saving fraud results: {e}")
    print("   Make sure 'fraudulent_wallets' DataFrame exists")

# -------------------------------
# 5. Save Training Loss History
# -------------------------------
print("\n[5/7] Saving loss history...")
try:
    # Assuming you have 'loss_history' list
    np.save("loss_history.npy", np.array(loss_history))
    print(f"✅ Saved loss_history.npy - Epochs: {len(loss_history)}")
except Exception as e:
    print(f"❌ Error saving loss history: {e}")
    print("   Make sure 'loss_history' list exists")
    print("   If not, you need to re-train with loss tracking")

# -------------------------------
# 6. Save ROC Curve Data
# -------------------------------
print("\n[6/7] Saving ROC curve data...")
try:
    # Generate ROC data if not already done
    # You need y_true and y_score from your test evaluation
    
    # If you already have these from test(), use them directly
    # Otherwise, compute them:
    model.eval()
    with torch.no_grad():
        z = model.encode(
            data_lp.x.to(device),
            data_lp.train_pos_edge_index.to(device)
        )
        
        pos_out = model.decode(z, data_lp.test_pos_edge_index.to(device))
        neg_out = model.decode(z, data_lp.test_neg_edge_index.to(device))
        
        y_true = torch.cat([
            torch.ones(pos_out.size(0)),
            torch.zeros(neg_out.size(0))
        ]).cpu().numpy()
        
        y_score = torch.cat([pos_out, neg_out]).cpu().numpy()
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc_value = auc(fpr, tpr)
    
    # Save
    np.savez("roc_data.npz", fpr=fpr, tpr=tpr, auc=roc_auc_value)
    print(f"✅ Saved roc_data.npz - AUC: {roc_auc_value:.4f}")
except Exception as e:
    print(f"❌ Error saving ROC data: {e}")
    print("   Make sure model, data_lp, and device variables exist")

# -------------------------------
# 7. Create Requirements File
# -------------------------------
print("\n[7/7] Creating requirements.txt...")
try:
    requirements = """streamlit==1.31.0
pandas==2.0.3
numpy==1.24.3
plotly==5.18.0
scikit-learn==1.3.2
networkx==3.2.1
torch==2.1.2
"""
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✅ Saved requirements.txt")
except Exception as e:
    print(f"❌ Error saving requirements: {e}")

# -------------------------------
# Summary
# -------------------------------
print("\n" + "=" * 60)
print("SUMMARY OF FILES")
print("=" * 60)

import os
files_to_check = [
    "node_embeddings.npy",
    "label_encoder.pkl",
    "edges.csv",
    "fraudulent_wallets.csv",
    "loss_history.npy",
    "roc_data.npz",
    "requirements.txt"
]

print("\nFiles created:")
for filename in files_to_check:
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"  ✅ {filename} ({size:,} bytes)")
    else:
        print(f"  ❌ {filename} - NOT FOUND")

print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("""
1. Download all the files above from Colab
2. Place them in the same directory as app.py
3. Run the Streamlit dashboard:
   
   streamlit run app.py

4. Your dashboard should open in the browser!
""")

print("\n" + "=" * 60)
print("DOWNLOAD INSTRUCTIONS FOR COLAB")
print("=" * 60)
print("""
To download all files at once:

from google.colab import files
import zipfile

# Create a zip file
with zipfile.ZipFile('dashboard_files.zip', 'w') as zipf:
    for filename in {}:
        if os.path.exists(filename):
            zipf.write(filename)

# Download the zip
files.download('dashboard_files.zip')
""".format(files_to_check))
