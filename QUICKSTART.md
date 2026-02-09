# 🚀 QUICK START GUIDE

## For Students: Running Your Blockchain GNN Dashboard

Follow these steps to get your dashboard running in **less than 10 minutes**.

---

## ⚡ Step 1: Save Data from Your Colab Notebook

### Option A: Run the Save Script (Recommended)

At the **END** of your Colab notebook (after all training is complete), add and run these cells:

```python
# Cell 1: Save all data files
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.metrics import roc_curve, auc
import os

print("Saving data files...")

# 1. Save embeddings
np.save("node_embeddings.npy", embeddings)
print("✓ Saved embeddings")

# 2. Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("✓ Saved label encoder")

# 3. Save edges
df[["from_id", "to_id"]].to_csv("edges.csv", index=False)
print("✓ Saved edges")

# 4. Save fraud results
fraudulent_wallets.to_csv("fraudulent_wallets.csv", index=False)
print("✓ Saved fraud results")

# 5. Save loss history
np.save("loss_history.npy", np.array(loss_history))
print("✓ Saved loss history")

# 6. Save ROC data
model.eval()
with torch.no_grad():
    z = model.encode(data_lp.x.to(device), data_lp.train_pos_edge_index.to(device))
    pos_out = model.decode(z, data_lp.test_pos_edge_index.to(device))
    neg_out = model.decode(z, data_lp.test_neg_edge_index.to(device))
    y_true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).cpu().numpy()
    y_score = torch.cat([pos_out, neg_out]).cpu().numpy()

fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc_value = auc(fpr, tpr)
np.savez("roc_data.npz", fpr=fpr, tpr=tpr, auc=roc_auc_value)
print("✓ Saved ROC data")

print("\nAll files saved!")
```

```python
# Cell 2: Download all files as a zip
from google.colab import files
import zipfile

files_to_zip = [
    "node_embeddings.npy",
    "label_encoder.pkl",
    "edges.csv",
    "fraudulent_wallets.csv",
    "loss_history.npy",
    "roc_data.npz"
]

with zipfile.ZipFile('dashboard_files.zip', 'w') as zipf:
    for filename in files_to_zip:
        if os.path.exists(filename):
            zipf.write(filename)
            print(f"✓ Added {filename}")

print("\nDownloading zip file...")
files.download('dashboard_files.zip')
```

---

## 📥 Step 2: Setup Local Environment

### On Windows:

```cmd
# 1. Create project folder
mkdir blockchain-dashboard
cd blockchain-dashboard

# 2. Extract the zip file you downloaded
# (Right-click > Extract All)

# 3. Download the dashboard files
# Download app.py, requirements.txt from the files provided

# 4. Install dependencies
pip install -r requirements.txt
```

### On Mac/Linux:

```bash
# 1. Create project folder
mkdir blockchain-dashboard
cd blockchain-dashboard

# 2. Extract the zip
unzip ~/Downloads/dashboard_files.zip

# 3. Download app.py and requirements.txt to this folder

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 🎯 Step 3: Run the Dashboard

```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`

**That's it! 🎉**

---

## 🔍 Verify Files Checklist

Before running, ensure you have these files in your folder:

```
blockchain-dashboard/
├── ✅ app.py
├── ✅ requirements.txt
├── ✅ node_embeddings.npy
├── ✅ label_encoder.pkl
├── ✅ edges.csv
├── ✅ fraudulent_wallets.csv
├── ✅ loss_history.npy (optional but recommended)
└── ✅ roc_data.npz (optional but recommended)
```

---

## ❗ Common Issues & Fixes

### Issue 1: "ModuleNotFoundError: No module named 'streamlit'"

**Fix:**
```bash
pip install streamlit
```

### Issue 2: "FileNotFoundError: node_embeddings.npy"

**Fix:**
- Check that all files from the zip are in the same folder as `app.py`
- Re-run the save script in Colab
- Verify file names match exactly

### Issue 3: Dashboard shows "Error loading data"

**Fix:**
- Open terminal in the dashboard folder
- Run: `ls` (Mac/Linux) or `dir` (Windows)
- Confirm all required files are listed
- Check for typos in file names

### Issue 4: Dashboard is very slow

**Fix:**
Reduce dataset size in Colab before saving:

```python
# In Colab, before saving edges.csv
df_sample = df.head(10000)  # Use first 10k transactions
df_sample[["from_id", "to_id"]].to_csv("edges.csv", index=False)
```

---

## 🎨 First-Time Usage Tour

### 1. Overview Page
- Check that all metrics load correctly
- Verify wallet count, transaction count

### 2. Test Link Prediction
- Go to "Link Prediction" tab
- Click "Random Prediction" button
- You should see 10 predictions with probabilities

### 3. Check Fraud Detection
- Go to "Fraud Detection" tab
- View the fraud score distribution chart
- Browse top suspicious wallets

### 4. View Model Performance
- Go to "Model Performance" tab
- Check training loss curve (should be decreasing)
- Check ROC curve (AUC should be > 0.7)

### 5. Try Network Visualization
- Go to "Network Visualization" tab
- Enter wallet ID: 0
- Click "Generate Network Graph"
- You should see an interactive network

---

## 📊 Demo Mode (If You Don't Have Data Yet)

If you want to test the dashboard before training your model:

1. Run the Colab notebook completely
2. Save the data files as shown above
3. If you're missing some files, the dashboard will show info messages

---

## 🎓 For Your Viva/Presentation

### Key Points to Demonstrate:

1. **Start with Overview**
   - "This dashboard analyzes [X] Ethereum transactions using a GraphSAGE model"
   - Show the key metrics

2. **Show Link Prediction**
   - "We can predict future transactions with [X]% accuracy"
   - Do a live prediction

3. **Demonstrate Fraud Detection**
   - "The model identified [X] suspicious wallets"
   - Investigate one specific wallet

4. **Present Model Performance**
   - "Our model achieved [X] ROC-AUC score"
   - Show the curves

5. **End with Visualization**
   - "Here's the actual transaction network"
   - Show an interactive graph

---

## 🚀 Advanced: Deploy Online (Optional)

### Option 1: Streamlit Cloud (Free)

1. Create a GitHub account
2. Create a new repository
3. Upload all files (app.py, data files, requirements.txt)
4. Go to [share.streamlit.io](https://share.streamlit.io)
5. Connect your GitHub repo
6. Deploy! (You'll get a public URL)

**Note:** Be careful with data file sizes. GitHub has a 100MB limit per file.

### Option 2: Local Network Sharing

```bash
# Run with external access
streamlit run app.py --server.address=0.0.0.0

# Others on your WiFi can access via your IP
# Find your IP: ipconfig (Windows) or ifconfig (Mac/Linux)
# Share: http://YOUR_IP:8501
```

---

## ✅ Success Checklist

- [ ] Colab notebook runs completely without errors
- [ ] All data files saved and downloaded
- [ ] Dashboard files in one folder
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dashboard runs (`streamlit run app.py`)
- [ ] All 6 sections load without errors
- [ ] Can perform a link prediction
- [ ] Can view fraud detection results
- [ ] Can see training curves

**If all checked: You're ready to present! 🎉**

---

## 📞 Need Help?

1. Check the main README.md
2. Re-read error messages carefully
3. Verify all files exist
4. Try with a smaller dataset (sample 5000 transactions)

---

**Remember:** The dashboard is just a visualization tool. The actual ML work happens in your Colab notebook. Make sure that runs successfully first!

Good luck with your project! 🚀
