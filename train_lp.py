"""Leakage-free link-prediction evaluation for ChainIntel Pro.

The original notebook computed node degrees over the FULL edge list before splitting, so degree
alone separated real edges from random negatives. This script splits first, builds features and
message-passing edges from TRAIN edges only (with a supervision-only slice), and reports validation/test ROC-AUC next to two
non-learned baselines. It writes metrics.json and never touches the dashboard artifacts.

    python train_lp.py                    # random edge split -> metrics.json
    python train_lp.py --split temporal   # oldest 70% train / next 15% val / newest 15% test
                                          #   -> metrics_temporal.json
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["random", "temporal"], default="random")
SPLIT = ap.parse_args().split
SEED, HIDDEN, EPOCHS, LR = 42, 64, 100, 0.01
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# --- data: same cleaning as the notebook, plus de-duplication of repeated transfers
df = pd.read_csv("ethereum_transactions.csv").dropna(subset=["from_address", "to_address"])
df = df[df["from_address"] != df["to_address"]]
codes, uniq = pd.factorize(pd.concat([df["from_address"], df["to_address"]]))
n = len(uniq)
src, dst = codes[: len(df)], codes[len(df):]
pairs = np.stack([src, dst], 1)
edges = np.unique(pairs, axis=0)
if SPLIT == "temporal":
    ts = pd.to_datetime(df["block_timestamp"], utc=True, errors="coerce").astype("int64").to_numpy()
    # first time each unique edge was seen; order edges oldest -> newest
    first_seen = pd.Series(ts).groupby([src, dst]).min().reindex(list(map(tuple, edges.tolist()))).to_numpy()
    edges = edges[np.argsort(first_seen, kind="stable")]
print(f"nodes={n:,} unique directed edges={len(edges):,}")

# --- split FIRST (70/15/15), everything below sees train edges only
n_val = n_test = int(0.15 * len(edges))
if SPLIT == "random":
    perm = rng.permutation(len(edges))
    val_e, test_e, train_e = edges[perm[:n_val]], edges[perm[n_val:n_val + n_test]], edges[perm[n_val + n_test:]]
else:  # edges are ordered oldest -> newest: train = oldest, test = newest
    train_e, val_e, test_e = edges[: -(n_val + n_test)], edges[-(n_val + n_test): -n_test], edges[-n_test:]
existing = set(map(tuple, edges.tolist()))

def sample_neg(k):
    out = []
    while len(out) < k:
        a, b = rng.integers(0, n, 2)
        if a != b and (a, b) not in existing:
            out.append((a, b))
    return np.array(out)

val_neg, test_neg = sample_neg(len(val_e)), sample_neg(len(test_e))

# --- features and graph from TRAIN edges only
out_deg = np.bincount(train_e[:, 0], minlength=n)
in_deg = np.bincount(train_e[:, 1], minlength=n)
x = torch.tensor(np.log1p(np.stack([out_deg, in_deg], 1)), dtype=torch.float)
# Message passing uses 80% of the train edges; the other 20% are supervision-only, so the model is
# never scored on an edge it also aggregated over (the cause of the falling validation AUC before).
p2 = rng.permutation(len(train_e)); k = int(0.8 * len(train_e))
mp_e, sup_e = train_e[p2[:k]], train_e[p2[k:]]
ei = torch.tensor(np.concatenate([mp_e, mp_e[:, ::-1]]).T.copy(), dtype=torch.long)  # both directions
deg = torch.bincount(ei[1], minlength=n).clamp(min=1).float().unsqueeze(1)

class SAGE(torch.nn.Module):
    """GraphSAGE with mean aggregation (SAGEConv(2,64) -> ReLU -> SAGEConv(64,64)), pure torch."""
    def __init__(self, i, h):
        super().__init__()
        self.s1, self.n1 = torch.nn.Linear(i, h), torch.nn.Linear(i, h, bias=False)
        self.s2, self.n2 = torch.nn.Linear(h, h), torch.nn.Linear(h, h, bias=False)
    def agg(self, h):
        m = torch.zeros_like(h).index_add_(0, ei[1], h[ei[0]])
        return m / deg
    def forward(self, x):
        h = F.relu(self.s1(x) + self.n1(self.agg(x)))
        return self.s2(h) + self.n2(self.agg(h))

model = SAGE(2, HIDDEN)
opt = torch.optim.Adam(model.parameters(), lr=LR)
T = lambda a: torch.tensor(a, dtype=torch.long)
score = lambda z, e: (z[e[:, 0]] * z[e[:, 1]]).sum(1)

def auc(z, pos, neg):
    with torch.no_grad():
        s = torch.cat([score(z, T(pos)), score(z, T(neg))]).numpy()
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], s)

best, best_state, hist = 0, None, []
for ep in range(1, EPOCHS + 1):
    model.train(); opt.zero_grad()
    z = model(x)
    neg = T(sample_neg(len(sup_e)))
    loss = F.binary_cross_entropy_with_logits(
        torch.cat([score(z, T(sup_e)), score(z, neg)]),
        torch.cat([torch.ones(len(sup_e)), torch.zeros(len(neg))]))
    loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        z = model(x)
    v = auc(z, val_e, val_neg); hist.append(float(loss.detach()))
    if v > best: best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    if ep % 20 == 0: print(f"epoch {ep:3d} loss {loss:.4f} val AUC {v:.4f}")

model.load_state_dict(best_state); model.eval()
with torch.no_grad():
    z = model(x)

# --- baselines using train degrees only
pa = lambda e: np.log1p(out_deg[e[:, 0]]) + np.log1p(in_deg[e[:, 1]])
lab = np.r_[np.ones(len(test_e)), np.zeros(len(test_neg))]
res = {
    "nodes": int(n), "unique_edges": int(len(edges)), "train/val/test": [len(train_e), len(val_e), len(test_e)],
    "graphsage_val_auc": round(best, 4),
    "graphsage_test_auc": round(auc(z, test_e, test_neg), 4),
    "baseline_degree_product_test_auc": round(roc_auc_score(lab, np.r_[pa(test_e), pa(test_neg)]), 4),
    "loss_first_last": [round(hist[0], 4), round(hist[-1], 4)],
    "split": SPLIT,
    "note": f"features from train edges; message passing on 80% of train edges, loss on the other 20%; {SPLIT} edge split; random negatives",
}
print(json.dumps(res, indent=2))
json.dump(res, open("metrics.json" if SPLIT == "random" else "metrics_temporal.json", "w"), indent=2)
