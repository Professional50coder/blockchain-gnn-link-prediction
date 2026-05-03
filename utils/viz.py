"""
utils/viz.py — All Plotly visualisations: charts, network graph, PCA, architecture diagram.
"""
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

from utils.theme import DARK, LIGHT
from utils.ml_utils import risk_color_hex

logger = logging.getLogger(__name__)


def _p(dark: bool) -> dict:
    return DARK if dark else LIGHT


# ── Training loss ─────────────────────────────────────────────────────────

def plot_loss_curve(loss_history: np.ndarray, dark: bool = True) -> go.Figure:
    p = _p(dark)
    epochs = list(range(1, len(loss_history) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=loss_history, mode="lines", name="Training Loss",
        line=dict(color=p["primary"], width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({'88,166,255' if dark else '9,105,218'},0.07)",
    ))
    fig.add_annotation(
        x=epochs[-1], y=float(loss_history[-1]),
        text=f"Final: {loss_history[-1]:.4f}",
        showarrow=True, arrowhead=2, ax=-60, ay=-30,
        font=dict(color=p["primary"], size=11),
    )
    fig.update_layout(
        title="GNN Training Loss", xaxis_title="Epoch", yaxis_title="Loss",
        hovermode="x unified", template=p["plotly"], height=360,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=40, l=40, r=20),
    )
    return fig


# ── ROC curve ────────────────────────────────────────────────────────────

def plot_roc_curve(fpr, tpr, auc_score: float, dark: bool = True) -> go.Figure:
    p = _p(dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"ROC (AUC = {auc_score:.3f})",
        line=dict(color=p["primary"], width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({'88,166,255' if dark else '9,105,218'},0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(color=p["text_muted"], width=1.5, dash="dash"),
    ))
    fig.update_layout(
        title="ROC Curve — Link Prediction",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        template=p["plotly"], height=360,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=40, l=40, r=20),
    )
    return fig


# ── Fraud distribution ────────────────────────────────────────────────────

def plot_fraud_distribution(fraud_df: pd.DataFrame, dark: bool = True) -> go.Figure:
    p = _p(dark)
    fig = px.histogram(
        fraud_df, x="risk_score", nbins=50,
        title="Risk Score Distribution (0 = safe · 100 = critical)",
        labels={"risk_score": "Risk Score"},
        color_discrete_sequence=[p["danger"]],
    )
    for thresh, label, color in [(35, "Medium", p["warning"]), (60, "High", p["danger"]), (80, "Critical", "#ff7b72")]:
        fig.add_vline(x=thresh, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_position="top right",
                      annotation_font_color=color)
    fig.update_layout(template=p["plotly"], height=340,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=40, b=40, l=40, r=20))
    return fig


# ── Network subgraph ──────────────────────────────────────────────────────

def create_network_subgraph(
    edges_df: pd.DataFrame, wallet_id: int, le, fraud_df: pd.DataFrame,
    max_connections: int = 25, layout: str = "spring", show_labels: bool = True,
    dark: bool = True,
) -> go.Figure | None:
    p = _p(dark)
    related = edges_df[
        (edges_df["from_id"] == wallet_id) | (edges_df["to_id"] == wallet_id)
    ].head(max_connections)
    if len(related) == 0:
        return None

    fraud_risk  = fraud_df.set_index("wallet_id")["risk_score"].to_dict() if "wallet_id" in fraud_df.columns else {}
    fraud_level = fraud_df.set_index("wallet_id")["risk_level"].to_dict() if "wallet_id" in fraud_df.columns else {}
    fraud_ids   = set(fraud_risk.keys())

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
        if   u == wallet_id: out_ex += [x0,x1,None]; out_ey += [y0,y1,None]
        elif v == wallet_id: in_ex  += [x0,x1,None]; in_ey  += [y0,y1,None]
        else:                int_ex += [x0,x1,None]; int_ey += [y0,y1,None]
        mx, my = (x0+x1)/2, (y0+y1)/2
        arrows.append(dict(x=x1, y=y1, ax=mx, ay=my, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.2,
                           arrowcolor=(p["primary"] if u==wallet_id else p["accent"] if v==wallet_id else "rgba(148,163,184,0.35)")))

    def edge_trace(ex, ey, color, name):
        return go.Scatter(x=ex, y=ey, mode="lines", name=name,
                          line=dict(width=1.3, color=color), hoverinfo="none", showlegend=bool(ex))

    traces = [
        edge_trace(out_ex, out_ey, f"rgba({'88,166,255' if dark else '9,105,218'},0.5)", "Outgoing"),
        edge_trace(in_ex,  in_ey,  f"rgba({'163,113,247' if dark else '130,80,223'},0.5)", "Incoming"),
        edge_trace(int_ex, int_ey, "rgba(148,163,184,0.2)", "Internal"),
    ]

    cats = {
        "center_fraud": {"color":"#e3b341","border":"#d29922","size":34,"label":"⭐ Centre (Flagged)"},
        "center":       {"color":p["primary"],"border":p["primary2"],"size":32,"label":"● Centre Wallet"},
        "fraud":        {"color":p["danger"],"border":"#ff7b72","size":22,"label":"● Flagged"},
        "normal":       {"color":p["text_muted"],"border":p["border2"],"size":14,"label":"● Normal"},
    }
    cat_nodes: dict[str, list] = {k: [] for k in cats}
    for node in G.nodes():
        k = ("center_fraud" if node==wallet_id and node in fraud_ids else
             "center"       if node==wallet_id else
             "fraud"        if node in fraud_ids else "normal")
        cat_nodes[k].append(node)

    for cat, cfg in cats.items():
        nids = cat_nodes[cat]
        if not nids: continue
        xs, ys, hvr, sizes, syms = [], [], [], [], []
        for node in nids:
            x, y = pos[node]; xs.append(x); ys.append(y)
            try:    addr = le.inverse_transform([node])[0]
            except: addr = f"ID {node}"
            rs = fraud_risk.get(node); rl = fraud_level.get(node,"Clean")
            hvr.append(f"<b>{addr}</b><br>ID:{node}<br>Risk:{rl}"
                       + (f" ({rs:.1f}/100)" if rs else "")
                       + f"<br>Out:{out_deg.get(node,0)} In:{in_deg.get(node,0)}"
                       + (" ⭐CENTER" if node==wallet_id else ""))
            sizes.append(cfg["size"] + min(int((in_deg.get(node,0)+out_deg.get(node,0))**0.5)*2,14))
            syms.append("star" if node==wallet_id else "circle")
        traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text" if show_labels else "markers",
            name=cfg["label"], hoverinfo="text", hovertext=hvr,
            text=[(le.inverse_transform([n])[0][:8]+"…" if show_labels else "") for n in nids],
            textposition="top center",
            textfont=dict(size=7, color=p["text_muted"], family="JetBrains Mono"),
            marker=dict(size=sizes, color=cfg["color"], symbol=syms,
                        line=dict(width=2, color=cfg["border"]), opacity=0.92),
            showlegend=True,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        annotations=arrows,
        title=dict(text=f"Transaction Network — Wallet {wallet_id}",
                   font=dict(size=12, color=p["text_muted"], family="Inter")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=10, color=p["text_muted"]),
                    bgcolor=f"rgba({'13,17,23' if dark else '246,248,250'},0.85)",
                    bordercolor=p["border"], borderwidth=1),
        hovermode="closest", template=p["plotly"],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600, margin=dict(t=80, b=20, l=20, r=20),
    )
    return fig


# ── PCA Embedding Space ───────────────────────────────────────────────────

def plot_pca_embeddings(
    coords: np.ndarray,           # (N, 2 or 3) PCA coordinates
    sampled_ids: np.ndarray,      # wallet IDs for each point
    fraud_df: pd.DataFrame,
    le,
    dark: bool = True,
    highlight_id: int = -1,
    mode_3d: bool = False,
) -> go.Figure:
    p = _p(dark)

    fraud_map = fraud_df.set_index("wallet_id") if "wallet_id" in fraud_df.columns else pd.DataFrame()

    labels, colors, sizes, hover = [], [], [], []
    for i, wid in enumerate(sampled_ids):
        if len(fraud_map) > 0 and wid in fraud_map.index:
            rl = fraud_map.loc[wid, "risk_level"]
            rs = float(fraud_map.loc[wid, "risk_score"])
        else:
            rl = "Clean"; rs = 0.0
        try:    addr = le.inverse_transform([int(wid)])[0]
        except: addr = f"ID {wid}"

        labels.append(rl)
        colors.append({"Critical":"#f85149","High":"#ff7b72","Medium":"#e3b341",
                       "Low":"#3fb950","Clean":"#58a6ff"}.get(rl, p["text_muted"]))
        sizes.append(10 if rs > 60 else (7 if rs > 35 else 5))
        hover.append(f"<b>ID {wid}</b><br>{addr[:20]}…<br>Risk: {rl} ({rs:.1f})")

    # Highlight queried wallet
    hl_trace = None
    if highlight_id >= 0 and highlight_id in sampled_ids:
        idx = list(sampled_ids).index(highlight_id)
        try: hl_addr = le.inverse_transform([highlight_id])[0]
        except: hl_addr = f"ID {highlight_id}"
        if mode_3d and coords.shape[1] >= 3:
            hl_trace = go.Scatter3d(
                x=[coords[idx,0]], y=[coords[idx,1]], z=[coords[idx,2]],
                mode="markers+text",
                marker=dict(size=16, color="#e3b341", symbol="diamond",
                            line=dict(width=3, color="#ffffff")),
                text=[f"  {hl_addr[:12]}…"], textfont=dict(color="#e3b341", size=10),
                hovertext=f"<b>QUERIED</b><br>ID {highlight_id}", hoverinfo="text",
                name="Queried Wallet", showlegend=True,
            )
        else:
            hl_trace = go.Scatter(
                x=[coords[idx,0]], y=[coords[idx,1]],
                mode="markers+text",
                marker=dict(size=18, color="#e3b341", symbol="star",
                            line=dict(width=3, color="#ffffff")),
                text=[f"  {hl_addr[:12]}…"], textfont=dict(color="#e3b341", size=10),
                hovertext=f"<b>QUERIED</b><br>ID {highlight_id}", hoverinfo="text",
                name="Queried Wallet", showlegend=True,
            )

    if mode_3d and coords.shape[1] >= 3:
        trace = go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(size=sizes, color=colors, opacity=0.75,
                        line=dict(width=0.3, color="rgba(0,0,0,0.2)")),
            hovertext=hover, hoverinfo="text",
            name="Wallets", showlegend=False,
        )
        data = [trace] + ([hl_trace] if hl_trace else [])
        fig = go.Figure(data=data)
        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=True, gridcolor=p["border"], title="PC1",
                           backgroundcolor="rgba(0,0,0,0)"),
                yaxis=dict(showgrid=True, gridcolor=p["border"], title="PC2",
                           backgroundcolor="rgba(0,0,0,0)"),
                zaxis=dict(showgrid=True, gridcolor=p["border"], title="PC3",
                           backgroundcolor="rgba(0,0,0,0)"),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=600, template=p["plotly"],
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20, l=10, r=10),
            title=dict(text="Wallet Embedding Space — 3D PCA",
                       font=dict(size=13, color=p["text_muted"])),
        )
    else:
        x, y = coords[:, 0], coords[:, 1]
        trace = go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=sizes, color=colors, opacity=0.72,
                        line=dict(width=0.3, color="rgba(0,0,0,0.15)")),
            hovertext=hover, hoverinfo="text",
            name="Wallets", showlegend=False,
        )
        data = [trace] + ([hl_trace] if hl_trace else [])
        fig = go.Figure(data=data)
        fig.update_layout(
            xaxis_title="PC 1", yaxis_title="PC 2",
            height=560, template=p["plotly"],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor=p["border"], zeroline=False),
            yaxis=dict(showgrid=True, gridcolor=p["border"], zeroline=False),
            margin=dict(t=40, b=40, l=40, r=20),
            title=dict(text="Wallet Embedding Space — 2D PCA",
                       font=dict(size=13, color=p["text_muted"])),
        )

    # Colour legend as invisible traces
    for label, color in [("Critical","#f85149"),("High","#ff7b72"),
                         ("Medium","#e3b341"),("Low","#3fb950"),("Clean","#58a6ff")]:
        ScatterType = go.Scatter3d if mode_3d else go.Scatter
        kw = dict(x=[None], y=[None]) if not mode_3d else dict(x=[None], y=[None], z=[None])
        fig.add_trace(ScatterType(
            **kw, mode="markers",
            marker=dict(size=9, color=color),
            name=label, showlegend=True,
        ))
    return fig


# ── GNN Architecture Diagram ──────────────────────────────────────────────

def plot_architecture_diagram(dark: bool = True) -> go.Figure:
    """
    Draw an interactive GraphSAGE architecture diagram using Plotly shapes + scatter.
    """
    p = _p(dark)
    bg  = p["bg"]
    muted = p["text_muted"]

    # Layer definitions: (x, label, sublabel, color, num_nodes_to_draw, y_spread)
    layers = [
        {"x": 0,    "label": "Input",             "sub": "2 features\nin-degree · out-degree",
         "color": p["primary"], "n": 2,  "spread": 1.4},
        {"x": 2.5,  "label": "GraphSAGE\nLayer 1","sub": "64 dims\nMEAN aggregation",
         "color": p["accent"],  "n": 9,  "spread": 0.55},
        {"x": 5.0,  "label": "GraphSAGE\nLayer 2","sub": "64 dims\nMEAN aggregation",
         "color": p["accent"],  "n": 9,  "spread": 0.55},
        {"x": 7.5,  "label": "Node\nEmbedding",   "sub": "64-dim vector\nrepresentation",
         "color": p["success"], "n": 9,  "spread": 0.55},
        {"x": 9.8,  "label": "Decoder\n(3 signals)","sub": "dot · cosine · L2\nweighted combine",
         "color": p["warning"], "n": 3,  "spread": 1.2},
        {"x": 11.8, "label": "Output",             "sub": "Link probability\n∈ [0, 1]",
         "color": "#e3b341",    "n": 1,  "spread": 0},
    ]

    fig = go.Figure()

    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        """Convert #rrggbb hex to rgba() string."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    # ── Background layer boxes ──────────────────────────────────────────
    box_w = 1.0
    for layer in layers:
        x0 = layer["x"] - box_w / 2
        x1 = layer["x"] + box_w / 2
        n  = layer["n"]
        y_half = (n - 1) * layer["spread"] / 2 + 0.8
        fig.add_shape(type="rect",
                      x0=x0, x1=x1, y0=-y_half - 0.4, y1=y_half + 0.4,
                      fillcolor=_hex_to_rgba(layer["color"], 0.10 if dark else 0.08),
                      line=dict(color=_hex_to_rgba(layer["color"], 0.35 if dark else 0.40), width=1.2),
                      layer="below")

    # ── Connection lines between adjacent layers ───────────────────────
    for i in range(len(layers) - 1):
        l1, l2 = layers[i], layers[i + 1]
        ys1 = (np.linspace(0, l1["n"]-1, l1["n"]) - (l1["n"]-1)/2) * l1["spread"]
        ys2 = (np.linspace(0, l2["n"]-1, l2["n"]) - (l2["n"]-1)/2) * l2["spread"]
        line_alpha = 0.06 if dark else 0.05
        for y1 in ys1:
            for y2 in ys2:
                fig.add_shape(type="line",
                              x0=l1["x"] + box_w/2, y0=y1,
                              x1=l2["x"] - box_w/2, y1=y2,
                              line=dict(color=_hex_to_rgba(l1["color"], line_alpha), width=0.8))

    # ── Layer nodes ─────────────────────────────────────────────────────
    for layer in layers:
        n = layer["n"]
        ys = (np.linspace(0, n-1, n) - (n-1)/2) * layer["spread"]
        hover_texts = [
            f"<b>{layer['label'].replace(chr(10),' ')}</b><br>{layer['sub'].replace(chr(10),'<br>')}"
        ] * n

        fig.add_trace(go.Scatter(
            x=[layer["x"]] * n, y=list(ys),
            mode="markers",
            marker=dict(
                size=22 if n <= 2 else (16 if n <= 4 else 11),
                color=layer["color"],
                line=dict(width=2, color=bg),
                opacity=0.95,
            ),
            hoverinfo="text", hovertext=hover_texts,
            showlegend=False,
        ))

    # ── Flow arrows between layers ───────────────────────────────────────
    for i in range(len(layers) - 1):
        l1, l2 = layers[i], layers[i + 1]
        fig.add_annotation(
            x=l2["x"] - box_w / 2 - 0.05,
            y=0,
            ax=l1["x"] + box_w / 2 + 0.05,
            ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.4,
            arrowwidth=2.5,
            arrowcolor=f"rgba({'88,166,255' if dark else '9,105,218'},0.55)",
        )

    # ── Layer name labels below ──────────────────────────────────────────
    for layer in layers:
        n = layer["n"]
        y_bottom = -(n - 1) * layer["spread"] / 2 - 1.4
        fig.add_annotation(
            x=layer["x"], y=y_bottom - 0.1,
            text=f"<b>{layer['label']}</b>",
            font=dict(size=11, color=layer["color"], family="Inter"),
            showarrow=False, yanchor="top", align="center",
        )
        fig.add_annotation(
            x=layer["x"], y=y_bottom - 0.75,
            text=layer["sub"],
            font=dict(size=8.5, color=muted, family="Inter"),
            showarrow=False, yanchor="top", align="center",
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=460,
        margin=dict(t=20, b=130, l=20, r=20),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.8, 12.8]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-6.5, 4.5]),
        hovermode="closest",
    )
    return fig


# ── Decoder breakdown ─────────────────────────────────────────────────────

def plot_decoder_signals(emb_a: np.ndarray, emb_b: np.ndarray, dark: bool = True) -> go.Figure:
    """Bar chart showing the three decoder signal contributions."""
    p = _p(dark)
    dot    = float(np.clip(np.dot(emb_a, emb_b), -500, 500))
    norm_a = np.linalg.norm(emb_a) + 1e-9
    norm_b = np.linalg.norm(emb_b) + 1e-9
    cosine = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
    l2_sim = 1.0 / (1.0 + float(np.linalg.norm(emb_a - emb_b)))
    raw    = np.clip(0.5*dot + 0.3*cosine*abs(dot) + 0.2*l2_sim*abs(dot), -500, 500)
    prob   = float(1.0 / (1.0 + np.exp(-raw)))

    signals = ["Dot Product\n(weight 0.5)", "Cosine Similarity\n(weight 0.3)", "L2 Similarity\n(weight 0.2)"]
    raw_vals = [dot * 0.5, cosine * abs(dot) * 0.3, l2_sim * abs(dot) * 0.2]
    colors = [p["primary"], p["accent"], p["success"]]

    fig = go.Figure()
    for sig, val, col in zip(signals, raw_vals, colors):
        fig.add_trace(go.Bar(
            name=sig, x=[sig.split("\n")[0]], y=[val],
            marker_color=col,
            text=[f"{val:.3f}"],
            textposition="outside",
        ))
    fig.add_hline(y=0, line=dict(color=p["border2"], width=1))
    fig.update_layout(
        template=p["plotly"], height=280,
        title=dict(text=f"Decoder Signals  →  σ(combined) = {prob:.4f}",
                   font=dict(size=12, color=p["text_muted"])),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, bargap=0.35,
        margin=dict(t=45, b=40, l=40, r=20),
        yaxis_title="Weighted Contribution",
    )
    return fig


# ── Transaction timeline ──────────────────────────────────────────────────

def plot_transaction_timeline(tx_df: pd.DataFrame, dark: bool = True) -> go.Figure:
    """Bubble timeline of recent transactions by block number."""
    p = _p(dark)
    if tx_df.empty or "Block" not in tx_df.columns:
        return go.Figure()

    df = tx_df.copy()
    df["Colour"] = df["Status"].map({"✅": p["success"], "❌": p["danger"]}).fillna(p["primary"])
    df["Size"]   = df["Value (ETH)"].apply(lambda v: max(8, min(30, int(v * 200 + 8))))

    fig = go.Figure()
    for status, grp in df.groupby("Status"):
        col = p["success"] if status == "✅" else p["danger"]
        fig.add_trace(go.Scatter(
            x=grp["Block"], y=[0.5] * len(grp),
            mode="markers+text",
            marker=dict(size=grp["Size"], color=col, opacity=0.8,
                        line=dict(width=1.5, color="rgba(255,255,255,0.2)")),
            text=grp["Value (ETH)"].apply(lambda v: f"{v:.4f}" if v > 0 else ""),
            textfont=dict(size=7, color=p["text_muted"]),
            textposition="top center",
            hovertext=grp.apply(
                lambda r: f"Block {r['Block']}<br>To: {r['To'][:12]}…<br>"
                          f"Value: {r['Value (ETH)']} ETH<br>Protocol: {r['Protocol']}", axis=1),
            hoverinfo="text",
            name=f"{'Success' if status=='✅' else 'Failed'}",
        ))

    fig.update_layout(
        template=p["plotly"], height=220,
        xaxis_title="Block Number",
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=40, l=10, r=10),
        title=dict(text="Transaction Timeline (bubble size ∝ ETH value)",
                   font=dict(size=11, color=p["text_muted"])),
    )
    return fig


# ── Event log chart ───────────────────────────────────────────────────────

def plot_event_log_chart(events_df: pd.DataFrame, dark: bool = True) -> go.Figure:
    """Bar chart of event counts by block range (histogram)."""
    p = _p(dark)
    if events_df.empty or "Block" not in events_df.columns:
        return go.Figure()

    fig = go.Figure()
    for direction, col in [("Received", p["success"]), ("Sent", p["primary"])]:
        grp = events_df[events_df["Direction"] == direction]
        if grp.empty: continue
        fig.add_trace(go.Histogram(
            x=grp["Block"], nbinsx=30,
            name=direction,
            marker_color=col, opacity=0.8,
        ))
    fig.update_layout(
        barmode="overlay",
        template=p["plotly"], height=240,
        xaxis_title="Block Number", yaxis_title="Transfer Events",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=40, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        title=dict(text="ERC-20 Transfer Event Distribution",
                   font=dict(size=11, color=p["text_muted"])),
    )
    return fig


# ── Degree distribution ───────────────────────────────────────────────────

def plot_degree_distributions(edges_df: pd.DataFrame, dark: bool = True) -> tuple[go.Figure, go.Figure]:
    p = _p(dark)
    out_deg = edges_df.groupby("from_id").size()
    in_deg  = edges_df.groupby("to_id").size()

    def _hist(series, title, color):
        fig = px.histogram(series, nbins=60, title=title,
                           labels={"value": "Degree"},
                           color_discrete_sequence=[color])
        fig.update_layout(template=p["plotly"], height=300,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=40,b=40,l=40,r=20),
                          showlegend=False)
        return fig

    return (_hist(out_deg, "Out-Degree Distribution", p["primary"]),
            _hist(in_deg,  "In-Degree Distribution",  p["accent"]))
