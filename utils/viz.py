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
    epochs     = list(range(1, len(loss_history) + 1))
    best_epoch = int(np.argmin(loss_history)) + 1
    best_loss  = float(loss_history[best_epoch - 1])
    init_loss  = float(loss_history[0])
    final_loss = float(loss_history[-1])
    reduction  = (init_loss - final_loss) / init_loss * 100

    fig = go.Figure()
    # Shaded area + main line
    fig.add_trace(go.Scatter(
        x=epochs, y=loss_history, mode="lines", name="Training Loss",
        line=dict(color=p["primary"], width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({'88,166,255' if dark else '9,105,218'},0.07)",
        hovertemplate=(
            "<b>Training Epoch %{x}</b><br>"
            "BCE Loss: %{y:,.3f}<br>"
            "<span style='color:#8b949e;font-size:11px;'>"
            "L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]<br>"
            "y=true label · ŷ=predicted probability"
            "</span><extra></extra>"
        ),
    ))
    # Best epoch star marker
    fig.add_trace(go.Scatter(
        x=[best_epoch], y=[best_loss],
        mode="markers", name=f"Best (epoch {best_epoch})",
        marker=dict(color=p["success"], size=12, symbol="star",
                    line=dict(color=p["bg"], width=1.5)),
        hovertemplate=(
            f"<b>⭐ Best Epoch: {best_epoch}</b><br>"
            f"Minimum BCE Loss: {best_loss:,.3f}<br>"
            f"Reduction from start: {(init_loss - best_loss)/init_loss*100:.1f}%<extra></extra>"
        ),
    ))
    # Initial loss annotation (top-left)
    fig.add_annotation(
        x=1, y=init_loss,
        text=f"Start: {init_loss:,.0f}",
        showarrow=True, arrowhead=2, ax=50, ay=0,
        font=dict(color=p["warning"], size=11),
        bgcolor=p["card"], bordercolor=p["warning"], borderwidth=1,
    )
    # Best epoch annotation
    fig.add_annotation(
        x=best_epoch, y=best_loss,
        text=f"Best: {best_loss:.1f}",
        showarrow=True, arrowhead=2, ax=45, ay=-38,
        font=dict(color=p["success"], size=11),
        bgcolor=p["card"], bordercolor=p["success"], borderwidth=1,
    )
    # Final loss annotation
    fig.add_annotation(
        x=epochs[-1], y=final_loss,
        text=f"Final: {final_loss:.1f}",
        showarrow=True, arrowhead=2, ax=-60, ay=-30,
        font=dict(color=p["primary"], size=11),
        bgcolor=p["card"], bordercolor=p["border"], borderwidth=1,
    )
    # Reduction callout (centre of chart)
    mid_epoch = len(epochs) // 2
    fig.add_annotation(
        xref="paper", yref="paper", x=0.5, y=0.72,
        text=f"▼ {reduction:.0f}% loss reduction over {len(epochs)} epochs",
        showarrow=False,
        font=dict(color=p["text_muted"], size=10),
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title=f"GNN Training Loss (GraphSAGE) — {len(epochs)} Epochs",
        xaxis_title="Epoch", yaxis_title="Loss",
        hovermode="x unified", template=p["plotly"], height=360,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=40, l=40, r=20),
        legend=dict(orientation="h", yanchor="top", y=1.08, xanchor="right", x=1),
    )
    return fig


# ── ROC curve ────────────────────────────────────────────────────────────

def plot_roc_curve(fpr, tpr, auc_score: float, dark: bool = True) -> go.Figure:
    p = _p(dark)
    auc_color = p["success"] if auc_score >= 0.9 else (p["warning"] if auc_score >= 0.7 else p["danger"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"Isolation Forest (AUC = {auc_score:.4f})",
        line=dict(color=p["primary"], width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({'88,166,255' if dark else '9,105,218'},0.12)",
        hovertemplate=(
            "<b>ROC Operating Point</b><br>"
            "FPR = FP / (FP + TN) : %{x:.4f}<br>"
            "TPR = TP / (TP + FN) : %{y:.4f}<br>"
            "<i style='color:#8b949e;font-size:11px;'>"
            "Sensitivity = Recall = Hit Rate = TPR<br>"
            "Specificity = TN / (TN + FP) = 1 − FPR"
            "</i><extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random (AUC = 0.50)",
        line=dict(color=p["text_muted"], width=1.5, dash="dash"),
        hoverinfo="skip",
    ))
    # AUC badge — top-left near the perfect curve
    fig.add_annotation(
        x=0.04, y=0.96, xref="paper", yref="paper",
        text=f"AUC = {auc_score:.4f}",
        showarrow=False,
        font=dict(color=auc_color, size=15, family="JetBrains Mono, monospace"),
        bgcolor=p["card"], bordercolor=auc_color, borderwidth=1.5,
        borderpad=8,
    )
    quality = ("Perfect separation" if auc_score >= 0.99 else
               "Excellent" if auc_score >= 0.9 else
               "Good" if auc_score >= 0.7 else "Moderate")
    fig.add_annotation(
        x=0.04, y=0.85, xref="paper", yref="paper",
        text=f"★ {quality}",
        showarrow=False,
        font=dict(color=p["text_muted"], size=10),
        bgcolor="rgba(0,0,0,0)",
    )
    # AUC integral formula annotation
    fig.add_annotation(
        x=0.04, y=0.74, xref="paper", yref="paper",
        text="AUC = ∫₀¹ TPR(t) dt",
        showarrow=False,
        font=dict(color=p["text_subtle"], size=9, family="JetBrains Mono, monospace"),
        bgcolor="rgba(0,0,0,0)",
    )
    # Youden's J index: max(TPR - FPR) operating point
    j_index = np.array(tpr) - np.array(fpr)
    best_j = int(np.argmax(j_index))
    fig.add_trace(go.Scatter(
        x=[float(fpr[best_j])], y=[float(tpr[best_j])],
        mode="markers", name=f"Youden's J (FPR={float(fpr[best_j]):.3f})",
        marker=dict(color=p["warning"], size=11, symbol="diamond",
                    line=dict(color=p["bg"], width=1.5)),
        hovertemplate=(
            "<b>Youden's J Operating Point</b><br>"
            f"J = TPR − FPR = {j_index[best_j]:.4f}<br>"
            f"FPR: {float(fpr[best_j]):.4f}  TPR: {float(tpr[best_j]):.4f}<br>"
            "<i style='color:#8b949e;font-size:11px;'>Optimal threshold by J-statistic</i>"
            "<extra></extra>"
        ),
    ))
    # Random baseline label
    fig.add_annotation(
        x=0.76, y=0.68, xref="paper", yref="paper",
        text="random",
        showarrow=False,
        font=dict(color=p["text_muted"], size=9),
        bgcolor="rgba(0,0,0,0)",
        textangle=-38,
    )
    fig.update_layout(
        title="ROC Curve — Isolation Forest on GNN Embeddings  (AUC = ∫₀¹ TPR(t) dt)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[-0.02, 1.02], constrain="domain",
                   tickformat=".1f", gridcolor=p["border"]),
        yaxis=dict(range=[-0.02, 1.02], scaleanchor="x",
                   tickformat=".1f", gridcolor=p["border"]),
        template=p["plotly"], height=360,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=40, l=40, r=20),
        legend=dict(orientation="h", yanchor="top", y=1.08, xanchor="right", x=1),
    )
    return fig


# ── Fraud distribution ────────────────────────────────────────────────────

def plot_fraud_distribution(fraud_df: pd.DataFrame, dark: bool = True) -> go.Figure:
    p = _p(dark)
    scores = fraud_df["risk_score"]
    n_low      = int((scores < 35).sum())
    n_medium   = int(((scores >= 35) & (scores < 60)).sum())
    n_high     = int(((scores >= 60) & (scores < 80)).sum())
    n_critical = int((scores >= 80).sum())

    fig = px.histogram(
        fraud_df, x="risk_score", nbins=50,
        title="Risk Score Distribution  (0 = safe · 100 = critical)  ·  s(x,n) = 2^(−E[h(x)] / c(n))",
        labels={"risk_score": "Risk Score", "count": "Wallets"},
        color_discrete_sequence=[p["danger"]],
        hover_data={"risk_score": ":.1f"},
    )
    fig.update_traces(
        hovertemplate=(
            "<b>Risk Score Band: %{x:.0f}</b><br>"
            "Wallets in band: %{y:,}<br>"
            "<span style='color:#8b949e;font-size:11px;'>"
            "IF Score: s(x,n) = 2^(−E[h(x)] / c(n))<br>"
            "Risk = (s_max − s) / (s_max − s_min) × 100<br>"
            "Higher risk score → shorter isolation path"
            "</span><extra></extra>"
        )
    )
    tiers = [
        (35, f"Medium<br><b>{n_medium}</b>", p["warning"]),
        (60, f"High<br><b>{n_high}</b>",     p["danger"]),
        (80, f"Critical<br><b>{n_critical}</b>", "#ff7b72"),
    ]
    for thresh, label, color in tiers:
        fig.add_vline(x=thresh, line_dash="dot", line_color=color, line_width=1.5,
                      annotation_text=label, annotation_position="top right",
                      annotation_font_color=color, annotation_font_size=11)
    fig.add_annotation(
        x=17, yref="paper", y=0.97,
        text=f"🟢 Low<br><b>{n_low}</b>",
        showarrow=False, font=dict(color=p["success"], size=11),
        align="center")
    fig.update_layout(
        template=p["plotly"], height=340,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=40, l=40, r=20),
        xaxis=dict(title="Risk Score (0–100)", range=[-2, 102]),
        yaxis_title="Number of Wallets",
    )
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
            except Exception: addr = f"ID {node}"
            rs = fraud_risk.get(node); rl = fraud_level.get(node,"Clean")
            hvr.append(
                f"<b>{addr[:22]}{'…' if len(addr)>22 else ''}</b>"
                f"<br>Wallet ID: {node}"
                f"<br>Risk Level: {rl}" + (f" · Score: {rs:.1f}/100" if rs else " · Score: —")
                + f"<br>Out-degree (sent txs): {out_deg.get(node,0)}"
                + f"<br>In-degree (recv txs): {in_deg.get(node,0)}"
                + f"<br>Total degree: {out_deg.get(node,0) + in_deg.get(node,0)}"
                + ("<br><b>⭐ QUERY CENTRE</b>" if node==wallet_id else "")
                + ("<br><i style='color:#f85149;'>⚠ Flagged by Isolation Forest</i>" if (rs and rs > 0) else "")
            )
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
        except Exception: addr = f"ID {wid}"

        labels.append(rl)
        colors.append({"Critical":"#f85149","High":"#ff7b72","Medium":"#e3b341",
                       "Low":"#3fb950","Clean":"#58a6ff"}.get(rl, p["text_muted"]))
        sizes.append(10 if rs > 60 else (7 if rs > 35 else 5))
        hover.append(
            f"<b>ID {wid}</b><br>{addr[:22]}{'…' if len(addr)>22 else ''}"
            f"<br>Risk: {rl} · Score: {rs:.1f}/100"
            f"<br><i style='color:#8b949e;font-size:10px;'>z = W·x via PCA · z ∈ ℝ²"
            f"<br>Wallets nearby = similar GNN embedding</i>"
        )

    # Highlight queried wallet
    hl_trace = None
    if highlight_id >= 0 and highlight_id in sampled_ids:
        idx = list(sampled_ids).index(highlight_id)
        try: hl_addr = le.inverse_transform([highlight_id])[0]
        except Exception: hl_addr = f"ID {highlight_id}"
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


# ── Live animated architecture diagram (HTML/Canvas) ──────────────────────

def render_architecture_animation(dark: bool = True) -> str:
    """Return a self-contained HTML string with a fully animated GraphSAGE
    architecture diagram: particle flow, pulsing nodes, animated dashes,
    hover tooltips, and click-to-burst interaction."""
    df = 'true' if dark else 'false'
    bg = '#0d1117' if dark else '#f0f6ff'
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{background:{bg};overflow:hidden;width:100%;height:100%}}
canvas{{display:block;cursor:crosshair}}
#tip{{position:fixed;background:rgba(13,17,23,.94);color:#e6edf3;
      border:1px solid #58a6ff55;border-radius:9px;padding:8px 12px;
      font:12px -apple-system,'Inter',sans-serif;pointer-events:none;
      white-space:pre;display:none;z-index:99;line-height:1.55;
      box-shadow:0 4px 20px rgba(0,0,0,.6)}}
#hdr{{position:absolute;top:10px;left:0;right:0;text-align:center;
      font:bold 12px -apple-system,'Inter',sans-serif;
      color:{'#8b949e' if dark else '#656d76'};letter-spacing:.06em;
      pointer-events:none;user-select:none}}
#hint{{position:absolute;bottom:8px;right:14px;
       font:9px -apple-system,sans-serif;color:{'#484f58' if dark else '#aaa'};
       pointer-events:none;user-select:none}}
</style></head>
<body>
<canvas id="c"></canvas>
<div id="tip"></div>
<div id="hdr">GRAPHSAGE ARCHITECTURE &nbsp;·&nbsp; LIVE DATA FLOW</div>
<div id="hint">click any layer to burst · hover for details</div>
<script>
const DARK={df};
const BG='{bg}';
const PAL={{
  bg:BG,
  primary:'#58a6ff',accent:'#bc8cff',success:'#3fb950',
  warning:'#e3b341',orange:'#f0883e',danger:'#f85149',
  muted:DARK?'#8b949e':'#656d76',text:DARK?'#e6edf3':'#24292f',
  card:DARK?'#161b22':'#ffffff',border:DARK?'#30363d':'#d0d7de',
}};

const LAYER_DEFS=[
  {{id:'input',    label:'Input',         subs:['2 features','in · out degree'],
    formula:'x ∈ ℝ²',              color:PAL.primary, n:2,
    detail:'Raw topological features\\n2 values per wallet:\\n  • in-degree (txs received)\\n  • out-degree (txs sent)'}},
  {{id:'sage1',    label:'GraphSAGE L1',  subs:['64 dims','MEAN aggregation'],
    formula:'h¹=σ(W¹·CONCAT(h⁰,MEAN(N)))', color:PAL.accent,   n:6,
    detail:'1-hop neighbourhood aggregation\\nEach wallet collects its\\ndirect counterparties\\' embeddings'}},
  {{id:'sage2',    label:'GraphSAGE L2',  subs:['64 dims','MEAN aggregation'],
    formula:'h²=σ(W²·CONCAT(h¹,MEAN(N)))', color:PAL.accent,   n:6,
    detail:'2-hop neighbourhood aggregation\\nNow sees the counterparties\\nof counterparties (depth-2)'}},
  {{id:'embed',    label:'Embedding',     subs:['64-dim vector','L2 normalised'],
    formula:'z = L2(h²) ∈ ℝ⁶⁴',   color:PAL.success, n:6,
    detail:'Final wallet representation\\n64-dimensional L2-normalised\\nvector · used for fraud scoring\\nand link prediction'}},
  {{id:'decoder',  label:'Decoder',       subs:['dot · cos · L2','weighted score'],
    formula:'s=0.5·dot+0.3·cos+0.2·L2',color:PAL.warning, n:3,
    detail:'3 complementary similarity signals:\\n  • dot product  (weight 0.5)\\n  • cosine sim   (weight 0.3)\\n  • L2 distance  (weight 0.2)'}},
  {{id:'output',   label:'Output',        subs:['Link probability','∈ [0, 1]'],
    formula:'p = σ(score)',         color:PAL.orange,  n:1,
    detail:'Transaction link probability\\np → 1.0 = very likely link\\np → 0.0 = unlikely link\\nThreshold 0.5 for classification'}},
];

const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
const tip=document.getElementById('tip');
let W=0,H=0,dpr=1,layers=[],particles=[],frame=0;
const BOX_W=74,NODE_R=8,PAD_X=52,PAD_TOP=52,PAD_BOT=90;
const NPART=100;

function hex2rgb(h){{
  const r=parseInt(h.slice(1,3),16),g=parseInt(h.slice(3,5),16),b=parseInt(h.slice(5,7),16);
  return [r,g,b];
}}
function rgba(h,a){{const[r,g,b]=hex2rgb(h);return`rgba(${{r}},${{g}},${{b}},${{a}})`;}}

function resize(){{
  dpr=window.devicePixelRatio||1;
  W=window.innerWidth;H=window.innerHeight;
  canvas.width=Math.round(W*dpr);canvas.height=Math.round(H*dpr);
  canvas.style.width=W+'px';canvas.style.height=H+'px';
  ctx.scale(dpr,dpr);
  buildLayout();
}}

function buildLayout(){{
  const usableW=W-PAD_X*2;
  const step=usableW/(LAYER_DEFS.length-1);
  const centerY=(H-PAD_TOP-PAD_BOT)/2+PAD_TOP;
  layers=LAYER_DEFS.map((def,i)=>{{
    const cx=PAD_X+i*step;
    const gap=Math.min(38,Math.max(24,(H-PAD_TOP-PAD_BOT-40)/(def.n)));
    const totalH=(def.n-1)*gap;
    const nodes=Array.from({{length:def.n}},(_,j)=>{{
      const x=cx,y=centerY-totalH/2+j*gap;
      return {{x,y,phase:Math.random()*Math.PI*2,active:0,flash:0}};
    }});
    return {{...def,cx,nodes,gap,step}};
  }});
  spawnAll();
}}

function spawnParticle(fromIdx){{
  const li=fromIdx!==undefined?fromIdx:Math.floor(Math.random()*(layers.length-1));
  if(li>=layers.length-1)return spawnParticle(0);
  const lA=layers[li],lB=layers[li+1];
  if(!lA||!lB)return;
  const nA=lA.nodes[Math.floor(Math.random()*lA.n)];
  const nB=lB.nodes[Math.floor(Math.random()*lB.n)];
  const speed=0.0045+Math.random()*0.0055;
  return {{
    x0:nA.x,y0:nA.y,x1:nB.x,y1:nB.y,
    t:Math.random(),speed,
    color:lA.color,r:1.8+Math.random()*1.6,
    alpha:0.65+Math.random()*0.35,
    trail:[],li,nBIdx:lB.nodes.indexOf(nB),
  }};
}}

function spawnAll(){{
  particles.length=0;
  for(let i=0;i<NPART;i++){{
    const p=spawnParticle();
    if(p)particles.push(p);
  }}
}}

let hoverLayer=-1,mx=-999,my=-999;
canvas.addEventListener('mousemove',e=>{{
  const r=canvas.getBoundingClientRect();
  mx=e.clientX-r.left;my=e.clientY-r.top;
  hoverLayer=-1;
  for(let i=0;i<layers.length;i++){{
    if(Math.abs(mx-layers[i].cx)<BOX_W/2+12){{hoverLayer=i;break;}}
  }}
  if(hoverLayer>=0){{
    const l=layers[hoverLayer];
    tip.style.display='block';
    tip.style.left=Math.min(mx+16,W-180)+'px';
    tip.style.top=Math.max(my-10,8)+'px';
    tip.textContent=l.label+'\\n\\n'+l.detail;
  }}else{{tip.style.display='none';}}
}});
canvas.addEventListener('mouseleave',()=>{{hoverLayer=-1;tip.style.display='none';}});
canvas.addEventListener('click',e=>{{
  const r=canvas.getBoundingClientRect();
  const cx=e.clientX-r.left;
  for(let i=0;i<layers.length-1;i++){{
    if(Math.abs(cx-layers[i].cx)<BOX_W/2+14){{
      for(let j=0;j<18;j++){{const p=spawnParticle(i);if(p){{p.t=0;particles.push(p);}}}}
      for(const n of layers[i].nodes){{n.flash=1;n.active=1;}}
      break;
    }}
  }}
}});

function rrect(x,y,w,h,r){{
  ctx.beginPath();
  ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);
  ctx.arcTo(x+w,y,x+w,y+r,r);ctx.lineTo(x+w,y+h-r);
  ctx.arcTo(x+w,y+h,x+w-r,y+h,r);ctx.lineTo(x+r,y+h);
  ctx.arcTo(x,y+h,x,y+h-r,r);ctx.lineTo(x,y+r);
  ctx.arcTo(x,y,x+r,y,r);ctx.closePath();
}}

function draw(){{
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle=BG;ctx.fillRect(0,0,W,H);

  /* dot grid */
  if(DARK){{
    for(let gx=22;gx<W;gx+=26)for(let gy=22;gy<H;gy+=26){{
      ctx.beginPath();ctx.arc(gx,gy,0.7,0,Math.PI*2);
      ctx.fillStyle='rgba(255,255,255,0.028)';ctx.fill();
    }}
  }}

  if(!layers.length){{requestAnimationFrame(draw);return;}}

  /* ── connection lines ───────────────────────────────────── */
  for(let i=0;i<layers.length-1;i++){{
    const lA=layers[i],lB=layers[i+1];
    for(const nA of lA.nodes)for(const nB of lB.nodes){{
      ctx.beginPath();ctx.moveTo(nA.x,nA.y);ctx.lineTo(nB.x,nB.y);
      ctx.strokeStyle=rgba(lA.color,hoverLayer===i?0.10:0.032);
      ctx.lineWidth=0.75;ctx.stroke();
    }}
  }}

  /* ── particles ──────────────────────────────────────────── */
  for(let i=particles.length-1;i>=0;i--){{
    const p=particles[i];
    p.t+=p.speed;
    if(p.t>=1){{
      /* activate destination node */
      const lB=layers[p.li+1];
      if(lB&&lB.nodes[p.nBIdx]){{
        const nd=lB.nodes[p.nBIdx];
        nd.active=Math.min(1,(nd.active||0)+0.45);
      }}
      particles.splice(i,1);
      const np=spawnParticle();if(np)particles.push(np);
      continue;
    }}
    const x=p.x0+(p.x1-p.x0)*p.t,y=p.y0+(p.y1-p.y0)*p.t;
    p.trail.push({{x,y}});if(p.trail.length>12)p.trail.shift();

    /* trail */
    for(let j=0;j<p.trail.length;j++){{
      const a=(j/p.trail.length)*p.alpha*0.38;
      const r=p.r*(0.3+0.7*j/p.trail.length);
      ctx.beginPath();ctx.arc(p.trail[j].x,p.trail[j].y,r,0,Math.PI*2);
      ctx.fillStyle=rgba(p.color,a);ctx.fill();
    }}
    /* outer glow */
    const g=ctx.createRadialGradient(x,y,0,x,y,p.r*4.5);
    g.addColorStop(0,rgba(p.color,p.alpha*0.35));
    g.addColorStop(1,rgba(p.color,0));
    ctx.beginPath();ctx.arc(x,y,p.r*4.5,0,Math.PI*2);
    ctx.fillStyle=g;ctx.fill();
    /* core */
    ctx.beginPath();ctx.arc(x,y,p.r,0,Math.PI*2);
    ctx.fillStyle=rgba(p.color,p.alpha);ctx.fill();
  }}

  /* ── layer boxes + nodes + labels ──────────────────────── */
  for(let i=0;i<layers.length;i++){{
    const l=layers[i];
    const isHov=hoverLayer===i;
    const pulse=0.5+0.5*Math.sin(frame*0.022+i*1.05);
    const minY=Math.min(...l.nodes.map(n=>n.y));
    const maxY=Math.max(...l.nodes.map(n=>n.y));
    const bx=l.cx-BOX_W/2,by=minY-NODE_R-18;
    const bh=(maxY-minY)+(NODE_R+18)*2;

    /* box shadow + fill */
    ctx.save();
    ctx.shadowBlur=isHov?28:12+pulse*9;
    ctx.shadowColor=rgba(l.color,.38);
    rrect(bx,by,BOX_W,bh,11);
    ctx.fillStyle=rgba(l.color,isHov?0.13:0.065);ctx.fill();
    ctx.strokeStyle=rgba(l.color,isHov?0.65:0.22+pulse*0.16);
    ctx.lineWidth=isHov?2:1.25;ctx.stroke();
    ctx.restore();

    /* nodes */
    for(const nd of l.nodes){{
      nd.active=Math.max(0,(nd.active||0)-0.018);
      nd.flash=Math.max(0,(nd.flash||0)-0.04);
      const np=0.5+0.5*Math.sin(frame*0.028+nd.phase);
      const act=nd.active||0;

      /* outer pulse ring */
      ctx.beginPath();ctx.arc(nd.x,nd.y,NODE_R+5+np*3.5+act*7,0,Math.PI*2);
      ctx.fillStyle=rgba(l.color,0.042+act*0.08+nd.flash*0.15);ctx.fill();

      /* flash ring on activation */
      if(nd.flash>0.05){{
        ctx.beginPath();ctx.arc(nd.x,nd.y,NODE_R+10+nd.flash*8,0,Math.PI*2);
        ctx.strokeStyle=rgba(l.color,nd.flash*0.5);ctx.lineWidth=1.2;ctx.stroke();
      }}

      /* body glow */
      ctx.save();
      ctx.shadowBlur=7+np*5+act*14+(nd.flash||0)*10;
      ctx.shadowColor=rgba(l.color,.75);
      ctx.beginPath();ctx.arc(nd.x,nd.y,NODE_R+act*2.5,0,Math.PI*2);
      const ng=ctx.createRadialGradient(nd.x-2.5,nd.y-2.5,1,nd.x,nd.y,NODE_R+act*2.5);
      ng.addColorStop(0,rgba(l.color,0.92+act*.08));
      ng.addColorStop(1,rgba(l.color,0.52));
      ctx.fillStyle=ng;ctx.fill();
      ctx.restore();

      /* highlight ring */
      ctx.beginPath();ctx.arc(nd.x,nd.y,NODE_R,0,Math.PI*2);
      ctx.strokeStyle=`rgba(255,255,255,${{DARK?0.18:0.1}})`;
      ctx.lineWidth=1;ctx.stroke();
    }}

    /* formula label above box */
    ctx.textAlign='center';
    ctx.fillStyle=rgba(l.color,isHov?0.55:0.28);
    ctx.font=`italic 7.5px 'JetBrains Mono','Courier New',monospace`;
    ctx.fillText(l.formula,l.cx,by-7);

    /* layer name + subs below box */
    const lby=maxY+NODE_R+17;
    ctx.fillStyle=isHov?l.color:rgba(l.color,0.92);
    ctx.font=`bold 10.5px -apple-system,'Inter',sans-serif`;
    ctx.fillText(l.label,l.cx,lby);
    ctx.fillStyle=PAL.muted;
    ctx.font=`9px -apple-system,'Inter',sans-serif`;
    l.subs.forEach((s,si)=>ctx.fillText(s,l.cx,lby+13+si*11));

    /* node count badge */
    const badge=`${{l.n}} node${{l.n>1?'s':''}}`;
    ctx.fillStyle=rgba(l.color,isHov?0.55:0.22);
    ctx.font=`8px -apple-system,sans-serif`;
    ctx.fillText(badge,l.cx,by-19);
  }}

  /* ── animated dashed arrows between boxes ───────────────── */
  for(let i=0;i<layers.length-1;i++){{
    const x0=layers[i].cx+BOX_W/2+4,x1=layers[i+1].cx-BOX_W/2-4;
    const ay=(layers[i].nodes[0].y+layers[i].nodes[layers[i].n-1].y)/2;
    const pulse=0.5+0.5*Math.sin(frame*0.04+i*0.85);
    const col=layers[i].color;
    const dashOff=-(frame*1.1%16);

    /* animated dash line */
    ctx.save();
    ctx.setLineDash([5,5]);ctx.lineDashOffset=dashOff;
    ctx.beginPath();ctx.moveTo(x0,ay);ctx.lineTo(x1-9,ay);
    ctx.strokeStyle=rgba(col,0.32+pulse*0.22);ctx.lineWidth=1.4;ctx.stroke();
    ctx.restore();

    /* arrowhead */
    ctx.beginPath();
    ctx.moveTo(x1,ay);ctx.lineTo(x1-9,ay-4.5);ctx.lineTo(x1-9,ay+4.5);
    ctx.closePath();
    ctx.fillStyle=rgba(col,0.48+pulse*0.28);ctx.fill();
  }}

  frame++;
  requestAnimationFrame(draw);
}}

window.addEventListener('resize',()=>{{resize();}});
resize();draw();
</script></body></html>"""


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
    formulas = [
        f"0.5 × dot(a,b) = 0.5 × {dot:.4f} = {dot*0.5:.4f}",
        f"0.3 × cos(a,b) × |dot| = 0.3 × {cosine:.4f} × {abs(dot):.4f} = {raw_vals[1]:.4f}",
        f"0.2 × L2sim(a,b) × |dot| = 0.2 × {l2_sim:.4f} × {abs(dot):.4f} = {raw_vals[2]:.4f}",
    ]
    full_names = ["Dot Product (weight 0.50)", "Cosine Similarity (weight 0.30)", "L2 Similarity (weight 0.20)"]
    for sig, val, col, formula, fname in zip(signals, raw_vals, colors, formulas, full_names):
        fig.add_trace(go.Bar(
            name=sig, x=[sig.split("\n")[0]], y=[val],
            marker_color=col,
            text=[f"{val:.3f}"],
            textposition="outside",
            hovertemplate=(
                f"<b>{fname}</b><br>"
                f"Weighted contribution: {val:.5f}<br>"
                f"Formula: {formula}<br>"
                "<i style='color:#8b949e;font-size:11px;'>"
                "P(link) = σ(0.5·dot + 0.3·cos·|dot| + 0.2·L2sim·|dot|)"
                "</i><extra></extra>"
            ),
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
                lambda r: (
                    f"<b>Block {r['Block']}</b><br>"
                    f"To: {str(r['To'])[:16]}…<br>"
                    f"Value: {r['Value (ETH)']:.6f} ETH<br>"
                    f"Gas: {r.get('Gas (Gwei)', '—')} Gwei<br>"
                    f"Status: {r.get('Status','—')}<br>"
                    f"Protocol: {r.get('Protocol','—')}<br>"
                    f"<i style='color:#8b949e;font-size:10px;'>Bubble size ∝ ETH value</i>"
                ), axis=1),
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
    out_deg = edges_df.groupby("from_id").size().sort_values()
    in_deg  = edges_df.groupby("to_id").size().sort_values()

    def _make_fig(series, title, color, accent, top_id):
        # Value counts for clean bar chart (degree → wallet count)
        vc = series.value_counts().sort_index()
        # Cap x-axis at 99th percentile for readability (exclude extreme hubs)
        cap = int(np.percentile(series.values, 99))
        vc_capped = vc[vc.index <= cap]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=vc_capped.index.tolist(),
            y=vc_capped.values.tolist(),
            marker=dict(
                color=vc_capped.index.map(
                    lambda d: accent if d > 10 else color
                ).tolist(),
                opacity=0.85,
                line=dict(width=0),
            ),
            name="Wallets",
            hovertemplate="Degree %{x}: %{y:,} wallets<extra></extra>",
        ))

        # 99th-pct cap note
        if series.max() > cap:
            fig.add_annotation(
                x=cap, y=vc_capped[cap] if cap in vc_capped.index else vc_capped.iloc[-1],
                text=f"▶ max={series.max():,}",
                showarrow=False,
                font=dict(color=accent, size=9, family="JetBrains Mono, monospace"),
                xanchor="left", yanchor="bottom",
            )

        # Degree=1 dominance annotation
        if 1 in vc.index:
            pct1 = vc[1] / series.size * 100
            fig.add_annotation(
                x=1, y=vc[1],
                text=f"{pct1:.0f}% single-tx",
                showarrow=True, arrowhead=2, ax=35, ay=-28,
                font=dict(color=p["text_muted"], size=9),
                bgcolor=p["card"], bordercolor=p["border"], borderwidth=1,
            )

        # Hub threshold line
        fig.add_vline(
            x=10, line_dash="dot", line_color=accent, line_width=1.2,
            annotation_text="Hub threshold (>10)",
            annotation_position="top right",
            annotation_font=dict(color=accent, size=8),
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=13, color=p["text"], family="Inter")),
            xaxis_title="Degree",
            yaxis_title="Wallet count",
            yaxis_type="log",
            template=p["plotly"], height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=44, b=44, l=50, r=20),
            showlegend=False,
            bargap=0.05,
        )
        return fig

    fig_out = _make_fig(out_deg, "Out-Degree Distribution  (log scale)",
                        p["primary"], p["accent"],   out_deg.idxmax())
    fig_in  = _make_fig(in_deg,  "In-Degree Distribution   (log scale)",
                        p["teal"],   p["danger"],    in_deg.idxmax())
    return fig_out, fig_in
