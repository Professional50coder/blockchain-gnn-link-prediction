# Enhancement Roadmap

A prioritised map of everything worth doing to this project, derived from reading the source rather than the documentation. Each item carries a **priority**, an **effort** estimate, and the **payoff** — so you can cut the list at whatever line your time budget lands on.

| Priority | Meaning |
|---|---|
| **P0** | Fix before showing this to anyone technical. Correctness or security. |
| **P1** | Substantially raises the quality of the work. Do these next. |
| **P2** | Polish and product surface. Good, not urgent. |
| **P3** | Ambitious extensions. Portfolio or paper material. |

---

## A · Scientific correctness

The category that decides whether the results are defensible.

| # | Item | Pri | Effort | Payoff |
|---|---|---|---|---|
| A1 | **Fix degree-feature leakage.** Node features are in/out-degree computed over the *full* edge list including test edges. Recompute from `train_pos_edge_index` only. | **P0** | 30 min | Turns a meaningless AUC 1.000 into a number you can defend |
| A2 | **Re-report all metrics after A1** and delete the leaked `roc_data.npz`. | **P0** | 20 min | Every claim downstream depends on this |
| A3 | **Reconcile the two decoders.** The notebook trains with plain dot product; the app serves a 0.5/0.3/0.2 blend. Either train the blend or serve the trained decoder. | **P0** | 1–2 h | Served probabilities currently do not come from the optimised objective |
| A4 | **Normalise node features.** Raw degrees produce logits large enough to need `clip(±500)` before the sigmoid — that clip is a symptom, not a fix. Standardise or log1p the features. | **P0** | 30 min | Stabilises training; loss currently oscillates between 823 and 20,741 |
| A5 | **Add ranking metrics** — Hits@K, MRR, precision@K. AUC alone is a weak signal for link prediction on sparse graphs. | P1 | 2 h | The metrics reviewers actually ask for |
| A6 | **Validation-based early stopping.** 100 fixed epochs with an unstable loss means the saved weights are whichever epoch happened to be last. | P1 | 1 h | Reproducibility |
| A7 | **Deterministic nearest neighbours.** `find_top_k_similar` samples 3,000 random wallets, so "top 5 similar" changes between calls. Use the full matrix or a fixed index. | P1 | 1 h | Users notice inconsistent answers |
| A8 | **Hard negative sampling.** Uniform random negatives are trivially separable. Sample negatives from 2-hop neighbourhoods. | P1 | 2 h | Removes the other half of the "too easy" problem |

---

## B · Model quality

| # | Item | Pri | Effort | Payoff |
|---|---|---|---|---|
| B1 | **Richer node features** — transaction count, total value in/out, unique counterparties, wallet age, gas percentiles, contract-vs-EOA flag. | P1 | 3–4 h | Biggest single accuracy lever available |
| B2 | **Temporal edge split.** Train on earlier transactions, test on later ones. This is the only split that answers "will they transact *next*". | P1 | 2 h | Converts a plausibility model into a prediction model |
| B3 | **Edge features** — value and timestamp per transfer, via `SAGEConv` alternatives that accept edge attributes. | P2 | 3 h | Value-aware predictions |
| B4 | **Compare architectures** — GAT and GIN against GraphSAGE on the same split, reported as a table. | P2 | 3 h | Turns one model into an experiment; GAT attention also gives interpretability |
| B5 | **Probability calibration** — Platt scaling or isotonic regression, so 0.7 actually means 70%. | P2 | 1 h | The dashboard presents these as probabilities; they should behave like probabilities |
| B6 | **Validate the fraud flags.** Cross-check the 507 against Etherscan address tags and public scam lists to report real precision. | P1 | 3 h | Converts "anomalous" into a defensible fraud claim |

---

## C · Data

| # | Item | Pri | Effort | Payoff |
|---|---|---|---|---|
| C1 | **Scale the graph.** 29,023 transactions is small; pull 500k–1M from BigQuery. GraphSAGE was designed to sample, so it scales. | P1 | 3 h | Every result gets more credible |
| C2 | **Reproducible ingestion script** — the BigQuery SQL committed alongside the notebook, not just its output. | P1 | 1 h | Nobody can currently reproduce your dataset |
| C3 | **Token transfers, not just ETH.** ERC-20 transfers are the majority of interesting activity. | P3 | 4 h | Much richer graph |
| C4 | **Data card** — provenance, date range, block range, filters applied. | P2 | 1 h | Required for any serious write-up |

---

## D · Engineering

| # | Item | Pri | Effort | Payoff |
|---|---|---|---|---|
| D1 | **Delete `app_fixed.py`.** 1,896 lines duplicating `app.py`. Two sources of truth is worse than one imperfect one. | **P0** | 5 min | Immediate clarity |
| D2 | **Remove `.local/` from tracking** — 607 committed files of agent scaffolding that have nothing to do with the project. Add to `.gitignore`. | **P0** | 10 min | The repo currently looks unmaintained at first glance |
| D3 | **Pin dependency versions.** `requirements.txt` is unpinned, and Streamlit has already broken `use_container_width` once (see `FIXES.md`). | **P0** | 15 min | Stops the app from breaking on a random Tuesday |
| D4 | **Split `app.py`.** 2,181 lines and 9 sections → Streamlit multipage under `pages/`. | P1 | 3 h | Navigable code; smaller reruns |
| D5 | **Tests.** `pytest` over `ml_utils` maths, the Etherscan response parser, and artifact loading. | P1 | 3 h | The decoder is pure functions — it is trivially testable and currently untested |
| D6 | **CI** — GitHub Actions running lint, tests, and a headless smoke-launch of the app. | P1 | 2 h | Catches breakage before a viewer does |
| D7 | **Fold `FIXES.md` into git history.** A file documenting bugs already fixed belongs in commits, not the repo root. | P2 | 10 min | Cleaner surface |
| D8 | **Clean the notebook.** Duplicate cells, out-of-order execution, dead `precision_at_k` that returns a constant. | P1 | 1 h | The notebook is the scientific record; it should read as one |
| D9 | **Type hints and docstrings** across `utils/`. Partly present in `ml_utils.py`, absent elsewhere. | P2 | 2 h | Maintainability |

---

## E · Security

| # | Item | Pri | Effort | Payoff |
|---|---|---|---|---|
| E1 | **Rotate the leaked Etherscan key** committed in `replit.md`, and remove it from the file. | **P0** | 10 min | It is public in the git history right now |
| E2 | **Rotate the GitHub PAT** if one has ever been pasted into a chat, terminal history, or file. | **P0** | 5 min | Full write access to every repo on the account |
| E3 | **`.env.example`** documenting every variable the app reads, with no real values. | P1 | 15 min | Onboarding without leaking |
| E4 | **`gitleaks` pre-commit hook.** | P2 | 20 min | Stops E1 from happening again |
| E5 | **Purge secrets from git history** with `git filter-repo` — rotation alone leaves the old key readable in old commits. | P2 | 30 min | Complete remediation |

---

## F · Performance

| # | Item | Pri | Effort | Payoff |
|---|---|---|---|---|
| F1 | **Persist the PCA projection** to disk instead of recomputing per session over 6,000 sampled wallets. | P2 | 30 min | Faster first paint on the Embedding Space tab |
| F2 | **Approximate nearest neighbours** (FAISS or Annoy) replacing the 3,000-row scan. Also fixes A7. | P2 | 1.5 h | Exact and fast, instead of approximate and slow |
| F3 | **Parquet instead of CSV** for `edges.csv` via pyarrow. | P3 | 20 min | Marginal at 29k rows, essential at 1M (see C1) |

---

## G · Product surface

| # | Item | Pri | Effort | Payoff |
|---|---|---|---|---|
| G1 | **Explain the flag.** For each suspicious wallet, show *why* — nearest flagged neighbours, degree percentile, which behaviour is unusual. | P1 | 3 h | The single biggest usability gap; a score with no reason is not actionable |
| G2 | **Export.** CSV/JSON download of the flagged list and of any investigation. | P2 | 1 h | Makes the tool usable in a real workflow |
| G3 | **Batch address upload** — paste 50 addresses, get 50 scores. | P2 | 2 h | Moves from demo to tool |
| G4 | **Deep links** — `?wallet=0x…` so a finding can be shared. | P2 | 1 h | Collaboration |
| G5 | **Empty and error states** for unknown wallets, API failures, rate limits. | P1 | 2 h | Currently the failure path is a stack trace |
| G6 | **Watchlist** persisted across sessions. | P3 | 3 h | Repeat usage |

---

## H · Documentation

| # | Item | Pri | Effort | Payoff |
|---|---|---|---|---|
| H1 | **README rewrite with architecture diagrams.** | **P0** | — | ✅ Done |
| H2 | **Model card** — intended use, out-of-scope use, training data, metrics, ethical considerations. | P1 | 1.5 h | Expected for any published ML artefact |
| H3 | **Reconcile `replit.md` and `README.md`.** They describe the same system with different section counts and different decoders. | P1 | 30 min | Contradictory docs undermine both |
| H4 | **Notebook narrative** — markdown cells explaining each step, so the notebook reads as a report. | P1 | 2 h | Doubles as your written submission |
| H5 | **`CONTRIBUTING.md`** and issue templates. | P3 | 30 min | Signals a maintained project |

---

## I · Deployment

| # | Item | Pri | Effort | Payoff |
|---|---|---|---|---|
| I1 | **Dockerfile** + compose for one-command local run. | P2 | 1 h | Reproducible anywhere |
| I2 | **Live deploy** on Streamlit Community Cloud or Hugging Face Spaces, linked from the README. | P1 | 1 h | A URL beats a screenshot in every conversation |
| I3 | **Health endpoint and structured logging.** | P3 | 1 h | Operability |

---

## J · Research extensions

| # | Item | Pri | Effort | Payoff |
|---|---|---|---|---|
| J1 | **Temporal GNN** (TGN / TGAT) — model *when*, not just *whether*. | P3 | 1–2 wk | The natural paper-shaped next step |
| J2 | **Entity clustering** — group wallets likely controlled by one actor, from embedding proximity plus co-spending heuristics. | P3 | 1 wk | The commercially valuable capability |
| J3 | **Streaming ingestion** — subscribe to new blocks and update the graph incrementally. | P3 | 2 wk | Removes the snapshot limitation entirely |
| J4 | **Subgraph explanations** via GNNExplainer — highlight the edges responsible for a prediction. | P3 | 1 wk | Interpretability, and a strong demo |

---

## Suggested order

**Weekend one — make it defensible.**
A1 → A2 → A4 → A3 → D1 → D2 → D3 → E1 → E2

Nine items, roughly a day of work, and they change the project from "impressive-looking but leaky" to "honest and solid". Everything else builds on a foundation that currently has a hole in it.

**Weekend two — make it good.**
B1 → B2 → C1 → A5 → G1 → D5 → I2

Richer features on a temporal split over a larger graph, with real metrics, explanations for each flag, tests, and a live URL.

**After that**, pick from B4, G2–G5, and the J-series depending on whether you are optimising for a portfolio, a grade, or a paper.
