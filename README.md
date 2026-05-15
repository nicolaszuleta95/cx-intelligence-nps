# CX Intelligence — NPS & Complaint Severity Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Production-FF6600?style=flat)
![Streamlit](https://img.shields.io/badge/Demo-Live-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5?style=flat)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat)

**End-to-end ML system for banking customer experience analytics —
complaint severity prediction, NPS analysis, topic mining, and customer segmentation.**

[Live Demo](https://nicolaszuleta95-cx-intelligence-nps.streamlit.app) · [Notebooks](notebooks/) · [Model Card](reports/model_card.md)

</div>

---

## Business Problem

Customer Experience teams at banks receive thousands of complaints every month. **NPS tells you *how many* customers are dissatisfied — but not *why*, and not *which ones to prioritize*.**

Reading complaints manually is impossible at scale. A team analyzing 10,000 monthly complaints would need weeks just to triage them — let alone extract actionable insights.

**This project automates that process end-to-end:**

1. Calculates and tracks NPS by product, channel, and time period
2. **Predicts complaint severity (LOW / MEDIUM / HIGH) to prioritize CX action**
3. Discovers the recurring themes driving dissatisfaction (LDA topic modeling)
4. Segments customers by their satisfaction and severity profile (K-Means)
5. Delivers all insights through an interactive Streamlit dashboard

---

## Results

| Metric | Value |
|--------|-------|
| **Severity model — XGBoost accuracy** | **74.7%** |
| **Severity model — F1-weighted** | **0.726** |
| **Baseline — Logistic Regression F1** | 0.713 |
| **Model lift over random triage** | **4.4×** |
| **HIGH severity recall (top-20% review)** | **87%** |
| **Estimated cost saving (test set)** | **$24,500** |
| **Overall NPS (complaint dataset)** | **−5.59** (expected — 100% complaints) |
| **Records analysed** | **35,226** banking complaints with narrative |
| **Topics discovered** | **4** (LDA, k optimised by perplexity) |
| **Customer segments** | **7** (K-Means, k optimised by silhouette) |

> **Key finding:** The #1 severity predictor is `response_type_encoded` (73.4% importance) — *how* the bank resolved the complaint matters more than any other variable. `timely_response` is #2 (21.3%). Complaint text contributes only 2.2% via word count. Correct problem framing outperforms NLP complexity.

---

## Model Performance Detail

### Severity Predictor (XGBoost, Optuna-tuned)

| Model | Accuracy | F1-weighted | Notes |
|-------|----------|-------------|-------|
| Logistic Regression | 73.0% | 0.713 | Interpretable baseline |
| **XGBoost (tuned)** | **74.7%** | **0.726** | Production · +1.3 pp F1 |

5-fold CV: accuracy **74.1% ± 0.4%**

### Feature Importance

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `response_type_encoded` | **73.4%** |
| 2 | `timely_response_binary` | 21.3% |
| 3 | `complaint_length` | 2.2% |
| 4 | `product_encoded` | 2.2% |
| 5 | `days_to_resolution` | 0.8% |

### Business Impact (conservative assumptions)

| Triage method | HIGH detected | Rate | Est. churn cost |
|---------------|--------------|------|-----------------|
| Random 20% review | 41 / 208 | 20% | $29,225 |
| **Model top-20% by P(HIGH)** | **181 / 208** | **87%** | **$4,725** |
| **Lift / Saving** | | **4.4×** | **$24,500** |

*Assumptions: $500 per lost customer · 35% churn if HIGH unaddressed · 8% if resolved within 24h*

---

## Key Findings

**NPS by product (simulated, complaint corpus):**

| Product | NPS |
|---------|-----|
| Credit card | **+11.1** |
| Checking / Savings | +3.5 |
| Personal loan | −12.8 |
| Student loan | −14.9 |
| Mortgage | **−15.8** |

**Top dissatisfaction drivers by topic (NPS):**

| Topic | NPS | Complaints | % Detractors |
|-------|-----|-----------|-------------|
| Incorrect Charges & Unauthorized Debits | **−17.9** | 6,894 | 29.1% |
| Mortgage & Loan Servicing | −6.8 | 1,774 | 25.4% |
| Credit Reporting & Disputes | −6.4 | 19,493 | 25.7% |
| Customer Service & Communication | **+8.9** | 7,059 | 20.2% |

> Volume ≠ urgency: Credit Reporting dominates volume (55%) but ranks only #2 by NPS damage. Incorrect Charges is 3.5× smaller but generates 2.7× more NPS loss per complaint.

**Customer segments (K-Means, k=7):**

| Segment | Avg NPS | Dominant Severity | % Detractors | % HIGH | n |
|---------|---------|------------------|--------------|--------|---|
| Critical Risk | 5.1 | **HIGH** | 68.6% | **100%** | 905 |
| Silent Dissatisfied | 6.5 | MEDIUM | 43.4% | 0% | 6,200 |
| Neutral Observers | 6.5 | MEDIUM | 43.5% | 0% | 7,471 |
| Promoter Candidates | 6.5 | MEDIUM | 43.9% | 0% | 5,287 |
| Active Promoters | 8.4 | MEDIUM | 0.0% | 0% | 3,073 |
| Promoter | 8.5 | MEDIUM | 0.1% | 0% | 4,908 |
| Promoter | 8.7 | LOW | 0.0% | 0% | 7,376 |

> The Critical Risk segment (2.6% of customers) concentrates 100% of HIGH-severity complaints. NPS alone (5.1) would not distinguish them from the other 6.5-NPS clusters — severity is the differentiating signal.

---

## System Architecture

```
consumer_complaints.csv (555K records)
        │
        ▼
[01] EDA & NPS Analysis ──────────────────────────── banking_complaints.csv
        │                                             (35,226 with narrative)
        ▼
[02] Severity Predictor (XGBoost) ───────────────── severity_model.joblib
        │   8 structured features                     features_nlp.csv
        │   Optuna hyperparameter tuning              + severity columns
        ▼
[03] Topic Mining (LDA k=4) ─────────────────────── lda_model.pkl
        │   TF-IDF + spaCy preprocessing              + topic columns
        ▼
[04] Customer Segmentation (K-Means k=7) ────────── kmeans_model.joblib
        │   severity + NPS + topic features            + cluster columns
        ▼
[app] Streamlit Dashboard (4 sections)
```

| Layer | Tool | Role |
|-------|------|------|
| Preprocessing | spaCy `en_core_web_sm` | Tokenization, lemmatization for LDA |
| NPS Analysis | Custom logic | Simulate and calculate NPS from CFPB resolution data |
| Severity Prediction | XGBoost + Optuna | Predict complaint severity from 8 structured features |
| Topic Mining | sklearn LDA + TF-IDF | Discover 4 recurring complaint themes |
| Segmentation | sklearn K-Means | 7 CX-actionable customer segments |
| Dashboard | Streamlit + Plotly | Interactive analytics app |

---

## Project Structure

```
cx-intelligence-nps/
│
├── data/
│   ├── raw/
│   │   └── consumer_complaints.csv      # CFPB dataset — never modified
│   └── processed/
│       ├── banking_complaints.csv       # Filtered · 35,226 rows with narrative
│       └── features_nlp.csv            # Severity + topics + clusters (accumulated)
│
├── notebooks/
│   ├── 01_eda_nps_analysis.ipynb        # EDA · NPS −5.59 · by product / time / channel
│   ├── 02_complaint_severity.ipynb      # XGBoost severity · 74.7% acc · 4.4× lift
│   ├── 03_topic_mining.ipynb            # LDA k=4 · NPS by topic · top drivers
│   └── 04_customer_segmentation.ipynb   # K-Means k=7 · CX segment profiles
│
├── src/
│   ├── data_processing.py    # Load, filter, clean CFPB data
│   ├── nps_calculator.py     # NPS simulation, classification, trend
│   ├── severity.py           # Feature engineering + SeverityPredictor class
│   ├── topics.py             # TF-IDF + LDA + topic assignment
│   └── segmentation.py       # K-Means + cluster profiles + naming
│
├── models/
│   ├── severity_model.joblib    # XGBoost · 1.3 MB
│   ├── severity_features.json   # Feature order for inference reproducibility
│   ├── lda_model.pkl            # LDA + TF-IDF vectorizer · 512 KB
│   └── kmeans_model.joblib      # K-Means · 139 KB
│
├── app/
│   └── app.py               # Streamlit dashboard · 4 sections
│
├── reports/
│   └── model_card.md        # Model documentation · decisions · limitations
│
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** [US Consumer Finance Complaints — CFPB](https://www.kaggle.com/datasets/cfpb/us-consumer-finance-complaints)

| Property | Value |
|----------|-------|
| Provider | Consumer Financial Protection Bureau (US Government) |
| Raw records | 555,957 |
| After banking product filter | 354,805 |
| After narrative text filter | **35,226** (9.9% have free-text narrative) |
| Submission channel | 100% Web (CFPB requires consent to share narratives) |
| Period | March 2015 – April 2016 |

Banking products included:
```python
["Checking or savings account", "Credit card or prepaid card",
 "Mortgage", "Personal loan", "Student loan"]
```

> **Data note:** NPS scores are **simulated** from CFPB resolution data (timeliness + outcome → proxy satisfaction score). This is documented explicitly — not presented as measured NPS.

---

## Key Technical Decisions

**1. Severity over sentiment**  
Sentiment classification was empirically rejected after 6 models achieved 33–41% accuracy (3-class random baseline: 33%). Root cause: CFPB is 100% complaints — no polarity signal. Severity from structured variables solved the problem correctly.

**2. XGBoost as primary model**  
Consistent with [Project 1: Banking Churn Prediction](https://github.com/nicolaszuleta95/banking-churn-prediction). Handles structured tabular banking data without normalization. Built-in feature importance without SHAP.

**3. Optuna for hyperparameter tuning**  
More efficient than GridSearchCV over XGBoost's hyperparameter space. Falls back to GridSearchCV automatically if Optuna is not installed.

**4. sklearn LDA over gensim**  
gensim 3.8 is incompatible with scipy ≥ 1.14 (`triu` removed). Coherence scoring reimplemented using sklearn held-out perplexity — no external dependency, same directional signal for k selection.

**5. NPS simulated from resolution data**  
CFPB doesn't include satisfaction scores. NPS simulated from response timeliness + resolution outcome as proxy. Documented transparently in model card.

---

## How to Run

**1. Clone and install**
```bash
git clone https://github.com/nicolaszuleta95/cx-intelligence-nps
cd cx-intelligence-nps
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**2. Prepare data**
```bash
# Place consumer_complaints.csv in data/raw/
# (download from Kaggle link above)
# Run notebooks in order:
jupyter notebook notebooks/01_eda_nps_analysis.ipynb
jupyter notebook notebooks/02_complaint_severity.ipynb
jupyter notebook notebooks/03_topic_mining.ipynb
jupyter notebook notebooks/04_customer_segmentation.ipynb
```

**3. Launch the dashboard**
```bash
streamlit run app/app.py
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data | `pandas`, `numpy` |
| NLP preprocessing | `spaCy` (`en_core_web_sm`) |
| Severity prediction | `xgboost`, `lightgbm`, `scikit-learn` |
| Hyperparameter tuning | `optuna` (GridSearchCV fallback) |
| Topic modeling | `scikit-learn` (TF-IDF + LDA) |
| Segmentation | `scikit-learn` (K-Means) |
| Visualization | `plotly`, `seaborn`, `matplotlib`, `wordcloud` |
| App | `streamlit` |
| Serialization | `joblib` |

---

## Model Card

See [`reports/model_card.md`](reports/model_card.md) for:
- Intended use and out-of-scope applications
- Feature engineering decisions and encoding logic
- Severity label definition (rule-based + NPS refinement)
- Evaluation metrics and business impact
- Known limitations and ethical considerations

---

## Author

**Nicolás Zuleta Sierra**  
Data Scientist · 7+ years · Banking & CX Analytics · Medellín, Colombia

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nicolaszuletasierra/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github&logoColor=white)](https://github.com/nicolaszuleta95)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
