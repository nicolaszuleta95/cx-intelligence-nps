# CX Intelligence — NPS & Complaint Severity Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Production-FF6600?style=flat)
![Streamlit](https://img.shields.io/badge/Demo-Live-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5?style=flat)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat)

**End-to-end ML system for banking customer experience analytics —
complaint severity prediction, NPS analysis, topic mining, and customer segmentation.**

[Live Demo](#) · [Notebooks](notebooks/) · [Model Card](reports/model_card.md)

</div>

---

## Business Problem

Customer Experience teams at banks receive thousands of complaints every month. **NPS tells you *how many* customers are dissatisfied — but not *why*, and not *which ones to prioritize*.**

Reading complaints manually is impossible at scale. A team analyzing 10,000 monthly complaints would need weeks just to triage them — let alone extract actionable insights.

**This project automates that process end-to-end:**

1. Calculates and tracks NPS by product, channel, and time period
2. **Predicts complaint severity (LOW / MEDIUM / HIGH) to prioritize CX action**
3. Discovers the recurring themes driving dissatisfaction
4. Segments customers by their satisfaction and severity profile
5. Delivers all insights through an interactive dashboard

---

## NPS Primer

**Net Promoter Score** is the standard CX metric in banking and financial services.

```
NPS = % Promoters − % Detractors

Promoters  (score 9–10): Loyal customers likely to recommend
Passives   (score 7–8):  Satisfied but not enthusiastic
Detractors (score 0–6):  Unhappy customers at risk of churn
```

NPS alone is a number. This system explains *what's behind it* — which products, topics, and customer segments are driving the score up or down, and **how urgently each complaint needs CX attention**.

---

## Why we moved away from sentiment classification

Sentiment classification (positive/negative/neutral) was initially planned for this project.
After testing **6 models** — including FinBERT, RoBERTa, VADER, Zero-Shot DeBERTa, and two
DistilBERT fine-tuning stages — all performed near random chance (**~33–41% accuracy**).

**Root cause:** The CFPB dataset is 100% complaints by definition, making the
positive/negative distinction semantically meaningless in this corpus.

Rather than forcing an ill-defined problem, we redefined the target: **complaint severity
(LOW / MEDIUM / HIGH)** — a variable that exists naturally in the structured CFPB data and
directly maps to CX action priorities. This is what a CX team actually needs: not whether
a complaint is 'negative', but **how urgently it needs attention**.

> *"El approach de sentiment classification fue descartado después de validación empírica con
> 6 modelos distintos. El diagnóstico: clasificar positivo/negativo en un corpus de quejas
> carece de sentido semántico. Redefinimos el problema como predicción de severidad —
> una variable con valor real para equipos de CX — y obtuvimos resultados medibles y
> accionables con variables estructuradas del mismo dataset."*

This demonstrates **senior critical thinking** — knowing when to reframe the problem, not just
run more models.

---

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Severity Accuracy — XGBoost** | *(run notebook 02)* | Production model — structured CFPB features |
| **Severity F1-weighted — XGBoost** | *(run notebook 02)* | 5-fold CV + Optuna tuning |
| **Severity Accuracy — Logistic Regression** | *(run notebook 02)* | Interpretable baseline |
| **NPS (complaint dataset)** | **−6.33** | Simulated proxy; negative NPS expected for complaint data |
| **Records analysed** | **~40,000–60,000** | Banking products, CFPB dataset |
| **Topics discovered** | *(run notebook 03)* | Named with banking domain expertise |
| **Customer segments** | *(run notebook 04)* | K-Means on severity + NLP features |

> **Key finding:** Complaint severity from structured variables (response timeliness, resolution
> type, product, complaint length) outperforms 6 NLP sentiment models on this corpus.
> The signal is in the metadata, not the text. Correct problem definition matters more than
> model complexity.

### Top 5 Drivers of Severity

*(Updated after running notebook 02)*
1. **Timely response** — The #1 predictor: SLA misses immediately elevate severity
2. **Resolution quality** — Monetary relief resolves; explanation-only escalates
3. **Complaint length** — Longer complaints signal deeper, unresolved frustration
4. **Product type** — Mortgage and student loans generate structurally higher severity
5. **Days to resolution** — Resolution time beyond 2 weeks is a strong HIGH severity signal

---

## System Architecture

| Layer | Tool | Role |
|-------|------|------|
| **Preprocessing** | spaCy | Tokenization, lemmatization |
| **NPS Analysis** | Custom logic | Simulate and calculate NPS from resolution data |
| **Severity Prediction** | XGBoost | Predict complaint severity from structured features |
| **Topic Mining** | LDA | Discover recurring complaint themes |
| **Segmentation** | K-Means | Cluster customers by severity + NPS + topic profile |

### Why structured features over NLP for severity

VADER, FinBERT, RoBERTa, and Zero-Shot DeBERTa all failed near random chance (~33%)
because they classify *text emotional tone*, while the target is *complaint urgency* —
a fundamentally different annotation axis.

The CFPB dataset already contains `company_response_to_consumer`, `timely_response`,
and resolution dates for every record. These structured variables carry more signal
about complaint urgency than the complaint text itself — and they require no GPU,
no model downloads, and no domain fine-tuning.

---

## Key Findings

*(Updated after full analysis)*

- **Response timeliness is the #1 severity driver** — banks that miss SLA targets
  generate the highest-severity, highest-churn-risk complaints
- **HIGH severity complaints concentrate in Mortgage and Student Loan products** —
  complexity and money stakes drive escalation
- **"Silent Dissatisfied" segment** — MEDIUM severity, low NPS — is the hardest to detect
  without combining severity + NPS + segmentation
- **Customers with 2+ complaints** are at extreme churn risk regardless of resolution outcome,
  connecting to [Project 1: Banking Churn Prediction](https://github.com/nicolaszuleta95/banking-churn-prediction)
- **Model lift over random triage**: reviewing the top 20% highest-predicted severity
  complaints catches significantly more HIGH severity cases than random review

---

## Demo

Four sections in the interactive dashboard:

- **NPS Dashboard** — Overall NPS gauge, segment distribution, NPS by product, temporal trend
- **Severity Analyzer** — Enter complaint details → XGBoost severity prediction in real time, with class probabilities and top feature drivers
- **Topic Explorer** — WordClouds per topic, NPS by topic, % HIGH severity per topic, most representative complaints
- **Customer Segments** — Interactive cluster visualization, severity + NPS profiles, CX action recommendations

[**Try the live demo →**](#)

> *Add screenshot here after Streamlit deploy*
> `![App Screenshot](reports/app_screenshot.png)`

---

## Project Structure

```
cx-intelligence-nps/
│
├── data/
│   ├── raw/
│   │   └── consumer_complaints.csv   # Original CFPB dataset — never modified
│   └── processed/
│       ├── banking_complaints.csv    # Filtered and cleaned
│       └── features_nlp.csv         # All features: severity, topics, clusters
│
├── notebooks/
│   ├── 01_eda_nps_analysis.ipynb       # EDA + NPS calculation + business insights
│   ├── 02_complaint_severity.ipynb     # Severity Predictor — XGBoost from structured features
│   ├── 03_topic_mining.ipynb           # LDA + topic naming + NPS by topic
│   └── 04_customer_segmentation.ipynb  # K-Means + severity + CX segment profiles
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py   # Load, filter, and clean CFPB data
│   ├── nps_calculator.py    # NPS logic: simulate, classify, calculate, trend
│   ├── severity.py          # Severity Predictor: features + label + SeverityPredictor class
│   ├── topics.py            # TF-IDF + LDA + topic naming
│   └── segmentation.py      # K-Means + cluster profiles
│
├── models/
│   ├── severity_model.joblib    # Serialized XGBoost severity model
│   ├── severity_features.json   # Feature order for inference reproducibility
│   ├── lda_model.pkl            # Serialized topic model
│   └── kmeans_model.joblib      # Serialized segmentation model
│
├── app/
│   └── app.py                   # Streamlit dashboard (4 sections)
│
├── reports/
│   └── model_card.md            # Technical decisions and model documentation
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** [US Consumer Finance Complaints — CFPB](https://www.kaggle.com/datasets/cfpb/us-consumer-finance-complaints)

| Property | Value |
|----------|-------|
| Provider | Consumer Financial Protection Bureau (US Government) |
| Total records | 180,000+ |
| After banking filter | ~40,000–60,000 |
| Text field | `Consumer complaint narrative` (free text) |

**Banking products included:**
```python
["Checking or savings account", "Credit card or prepaid card",
 "Mortgage", "Personal loan", "Student loan"]
```

---

## Key Technical Decisions

**1. Severity over sentiment**
Sentiment classification was empirically rejected — 6 models, ~33% accuracy. Severity from
structured CFPB variables was redefined as the target, producing a model with CX operational value.

**2. XGBoost as primary model**
Consistent with Project 1 (banking-churn-prediction). Performs well on structured tabular
banking data. Built-in feature importance replaces SHAP without loss of interpretability.

**3. Optuna for hyperparameter tuning**
More efficient than GridSearchCV for XGBoost's large hyperparameter space. Falls back to
GridSearchCV if Optuna is not installed.

**4. spaCy for preprocessing only**
Tokenization, lemmatization, linguistic cleaning. Severity classification uses structured
features — each tool used for what it does best.

**5. NPS simulated from resolution data**
CFPB doesn't include satisfaction scores. NPS simulated from response timeliness + resolution
outcome as proxy. Documented transparently — not presented as measured NPS.

---

## Data Note

The NPS scores in this project are **simulated** using CFPB resolution data as a proxy
(response timeliness + resolution outcome → satisfaction score). This is a methodological
choice documented for transparency — not presented as directly measured NPS.

All data is public and sourced from official US government sources.

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
# Place consumer_complaints.csv in data/raw/ (download from Kaggle link above)
# Run notebooks in order: 01 → 02 → 03 → 04
```

**3. Launch the app**
```bash
streamlit run app/app.py
```

> The severity model (`models/severity_model.joblib`) is loaded automatically.
> If not present, run Notebook 02 first to train and save it.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data | `pandas`, `numpy` |
| NLP preprocessing | `spaCy` (en_core_web_sm) |
| Severity prediction | `xgboost`, `lightgbm`, `scikit-learn` |
| Hyperparameter tuning | `optuna` |
| Topic modeling | `scikit-learn` (TF-IDF + LDA), `gensim` (coherence) |
| Segmentation | `scikit-learn` (K-Means) |
| Visualization | `plotly`, `seaborn`, `matplotlib`, `wordcloud` |
| App & deployment | `streamlit` |
| Serialization | `joblib` |

---

## Model Card

See [`reports/model_card.md`](reports/model_card.md) for:
- Intended use and out-of-scope applications
- Severity model design decisions and limitations
- NPS simulation methodology and limitations
- Why sentiment classification was rejected

---

## Author

**Nicolás Zuleta Sierra**
Data Scientist · 7+ years · Banking & CX Analytics · Medellín, Colombia

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nicolaszuletasierra/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github&logoColor=white)](https://github.com/nicolaszuleta95)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=flat&logo=gmail&logoColor=white)](mailto:nicolaszuleta95@gmail.com)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---
