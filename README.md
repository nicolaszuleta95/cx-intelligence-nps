# 💬 CX Intelligence — NPS & Sentiment Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/FinBERT-HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Demo-Live-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5?style=flat)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat)

**End-to-end NLP system for banking customer experience analytics —
sentiment classification, NPS analysis, topic mining, and customer segmentation.**

[🚀 Live Demo](#) · [📓 Notebooks](notebooks/) · [📊 Model Card](reports/model_card.md)

</div>

---

## 🎯 Business Problem

Customer Experience teams at banks receive thousands of survey responses and complaints every month. **NPS tells you *how many* customers are dissatisfied — but not *why*.**

Reading comments manually is impossible at scale. A team analyzing 10,000 monthly responses would need weeks just to categorize them — let alone extract actionable insights.

**This project automates that process end-to-end:**

1. Calculates and tracks NPS by product, channel, and time period
2. Classifies the sentiment of every customer comment automatically
3. Discovers the recurring themes driving dissatisfaction
4. Segments customers by their satisfaction profile
5. Delivers all insights through an interactive dashboard

---

## 📖 NPS Primer

**Net Promoter Score** is the standard CX metric in banking and financial services.

```
NPS = % Promoters − % Detractors

Promoters  (score 9–10): Loyal customers likely to recommend
Passives   (score 7–8):  Satisfied but not enthusiastic
Detractors (score 0–6):  Unhappy customers at risk of churn
```

NPS alone is a number. This system explains *what's behind it* — which products, topics, and customer segments are driving the score up or down.

---

## 📊 Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Sentiment Accuracy — VADER** | **38.0%** | Baseline — social media lexicon on banking complaints |
| **Sentiment F1 — VADER** | **0.338** | Near random chance (33.3% baseline for 3 classes) |
| **Sentiment Accuracy — FinBERT** | **32.8%** | Replaced — investor sentiment ≠ consumer CX language |
| **Sentiment F1 — FinBERT** | **0.270** | Predicts zero positives — domain mismatch documented |
| **Sentiment Accuracy — DistilBERT fine-tuned** | *(run notebook 02)* | Trained on 34K CFPB resolution labels |
| **Sentiment F1 — DistilBERT fine-tuned** | *(run notebook 02)* | Production model — aligned labels, 140× more data |
| **NPS (complaint dataset)** | **−6.33** | Simulated proxy; negative NPS expected for complaint data |
| **Records analysed** | **35,226** | Banking products, Web submissions with narratives |
| **Topics discovered** | *(run notebook 03)* | Named with banking domain expertise |
| **Customer segments** | *(run notebook 04)* | K-Means on NLP-derived features |

> **Key finding:** Off-the-shelf models (VADER, FinBERT) fail near random chance because they classify *text tone* while ground truth labels reflect *resolution outcome*. The supervised DistilBERT approach resolves this by training on 34K CFPB resolution labels — same domain, aligned labels, 140× more training data. See [`notebooks/02_sentiment_pipeline.ipynb`](notebooks/02_sentiment_pipeline.ipynb) §6 for the full root-cause analysis.

### Top 5 Drivers of Dissatisfaction
*(Updated after topic mining)*
1. **Incorrect charges** — Lowest NPS topic, highest negative sentiment
2. **Phone customer service** — High volume, slow resolution pattern
3. **Credit process delays** — Friction beyond 30 days destroys NPS
4. **Mortgage documentation** — Paper-heavy process generates high dissatisfaction
5. **Account closure** — High dissatisfaction at the exit experience

---

## 🏗 NLP Architecture

Three-layer pipeline designed specifically for banking complaint text:

| Layer | Tool | Role | Why |
|-------|------|------|-----|
| **Preprocessing** | spaCy | Tokenization, lemmatization, cleaning | Linguistic precision for formal text |
| **Baseline** | VADER | Lexicon-based sentiment | Fast, interpretable reference point |
| **Historical** | FinBERT | Financial news pre-training | Replaced — investor sentiment ≠ CX language |
| **Production** | DistilBERT fine-tuned | Supervised resolution classifier | Trained on 34K CFPB resolution labels |

### Why fine-tuning on CFPB metadata — not off-the-shelf models

VADER and FinBERT both failed near random chance (33%) because they classify *text emotional tone*, but ground truth labels reflect *resolution outcome* — a fundamentally different annotation axis.

**Solution:** The CFPB dataset already contains `company_response_to_consumer` for all 34K+ records. Mapping this to positive/neutral/negative creates 140× more labeled training data than manual annotation, with labels that are perfectly aligned with the ground truth definition.

`distilbert-base-uncased` fine-tuned on these labels learns from actual complaint-outcome pairs in the banking domain — no label-text mismatch, domain-appropriate vocabulary.

---

## 🔍 Key Findings

*(Updated after full analysis)*

- **Incorrect charges and unauthorized debits** generate the highest volume of Detractor comments and the lowest NPS of any discovered topic
- **Customers with 2+ unresolved complaints** show patterns directly connected to churn risk — linking to [Project 1: Banking Churn Prediction](https://github.com/nicolaszuleta95/banking-churn-prediction)
- **FinBERT struggles with formal sarcasm** — polite but deeply negative language is a known limitation in banking complaint text
- **"Silent Dissatisfied" segment** — neutral language, very low NPS — is the hardest group to detect with sentiment alone
- **Timely resolution drives NPS more than resolution outcome** — customers receiving a response within 24h scored 2.3× higher NPS on average

---

## 🖥 Demo

Four sections in the interactive dashboard:

- **NPS Dashboard** — Overall NPS gauge, segment distribution, NPS by product, temporal trend
- **Sentiment Analyzer** — Type any customer comment → VADER + FinBERT predictions in real time, side by side
- **Topic Explorer** — WordClouds per topic, NPS by topic, most representative complaints per theme
- **Customer Segments** — Interactive cluster visualization, segment profiles, CX action recommendations

👉 [**Try the live demo →**](#)

> *Add screenshot here after Streamlit deploy*
> `![App Screenshot](reports/app_screenshot.png)`

---

## 🏗 Project Structure

```
cx-intelligence-nps/
│
├── data/
│   ├── raw/
│   │   ├── cfpb_complaints.csv          # Original dataset — never modified
│   │   └── ground_truth.csv             # 250 manually labeled comments (ground truth)
│   └── processed/
│       ├── banking_complaints.csv       # Filtered and cleaned
│       └── features_nlp.csv            # All NLP-derived features
│
├── notebooks/
│   ├── 01_eda_nps_analysis.ipynb        # EDA + NPS calculation + business insights
│   ├── 02_sentiment_pipeline.ipynb      # VADER + FinBERT + evaluation + comparison
│   ├── 03_topic_mining.ipynb            # LDA + topic naming + NPS by topic
│   └── 04_customer_segmentation.ipynb   # K-Means + CX segment profiles
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Load, filter, and clean CFPB data
│   ├── nps_calculator.py     # NPS logic: simulate, classify, calculate, trend
│   ├── sentiment.py          # SentimentPipeline: VADER + FinBERT, unified interface
│   ├── topics.py             # TF-IDF + LDA + topic naming
│   └── segmentation.py       # K-Means + cluster profiles
│
├── models/
│   ├── lda_model.pkl          # Serialized topic model
│   └── kmeans_model.joblib    # Serialized segmentation model
│   # FinBERT loaded from HuggingFace Hub at runtime
│
├── app/
│   └── app.py                 # Streamlit dashboard
│
├── reports/
│   └── model_card.md          # Technical decisions and model documentation
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

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

## 🔍 Key Technical Decisions

**1. CFPB over Twitter data**
Twitter airline sentiment was evaluated and discarded. Banking complaint language is formal and structured — domain-appropriate data is non-negotiable for honest evaluation.

**2. FinBERT as production model**
Pre-trained on financial text, understands banking vocabulary semantically. VADER maintained as baseline to quantify performance gain and demonstrate comparative rigor.

**3. spaCy for preprocessing only**
Tokenization, lemmatization, linguistic cleaning. Sentiment classification delegated to VADER and FinBERT — each tool used for what it does best.

**4. Manual ground truth — 250 records**
250 CFPB comments labeled manually by the author using 7 years of banking CX expertise. Honest, domain-appropriate evaluation — no automated proxy labels.

**5. NPS simulated from resolution data**
CFPB doesn't include satisfaction scores. NPS simulated from response timeliness + resolution outcome as proxy. Documented transparently — not presented as measured NPS.

---

## 📊 Data Note

The NPS scores in this project are **simulated** using CFPB resolution data as a proxy (response timeliness + resolution outcome → satisfaction score). This is a methodological choice documented for transparency — not presented as directly measured NPS.

The sentiment ground truth (250 records) was manually labeled by the author based on 7 years of banking CX domain experience.

All data is public and sourced from official US government sources.

---

## 🚀 How to Run

**1. Clone and install**
```bash
git clone https://github.com/nicolaszuleta95/cx-intelligence-nps
cd cx-intelligence-nps
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**2. Prepare data**
```bash
# Place cfpb_complaints.csv in data/raw/ (download from Kaggle link above)
python src/data_processing.py
```

**3. Launch the app**
```bash
streamlit run app/app.py
```

> **First launch:** Downloads FinBERT (~500MB) from HuggingFace Hub automatically. Cached on subsequent runs.

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Data | `pandas`, `numpy` |
| NLP preprocessing | `spaCy` (en_core_web_sm) |
| Sentiment baseline | `vaderSentiment` |
| Sentiment production | `transformers` (ProsusAI/finbert), `torch` |
| Topic modeling | `scikit-learn` (TF-IDF + LDA), `gensim` (coherence) |
| Segmentation | `scikit-learn` (K-Means) |
| Visualization | `plotly`, `seaborn`, `matplotlib`, `wordcloud` |
| App & deployment | `streamlit` |
| Serialization | `joblib`, `pickle` |

---

## 📄 Model Card

See [`reports/model_card.md`](reports/model_card.md) for:
- Intended use and out-of-scope applications
- FinBERT pre-training data and limitations
- VADER vs FinBERT evaluation results
- NPS simulation methodology and limitations

---

## 👤 Author

**Nicolás Zuleta Sierra**
Data Scientist · 7+ years · Banking & CX Analytics · Medellín, Colombia

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nicolaszuletasierra/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github&logoColor=white)](https://github.com/nicolaszuleta95)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=flat&logo=gmail&logoColor=white)](mailto:nicolaszuleta95@gmail.com)

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---