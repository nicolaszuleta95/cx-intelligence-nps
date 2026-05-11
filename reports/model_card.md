# Model Card — Sentiment Analysis for Banking Complaints

> CX Intelligence: NPS & Sentiment Analysis
> Author: Nicolás Zuleta Sierra · [LinkedIn](https://www.linkedin.com/in/nicolaszuletasierra/)

---

## 0. Model Evolution

| Version | Model | Training data | Accuracy | F1 weighted | Status |
|---------|-------|--------------|----------|-------------|--------|
| v1 | VADER (baseline) | Social media lexicon | 38.0% | 0.338 | Baseline only |
| v1 | FinBERT | Financial news (4.9B tokens) | 32.8% | 0.270 | Replaced |
| **v2** | **DistilBERT fine-tuned** | **34K CFPB resolution labels** | ***(run nb02 §7)*** | ***(run nb02 §7)*** | **Production** |

**Why v2 replaced v1:** Both off-the-shelf models (VADER, FinBERT) performed near random chance (33%) because they classify *text sentiment* but the ground truth labels reflect *resolution outcome*. Reframing the task as supervised fine-tuning on CFPB metadata resolved the label-text mismatch and increased training data 140×. See notebook 02 §6 for root cause analysis.

---

## 1. Intended Use

**Primary use case:** Binary/ternary sentiment classification (positive / negative / neutral) applied to formal banking complaint narratives in English.

**Target users:** CX analytics teams at financial institutions who need to understand the emotional tone of customer feedback at scale.

**Supported input:** Free-text English complaint narratives from consumer-facing banking products (checking accounts, credit cards, mortgages, personal loans, student loans). Optimal performance is on texts between 20 and 512 tokens.

---

## 2. Out-of-Scope Uses

| Scenario | Why out of scope |
|----------|-----------------|
| Non-English text | Models are trained/tuned on English financial text only |
| Social media / tweets | Domain mismatch — see Technical Decision §7 |
| Texts shorter than 5 words | Insufficient context for reliable classification |
| Real-time fraud detection | Not a safety-critical classification system |
| Non-financial domains | FinBERT vocabulary is finance-specific |

---

## 3. Models in This Pipeline

### 3.1 VADER — Baseline Model

| Property | Value |
|----------|-------|
| **Type** | Rule-based lexicon (VADER) |
| **Library** | `vaderSentiment` |
| **Pre-training** | Social media text (Twitter, product reviews, movie reviews) |
| **Classification** | compound ≥ 0.05 → positive · ≤ −0.05 → negative · else → neutral |
| **Role** | Baseline for performance comparison; zero-compute reference |

**Why VADER is kept:** Interpretability and speed. No GPU required. Useful as a sanity check and for environments without HuggingFace access.

### 3.2 DistilBERT Fine-Tuned — Production Model (v2)

| Property | Value |
|----------|-------|
| **Model ID** | `distilbert-base-uncased` (fine-tuned) |
| **Saved path** | `models/distilbert_cx/` |
| **Base architecture** | DistilBERT (6 layers, 66M parameters — 40% smaller than BERT-base) |
| **Training data** | 34K+ CFPB complaints with `company_response_to_consumer` as label proxy |
| **Label mapping** | monetary relief → positive · non-monetary → neutral · explanation/untimely → negative |
| **Class balancing** | Undersampled to equal class counts (min class × 3 total) |
| **Fine-tuning** | 3 epochs · lr=2e-5 · batch=16 · max_length=256 · warmup=100 steps |
| **Input limit** | 256 tokens (truncation applied; covers ~80% of complaint lengths) |
| **Role** | Production model — trained on domain-aligned labels |

**Why DistilBERT over FinBERT:** FinBERT was trained on investor-facing financial news. DistilBERT fine-tuned on CFPB complaints learns from the actual domain, resolving the training-evaluation mismatch.

### 3.3 FinBERT — Historical Baseline

| Property | Value |
|----------|-------|
| **Model ID** | `ProsusAI/finbert` |

| Property | Value |
|----------|-------|
| **Model ID** | `ProsusAI/finbert` |
| **Hub** | [HuggingFace](https://huggingface.co/ProsusAI/finbert) |
| **Base architecture** | BERT-base (12 layers, 110M parameters) |
| **Pre-training corpus** | Financial PhraseBank + Reuters financial news + Bloomberg articles (~4.9B tokens) |
| **Fine-tuning task** | Sentiment classification on Financial PhraseBank (positive/negative/neutral) |
| **Input limit** | 512 tokens (hard limit — truncation always applied) |
| **Batching** | batch_size=32 for inference efficiency |
| **Role** | Production model — higher F1 on formal banking text |

---

## 4. Training and Evaluation Data

### Pre-training (external — not modified in this project)

- **FinBERT:** Pre-trained by ProsusAI on financial news corpora (Reuters, Bloomberg). Fine-tuned on the [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank) dataset.
- **VADER:** Lexicon built from human ratings of social media text. Published by Hutto & Gilbert (2014).

### Evaluation Data (produced in this project)

| Property | Value |
|----------|-------|
| **Source** | CFPB Consumer Financial Protection Bureau complaints |
| **Subset** | Banking products only (checking, credit card, mortgage, personal loan, student loan) |
| **Size** | 250 complaint narratives |
| **Labeling method** | Manual annotation by the author |
| **Annotator** | Nicolás Zuleta Sierra (7 years banking CX experience) |
| **Label distribution** | 100 negative · 80 positive · 70 neutral |
| **Desambiguation rule** | In case of doubt → `negative` (Type II errors are costlier in CX banking) |
| **File** | `data/raw/ground_truth.csv` |

> **Limitation:** 250 records is exploratory, not statistically robust. Results should be interpreted directionally, not as definitive benchmarks.

---

## 5. Evaluation Metrics

| Metric | VADER | FinBERT | Notes |
|--------|-------|---------|-------|
| **Accuracy** | **38.0%** | 32.8% | VADER wins; both near 33% random baseline |
| **F1 (weighted)** | **0.338** | 0.270 | VADER wins by +0.068 |
| **F1 (positive class)** | 0.41 | 0.00 | FinBERT predicts zero positives |
| **F1 (negative class)** | **0.46** | 0.43 | Near-even |
| **F1 (neutral class)** | 0.07 | **0.35** | FinBERT better at identifying neutral text |
| **Full dataset % neutral** | 4.0% | **63.6%** | FinBERT is more conservative |

**Ground truth: n=250 · 100 negative · 80 positive · 70 neutral**

### Why F1-weighted is the primary metric

The label distribution is imbalanced (40% negative, 32% positive, 28% neutral). Weighted F1 accounts for class frequency and is more informative than accuracy on imbalanced datasets.

### Key finding — counter-intuitive result

VADER outperforms FinBERT on this evaluation — the opposite of the original hypothesis. Root causes:
1. FinBERT was pre-trained on financial NEWS (investor sentiment), not consumer complaint language
2. Ground truth labels partially reflect resolution outcome rather than pure text sentiment
3. Both models operate near random-chance (33%) baseline — domain fine-tuning is required for production use

See `notebooks/02_sentiment_pipeline.ipynb` Section 6 for full error analysis.

---

## 6. Known Limitations

### FinBERT limitations
- Pre-trained on **institutional financial text** (news, earnings calls), not on **retail consumer complaints**. There is a subdomain shift even within finance.
- **Formal sarcasm** ("They were extremely helpful — if your definition of helpful includes charging you twice") is not reliably detected.
- **Long complaints** (>512 tokens) are truncated from the right, potentially losing resolution information that appears at the end of the text.
- Labels in Financial PhraseBank reflect *investor sentiment*, not *customer satisfaction*. Transfer to consumer complaints is an approximation.

### VADER limitations
- Lexicon tuned on social media — does not understand domain-specific negation patterns in formal banking language (e.g., "the account was not disputed" scored as ambiguous).
- No understanding of context or word order beyond simple negation modifiers.

### NPS limitations
- **NPS scores in this project are simulated**, not measured. They are derived from CFPB resolution metadata (response timeliness + resolution outcome) as a proxy for customer satisfaction.
- Simulation is documented in `src/nps_calculator.py` and disclosed in the README under **Data Note**.
- Gaussian noise (σ=1) is added for realism, but the simulation may not capture the full variance of real NPS distributions.

### Evaluation data limitations
- 250 records is insufficient for statistical significance. Confidence intervals are wide.
- Labels reflect a single annotator's judgement — no inter-annotator agreement was measured.
- Sampling strategy (resolution-based pool selection) may introduce selection bias toward more extreme cases.

---

## 7. Key Technical Decision: Why Not Train from Scratch on Twitter Data?

This decision was evaluated and explicitly discarded for the following reasons:

| Factor | Twitter Airline Sentiment | CFPB Banking Complaints |
|--------|--------------------------|------------------------|
| **Language register** | Informal, abbreviated, emoji-heavy | Formal, detailed, narrative |
| **Domain** | Aviation customer service | Banking / financial services |
| **Text length** | 140–280 characters | Typically 200–800 words |
| **Vocabulary** | Colloquial, slang | Legal and financial terminology |
| **Annotation quality** | Crowdsourced | Domain expert (this project) |

Training on Twitter airline data and deploying on CFPB banking complaints would constitute a **domain mismatch** that would invalidate the evaluation. Using FinBERT's financial pre-training as the starting point is the methodologically correct choice.

**Alternative considered:** Fine-tuning FinBERT on the 250 ground truth records. Discarded because 250 samples is too few for stable fine-tuning and would risk overfitting. Retained as a future enhancement once more annotated data is available.

---

## 8. Ethical Considerations

- All data is sourced from official US government public records (CFPB). No private data is used.
- No personally identifiable information (PII) is present — CFPB redacts names, account numbers, and addresses before publication.
- The model is not used for automated decision-making affecting consumers. It is an analytics tool for CX teams.
- NPS simulation is disclosed — results should not be reported as measured NPS without this caveat.

---

## 9. Citation and Attribution

```
FinBERT:
  Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.
  arXiv:1908.10063

VADER:
  Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
  Sentiment Analysis of Social Media Text. ICWSM.

Dataset:
  Consumer Financial Protection Bureau. CFPB Consumer Complaint Database.
  https://www.consumerfinance.gov/data-research/consumer-complaints/
```

---

*Last updated: May 2026 — Active project*
