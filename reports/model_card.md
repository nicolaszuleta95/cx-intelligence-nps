# Model Card — Complaint Severity Predictor

> CX Intelligence: NPS & Complaint Severity Analysis  
> Author: Nicolás Zuleta Sierra · [LinkedIn](https://www.linkedin.com/in/nicolaszuletasierra/)

---

## 0. Why This Model Exists — The Pivot Decision

Sentiment classification (positive/negative/neutral) was the original approach.
After testing **6 models**, all performed near random chance:

| Model | Accuracy | F1-weighted | Verdict |
|-------|----------|-------------|---------|
| VADER | 38.0% | 0.338 | Near-random (33.3% baseline) |
| FinBERT | 32.8% | 0.270 | Predicts zero positives |
| RoBERTa Twitter | ~35% | ~0.31 | Domain mismatch |
| Zero-Shot DeBERTa | ~37% | ~0.33 | No signal |
| DistilBERT fine-tuned v1 | ~39% | ~0.31 | Collapses to negative |
| DistilBERT fine-tuned v2 | ~41% | ~0.34 | Marginal — not production-ready |

**Root cause:** The CFPB dataset is 100% complaints by definition. Classifying
positive/negative/neutral in a corpus where every record describes a problem has no
semantic validity. All variation is in *urgency*, not *polarity*.

**Decision:** Redefine the target as **complaint severity (LOW / MEDIUM / HIGH)** —
a variable that exists naturally in CFPB structured metadata and directly maps to CX action priorities.

---

## 1. Intended Use

**Primary use case:** Triaging incoming banking complaints by urgency to prioritize
CX team action — enabling automated routing and capacity allocation.

**Target users:** CX analytics teams and operations managers at financial institutions
who need to prioritize complaint resolution at scale.

**Supported input:** Structured CFPB complaint metadata fields:
- `timely_response` (Yes/No)
- `company_response_to_consumer` (resolution type)
- `product` (banking product category)
- `consumer_complaint_narrative` (free text — used for word count only)
- `submitted_via` (submission channel)
- `date_received`, `date_sent_to_company` (for days_to_resolution)
- `complaint_id` (for multi-complaint detection)

---

## 2. Out-of-Scope Uses

| Scenario | Why out of scope |
|----------|-----------------|
| Real-time fraud detection | Not a safety-critical system |
| Non-English complaints | CFPB is English-only corpus |
| Measuring customer satisfaction (NPS) | Severity ≠ NPS — separate signal |
| Replacing human case managers | Triage tool, not autonomous decision-maker |
| Non-banking complaints | Product encoding is banking-specific |

---

## 3. Model Architecture

### Primary Model — XGBoost Classifier

| Property | Value |
|----------|-------|
| **Algorithm** | XGBoost (`XGBClassifier`) |
| **Task** | Multi-class classification (3 classes: LOW / MEDIUM / HIGH) |
| **Training data** | CFPB Consumer Complaints — banking products filter (~35K records) |
| **Features** | 8 structured features (see Section 4) |
| **Tuning** | Optuna hyperparameter optimization (30 trials, 3-fold CV) |
| **Evaluation** | 5-fold cross-validation + held-out test set (20%) |
| **Serialization** | `models/severity_model.joblib` (joblib) |
| **Feature list** | `models/severity_features.json` (ordered list for inference) |
| **Why XGBoost** | Consistent with Project 1 (banking-churn-prediction); strong on structured tabular banking data; built-in feature importance without SHAP |

### Baseline Model — Logistic Regression

| Property | Value |
|----------|-------|
| **Algorithm** | Logistic Regression (`multi_class='multinomial'`, `class_weight='balanced'`) |
| **Role** | Interpretable baseline for performance comparison |
| **Why kept** | Transparency — documents the minimum bar that XGBoost must exceed |

---

## 4. Features

| Feature | Source | Engineering |
|---------|--------|-------------|
| `timely_response_binary` | `timely_response` | 1 = Yes, 0 = No |
| `response_type_encoded` | `company_response_to_consumer` | Ordinal favorability: 5 (monetary relief) → 0 (unknown) |
| `product_encoded` | `product` | Ordinal severity weight: Mortgage=4, Student loan=3, Credit card=2, Checking=2, Personal=1 |
| `complaint_length` | `consumer_complaint_narrative` | Word count (0 if no narrative) |
| `has_narrative` | `consumer_complaint_narrative` | Binary: 1 = has text, 0 = absent |
| `submission_channel_encoded` | `submitted_via` | Label encoded |
| `days_to_resolution` | `date_received`, `date_sent_to_company` | Delta in calendar days |
| `multi_complaint_flag` | `complaint_id` | 1 = same complaint_id appears >1 times |

**Missing value strategy:** `days_to_resolution` imputed with median. All others default to 0.

---

## 5. Target Variable — Severity Label

Severity is defined by rule-based logic from CFPB structured variables:

```
HIGH  = (timely_response == No OR response_type_encoded <= 1) AND complaint_length > 150
LOW   = timely_response == Yes AND response_type_encoded >= 4 AND complaint_length <= 150
MEDIUM = everything else
```

**NPS refinement** (applied after base rules):
- `nps_score <= 4` (Detractor) in MEDIUM zone → upgrade to HIGH
- `nps_score >= 8` (Promoter) in MEDIUM zone → downgrade to LOW

**Label encoding:** LOW=0, MEDIUM=1, HIGH=2

---

## 6. Evaluation Metrics

*(Populated after running notebook 02)*

| Model | Accuracy | F1-weighted | Notes |
|-------|----------|-------------|-------|
| Logistic Regression | *(run nb02)* | *(run nb02)* | Baseline |
| XGBoost (tuned) | *(run nb02)* | *(run nb02)* | Production |

**Primary metric:** F1-weighted — accounts for class imbalance.

**Business metric:** Detection rate of HIGH severity complaints in top-20% review
(lift over random triage). See notebook 02 Section 7.

---

## 7. Known Limitations

### Target variable is rule-based, not human-labeled
The severity label is derived from CFPB structured fields using business rules —
not from manual annotation. This means the model learns to reproduce a rule, not
a ground truth. The value is in generalizing the pattern to cases where the rule
logic might be ambiguous.

### NPS score dependency
The NPS refinement step in `define_severity_label()` uses simulated NPS scores
(not measured NPS). If NPS simulation logic changes, severity labels shift accordingly.

### CFPB-specific encoding
`product_encoded` and `response_type_encoded` use hardcoded CFPB category mappings.
The model will require re-encoding if applied to non-CFPB complaint systems.

### complaint_length as a frustration proxy
Word count is a crude signal. A short but profanity-laden complaint may be high severity;
a long but polite complaint may be low. This is a known limitation accepted for simplicity.

### Class imbalance
Depending on the data distribution, MEDIUM may dominate. The Logistic Regression baseline
uses `class_weight='balanced'`; XGBoost handles imbalance through its loss function and
hyperparameter tuning. Verify distribution in notebook 02 Section 3.

---

## 8. Ethical Considerations

- All data is from public US government sources (CFPB). No PII is present — CFPB redacts names, account numbers, and addresses.
- The model is a triage tool for internal CX teams — not used for automated customer-facing decisions.
- NPS simulation is documented — severity labels should not be presented as ground truth.
- No demographic data is used as a feature — severity is based purely on complaint characteristics.

---

## 9. Dataset Citation

```
Consumer Financial Protection Bureau (CFPB).
US Consumer Complaint Database.
https://www.consumerfinance.gov/data-research/consumer-complaints/
Available on Kaggle: https://www.kaggle.com/datasets/cfpb/us-consumer-finance-complaints
```

---

*Last updated: May 2026 — Active project*
