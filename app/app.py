"""
CX Intelligence — NPS & Complaint Severity Analysis · Streamlit Dashboard

Four interactive sections:
  1. NPS Dashboard         — overall NPS gauge, segment breakdown, NPS by product/time
  2. Severity Analyzer     — real-time complaint severity prediction (XGBoost)
  3. Topic Explorer        — LDA topic keywords, NPS by topic, representative complaints
  4. Customer Segments     — K-Means cluster profiles, PCA scatter, CX recommendations

Run:
    streamlit run app/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths — relative to repo root
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "banking_complaints.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "features_nlp.csv"
LDA_MODEL_PATH = ROOT / "models" / "lda_model.pkl"
KMEANS_MODEL_PATH = ROOT / "models" / "kmeans_model.joblib"
SEVERITY_MODEL_PATH = ROOT / "models" / "severity_model.joblib"
SEVERITY_FEATURES_PATH = ROOT / "models" / "severity_features.json"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CX Intelligence — NPS & Severity",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached data & model loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading complaints data…")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed and NLP-feature datasets."""
    if not PROCESSED_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()

    processed = pd.read_csv(PROCESSED_PATH, low_memory=False)
    features = pd.read_csv(FEATURES_PATH, low_memory=False) if FEATURES_PATH.exists() else pd.DataFrame()
    return processed, features


@st.cache_resource(show_spinner="Loading severity model…")
def load_severity_model():
    """Load SeverityPredictor — cached across Streamlit reruns."""
    import json
    import sys
    sys.path.insert(0, str(ROOT))
    from src.severity import SeverityPredictor, FEATURE_COLS
    if not SEVERITY_MODEL_PATH.exists():
        return None, FEATURE_COLS
    predictor = SeverityPredictor(model_type="xgboost")
    predictor.load(SEVERITY_MODEL_PATH)
    feature_names = FEATURE_COLS
    if SEVERITY_FEATURES_PATH.exists():
        with open(SEVERITY_FEATURES_PATH) as f:
            feature_names = json.load(f)
    return predictor, feature_names


@st.cache_resource(show_spinner=False)
def load_lda_models():
    """Load LDA model and vectorizer if available."""
    if not LDA_MODEL_PATH.exists():
        return None, None
    import sys
    sys.path.insert(0, str(ROOT))
    from src.topics import load_lda_model
    return load_lda_model(LDA_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_kmeans():
    """Load K-Means model if available."""
    if not KMEANS_MODEL_PATH.exists():
        return None
    import sys
    sys.path.insert(0, str(ROOT))
    from src.segmentation import load_kmeans_model
    return load_kmeans_model(KMEANS_MODEL_PATH)


# ---------------------------------------------------------------------------
# Helper: severity color mapping
# ---------------------------------------------------------------------------

SEVERITY_COLORS = {
    "HIGH": "#ef4444",
    "MEDIUM": "#f59e0b",
    "LOW": "#22c55e",
}

SEVERITY_ACTIONS = {
    "HIGH": "Escalate immediately — assign dedicated case manager",
    "MEDIUM": "Follow up within 24h — standard resolution process",
    "LOW": "Queue for batch resolution — monitor NPS impact",
}


def severity_badge(label: str) -> str:
    color = SEVERITY_COLORS.get(label.upper(), "#6b7280")
    return f'<span style="background:{color};color:white;padding:6px 14px;border-radius:6px;font-weight:bold;font-size:1.1em;">{label.upper()}</span>'


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.image("https://img.shields.io/badge/CX_Intelligence-NPS_%26_Severity-1e40af?style=for-the-badge", use_column_width=True)
st.sidebar.markdown("## Navigation")

page = st.sidebar.radio(
    "Go to",
    ["NPS Dashboard", "Severity Analyzer", "Topic Explorer", "Customer Segments"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Author:** Nicolás Zuleta Sierra  \n"
    "[![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/nicolaszuletasierra/) "
    "[![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github)](https://github.com/nicolaszuleta95)"
)

# ---------------------------------------------------------------------------
# Load shared data
# ---------------------------------------------------------------------------

df_processed, df_features = load_data()
data_loaded = not df_processed.empty

if not data_loaded:
    st.warning(
        "Processed data not found. Run the notebooks first to generate "
        "`data/processed/banking_complaints.csv` and `data/processed/features_nlp.csv`."
    )

# ============================================================
# PAGE 1 — NPS DASHBOARD
# ============================================================

if page == "NPS Dashboard":
    st.title("NPS Dashboard")
    st.caption("Net Promoter Score analysis across banking products, channels, and time")

    if not data_loaded:
        st.info("Run Notebook 01 to generate the NPS data.")
        st.stop()

    import sys
    sys.path.insert(0, str(ROOT))
    from src.nps_calculator import calculate_nps, nps_by_group, nps_trend, classify_nps

    # Ensure NPS columns exist
    if "nps_score" not in df_processed.columns:
        st.warning("NPS scores not found. Ensure Notebook 01 has been executed.")
        st.stop()

    if "nps_segment" not in df_processed.columns:
        df_processed["nps_segment"] = df_processed["nps_score"].apply(classify_nps)

    nps_val = calculate_nps(df_processed)
    n_total = len(df_processed)
    n_promoters = (df_processed["nps_segment"] == "Promoter").sum()
    n_passives = (df_processed["nps_segment"] == "Passive").sum()
    n_detractors = (df_processed["nps_segment"] == "Detractor").sum()

    # --- KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Complaints", f"{n_total:,}")
    col2.metric("% Promoters", f"{n_promoters / n_total * 100:.1f}%")
    col3.metric("% Passives", f"{n_passives / n_total * 100:.1f}%")
    col4.metric("% Detractors", f"{n_detractors / n_total * 100:.1f}%")

    # --- Gauge ---
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=nps_val,
        title={"text": "Net Promoter Score"},
        gauge={
            "axis": {"range": [-100, 100]},
            "bar": {"color": "#1e40af"},
            "steps": [
                {"range": [-100, 0], "color": "#fee2e2"},
                {"range": [0, 50], "color": "#fef9c3"},
                {"range": [50, 100], "color": "#dcfce7"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.75, "value": nps_val},
        },
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # --- NPS by Product ---
    if "product" in df_processed.columns:
        with col_a:
            st.subheader("NPS by Product")
            nps_prod = nps_by_group(df_processed, "product")
            fig_prod = px.bar(
                nps_prod, x="nps", y="product", orientation="h",
                color="nps", color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                labels={"nps": "NPS", "product": ""},
                text="nps",
            )
            fig_prod.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_prod.update_layout(coloraxis_showscale=False, height=350)
            st.plotly_chart(fig_prod, use_container_width=True)

    # --- Segment pie ---
    with col_b:
        st.subheader("Segment Distribution")
        segment_counts = df_processed["nps_segment"].value_counts().reset_index()
        segment_counts.columns = ["segment", "count"]
        color_map = {"Promoter": "#22c55e", "Passive": "#f59e0b", "Detractor": "#ef4444"}
        fig_pie = px.pie(
            segment_counts, values="count", names="segment",
            color="segment", color_discrete_map=color_map,
            hole=0.4,
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- NPS trend ---
    st.subheader("NPS Trend Over Time")
    if "date_received" in df_processed.columns:
        trend = nps_trend(df_processed, date_col="date_received", freq="M")
        if not trend.empty:
            fig_trend = px.line(
                trend, x="period", y="nps",
                labels={"period": "Month", "nps": "NPS"},
                markers=True,
                color_discrete_sequence=["#1e40af"],
            )
            fig_trend.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_trend.update_layout(height=350)
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Date column not found.")

    # --- NPS by channel ---
    if "submitted_via" in df_processed.columns:
        st.subheader("NPS by Complaint Channel")
        nps_channel = nps_by_group(df_processed, "submitted_via")
        fig_ch = px.bar(
            nps_channel, x="submitted_via", y="nps",
            color="nps", color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
            labels={"submitted_via": "Channel", "nps": "NPS"},
            text="nps",
        )
        fig_ch.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_ch.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig_ch, use_container_width=True)

# ============================================================
# PAGE 2 — SEVERITY ANALYZER
# ============================================================

elif page == "Severity Analyzer":
    st.title("Complaint Severity Analyzer")
    st.caption("Predict complaint severity in real time using structured features — XGBoost model")

    st.info(
        "Enter complaint details below. The model predicts severity (LOW / MEDIUM / HIGH) "
        "from structured CFPB variables — no NLP processing required. "
        "Sentiment classification was discarded after testing 6 models (~33–41% accuracy = near random)."
    )

    import sys
    sys.path.insert(0, str(ROOT))
    from src.severity import (
        FEATURE_COLS,
        PRODUCT_SEVERITY_WEIGHT,
        RESPONSE_FAVORABILITY,
        build_severity_features,
    )

    BANKING_PRODUCTS = list(PRODUCT_SEVERITY_WEIGHT.keys())
    RESPONSE_TYPES = list(RESPONSE_FAVORABILITY.keys())

    col_left, col_right = st.columns([2, 1])

    with col_left:
        complaint_text = st.text_area(
            "Complaint narrative (optional — used for word count feature):",
            placeholder="e.g. I have been charged an overdraft fee every month for the past year despite maintaining the minimum balance. I called three times and each time I was told it would be resolved, but nothing changed...",
            height=150,
        )

    with col_right:
        product = st.selectbox("Banking product:", BANKING_PRODUCTS)
        timely_response = st.radio("Was the response timely?", ["Yes", "No"], horizontal=True)
        response_type = st.selectbox("Company resolution type:", RESPONSE_TYPES)

    if st.button("Analyze Severity", type="primary"):
        # Build a single-row DataFrame from user inputs
        complaint_length = len(complaint_text.split()) if complaint_text.strip() else 0
        has_narrative = 1 if complaint_length > 0 else 0
        timely_binary = 1 if timely_response == "Yes" else 0
        response_encoded = RESPONSE_FAVORABILITY.get(response_type, 0)
        product_encoded = PRODUCT_SEVERITY_WEIGHT.get(product, 1)

        input_df = pd.DataFrame([{
            "timely_response_binary": timely_binary,
            "response_type_encoded": response_encoded,
            "product_encoded": product_encoded,
            "complaint_length": complaint_length,
            "has_narrative": has_narrative,
            "submission_channel_encoded": 3,  # Web (most common for detailed complaints)
            "days_to_resolution": 2.0,        # Neutral default
            "multi_complaint_flag": 0,
        }])

        predictor, feature_names = load_severity_model()

        if predictor is None:
            st.warning(
                "Severity model not found. Run Notebook 02 first to train and save the model."
            )
        else:
            pred_label = predictor.predict_labels(input_df[feature_names])[0]
            pred_proba = predictor.predict_proba(input_df[feature_names])[0]

            st.markdown("---")

            # Severity badge
            col_sev, col_action = st.columns([1, 2])
            with col_sev:
                st.subheader("Severity Level")
                st.markdown(severity_badge(pred_label), unsafe_allow_html=True)

            with col_action:
                st.subheader("CX Action Required")
                action = SEVERITY_ACTIONS[pred_label]
                action_color = SEVERITY_COLORS[pred_label]
                st.markdown(
                    f'<div style="background:{action_color}22;border-left:4px solid {action_color};'
                    f'padding:12px 16px;border-radius:4px;font-size:1.0em;">{action}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Class probabilities
            st.subheader("Probability by Severity Level")
            prob_cols = st.columns(3)
            for col, (label, prob) in zip(prob_cols, zip(["LOW", "MEDIUM", "HIGH"], pred_proba)):
                with col:
                    color = SEVERITY_COLORS[label]
                    st.markdown(f"**{label}**")
                    st.progress(float(prob), text=f"{prob:.1%}")

            st.markdown("---")

            # Top 3 feature contributions
            st.subheader("Top 3 Drivers of This Prediction")
            fi_df = predictor.get_feature_importance().head(3)
            for _, row in fi_df.iterrows():
                val = input_df[row["feature_name"]].values[0]
                st.markdown(f"- **{row['feature_name']}** = `{val}` — {row['importance_pct']:.1f}% model weight")

# ============================================================
# PAGE 3 — TOPIC EXPLORER
# ============================================================

elif page == "Topic Explorer":
    st.title("Topic Explorer")
    st.caption("LDA topic model — recurring themes in banking complaints")

    if df_features.empty:
        st.info("Run Notebook 03 to generate topic assignments.")
        st.stop()

    if "dominant_topic" not in df_features.columns:
        st.info("Topic column not found. Run Notebook 03 first.")
        st.stop()

    # Topic name mapping (populated after manual naming in notebook 03)
    topic_names = df_features.get("topic_name", df_features["dominant_topic"].astype(str)).unique()
    if "topic_name" in df_features.columns:
        topics_available = sorted(df_features["topic_name"].dropna().unique())
        topic_col = "topic_name"
    else:
        topics_available = sorted(df_features["dominant_topic"].dropna().unique())
        topic_col = "dominant_topic"

    col_sel, col_desc = st.columns([1, 2])
    with col_sel:
        selected_topic = st.selectbox("Select topic:", topics_available)

    topic_df = df_features[df_features[topic_col] == selected_topic]

    st.markdown("---")
    col_wc, col_bar = st.columns([1, 1])

    with col_wc:
        st.subheader(f"WordCloud — {selected_topic}")
        text_col_name = "text_clean" if "text_clean" in df_features.columns else "consumer_complaint_narrative"
        if text_col_name in topic_df.columns:
            all_text = " ".join(topic_df[text_col_name].dropna().astype(str))
            if all_text.strip():
                wc = WordCloud(
                    width=600, height=400,
                    background_color="white",
                    colormap="Blues",
                    max_words=80,
                ).generate(all_text)
                fig_wc, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig_wc, use_container_width=True)

    with col_bar:
        st.subheader("NPS by Topic")
        if "nps_score" in df_features.columns:
            import sys
            sys.path.insert(0, str(ROOT))
            from src.nps_calculator import nps_by_group
            nps_topic = nps_by_group(df_features, topic_col)
            fig_nt = px.bar(
                nps_topic.sort_values("nps"), x="nps", y=topic_col,
                orientation="h",
                color="nps", color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                labels={"nps": "NPS", topic_col: "Topic"},
                text="nps",
            )
            fig_nt.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_nt.update_layout(coloraxis_showscale=False, height=400)
            st.plotly_chart(fig_nt, use_container_width=True)

    # --- Summary table ---
    st.subheader("Topic Summary")
    if "nps_score" in df_features.columns:
        summary_rows = []
        for topic, gdf in df_features.groupby(topic_col):
            row_data = {
                "Topic": topic,
                "% of Complaints": f"{len(gdf) / len(df_features) * 100:.1f}%",
                "Avg NPS": round(gdf["nps_score"].mean(), 1),
                "N Complaints": len(gdf),
            }
            if "severity_label" in gdf.columns:
                pct_high = (gdf["severity_label"] == "HIGH").sum() / len(gdf) * 100
                row_data["% HIGH Severity"] = f"{pct_high:.1f}%"
            summary_rows.append(row_data)
        st.dataframe(
            pd.DataFrame(summary_rows).sort_values("Avg NPS"),
            use_container_width=True,
            hide_index=True,
        )

    # --- Representative complaints ---
    st.subheader(f"Top 5 Representative Complaints — {selected_topic}")
    narr_col = "consumer_complaint_narrative" if "consumer_complaint_narrative" in topic_df.columns else "text_clean"
    if narr_col in topic_df.columns:
        sample = topic_df.nlargest(5, "topic_probability") if "topic_probability" in topic_df.columns else topic_df.head(5)
        for i, row in sample.iterrows():
            with st.expander(f"Complaint #{row.get('complaint_id', i)}"):
                st.write(row[narr_col])

# ============================================================
# PAGE 4 — CUSTOMER SEGMENTS
# ============================================================

elif page == "Customer Segments":
    st.title("Customer Segments")
    st.caption("K-Means clustering on NLP-derived features — CX action profiles")

    if df_features.empty or "cluster" not in df_features.columns:
        st.info("Run Notebook 04 to generate cluster assignments.")
        st.stop()

    # CX names mapping (populated after notebook 04)
    cx_names = {
        0: "Critical Risk",
        1: "Silent Dissatisfied",
        2: "Neutral Observers",
        3: "Promoter Candidates",
        4: "Active Promoters",
    }
    if "cluster_name" in df_features.columns:
        segment_col = "cluster_name"
    else:
        df_features = df_features.copy()
        df_features["cluster_name"] = df_features["cluster"].map(cx_names).fillna(df_features["cluster"].astype(str))
        segment_col = "cluster_name"

    segment_options = sorted(df_features[segment_col].unique())
    selected_segment = st.selectbox("Select customer segment:", segment_options)

    seg_df = df_features[df_features[segment_col] == selected_segment]

    # --- Cluster profile ---
    col_profile, col_scatter = st.columns([1, 2])
    with col_profile:
        st.subheader(f"Profile — {selected_segment}")
        if "nps_score" in seg_df.columns:
            st.metric("Avg NPS Score", f"{seg_df['nps_score'].mean():.1f}")
        if "severity_label" in seg_df.columns:
            dominant = seg_df["severity_label"].mode().iloc[0] if len(seg_df) > 0 else "—"
            pct_high = (seg_df["severity_label"] == "HIGH").sum() / len(seg_df) * 100
            st.metric("Dominant Severity", dominant)
            st.metric("% HIGH Severity", f"{pct_high:.1f}%")
        if "product" in seg_df.columns:
            top_product = seg_df["product"].mode().iloc[0] if len(seg_df) > 0 else "—"
            st.metric("Top Product", top_product)
        if "nps_segment" in seg_df.columns:
            pct_det = (seg_df["nps_segment"] == "Detractor").sum() / len(seg_df) * 100
            st.metric("% Detractors", f"{pct_det:.1f}%")
        st.metric("Total Complaints", f"{len(seg_df):,}")

        # CX Recommendations
        st.markdown("#### CX Recommendation")
        recommendations = {
            "Critical Risk": "Immediate outreach required. Prioritize personal follow-up within 48h. "
                             "Escalate to specialized retention team.",
            "Silent Dissatisfied": "Proactive NPS survey recommended. Neutral language masks high churn risk. "
                                   "Segment for targeted recovery campaigns.",
            "Neutral Observers": "Monitor for sentiment drift. Opportunity for proactive engagement "
                                  "to move toward Promoter territory.",
            "Promoter Candidates": "Leverage for referral programs. Request public reviews. "
                                    "Cross-sell premium products.",
            "Active Promoters": "Activate as brand ambassadors. Referral incentives apply. "
                                 "Low intervention needed.",
        }
        rec = recommendations.get(selected_segment, "Analyze segment characteristics for tailored action.")
        st.info(rec)

    # --- PCA scatter plot ---
    with col_scatter:
        st.subheader("Cluster Visualization (PCA)")
        if "pca_1" in df_features.columns and "pca_2" in df_features.columns:
            fig_pca = px.scatter(
                df_features, x="pca_1", y="pca_2",
                color=segment_col, symbol=segment_col,
                opacity=0.6,
                labels={"pca_1": "PCA Component 1", "pca_2": "PCA Component 2"},
                title="Customer Segments — PCA Projection",
            )
            fig_pca.update_layout(height=450, legend_title="Segment")
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.info("PCA coordinates not found. Run Notebook 04 with PCA export enabled.")

    # --- Segment summary table ---
    st.subheader("All Segments — Summary")
    import sys
    sys.path.insert(0, str(ROOT))
    from src.segmentation import build_cluster_profiles

    if "nps_segment" in df_features.columns:
        profiles = build_cluster_profiles(df_features)
        profiles["segment_name"] = profiles["cluster"].map(cx_names).fillna(profiles["cluster"].astype(str))
        display_cols = [c for c in ["segment_name", "n_complaints", "avg_nps", "dominant_severity",
                                     "avg_severity_proba_high", "top_product", "pct_detractors",
                                     "pct_high_severity"] if c in profiles.columns]
        st.dataframe(
            profiles[display_cols],
            use_container_width=True,
            hide_index=True,
        )

    # --- Product distribution by cluster ---
    if "product" in df_features.columns:
        st.subheader("Product Distribution by Segment")
        prod_dist = (
            df_features.groupby([segment_col, "product"])
            .size()
            .reset_index(name="count")
        )
        fig_prod = px.bar(
            prod_dist, x=segment_col, y="count", color="product",
            barmode="stack",
            labels={"count": "Complaints", segment_col: "Segment"},
        )
        fig_prod.update_layout(height=400, xaxis_tickangle=-20)
        st.plotly_chart(fig_prod, use_container_width=True)
