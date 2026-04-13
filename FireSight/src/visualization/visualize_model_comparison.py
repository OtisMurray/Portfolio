from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]

# new advanced results
MODEL_COMPARISON_PATH = BASE_DIR / "data/processed/advanced_model_comparison_test_2024_2025.csv"

# optional old results for improvement comparison
OLD_MODEL_COMPARISON_PATH = BASE_DIR / "data/processed/model_comparison_test_2024_2025.csv"

st.set_page_config(
    page_title="FireSight Model Comparison",
    page_icon="📊",
    layout="wide",
)

st.title("📊 FireSight Model Comparison")
st.caption("Comparison of advanced wildfire prediction models on the 2024–2025 test period")

if not MODEL_COMPARISON_PATH.exists():
    st.error(f"Could not find model comparison file:\n{MODEL_COMPARISON_PATH}")
    st.stop()

df = pd.read_csv(MODEL_COMPARISON_PATH)

required_cols = [
    "model",
    "best_threshold_by_f1",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "predicted_positives",
    "true_positives",
]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    st.error(f"Missing required columns in advanced model comparison file: {missing}")
    st.stop()

df = df.sort_values("f1", ascending=False).reset_index(drop=True)

metric_labels = {
    "roc_auc": "ROC-AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
}

# ======================
# TOP SUMMARY
# ======================
best_row = df.iloc[0]
best_auc_row = df.loc[df["roc_auc"].idxmax()]
best_precision_row = df.loc[df["precision"].idxmax()]
best_recall_row = df.loc[df["recall"].idxmax()]
best_f1_row = df.loc[df["f1"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best Overall by F1", best_f1_row["model"])
c2.metric("Best Threshold", f"{best_f1_row['best_threshold_by_f1']:.2f}")
c3.metric("Best F1", f"{best_f1_row['f1']:.3f}")
c4.metric("Best Precision", f"{best_precision_row['precision']:.3f}")

st.markdown("---")

# ======================
# OPTIONAL IMPROVEMENT VS OLD RESULTS
# ======================
if OLD_MODEL_COMPARISON_PATH.exists():
    old_df = pd.read_csv(OLD_MODEL_COMPARISON_PATH)

    old_required = [
        "model",
        "roc_auc",
        "precision",
        "recall",
        "f1",
    ]
    old_missing = [col for col in old_required if col not in old_df.columns]

    if not old_missing:
        old_best_f1 = old_df["f1"].max()
        old_best_precision = old_df["precision"].max()
        old_best_recall = old_df["recall"].max()
        old_best_auc = old_df["roc_auc"].max()

        f1_delta = best_f1_row["f1"] - old_best_f1
        precision_delta = best_precision_row["precision"] - old_best_precision
        recall_delta = best_recall_row["recall"] - old_best_recall
        auc_delta = best_auc_row["roc_auc"] - old_best_auc

        st.subheader("Improvement from earlier run")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Best F1 change", f"{best_f1_row['f1']:.3f}", delta=f"{f1_delta:+.3f}")
        d2.metric("Best Precision change", f"{best_precision_row['precision']:.3f}", delta=f"{precision_delta:+.3f}")
        d3.metric("Best Recall change", f"{best_recall_row['recall']:.3f}", delta=f"{recall_delta:+.3f}")
        d4.metric("Best ROC-AUC change", f"{best_auc_row['roc_auc']:.3f}", delta=f"{auc_delta:+.3f}")

        st.info(
            f"The advanced run improved practical classification performance. "
            f"The top F1 is now {best_f1_row['f1']:.3f}, led by {best_f1_row['model']} "
            f"at threshold {best_f1_row['best_threshold_by_f1']:.2f}."
        )

        st.markdown("---")

# ======================
# TABLE
# ======================
st.subheader("Advanced model comparison table")

display_df = df.copy()
display_df = display_df.rename(columns={
    "best_threshold_by_f1": "best_threshold",
    "predicted_positives": "predicted_positives",
    "true_positives": "true_positives",
    "fit_seconds": "fit_seconds",
    "predict_seconds": "predict_seconds",
})

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ======================
# BEST MODEL CALLOUT
# ======================
st.subheader("Current takeaway")
st.success(
    f"{best_row['model']} is currently the best overall model by F1 "
    f"({best_row['f1']:.3f}) at threshold {best_row['best_threshold_by_f1']:.2f}. "
    f"It also achieved precision {best_row['precision']:.3f} and recall {best_row['recall']:.3f}."
)

# ======================
# SINGLE METRIC CHART
# ======================
st.subheader("Metric comparison")

selected_metric = st.selectbox(
    "Metric to compare",
    ["f1", "precision", "recall", "roc_auc"],
    format_func=lambda x: metric_labels[x],
    index=0,
)

fig = px.bar(
    df,
    x="model",
    y=selected_metric,
    text=selected_metric,
    color=selected_metric,
    title=f"Advanced Model Comparison: {metric_labels[selected_metric]}",
)

fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig.update_layout(
    yaxis_range=[0, max(1.0, df[selected_metric].max() * 1.20)],
    xaxis_title="Model",
    yaxis_title=metric_labels[selected_metric],
    coloraxis_showscale=False,
)

st.plotly_chart(fig, use_container_width=True)

# ======================
# ALL METRICS CHART
# ======================
st.subheader("All key metrics")

metric_df = df.melt(
    id_vars=["model", "best_threshold_by_f1"],
    value_vars=["roc_auc", "precision", "recall", "f1"],
    var_name="metric",
    value_name="value",
)
metric_df["metric_label"] = metric_df["metric"].map(metric_labels)

fig2 = px.bar(
    metric_df,
    x="model",
    y="value",
    color="metric_label",
    barmode="group",
    text="value",
    title="ROC-AUC, Precision, Recall, and F1 by Model",
)

fig2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig2.update_layout(
    xaxis_title="Model",
    yaxis_title="Score",
    yaxis_range=[0, max(1.0, metric_df["value"].max() * 1.20)],
)

st.plotly_chart(fig2, use_container_width=True)

# ======================
# PRECISION VS RECALL
# ======================
st.subheader("Precision vs recall tradeoff")

scatter_df = df.copy()
scatter_df["label"] = scatter_df.apply(
    lambda row: f"{row['model']}<br>Threshold: {row['best_threshold_by_f1']:.2f}<br>F1: {row['f1']:.3f}",
    axis=1,
)

fig3 = px.scatter(
    scatter_df,
    x="recall",
    y="precision",
    size="f1",
    color="model",
    hover_name="model",
    text="model",
    title="Precision vs Recall (bubble size = F1)",
)

fig3.update_traces(textposition="top center")
fig3.update_layout(
    xaxis_title="Recall",
    yaxis_title="Precision",
)

st.plotly_chart(fig3, use_container_width=True)

# ======================
# COUNTS
# ======================
st.subheader("Prediction volume")

volume_df = df.melt(
    id_vars=["model"],
    value_vars=["predicted_positives", "true_positives"],
    var_name="count_type",
    value_name="count",
)
volume_df["count_type"] = volume_df["count_type"].replace({
    "predicted_positives": "Predicted Positives",
    "true_positives": "True Positives",
})

fig4 = px.bar(
    volume_df,
    x="model",
    y="count",
    color="count_type",
    barmode="group",
    text="count",
    title="Predicted Positive Cells vs True Positive Cells",
)

fig4.update_traces(texttemplate="%{text:,}", textposition="outside")
fig4.update_layout(
    xaxis_title="Model",
    yaxis_title="Count",
)

st.plotly_chart(fig4, use_container_width=True)

# ======================
# SPEED
# ======================
if {"fit_seconds", "predict_seconds"}.issubset(df.columns):
    st.subheader("Training speed")

    speed_df = df.melt(
        id_vars=["model"],
        value_vars=["fit_seconds", "predict_seconds"],
        var_name="time_type",
        value_name="seconds",
    )
    speed_df["time_type"] = speed_df["time_type"].replace({
        "fit_seconds": "Fit Time (s)",
        "predict_seconds": "Predict Time (s)",
    })

    fig5 = px.bar(
        speed_df,
        x="model",
        y="seconds",
        color="time_type",
        barmode="group",
        text="seconds",
        title="Model runtime comparison",
    )

    fig5.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig5.update_layout(
        xaxis_title="Model",
        yaxis_title="Seconds",
    )

    st.plotly_chart(fig5, use_container_width=True)

# ======================
# FOOTER METRICS
# ======================
st.subheader("Best metric leaders")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Best ROC-AUC", f"{best_auc_row['model']} ({best_auc_row['roc_auc']:.3f})")
m2.metric("Best Precision", f"{best_precision_row['model']} ({best_precision_row['precision']:.3f})")
m3.metric("Best Recall", f"{best_recall_row['model']} ({best_recall_row['recall']:.3f})")
m4.metric("Best F1", f"{best_f1_row['model']} ({best_f1_row['f1']:.3f})")