import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="SpaceGuard", page_icon="🛰️", layout="wide")

FINAL = Path("outputs/final")
EVAL = Path("outputs/evaluation/saved_predictions_evaluation.csv")
SWEEP = Path("outputs/temporal_if/smoothing_sweep.csv")
GEN = Path("outputs/generalization/generalization_results.csv")
GEN_SUM = Path("outputs/generalization/generalization_summary.csv")


@st.cache_data
def load_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_config():
    p = FINAL / "config.json"
    return json.load(open(p)) if p.exists() else None


pred = load_csv(FINAL / "predictions.csv")
metrics = load_csv(FINAL / "metrics.csv")
cfg = load_config()

if pred is None or metrics is None or cfg is None:
    st.error("Missing files in outputs/final. Run final_model.py first.")
    st.stop()

regions = cfg["regions"]

st.sidebar.title("🛰️ SpaceGuard")
page = st.sidebar.radio("Page", [
    "Overview", "Telemetry & ground truth", "Scores & thresholds",
    "Confusion matrix & metrics", "Model comparison",
    "Generalization", "Detected anomalies & limitations"])
pct = st.sidebar.selectbox("Threshold (validation percentile)",
                           cfg["threshold_percentiles"], index=len(cfg["threshold_percentiles"]) - 1)
pred_col = f"pred_p{pct}"
mrow = metrics[metrics["percentile"] == pct].iloc[0]
st.sidebar.caption("99th = low false-alarm setting, 95th = more sensitive setting.")


def shade(fig):
    for i, (s, e) in enumerate(regions):
        fig.add_vrect(x0=s, x1=e, fillcolor="orange", opacity=0.25, line_width=0,
                      annotation_text=f"true anomaly {i + 1}", annotation_position="top left")
    return fig


if page == "Overview":
    st.title("SpaceGuard: Spacecraft Telemetry Anomaly Detection")
    st.caption("University AI project on a public benchmark dataset.")
    st.warning("This is an offline research project on the public Telemanom benchmark. "
               "It is NOT connected to live spacecraft, NOT endorsed by NASA, and NOT "
               "flight-certified or suitable for spacecraft control.")
    c = st.columns(4)
    c[0].metric("ROC-AUC", f"{mrow['ROC-AUC']:.3f}", help="0.5 = random")
    c[1].metric("PR-AUC", f"{mrow['PR-AUC']:.3f}", help=f"random = {mrow['random PR-AUC']:.3f}")
    c[2].metric("Regions detected", mrow["regions detected"])
    c[3].metric("False-alarm events", int(mrow["false-alarm events"]))
    st.subheader("Method")
    st.markdown(f"""
- **Data:** channel **{cfg['channel']}** (SMAP) from the Telemanom benchmark (Hundman et al., 2018).
- **Features:** rolling mean, standard deviation, min, max (10 and 50 steps), recent changes, and command activity. Windows only look at the past.
- **Model:** Isolation Forest (n_estimators={cfg['n_estimators']}, seed {cfg['random_state']}), fitted on the first {int(cfg['fit_fraction'] * 100)}% of the training data.
- **Scoring:** anomaly scores are smoothed with a causal moving average (EWMA span {cfg['ewma_span']}).
- **Thresholds:** percentiles of scores on a chronological validation slice. Test labels are used only for evaluation.
""")

elif page == "Telemetry & ground truth":
    st.header("Telemetry, true anomaly intervals and alarms")
    lo, hi = st.slider("Timestep range", 0, len(pred) - 1, (0, len(pred) - 1))
    sub = pred.iloc[lo:hi + 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["timestep"], y=sub["value"], mode="lines",
                             name="telemetry value", line=dict(width=1)))
    alarms = sub[sub[pred_col] == 1]
    fig.add_trace(go.Scatter(x=alarms["timestep"], y=alarms["value"], mode="markers",
                             name=f"alarm ({pct}th pct)", marker=dict(color="red", size=4)))
    shade(fig)
    fig.update_layout(height=500, xaxis_title="test timestep", yaxis_title="normalized value")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Orange = labeled anomaly intervals. Red dots = points the model flags.")

elif page == "Scores & thresholds":
    st.header("Anomaly score over time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred["timestep"], y=pred["raw_score"], mode="lines",
                             name="raw score", line=dict(width=0.7, color="lightgray")))
    fig.add_trace(go.Scatter(x=pred["timestep"], y=pred["smoothed_score"], mode="lines",
                             name=f"smoothed (EWMA span {cfg['ewma_span']})", line=dict(width=1.6)))
    fig.add_hline(y=cfg["thresholds"][str(pct)], line_dash="dash", line_color="red",
                  annotation_text=f"{pct}th pct validation threshold")
    shade(fig)
    fig.update_layout(height=500, xaxis_title="test timestep", yaxis_title="score (higher = more unusual)")
    st.plotly_chart(fig, use_container_width=True)
    val = load_csv(FINAL / "validation_scores.csv")
    if val is not None:
        st.subheader("Validation score distribution (used to set thresholds)")
        h = px.histogram(val, x="val_smoothed_score", nbins=40)
        for q, t in cfg["thresholds"].items():
            h.add_vline(x=t, line_dash="dash", annotation_text=f"{q}th")
        h.update_layout(height=300)
        st.plotly_chart(h, use_container_width=True)

elif page == "Confusion matrix & metrics":
    st.header(f"Test-set results at the {pct}th percentile threshold")
    z = [[mrow["TN"], mrow["FP"]], [mrow["FN"], mrow["TP"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted normal", "Predicted anomaly"],
                               y=["Actual normal", "Actual anomaly"],
                               text=z, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(height=380, yaxis_autorange="reversed")
    c1, c2 = st.columns([1, 1])
    c1.plotly_chart(fig, use_container_width=True)
    c2.dataframe(pd.DataFrame({
        "metric": ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC (random = %.3f)" % mrow["random PR-AUC"],
                   "Regions detected", "Mean detection delay (steps)", "False-alarm events"],
        "value": [mrow["precision"], mrow["recall"], mrow["F1"], mrow["ROC-AUC"], mrow["PR-AUC"],
                  mrow["regions detected"], mrow["mean delay"], mrow["false-alarm events"]]
    }).astype(str), hide_index=True, use_container_width=True)
    st.caption("Accuracy is intentionally not shown: anomalies are a small share of points, "
               "so accuracy would look high even for a useless detector.")
    st.dataframe(metrics, hide_index=True, use_container_width=True)

elif page == "Model comparison":
    st.header("Model comparison on the P-1 test set")
    ev = load_csv(EVAL)
    final_row = {"model": f"Final: Isolation Forest + EWMA {cfg['ewma_span']} ({pct}th pct)",
                 "ROC-AUC": mrow["ROC-AUC"], "PR-AUC": mrow["PR-AUC"],
                 "PR-AUC random baseline": mrow["random PR-AUC"],
                 "regions detected": mrow["regions detected"],
                 "mean delay (steps)": mrow["mean delay"],
                 "false-alarm events": mrow["false-alarm events"]}
    if ev is not None:
        comp = pd.concat([ev, pd.DataFrame([final_row])], ignore_index=True)
    else:
        comp = pd.DataFrame([final_row])
        st.info("outputs/evaluation file not found; showing only the final model.")
    st.dataframe(comp, hide_index=True, use_container_width=True)
    long = comp.melt(id_vars="model", value_vars=["ROC-AUC", "PR-AUC"])
    fig = px.bar(long, x="value", y="model", color="variable", barmode="group", orientation="h")
    fig.add_vline(x=0.5, line_dash="dot", annotation_text="ROC random")
    fig.add_vline(x=float(mrow["random PR-AUC"]), line_dash="dot", line_color="green",
                  annotation_text="PR random")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    sw = load_csv(SWEEP)
    if sw is not None:
        st.subheader("Effect of score smoothing (all features, from the span sweep)")
        s95 = sw[sw["val pct"] == 95]
        f2 = px.line(s95.melt(id_vars="span", value_vars=["ROC-AUC", "PR-AUC"]),
                     x="span", y="value", color="variable", markers=True)
        f2.update_layout(height=320)
        st.plotly_chart(f2, use_container_width=True)
        st.caption("The smoothing span was explored using test results, so the headline "
                   "numbers are somewhat optimistic.")

elif page == "Generalization":
    st.header("Does the same pipeline work on other channels?")
    g, gs = load_csv(GEN), load_csv(GEN_SUM)
    if g is None:
        st.info("Run generalization_test.py first.")
    else:
        st.write("Channels with constant training telemetry could not be modeled and were skipped.")
        if gs is not None:
            st.dataframe(gs, hide_index=True, use_container_width=True)
        fig = px.bar(g, x="channel", y="ROC-AUC", color="method", barmode="group")
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="chance")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(g, hide_index=True, use_container_width=True)

else:
    st.header(f"What was detected, region by region ({pct}th percentile)")
    rows = []
    for i, (s, e) in enumerate(regions, 1):
        seg = pred.iloc[s:e + 1]
        hits = np.where(seg[pred_col].values == 1)[0]
        rows.append({"region": i, "start-end": f"{s}-{e}", "length": e - s + 1,
                     "alarm points inside": int(len(hits)),
                     "share of region flagged": round(len(hits) / (e - s + 1), 3),
                     "detection delay (steps)": int(hits[0]) if len(hits) else None,
                     "max smoothed score": round(seg["smoothed_score"].max(), 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.markdown(f"""
**Reading this:** the model raised an alarm in **{mrow['regions detected']}** labeled regions,
but it typically flags only part of each region and reacts after a delay, because scores are
smoothed over a window. It also produced **{int(mrow['false-alarm events'])}** false-alarm events
on normal data at this threshold.

### Limitations
- Evaluated mostly on one channel with only 3 labeled anomaly regions.
- The smoothing span was explored using test results, so the numbers are optimistic.
- On other channels, results were uneven, and false alarms were frequent (see Generalization).
- Many Telemanom channels have constant training telemetry and cannot be modeled this way.
- Anomaly types differ per channel; one fixed rule does not fit all.
- Point-level metrics and event-level metrics tell different stories; both are shown.
""")