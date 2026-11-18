# SpaceGuard: Spacecraft Telemetry Anomaly Detection

Offline university project (Bachelor's in AI) on the public **Telemanom** benchmark (SMAP/MSL spacecraft telemetry, Hundman et al., 2018).

> **Disclaimer:** This project is not connected to live spacecraft, is not endorsed by NASA, and is not flight-certified or suitable for spacecraft control.

## Summary
Simple anomaly detectors on raw telemetry performed at chance level. Rolling-window features and smoothing of the anomaly scores gave a clear improvement on channel **P-1**. Results on other channels were uneven, and false alarms are a real limitation.

## Dataset
- Hugging Face mirror: `appleparan/telemanom` (original: github.com/khundman/telemanom)
- Main channel: **P-1** (SMAP): 2,872 training points, 8,505 test points, 3 labeled anomaly regions (751 anomalous points, 8.83%)
- Anomalies are labeled "contextual", not simple extreme values

## Method
- Features (past-only windows): rolling mean/std/min/max (10 and 50 steps), recent changes, command activity
- Model: Isolation Forest (300 trees, seed 42), fitted on the first 80% of the training data
- Scores smoothed with a causal EWMA (span 100)
- Thresholds: percentiles of validation scores (chronological last 20% of training data). Test labels are used only for evaluation.

## Results (channel P-1, test set)

| Experiment | ROC-AUC | PR-AUC |
|---|---|---|
| Univariate Isolation Forest | 0.496 | 0.088 |
| Multivariate Isolation Forest (raw) | 0.520 | 0.094 |
| LSTM autoencoder | 0.462 | 0.079 |
| Rolling features, no smoothing | 0.671 | 0.186 |
| **Final: rolling features + EWMA span 100** | **0.832** | **0.444** |

Random guessing gives PR-AUC of about 0.088.

| Threshold | Precision | Recall | F1 | Regions found | False-alarm events | Mean delay |
|---|---|---|---|---|---|---|
| 99th percentile | 0.729 | 0.293 | 0.418 | 3/3 | 2 | 67 steps |
| 95th percentile | 0.570 | 0.505 | 0.535 | 3/3 | 12 | 55 steps |

Confusion matrices (TN / FP / FN / TP): 99th = 7672 / 82 / 531 / 220; 95th = 7468 / 286 / 372 / 379.

**Other channels:** on 6 other channels (4 more had constant training telemetry and were skipped), the same pipeline had median ROC-AUC 0.695 and median PR-AUC about 2.7x random, but it was below chance on some channels and produced many false alarms.

## Limitations
- Mostly one channel with 3 anomaly regions; small sample.
- The smoothing span and feature choices were explored using test results, so reported numbers are optimistic.
- A hypothesis that anomalies on P-1 are "unusually quiet" stretches did **not** generalize to other channels.
- Validation set is short, so threshold transfer is imperfect; false alarms are frequent on some channels.
- False alarms appear in the warm-up period at the start of the test series.
- Many channels have constant training telemetry and can't be modeled this way.

## Run it
```bash
pip install -r requirements.txt
python final_model.py
python -m streamlit run dashboard.py
```
The dashboard reads saved results from `outputs/`. Experiment scripts are in the repo root (data download, EDA, baselines, LSTM, temporal features, generalization test). The raw data is not included; use the download scripts.

## References
- Hundman et al. (2018). *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.* https://arxiv.org/abs/1802.04431
- Telemanom: https://github.com/khundman/telemanom
- Dataset mirror: https://huggingface.co/datasets/appleparan/telemanom

Author WAJEEH RIZWAN
