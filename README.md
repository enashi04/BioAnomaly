<div align="center">

<img src="https://img.shields.io/badge/-%F0%9F%A7%AC%20BioAnomaly-028090?style=for-the-badge&logoColor=white" alt="BioAnomaly" height="40"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.9+-EEF8FF?style=flat-square&logo=python&logoColor=028090)
![PyTorch](https://img.shields.io/badge/PyTorch-Autoencoder-68C3D4?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-729EA1?style=flat-square&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Sklearn-IsolationForest-028090?style=flat-square&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-EEF8FF?style=flat-square)

<br/>

> **Anomaly detection system for biological gene expression data**
> combining Isolation Forest and Deep Autoencoders —
> robust identification of unusual patterns in complex RNA-seq datasets.

<br/>

</div>

---

## Overview

**BioAnomaly** is an end-to-end data science pipeline designed to detect anomalous biological samples in high-dimensional gene expression data (TCGA RNA-seq). It combines two complementary unsupervised methods — a tree-based outlier detector and a deep neural network — to maximise detection robustness while minimising false positives.

The project ships with a fully interactive **Streamlit dashboard** styled with a medical color palette.

---

## Features

| Feature | Description |
|---|---|
| **Isolation Forest** | Detects outliers via random partitioning — anomalies are isolated faster |
| **Deep Autoencoder** | Flags samples with high reconstruction error (unusual patterns) |
| **Combined Score** | Normalized union of both detectors for robust final ranking |
| **t-SNE Visualization** | 2D projection of 20 000+ gene dimensions |
| **Interactive Dashboard** | 5-tab Streamlit UI with Plotly charts |
| **Exportable Report** | Full per-sample anomaly report downloadable as CSV |

---

## Project Structure

```
BioAnomaly/
│
├── dashboard_bioanomaly.py                  # Streamlit dashboard (main entry point)
├── bioanomaly.py           # Core pipeline class (CLI usage)
│
├── data.csv               # Gene expression matrix (801 × 20 531)
│── labels.csv              # Cancer type labels (BRCA, KIRC, LUAD, PRAD, COAD)
│
└── README.md
```

---

## Quick Start

### 1 — Clone & install dependencies

```bash
git clone https://github.com/your-username/BioAnomaly.git
cd BioAnomaly

pip install -r requirements.txt
```

### 2 — Launch the dashboard

```bash
streamlit run dashboard_bioanomaly.py
```

> **Do not run** `python dashboard_bioanomaly.py` — Streamlit apps must be launched with the `streamlit run` command.

### 3 — Upload your data & run

1. Upload `data.csv` and `labels.csv` in the sidebar
2. Adjust parameters (PCA variance, contamination rate, epochs…)
3. Click **Run Analysis**

---

## Requirements

```txt
streamlit>=1.32
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
torch>=2.2
plotly>=5.20
```

Install all at once:

```bash
pip install streamlit pandas numpy scikit-learn torch plotly
```

---

## Pipeline — How It Works

```
data.csv ──► StandardScaler ──► PCA (95% variance)
                                      │
                    ┌─────────────────┴──────────────────┐
                    │                                    │
           Isolation Forest                       Deep Autoencoder
           (contamination=5%)               (encoder-decoder, MSE loss)
                    │                                    │
             iso_scores                         recon_errors
                    │                                    │
                    └─────────── Combined Score ─────────┘
                                      │
                              Final Anomaly Flag
                                      │
                              t-SNE + Dashboard
```

### Isolation Forest

Builds an ensemble of random trees. Anomalous samples live in sparse regions and are **isolated faster** (shorter average path length → lower decision score).

### Deep Autoencoder

Trained to compress then reconstruct normal gene expression profiles. Samples the network never learned to reconstruct well produce **high Mean Squared Error** → flagged above the 95th percentile threshold.

### Combined Score

Both scores are normalized to `[0, 1]` and averaged:

```
combined_score = (1 - iso_norm) × 0.5 + ae_norm × 0.5
```

Final flag = **union** of both detectors → a sample is anomalous if flagged by *either* model.

---

## Dataset

The project uses the **TCGA Pan-Cancer RNA-seq** dataset from Kaggle.

| Property | Value |
|---|---|
| Samples | 801 |
| Features | 20 531 genes |
| Classes | 5 cancer types |
| Source | [Kaggle — TCGA Gene Expression](https://www.kaggle.com/datasets/crawford/gene-expression) |

**Class distribution:**

```
BRCA (Breast)     ████████████████████  300 samples
KIRC (Kidney)     █████████░░░░░░░░░░░  146 samples
LUAD (Lung)       ████████░░░░░░░░░░░░  141 samples
PRAD (Prostate)   ████████░░░░░░░░░░░░  136 samples
COAD (Colon)      ████░░░░░░░░░░░░░░░░   78 samples
```

---

## Dashboard — Tab Overview

| Tab | Content |
|---|---|
| **t-SNE Projection** | Cluster map coloured by cancer type, anomalies circled in red, class distribution pie, anomaly rate table |
| **Isolation Forest** | Anomaly scatter, decision score histogram, stacked bar per class |
| **Autoencoder** | Reconstruction error heatmap, training loss curve, error distribution with threshold |
| **Combined Analysis** | Composite score map, detector agreement bar, top-10 most anomalous samples |
| **Sample Report** | Filterable full table, CSV export |

---

## Color Palette

<div align="center">

| Hex | Usage |
|:---:|---|
| `#028090` | Primary — deep teal |
|`#729EA1` | Secondary — muted teal |
|`#68C3D4` | Accent — sky blue |
| `#EEF8FF` | Background — ice white |
| `#E05263` | Alert — anomaly flag |

</div>

---

## Typical Results (TCGA dataset)

| Detector | Anomalies | Rate |
|---|---|---|
| Isolation Forest | 40 | 5.0% |
| Autoencoder | 40 | 5.0% |
| **Combined (union)** | **75** | **9.4%** |
| Confirmed by both | 5 | 0.6% |

Highest anomaly rate by class: **LUAD** (lung) at ~18% — consistent with the known high molecular heterogeneity of lung adenocarcinoma.

---

## CLI Usage (without dashboard)

```python
from bioanomaly import BioAnomaly

bio = BioAnomaly(
    data_path="data.csv",
    labels_path="labels.csv"
)
bio.run_pipeline()
```

---

## License

This project is licensed under the **MIT License** — free to use, modify and distribute.

---

<div align="center">

Made with 🧬 and passion

</div>
