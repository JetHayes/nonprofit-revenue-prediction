# Nonprofit Revenue Prediction

A full machine learning pipeline applied to IRS nonprofit organization data — predicting revenue using regression, classification, and a hybrid clustering approach. Built with scikit-learn on a real-world messy dataset of US charities and nonprofits.

Developed as a data science course final project at the University of Cincinnati.

---

## The Problem

Given organizational characteristics of US nonprofits registered with the IRS — tax classification, state, ruling year, activity codes, and more — can we predict their annual revenue? And can segmenting organizations into clusters before modeling improve performance?

This project tackles the problem three ways: as a regression task (predict exact revenue), as a classification task (predict revenue tier), and as a hybrid system using KMeans clustering to train specialized models per segment.

---

## Dataset

**US Charities and Nonprofits (IRS)**
Source: [Kaggle — US Charities and Nonprofits](https://www.kaggle.com/datasets/crawford/us-charities-and-nonprofits?select=eo3.csv)

Download `eo3.csv` from the link above and place it in the project root before running.

**Target variable:** `REVENUE_AMT` — total annual revenue of each nonprofit organization

**Features used:** Tax period, ruling year, group code, activity codes, state (label encoded), NTEE category code, subsection, classification, affiliation, deductibility, organization type, status

**Data cleaning:** Removed direct identifiers and data leakage columns, dropped missing values, removed bottom and top 10% of revenue values to reduce skew, applied per-feature outlier removal at 1%-99% quantiles.

**Final dataset size:** ~10,000+ records after cleaning

---

## Results

### Regression — Predicting Revenue Amount

| Model | MSE | MAE | R² |
|-------|-----|-----|----|
| Linear Regression (baseline) | higher | higher | 0.69 |
| MLP Regressor | 40,872,682,170 | $103,729 | **0.82** |
| Clustered MLP Regressor | slightly lower | slightly lower | ~0.82 |

### Classification — Predicting Revenue Tier (Low / Medium / High / Very High)

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Logistic Regression (baseline) | 0.69 | — | — | — |
| MLP Classifier | **0.7033** | **0.6979** | **0.7111** | **0.7033** |
| Clustered MLP Classifier | ~0.70 | ~0.70 | ~0.70 | ~0.70 |

**Key finding:** Clustering did not improve performance — the clustered models performed roughly 1% worse than the standard MLP on both tasks. The MLP was already capturing the non-linear structure that clustering was trying to exploit.

---

## Pipeline Architecture

```
Raw IRS Data (eo3.csv)
   │
   ▼
Data Cleaning
   ├── Drop identifiers and leakage columns
   ├── Remove missing values
   ├── Remove revenue outliers (10%-90% quantile)
   └── Remove feature outliers (1%-99% per feature)
   │
   ▼
Feature Engineering
   ├── Label encode STATE and NTEE_CD
   ├── StandardScaler normalization
   └── Train/test split (80/20)
   │
   ├── Regression Label: REVENUE_AMT (continuous)
   └── Classification Label: 4 quantile bins (Low/Medium/High/Very High)
   │
   ▼
Section 4 — Standard Models
   ├── Linear Regression (baseline)
   ├── MLP Regressor (64→32 hidden layers)
   ├── Logistic Regression (baseline)
   └── MLP Classifier (64→32 hidden layers)
   │
   ▼
Section 5 — Hybrid Clustering System
   ├── KMeans clustering (optimal k selected by elbow method)
   ├── Assign train/test samples to clusters
   ├── Train cluster-specific MLP Regressor per cluster
   ├── Train cluster-specific MLP Classifier per cluster
   └── Evaluate per cluster and overall
   │
   ▼
Section 6 — Analysis and Comparison
```

---

## Quickstart

**Clone**
```bash
git clone https://github.com/JetHayes/nonprofit-revenue-prediction.git
cd nonprofit-revenue-prediction
```

**Install**
```bash
pip install -r requirements.txt
```

**Download the dataset**
Download `eo3.csv` from [Kaggle](https://www.kaggle.com/datasets/crawford/us-charities-and-nonprofits?select=eo3.csv) and place it in the project root.

**Run**
```bash
jupyter notebook nonprofit_revenue_prediction.ipynb
```

Or if running as a script:
```bash
python nonprofit_revenue_prediction.py
```

---

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
jupyter
```

---

## Lessons Learned

The most important lesson was dataset selection and understanding. The IRS nonprofit dataset required aggressive outlier removal (cutting the top and bottom 10%) to produce a usable revenue distribution — an approach that wouldn't be appropriate in production but was necessary to make the regression tractable given the extreme skew in nonprofit revenue.

The clustering experiment was a valuable negative result. Cluster-specific models performed slightly worse than a single MLP trained on all data, suggesting that the MLP's hidden layers were already learning the segment structure implicitly. Negative results are still results worth documenting.

Neural networks outperformed linear baselines on both tasks (R² 0.82 vs 0.69 for regression, 70% vs 69% for classification) — a modest but consistent improvement attributable to the MLP's ability to capture non-linear relationships between organizational characteristics and revenue.

---

## Future Work

- Better dataset selection — revenue prediction from organizational metadata is inherently noisy. Financial time-series data or grant database enrichment would produce a more reliable signal.
- Feature importance analysis — SHAP values or permutation importance to identify which organizational characteristics actually drive revenue.
- Gradient boosting models — XGBoost or LightGBM would likely outperform MLP on this tabular dataset with less hyperparameter tuning required.

---

## License

MIT License. See `LICENSE` for details.

---

## Author

**John Cavanaugh**
University of Cincinnati

[LinkedIn](https://www.linkedin.com/in/privacy-evangelist/) · [Email](johnthecavanaugh@gmail.com)
