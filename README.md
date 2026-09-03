# Fraud Driver Analysis

What actually explains credit card fraud, not just whether a model can flag it.

Most fraud detection write-ups optimize purely for catching fraud. This one asks a
different question: which factors, in plain and specific terms, distinguish a
fraudulent transaction from a legitimate one — and does that answer hold up across
several very different types of models, or is it just one algorithm's quirk?

## Why simulated data

The standard public fraud dataset (the Kaggle/ULB credit card dataset) anonymizes
every feature into PCA components (`V1` through `V28`) to protect the underlying bank's
data. That's necessary for privacy, but it makes the dataset a dead end for an
interpretation-focused project — you can't explain to a stakeholder what `V17` means.

So this project uses a simulated transaction dataset instead: ~100k transactions
across ~8,000 customers, with named, interpretable features (merchant category,
distance from home, spending relative to personal baseline, transaction velocity,
etc.) and a fraud rate calibrated to ~1.6%, in line with real-world card fraud rates.
Fraud probability is generated from a mix of realistic risk factors plus noise, so
there's genuine signal to find without the classes being trivially separable.
`data/generate_data.py` has the full generation logic if you want to see exactly how
it's built or tweak it.

## What's here

```
data/
  generate_data.py       # the dataset generator (fixed seed, reproducible)
  transactions.csv       # generated dataset, ~100k rows
notebooks/
  fraud_driver_analysis.ipynb   # full analysis: EDA, feature engineering, 4 models
report/
  fraud_driver_analysis_report.pdf   # written report, standalone (assignment format)
requirements.txt
```

## The analysis

- **EDA** on class imbalance, fraud rate by merchant category / channel / hour, and
  transaction amount patterns.
- **Feature engineering**: haversine distance from home and from the previous
  transaction, 24-hour transaction velocity, spend-relative-to-personal-baseline
  ratio, and time-of-day features.
- **Four classifiers**, spanning the explainability-predictability spectrum:
  Logistic Regression, a depth-limited Decision Tree, Random Forest, and XGBoost.
  Interpreted via coefficients/odds ratios, tree rules, feature importances, and
  SHAP values respectively.
- **Model comparison and a specific recommendation** — see the notebook's final
  sections for the reasoning, not just the numbers.

## Headline finding

The same handful of features — distance from the previous transaction, spend
relative to personal baseline, and transaction velocity — come out on top across
all four models, despite the models working in completely different ways
(coefficients, split rules, impurity-based importance, SHAP). Merchant category
(jewelry and travel run several times the baseline fraud rate) matters too, but
ranks below the behavioral features in every model. Full breakdown, numbers, and
the model recommendation are in the notebook and report.

## Running it

```bash
pip install -r requirements.txt
python data/generate_data.py        # regenerates transactions.csv (optional, already included)
jupyter notebook notebooks/fraud_driver_analysis.ipynb
```

## Limitations

Simulated data is the big one — see the notebook's closing section for this plus
class-imbalance handling, validation approach, and other caveats in full.

## Coursera capstone project for Supervised Machine Learning 

