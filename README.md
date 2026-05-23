# Financial Time Series ML Pipeline

An end-to-end machine learning pipeline for financial time series data.

## Running the Pipeline
Install dependencies:
```bash
pip install -r requirements.txt
```
Run:
```bash
python src/run.py
```

## Project Structure
```text
.
├── notebooks
│   └── eda.ipynb
├── src
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── features.py
│   ├── models.py
│   └── run.py
└── README.md
```

## Data Source
Price data is downloaded using `yfinance` (Yahoo Finance).
Currently, the pipeline runs on AAPL daily data from 2019 to 2024.

## Features
- Log daily return and its lags (`lag_1`, `lag_2`, `lag_3`)
- Rolling mean return and volatility over 5- and 20-day windows
- 5-day average and relative trading volume
- Intraday price range (high − low normalized by close)

Label: 1 if next-day close is higher than today's close, else 0.

## Evaluation
Models output probabilities via `predict_proba`. Reported metrics:
- **ROC-AUC** — ranking quality, threshold-independent
- **Log loss** — proper scoring rule for probabilistic predictions
- Accuracy / macro-F1 / confusion matrix at the 0.5 threshold

The dummy baseline predicts the training base rate (`strategy='prior'`),
so its log loss is the entropy of the label distribution — a meaningful
floor for the real models to beat.

## Results

Expanding-window rolling slices (train on 2019 through year N−1, test on year N):

| Test year | Model | ROC-AUC | Log Loss |
|---|---|---:|---:|
| 2022 | Dummy | 0.500 | 0.705 |
| 2022 | Logistic Regression | 0.491 | 0.705 |
| 2022 | Random Forest | 0.508 | 0.696 |
| 2023 | Dummy | 0.500 | 0.688 |
| 2023 | Logistic Regression | 0.518 | 0.709 |
| 2023 | Random Forest | 0.548 | 0.752 |
| 2024 | Dummy | 0.500 | 0.686 |
| 2024 | Logistic Regression | 0.403 | 0.732 |
| 2024 | Random Forest | 0.455 | 0.744 |

Average AUC across slices is ~0.47 (LR) and ~0.50 (RF), and the dummy
beats both models on log loss in 2 of 3 slices. With price-only
features at a 1-day horizon, there is no clear predictive edge on AAPL
over this window — a useful negative result that the prior accuracy /
F1 view was hiding.
