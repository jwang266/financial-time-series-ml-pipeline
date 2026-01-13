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
