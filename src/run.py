import pandas as pd

from data_loader import load_price_data
from evaluation import evaluate_classifier
from features import build_features
from models import lr_pipeline, rf_pipeline, dummy_pipeline


def evaluate_rolling_slices(df_feat, slices):
    df = df_feat.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    all_metrics = []
    for train_start, train_end, test_year, slice_label in slices:
        train_mask = (df["Date"].dt.year >= train_start) & (df["Date"].dt.year <= train_end)
        test_mask = df["Date"].dt.year == test_year

        df_train = df.loc[train_mask].sort_values("Date").reset_index(drop=True)
        df_test = df.loc[test_mask].sort_values("Date").reset_index(drop=True)

        if len(df_train) == 0 or len(df_test) == 0:
            print(f"Skip {slice_label}: empty train or test")
            continue

        X_train = df_train.drop(columns=["Date", "label_up_next_day"])
        y_train = df_train["label_up_next_day"]
        X_test = df_test.drop(columns=["Date", "label_up_next_day"])
        y_test = df_test["label_up_next_day"]

        print(f"\n=== {slice_label} (train: {train_start}-{train_end}, test: {test_year}, n_train={len(X_train)}, n_test={len(X_test)}) ===")

        for name, pipeline in {
            "Baseline (Dummy)": dummy_pipeline(),
            "Logistic Regression": lr_pipeline(),
            "Random Forest": rf_pipeline(),
        }.items():
            pipeline.fit(X_train, y_train)

            metrics = evaluate_classifier(pipeline, X_test, y_test, model_name=f"{slice_label} | {name}")
            metrics["slice"] = slice_label
            metrics["Model"] = name  # keep clean model name for summaries
            all_metrics.append(metrics)

    return all_metrics


def main():
    df = load_price_data("AAPL", "2019-01-01", "2024-12-31")
    df_feat = build_features(df)

    df_main = df_feat.copy()
    X = df_main.drop(columns=["Date", "label_up_next_day"])
    y = df_main["label_up_next_day"]

    n = len(df_main)
    split_idx = int(n * 0.8)

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    print("Label distribution:\n", y_test.value_counts(normalize=True))

    print("\n" + "=" * 60)
    print("Single Split Evaluation (80/20)")
    print("=" * 60)

    for name, pipeline in {
        "Baseline (Dummy)": dummy_pipeline(),
        "Logistic Regression": lr_pipeline(),
        "Random Forest": rf_pipeline(),
    }.items():
        pipeline.fit(X_train, y_train)

        m = evaluate_classifier(pipeline, X_test, y_test, model_name=name)
        print(f"{name:25s} | Acc: {m['Accuracy']:.4f} | F1: {m['F1_macro']:.4f} | AUC: {m['ROC_AUC']:.4f} | LogLoss: {m['Log_Loss']:.4f} | Pred+: {m['Pred Positive Rate']:.2%} | True+: {m['True Positive Prevalence']:.2%}")

    print("\n" + "=" * 60)
    print("Rolling Time-Slice Evaluation (2019-2024)")
    print("=" * 60)

    slices = [
        (2019, 2021, 2022, "Slice 1: 2019-2021 → 2022"),
        (2019, 2022, 2023, "Slice 2: 2019-2022 → 2023"),
        (2019, 2023, 2024, "Slice 3: 2019-2023 → 2024"),
    ]

    rolling_metrics = evaluate_rolling_slices(df_feat, slices)

    print("\n" + "=" * 60)
    print("Rolling Slice Summary")
    print("=" * 60)

    for _, _, _, slice_label in slices:
        slice_results = [m for m in rolling_metrics if m["slice"] == slice_label]
        if not slice_results:
            continue

        print(f"\n{slice_label}:")
        for m in slice_results:
            print(
                f"  {m['Model']:25s} | Acc: {m['Accuracy']:.4f} | F1: {m['F1_macro']:.4f}"
                f" | AUC: {m['ROC_AUC']:.4f} | LogLoss: {m['Log_Loss']:.4f}"
                f" | Pred+: {m['Pred Positive Rate']:.2%} | True+: {m['True Positive Prevalence']:.2%}"
            )


if __name__ == "__main__":
    main()