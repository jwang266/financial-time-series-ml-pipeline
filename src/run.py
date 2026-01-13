from data_loader import load_price_data
from features import build_features
from sklearn.metrics import accuracy_score, classification_report
from models import lr_pipeline, rf_pipeline, dummy_pipeline


def main():
    df = load_price_data("AAPL", "2019-01-01", "2024-12-31")
    df_feat = build_features(df)

    df = df_feat.copy()
    X = df.drop(columns=["Date", "label_up_next_day"])
    y = df["label_up_next_day"]
    n = len(df)

    # time-based split: first 80% as train, last 20% as test
    split_idx = int(n * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]

    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    print("Train size:",len(X_train))
    print("Test size:",len(X_test))
    print("Lable Distribution: \n", y_test.value_counts(normalize=True))

    models_to_run = {
        "Baseline (Dummy)": dummy_pipeline(),
        "Logistic Regression": lr_pipeline(),
        "Random Forest": rf_pipeline()
    }
    for name, pipeline in models_to_run.items():
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"--> Accuracy: {acc:.4f}")

        if name != "Baseline (Dummy)":
            print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
