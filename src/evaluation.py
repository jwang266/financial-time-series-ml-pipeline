import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    log_loss,
)


def evaluate_classifier(model, X_test, y_true, model_name="Model"):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    pred_positive_rate = np.mean(y_pred == 1)
    true_positive_prevalence = np.mean(y_true == 1)
    tn, fp, fn, tp = cm.ravel()

    auc = roc_auc_score(y_true, y_proba)
    ll = log_loss(y_true, y_proba, labels=[0, 1])

    metrics = {
        "Model": model_name,
        "Accuracy": acc,
        "Precision_macro": precision,
        "Recall_macro": recall,
        "F1_macro": f1,
        "ROC_AUC": auc,
        "Log_Loss": ll,
        "Pred Positive Rate": pred_positive_rate,
        "True Positive Prevalence": true_positive_prevalence,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    print("-" * 20)
    print(f"{model_name} Evaluation: ")
    print(f"Accuracy:           {acc:.4f}")
    print(f"F1 Score (macro):   {f1:.4f}")
    print(f"ROC-AUC:            {auc:.4f}")
    print(f"Log Loss:           {ll:.4f}")
    print(f"Pred Positive Rate: {pred_positive_rate:.2%} (Positive Rate in data: {true_positive_prevalence:.2%})")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    return metrics
