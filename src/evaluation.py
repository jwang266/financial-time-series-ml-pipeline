import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate_classifier(y_true, y_pred, model_name="Model"):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    pred_positive_rate = np.mean(y_pred == 1)
    true_positive_prevalence = np.mean(y_true == 1)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "Model": model_name,
        "Accuracy": acc,
        "Precision_macro": precision,
        "Recall_macro": recall,
        "F1_macro": f1,
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
    print(f"F1 Score (macro):       {f1:.4f}")
    print(f"Pred Positive Rate: {pred_positive_rate:.2%} (Positive Rate in data: {true_positive_prevalence:.2%})")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    return metrics
