"""Metrics and diagnostic plots generated from held-out predictions."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             RocCurveDisplay)


def evaluate_models(models, X_test, y_test, results_dir: str | Path = "results"):
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    for name, model in models.items():
        raw_pred = model.predict(X_test)
        # IsolationForest and OneClassSVM use -1 for anomalies and +1 for inliers.
        is_outlier_detector = name in {"Isolation Forest", "One-Class SVM"}
        pred = (raw_pred == -1).astype(int) if is_outlier_detector else raw_pred
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            # decision_function is larger for inliers; invert it so larger means attack.
            scores = -model.decision_function(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, scores),
        })
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"],
                    yticklabels=["Normal", "Attack"], ax=ax)
        ax.set(title=f"{name} confusion matrix", xlabel="Predicted", ylabel="Actual")
        fig.tight_layout()
        fig.savefig(output / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close(fig)
        RocCurveDisplay.from_predictions(y_test, scores, name=name, ax=ax_roc)
    fig_roc.tight_layout()
    fig_roc.savefig(output / "roc_curve.png", dpi=150)
    plt.close(fig_roc)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "model_comparison.csv", index=False)
    ax = metrics.set_index("Model").plot(kind="bar", figsize=(9, 5), ylim=(0, 1), rot=0)
    ax.set_ylabel("Score")
    ax.set_title("Held-out model comparison")
    ax.figure.tight_layout()
    ax.figure.savefig(output / "model_comparison.png", dpi=150)
    plt.close(ax.figure)
    return metrics
