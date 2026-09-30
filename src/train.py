"""Train, tune, evaluate, and persist KDD Cup 99 intrusion classifiers."""
import argparse
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM, SVC

try:  # Support script execution and notebook/package imports.
    from .data_preprocessing import clean_data, describe_data, load_sampled_data
    from .evaluate import evaluate_models
    from .feature_engineering import make_preprocessor
except ImportError:
    from data_preprocessing import clean_data, describe_data, load_sampled_data
    from evaluate import evaluate_models
    from feature_engineering import make_preprocessor


def run(data_path: str = "/content/kddcup.data.corrected", output_dir: str = ".", sampled_df=None):
    root = Path(output_dir)
    models_dir, results_dir = root / "models", root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Streamlit can preview the uploaded file first and pass its already sampled
    # 12,000 rows here, avoiding a second read of the large compressed dataset.
    raw = load_sampled_data(data_path) if sampled_df is None else sampled_df.copy()
    if sampled_df is not None and len(raw) != 12_000:
        raise ValueError(f"Expected a previously sampled 12,000-row DataFrame; found {len(raw)} rows.")
    print("Raw data diagnostics")
    data = clean_data(raw)
    describe_data(data)
    print("\nCleaned target distribution:\n", data["target"].value_counts())
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=data["target"].map({0: "Normal", 1: "Attack"}), ax=ax)
    ax.set(title="12,000 sampled connections", xlabel="Class", ylabel="Count")
    fig.tight_layout()
    fig.savefig(results_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    X, y = data.drop(columns="target"), data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if y_train.nunique() < 2:
        raise ValueError("The sampled training partition has only one class; stratified binary training is impossible.")
    cv = 3
    rf_pipe = Pipeline([("preprocess", make_preprocessor()), ("model", RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    ))])
    svm_pipe = Pipeline([("preprocess", make_preprocessor(scale=True)), ("model", SVC(
        probability=True, class_weight="balanced", random_state=42
    ))])
    searches = {
        "Random Forest": GridSearchCV(rf_pipe, {
            "model__n_estimators": [150, 250], "model__max_depth": [None, 20],
            "model__min_samples_leaf": [1, 2],
        }, scoring="f1", cv=cv, n_jobs=-1, refit=True),
        "SVM": GridSearchCV(svm_pipe, {
            "model__C": [0.5, 2], "model__kernel": ["rbf"], "model__gamma": ["scale", "auto"],
        }, scoring="f1", cv=cv, n_jobs=-1, refit=True),
    }
    fitted = {}
    for name, search in searches.items():
        print(f"\nTuning {name} (3-fold CV, F1 scoring)...")
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        file_name = "random_forest.pkl" if name == "Random Forest" else "svm.pkl"
        joblib.dump(search.best_estimator_, models_dir / file_name)
        print("Best CV F1:", search.best_score_)
        print("Best parameters:", search.best_params_)
    # Unsupervised detectors share the train-only transforms but learn their
    # boundary without labels. A 30% contamination estimate is retained from
    # the prior notebook; verify/tune it against the operating false-alarm rate.
    anomaly_models = {
        "Isolation Forest": Pipeline([
            ("preprocess", make_preprocessor(scale=True)),
            ("model", IsolationForest(n_estimators=200, contamination=0.30,
                                       random_state=42, n_jobs=-1)),
        ]),
        "One-Class SVM": Pipeline([
            ("preprocess", make_preprocessor(scale=True)),
            ("model", OneClassSVM(kernel="rbf", gamma="scale", nu=0.10)),
        ]),
    }
    for name, model in anomaly_models.items():
        print(f"\nFitting {name} (unsupervised)...")
        model.fit(X_train)
        fitted[name] = model
        file_name = "isolation_forest.pkl" if name == "Isolation Forest" else "one_class_svm.pkl"
        joblib.dump(model, models_dir / file_name)
    metrics = evaluate_models(fitted, X_test, y_test, results_dir)
    print("\nHeld-out metrics (actual sampled experiment):\n", metrics.to_string(index=False))

    rf = fitted["Random Forest"]
    names = rf.named_steps["preprocess"].get_feature_names_out()
    importance = pd.Series(rf.named_steps["model"].feature_importances_, index=names)
    top = importance.nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    top.plot.barh(ax=ax)
    ax.set(title="Random Forest: top 20 feature importances", xlabel="Importance", ylabel="Encoded feature")
    fig.tight_layout()
    fig.savefig(results_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    importance.sort_values(ascending=False).rename("importance").to_csv(results_dir / "feature_importance.csv")
    print("\nTop features:\n", top.sort_values(ascending=False).head(10))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="/content/kddcup.data.corrected")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    run(args.data_path, args.output_dir)
