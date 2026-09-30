# Network Intrusion Detection Using Machine Learning

An end-to-end, reproducible intrusion-detection experiment for normal versus malicious network connections in the KDD Cup 99 corrected dataset. The training program samples exactly 12,000 rows with `random_state=42`, evaluates supervised Random Forest and Support Vector Machine pipelines alongside Isolation Forest and One-Class SVM detectors from the previous notebook, and calculates metrics and plots from held-out predictions.

## Problem statement

Network intrusion detection must identify anomalous connections while balancing missed attacks and false alarms. This project treats the KDD attack categories as one `attack` class and the `normal` label as the `normal` class. Labels are not included among model inputs.

## Objectives

- Explore the sample, feature types, missing values, and attack-label distribution.
- Apply reproducible cleaning, one-hot encoding, imputation, and scaling where needed.
- Compare tuned Random Forest and SVM models with imbalance-aware class weights, plus Isolation Forest and One-Class SVM anomaly detectors.
- Report held-out accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC curves, and feature importance.
- Save fitted end-to-end pipelines so new raw records receive the same preprocessing.

## Dataset

Use the KDD Cup 99 corrected connection dataset file `kddcup.data.corrected`. Put it at `data/kddcup.data.corrected`, or provide its path at runtime. In Google Colab the default is `/content/kddcup.data.corrected`. The project reads it with `pandas.read_csv(path, header=None)` and then selects exactly 12,000 records using `sample(n=12000, random_state=42)`. The dataset itself is not distributed in this repository.

## Technologies

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, joblib, Jupyter, Git, and GitHub.

## Workflow

```text
KDD Cup 99
     ↓
Data Loading
     ↓
12,000 Sample Selection
     ↓
Data Cleaning
     ↓
Encoding
     ↓
Feature Selection
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
Random Forest + SVC + Isolation Forest + One-Class SVM
     ↓
Hyperparameter Tuning
     ↓
Prediction
     ↓
Evaluation
     ↓
Normal / Malicious Traffic
```

Feature selection is provided as an optional `SelectKBest(mutual_info_classif)` helper; the default experiment retains all 41 input fields so that no feature information is discarded without evidence. Random Forest impurity importances show which transformed inputs were useful in the fitted forest.

## Data preprocessing and leakage control

The final `label` field is normalized (including stripping the KDD trailing period) and mapped to `target=0` for normal or `target=1` for every attack type. Numeric columns are safely coerced, invalid values become missing, and categorical values are normalized. Median/mode imputation and one-hot encoding are fitted inside each model pipeline, including separately within each cross-validation fold. The stratified 80/20 split is made before fitting. Numeric features and the full sparse feature matrix are scaled inside the SVM pipeline; Random Forest does not require scaling. `class_weight="balanced"` provides a simple training-only response to class imbalance.

## Models and tuning

- **Random Forest:** `n_estimators` controls the number of trees; `max_depth` limits tree depth and complexity; `random_state=42` makes randomized behavior reproducible. Grid search also checks minimum leaf size.
- **SVM:** an RBF-kernel `SVC` with probability estimates for ROC-AUC; grid search checks `C` and `gamma`.
- **Isolation Forest:** unsupervised detector with 200 trees and 30% contamination, matching the earlier KDD notebook's starting setting. Contamination directly affects its flagged anomaly share and should be checked against the desired false-alarm rate.
- **One-Class SVM:** unsupervised RBF detector with `nu=0.10`; its decision score is inverted for ROC-AUC so larger scores indicate more anomalous traffic.

Random Forest and SVC use three-fold `GridSearchCV` optimized for F1. F1 balances precision and recall; adjusting the objective or decision threshold can tune the precision-recall tradeoff and reduce false positives, but may increase missed attacks. The unsupervised detectors fit training features and their default thresholds are reported as-is.

## Evaluation and results

The program computes Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrices, and ROC curves from the held-out test set for all four methods. It writes the comparison table to `results/model_comparison.csv`, fitted pipelines to `models/`, and plots under `results/`. The old notebook's IoT dataset cells are outside this KDD project. Its test-set-derived autoencoder threshold was omitted because it makes test evaluation optimistic; an autoencoder can be added with a threshold selected using training-only validation data. No results are prefilled or claimed: metrics depend on running the experiment with the actual dataset and installed library versions.

## Install and run

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py --data-path data/kddcup.data.corrected
```

In Colab, upload the dataset to `/content/kddcup.data.corrected`, install with `!pip install -r requirements.txt`, then run `!python src/train.py`. The training command saves tuned pipeline files to `models/` and generated plots/CSV metrics to `results/`. Training includes cross-validation and can take several minutes.

## Streamlit app

The beginner-friendly frontend lets a new user upload the KDD file, preview the deterministic 12,000-row sample, train and compare the four models, inspect plots, and try a single-record prediction.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens in your browser. Upload the 42-column KDD Cup 99 connection dataset in the sidebar. The file named `KDD Cup 99.csv` that lists dataset descriptions is a catalog and cannot be used as the connection data. Training artifacts remain in the local `models/` and `results/` folders.

For a prediction in Python after training:

```python
from src.predict import predict_network_traffic

record = {
    "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "sf",
    "src_bytes": 181, "dst_bytes": 5450,
    # Remaining KDD features may be supplied; omitted fields are imputed.
}
print(predict_network_traffic(record, "models/random_forest.pkl"))
```

Output is `Normal Traffic` or `Malicious Traffic`. The saved artifact includes preprocessing as well as the estimator. Predictions require the same KDD feature schema and are not a substitute for validating a detector on current production traffic.

## Project structure

```text
network-intrusion-detection/
├── data/README.md
├── notebooks/intrusion_detection.ipynb
├── src/data_preprocessing.py
├── src/feature_engineering.py
├── src/train.py
├── src/evaluate.py
├── src/predict.py
├── models/                  # fitted joblib pipeline artifacts after training
├── results/                 # metrics and plots after training
├── requirements.txt
└── README.md
```

## Notebook

Open `notebooks/intrusion_detection.ipynb` in Jupyter or Colab. It documents the same project modules, inspects the sample, and invokes the shared training workflow. Make sure the dataset path is available before running.

## Future improvements

- Compare against newer intrusion datasets and test temporal/generalization splits.
- Report per-attack-category performance and calibration, and choose thresholds based on an explicit false-alarm budget.
- Explore cost-sensitive learning, explainability, drift monitoring, and inference latency.
- Add experiment tracking and a deployment interface after validating real operational data.

## GitHub upload

```bash
git init
git add README.md requirements.txt .gitignore data/README.md notebooks/ src/ results/.gitkeep
git commit -m "Add network intrusion detection project"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

Replace `USERNAME/REPOSITORY` with your GitHub repository. Dataset files and trained model binaries are ignored by default; publish them only if their licenses and your repository's size policy permit it.
#   N e t w o r k - A n a m o l y - D e t e c t i o n - S y s t e m  
 