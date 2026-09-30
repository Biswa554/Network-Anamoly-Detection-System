# Network Intrusion Detection Using Machine Learning

A Python project that compares supervised classifiers and unsupervised anomaly detectors on the KDD Cup 99 connection dataset. It includes a Streamlit interface for uploading data, exploring a sample, training models, reviewing metrics, and trying a single-connection prediction.

> **Scope:** This is an educational benchmark project. KDD Cup 99 is an old, highly duplicated dataset; the reported scores should not be interpreted as expected performance on modern network traffic.

## What it does

- Loads the 42-column KDD Cup 99 connection data, including `.gz` files.
- Selects a repeatable 12,000-row sample (`random_state=42`).
- Cleans labels and features, then maps `normal` to 0 and all attack labels to 1.
- Compares Random Forest and SVM classifiers with Isolation Forest and One-Class SVM detectors.
- Tunes the supervised classifiers using three-fold cross-validation with F1 scoring.
- Reports accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC curves, and feature importance.
- Saves fitted model pipelines and result files locally.

## Dataset

Use the **KDD Cup 99 corrected connection-level dataset** (`kddcup.data.corrected` or its gzip-compressed version). It must contain the KDD connection records with 41 feature columns and one label column. The dataset is not included in this repository.

A catalog file named `KDD Cup 99.csv` is not the connection dataset and cannot be used for training.

Place the dataset in `data/`, or pass its full path to the training command. This project has been run with `data/kddcup.data.gz`.

## Windows setup

Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train and evaluate

With the dataset at `data\kddcup.data.gz`:

```powershell
python src\train.py --data-path data\kddcup.data.gz
```

Or specify a different file path:

```powershell
python src\train.py --data-path "C:\path\to\kddcup.data.corrected"
```

Training runs cross-validation and can take a few minutes. Successful runs save fitted pipelines in `models/` and metrics and charts in `results/`. Dataset, model, and generated result files are excluded from Git by `.gitignore`.

## Launch the Streamlit app

```powershell
python -m streamlit run app.py
```

Streamlit opens the app in your browser. Use the sidebar to upload the 42-column KDD connection dataset, then:

1. Review the reproducible sample and class balance in **Data overview**.
2. Select **Train all four models**.
3. Compare held-out metrics and charts in **Model results**.
4. Use **Try a prediction** to classify a sample connection after training.

The app has a dark theme configured in `.streamlit/config.toml`. The uploaded file is processed locally by the app; it is not sent to an external service.

## Example results

One run using the fixed 12,000-row sample and an 80/20 stratified random split produced:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9992 | 1.0000 | 0.9990 | 0.9995 | 1.0000 |
| SVM | 0.9988 | 0.9990 | 0.9995 | 0.9992 | 1.0000 |
| Isolation Forest | 0.2596 | 0.6035 | 0.2340 | 0.3372 | 0.1665 |
| One-Class SVM | 0.1650 | 0.3435 | 0.0409 | 0.0731 | 0.6806 |

These are observed results from one sampled experiment, not a guarantee. KDD Cup 99 contains repeated records, and a random split can place very similar connections in both training and test sets. This can make supervised scores look unusually high. The unsupervised detectors also use preset contamination/`nu` values, so their performance is sensitive to threshold choice. Validate with deduplicated or time-based splits and newer data before drawing conclusions about generalization.

## Outputs

After training, generated files include:

- `models/random_forest.pkl`, `models/svm.pkl`
- `models/isolation_forest.pkl`, `models/one_class_svm.pkl`
- `results/model_comparison.csv`
- `results/roc_curve.png` and model confusion-matrix images
- `results/class_distribution.png` and `results/feature_importance.png`
- `results/feature_importance.csv`

These outputs are generated locally and are not committed by default.

## Project layout

```text
.
├── .streamlit/config.toml       # Streamlit theme
├── app.py                       # Browser interface
├── data/README.md               # Dataset placement notes
├── notebooks/intrusion_detection.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Tech stack

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, joblib, and Streamlit.

## Limitations

- KDD Cup 99 is historical and does not represent current network traffic or modern attack behavior.
- The project combines all attack categories into a binary `attack` class.
- The 12,000-row sample and random split are useful for a reproducible demo, but are not a deployment validation strategy.
- Unsupervised detector thresholds need to be calibrated against an operational false-alarm target.
- The prediction form fills unspecified features through the pipeline's imputers; useful real predictions require representative, complete connection data.

## License and dataset terms

Check the licenses and usage terms for the code and dataset before redistributing them. The dataset is intentionally not bundled here.
