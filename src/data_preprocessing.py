"""Load and clean the KDD Cup 99 corrected connection records."""
from pathlib import Path

import pandas as pd


KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label",
]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
NUMERIC_COLUMNS = [c for c in KDD_COLUMNS if c not in CATEGORICAL_COLUMNS + ["label"]]


def load_sampled_data(path: str | Path = "/content/kddcup.data.corrected") -> pd.DataFrame:
    """Load all records, then take the required deterministic 12,000-row sample."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(
            f"KDD dataset not found: {source}. Download kddcup.data.corrected and "
            "pass its path with --data-path (Colab default: /content/kddcup.data.corrected)."
        )
    # The original corrected file has no header. Some downloaded CSV copies add
    # the standard KDD field names; detect and remove that row while preserving
    # the exact header=None loading behavior for the original dataset.
    df = pd.read_csv(source, header=None)
    if df.shape[1] != len(KDD_COLUMNS):
        raise ValueError(
            f"Expected {len(KDD_COLUMNS)} KDD columns, found {df.shape[1]}. "
            "Check that this is the corrected connection-level KDD Cup 99 CSV, "
            "not a version with an extra index column or a different dataset."
        )
    first_row = [str(value).strip().lower() for value in df.iloc[0].tolist()]
    if first_row[0] == "duration" and first_row[-1] in {"label", "class", "attack"}:
        df = df.iloc[1:].reset_index(drop=True)
    df.columns = KDD_COLUMNS
    if len(df) < 12_000:
        raise ValueError(f"Expected at least 12,000 rows for the requested sample; found {len(df)}.")
    return df.sample(n=12_000, random_state=42).reset_index(drop=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize labels and coerce invalid feature values to missing values."""
    cleaned = df.copy()
    cleaned["label"] = cleaned["label"].astype("string").str.strip().str.rstrip(".").str.lower()
    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].astype("string").str.strip().str.lower()
    # KDD Cup 99 uses the label "normal." for normal connections; all other labels are attacks.
    cleaned["target"] = (cleaned["label"] != "normal").astype("int8")
    cleaned.drop(columns="label", inplace=True)
    return cleaned


def describe_data(df: pd.DataFrame) -> None:
    """Print useful first-pass data diagnostics."""
    print("Shape:", df.shape)
    print("\nFirst rows:\n", df.head())
    print("\nData types:\n", df.dtypes)
    print("\nStatistics:\n", df.describe(include="all").transpose())
    print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False).head(15))
    print("\nTarget counts (0=normal, 1=attack):\n", df["target"].value_counts())
