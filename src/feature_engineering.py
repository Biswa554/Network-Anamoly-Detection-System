"""Leakage-safe feature transformations shared by model pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # Support both `python src/train.py` and `import src.feature_engineering`.
    from .data_preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
except ImportError:
    from data_preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def make_preprocessor(scale: bool = False) -> ColumnTransformer:
    """Impute and encode within CV folds; optionally scale the complete feature matrix."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformed = ColumnTransformer(
        [("numeric", numeric, NUMERIC_COLUMNS), ("categorical", categorical, CATEGORICAL_COLUMNS)],
        remainder="drop",
    )
    if scale:
        # SVM needs all columns scaled; with_mean=False supports sparse one-hot output.
        from sklearn.preprocessing import StandardScaler as Scaler
        return Pipeline([("columns", transformed), ("scale_all", Scaler(with_mean=False))])
    return transformed


def make_feature_selector(k: int = 30) -> SelectKBest:
    """Optional mutual-information selection; fit only inside a training pipeline."""
    return SelectKBest(score_func=mutual_info_classif, k=k)
