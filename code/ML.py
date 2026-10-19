"""Publication-grouped classification of organic nanomaterial response.

Run from the repository root:
    python code/response_models.py
    python code/response_models.py --data datasets/ml_dataset.csv --out results/group_cv

Inner and outer validation keep each publication intact. Model families,
parameter grids and random-seed formulas follow the full research workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# 1. User settings
# ============================================================

SEED = 123
MAIN_REPEATS = 5
MAIN_SPLITS = 5
INNER_SPLITS = 5
N_JOBS = -1

# Resolve default paths relative to this script, independent of the working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "code" else SCRIPT_DIR
DATA_PATH = PROJECT_DIR / "datasets" / "ml_dataset.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "group_cv"

GROUP_COLUMN = "paper_id"
OUTCOME_COLUMN = "Tumor_Vol_Reduction_(%)"
TARGET_COLUMN = "binary_labels"


# Set to a subset such as ["RF", "AdaBoost"] for a quicker comparison.
MODELS_TO_RUN = [
    "NB",
    "kNN",
    "SVC",
    "DT",
    "RF",
    "XGBoost",
    "AdaBoost",
    "MLP",
]

# MLP's original grid has 540 combinations and can be extremely slow.
# This switch does not change the grid; it only makes the cost explicit.
ALLOW_VERY_LARGE_MLP_GRID = True

# After nested evaluation, tune the winning algorithm once more using grouped
# CV on all development data, fit it on all rows, and save it.
TRAIN_AND_SAVE_FINAL_MODEL = True


MODEL_NAMES = ["NB", "kNN", "SVC", "DT", "RF", "XGBoost", "AdaBoost", "MLP"]


def model_name(value: str) -> str:
    """Accept standard model abbreviations and the previous descriptive aliases."""
    aliases = {
        "forest": "RF", "svm": "SVC", "neighbors": "kNN", "tree": "DT",
        "bayes": "NB", "boost": "AdaBoost", "mlp": "MLP", "xgboost": "XGBoost",
    }
    canonical = {name.lower(): name for name in MODEL_NAMES}
    return aliases.get(value.lower(), canonical.get(value.lower(), value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Input CSV")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH, help="New results directory")
    parser.add_argument("--group", default=GROUP_COLUMN, help="Publication identifier column")
    parser.add_argument("--label", "--target", dest="target", default=TARGET_COLUMN,
                        help="Binary endpoint column")
    parser.add_argument("--outcome", default=OUTCOME_COLUMN, help="Continuous outcome column to exclude")
    parser.add_argument("--models", nargs="+", type=model_name, choices=MODEL_NAMES,
                        default=MODELS_TO_RUN, help="Classifiers to evaluate")
    parser.add_argument("--repeats", type=int, default=MAIN_REPEATS)
    parser.add_argument("--outer-folds", "--folds", dest="outer_folds", type=int, default=MAIN_SPLITS)
    parser.add_argument("--inner-folds", type=int, default=INNER_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--jobs", type=int, default=N_JOBS, help="Parallel grid-search workers")
    parser.add_argument("--features", nargs="+", help="Explicit predictor column names")
    parser.add_argument("--drop", nargs="+", default=[], help="Additional columns to exclude")
    parser.add_argument("--no-final-model", action="store_true", help="Evaluate without saving a final refit")
    args = parser.parse_args()
    if args.repeats < 1 or min(args.outer_folds, args.inner_folds) < 2:
        parser.error("Use positive repeats and at least two folds in each validation layer.")
    if args.jobs == 0 or len(set(args.models)) != len(args.models):
        parser.error("--jobs must be nonzero and model names must be unique.")
    if len({args.group, args.target, args.outcome}) != 3:
        parser.error("Group, target and outcome columns must be different.")
    args.data = args.data.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    return args


# ============================================================
# 2. Utility functions
# ============================================================

def clean_feature_names(columns: pd.Index) -> pd.Index:
    """Apply the same feature-name cleaning to all input data."""
    return (
        pd.Index(columns)
        .astype(str)
        .str.strip()
        .str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_", regex=False)
        .str.replace(">", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )


def make_one_hot_encoder() -> OneHotEncoder:
    """Support both new and old scikit-learn OneHotEncoder APIs."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Create a fresh preprocessing specification."""
    transformers = []

    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("num", numeric_transformer, numeric_features)
        )

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(
            ("cat", categorical_transformer, categorical_features)
        )

    if not transformers:
        raise ValueError("No usable numeric or categorical features found.")

    return ColumnTransformer(transformers=transformers)


def make_models_and_grids(
    preprocessor: ColumnTransformer,
    seed: int = SEED,
    selected_models: list[str] | None = None,
):
    """
    Build the original eight algorithms and original parameter grids.

    Estimator-level n_jobs is kept at 1 for RF/XGBoost because GridSearchCV
    already parallelizes candidate fits. This avoids nested CPU oversubscription.
    """
    models = {
        "NB": Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                ("model", GaussianNB()),
            ]
        ),
        "kNN": Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                ("model", KNeighborsClassifier()),
            ]
        ),
        "SVC": Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                ("model", SVC(probability=True, random_state=seed)),
            ]
        ),
        "DT": Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    DecisionTreeClassifier(random_state=seed),
                ),
            ]
        ),
        "RF": Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "AdaBoost": Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    AdaBoostClassifier(random_state=seed),
                ),
            ]
        ),
        "MLP": Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    MLPClassifier(random_state=seed),
                ),
            ]
        ),
    }

    param_grids = {
        "NB": {},
        "kNN": {
            "model__n_neighbors": [3, 4, 5],
        },
        "SVC": {
            "model__kernel": ["linear", "poly", "rbf", "sigmoid"],
        },
        "DT": {
            "model__criterion": ["gini", "entropy", "log_loss"],
        },
        "RF": {
            "model__n_estimators": [100, 500, 1000],
        },
        "XGBoost": {
            "model__n_estimators": [100, 500, 1000],
        },
        "AdaBoost": {
            "model__n_estimators": [50, 100, 500],
        },
        "MLP": {
            "model__hidden_layer_sizes": [
                (100,),
                (100, 100),
                (50, 50, 50),
                (50, 100, 50),
                (100, 100, 100),
            ],
            "model__activation": [
                "identity",
                "logistic",
                "tanh",
                "relu",
            ],
            "model__solver": ["lbfgs", "sgd", "adam"],
            "model__alpha": [0.0001, 0.0005, 0.001],
            "model__max_iter": [200, 500, 1000],
        },
    }

    if selected_models is None or "XGBoost" in selected_models:
        try:
            from xgboost import XGBClassifier
        except (ImportError, OSError, ValueError) as exc:
            raise RuntimeError(
                "XGBoost is unavailable. Install xgboost and its platform runtime, "
                "or select other models with --models."
            ) from exc
        models["XGBoost"] = Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    XGBClassifier(
                        random_state=seed,
                        eval_metric="logloss",
                        n_jobs=1,
                    ),
                ),
            ]
        )

    return models, param_grids


def randomized_group_codes(
    groups: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """
    Compatibility fallback for scikit-learn versions whose GroupKFold has no
    shuffle/random_state parameters.

    Re-encoding group identities changes deterministic tie-breaking while
    preserving publication integrity. It is less flexible than native shuffled
    GroupKFold but is compatible with older versions.
    """
    rng = np.random.default_rng(random_state)
    unique_groups = pd.unique(groups).copy()
    rng.shuffle(unique_groups)
    mapping = {
        group: code
        for code, group in enumerate(unique_groups)
    }
    return np.asarray([mapping[group] for group in groups])


def make_group_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create shuffled GroupKFold splits when supported, with an older-version
    compatibility fallback. This deliberately does not stratify by label.
    """
    n_groups = pd.Series(groups).nunique()
    if n_groups < n_splits:
        raise ValueError(
            f"Need at least {n_splits} publications, found {n_groups}."
        )

    try:
        cv = GroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(cv.split(X, y, groups=groups))
    except TypeError:
        encoded_groups = randomized_group_codes(
            groups,
            random_state=random_state,
        )
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, y, groups=encoded_groups))

    return splits


def validate_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    groups: np.ndarray,
    context: str,
) -> None:
    """Check publication isolation and binary-class availability."""
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups.intersection(test_groups)

        if overlap:
            preview = sorted(map(str, overlap))[:10]
            raise RuntimeError(
                f"{context}, fold {fold}: publication leakage: {preview}"
            )

        train_classes = np.unique(y[train_idx])
        test_classes = np.unique(y[test_idx])

        if len(train_classes) < 2:
            raise ValueError(
                f"{context}, fold {fold}: training data contains only "
                f"class {train_classes.tolist()}."
            )

        if len(test_classes) < 2:
            raise ValueError(
                f"{context}, fold {fold}: validation/test data contains only "
                f"class {test_classes.tolist()}, so ROC-AUC is undefined. "
                "Reduce n_splits, change the random seed, or use "
                "StratifiedGroupKFold later."
            )


def calculate_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Calculate the six metrics used by the original pipeline."""
    return {
        "roc_auc": roc_auc_score(y_true, probability),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, prediction
        ),
        "f1_score": f1_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "mcc_score": matthews_corrcoef(y_true, prediction),
        "pr_auc_score": average_precision_score(
            y_true, probability
        ),
        "cohen_kappa": cohen_kappa_score(
            y_true, prediction
        ),
    }


def serialize_params(params: dict) -> str:
    """Store parameter dictionaries in a stable JSON string."""
    return json.dumps(params, ensure_ascii=False, sort_keys=True)




# Shared input handling used by the companion LDA script.
EXCLUDED = {"paper_id", "binary_labels", "multiclass_labels", "Tumor_Vol_Reduction_(%)",
            "Tumor Inhibition Rate (%)", "Targeting rate", "Name", "DOI", "doi",
            "sample_id", "Unnamed: 0"}


def read_data(path, label, group="paper_id", features=None, drop=()):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f), [])
    names = [c.strip() for c in header]
    if not names or "" in names or len(set(names)) != len(names):
        raise ValueError("CSV headers must be nonempty and unique.")
    group_header = next((c for c in header if c.strip() == group), group)
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={group_header: "string"})
    frame.columns = names
    if frame.empty or label not in frame or frame[label].isna().any():
        raise ValueError(f"Need nonempty data and complete labels in {label!r}.")
    blocked = EXCLUDED | {label, group} | set(drop)
    features = features if features is not None else [c for c in frame if c not in blocked]
    if not features or len(set(features)) != len(features) or blocked.intersection(features):
        raise ValueError("Choose unique predictors excluding IDs, labels and outcome columns.")
    X = frame.loc[:, features].copy()
    for c in X:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(float)
            if np.isinf(X[c].to_numpy()).any():
                raise ValueError(f"Infinite predictor values in {c!r}.")
        else:
            X[c] = X[c].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
    return frame, X


def prepare(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X if c not in numeric]
    parts = []
    if numeric:
        parts.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler())]), numeric))
    if categorical:
        parts.append(("category", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__",
                                     keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    return ColumnTransformer(parts, sparse_threshold=0)

# ============================================================
# 3. Data loading and validation
# ============================================================

def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Dataset not found:\n{args.data}")

    with args.data.open(encoding="utf-8-sig", newline="") as stream:
        header = [name.strip() for name in next(csv.reader(stream), [])]
    if not header or "" in header or len(set(header)) != len(header):
        raise ValueError("Input CSV headers must be nonempty and unique after trimming.")
    df = pd.read_csv(args.data, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()

    required_columns = [
        args.group,
        args.outcome,
        args.target,
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[args.group].isna().any():
        raise ValueError(
            f"{args.group!r} contains missing publication IDs."
        )

    groups = (
        df[args.group]
        .astype(str)
        .str.strip()
        .to_numpy()
    )
    if np.any(groups == ""):
        raise ValueError(
            f"{args.group!r} contains empty publication IDs."
        )

    y_numeric = pd.to_numeric(
        df[args.target],
        errors="coerce",
    )
    if y_numeric.isna().any():
        raise ValueError(
            f"{args.target!r} contains missing/non-numeric labels."
        )
    if not y_numeric.isin([0, 1]).all():
        raise ValueError(f"{args.target!r} must contain only the exact values 0 and 1.")
    y = y_numeric.astype(int).to_numpy()

    unique_labels = sorted(np.unique(y).tolist())
    if unique_labels != [0, 1]:
        raise ValueError(
            f"{args.target!r} must contain exactly 0 and 1; "
            f"found {unique_labels}."
        )

    excluded = {args.group, args.outcome, args.target, "multiclass_labels", *args.drop}
    if args.features is not None:
        if len(set(args.features)) != len(args.features) or excluded.intersection(args.features):
            raise ValueError("Predictors must be unique and exclude IDs, labels and outcomes.")
        X = df.loc[:, args.features].copy()
    else:
        X = df.drop(columns=[column for column in excluded if column in df]).copy()
    X.columns = clean_feature_names(X.columns)

    duplicated_columns = (
        X.columns[X.columns.duplicated()].unique().tolist()
    )
    if duplicated_columns:
        raise ValueError(
            "Feature-name cleaning created duplicate columns: "
            f"{duplicated_columns}"
        )

    if X.shape[1] == 0:
        raise ValueError("No predictor columns remain.")

    numeric_features = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    categorical_features = [
        column
        for column in X.columns
        if column not in numeric_features
    ]

    print("=" * 72)
    print("Publication-level nested GroupKFold ML pipeline")
    print("=" * 72)
    print(f"Rows/formulations: {len(df)}")
    print(f"Publications: {pd.Series(groups).nunique()}")
    print(f"Positive samples: {int((y == 1).sum())}")
    print(f"Negative samples: {int((y == 0).sum())}")
    print(f"Predictors: {X.shape[1]}")
    print(f"Numeric predictors: {len(numeric_features)}")
    print(f"Categorical predictors: {len(categorical_features)}")
    print(f"Output directory: {args.out}")

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )
    models, param_grids = make_models_and_grids(
        preprocessor, seed=args.seed, selected_models=args.models
    )

    unknown_models = [
        name
        for name in args.models
        if name not in models
    ]
    if unknown_models:
        raise ValueError(
            f"Unknown model names: {unknown_models}"
        )

    mlp_candidates = 5 * 4 * 3 * 3 * 3
    if "MLP" in args.models:
        print(
            f"Warning: the MLP grid contains {mlp_candidates} "
            "hyperparameter combinations."
        )
        if not ALLOW_VERY_LARGE_MLP_GRID:
            raise RuntimeError(
                "Set ALLOW_VERY_LARGE_MLP_GRID=True to run the "
                "original MLP grid, or omit MLP from --models."
            )

    args.out.mkdir(parents=True, exist_ok=False)

    # --------------------------------------------------------
    # Precompute outer folds once so all models use identical data.
    # --------------------------------------------------------
    outer_splits_by_repeat = {}
    assignment_entries = []

    for repeat in range(args.repeats):
        splits = make_group_splits(
            X=X,
            y=y,
            groups=groups,
            n_splits=args.outer_folds,
            random_state=args.seed + repeat,
        )
        validate_splits(
            splits,
            y,
            groups,
            context=f"Outer CV repeat {repeat + 1}",
        )
        outer_splits_by_repeat[repeat] = splits

        for fold, (_, test_idx) in enumerate(splits, start=1):
            for index in test_idx:
                assignment_entries.append(
                    {
                        "Repeat": repeat + 1,
                        "Fold": fold,
                        "Sample_index": int(index),
                        "CSV_row_number": int(index) + 2,
                        "Name": groups[index],
                        "True_label": int(y[index]),
                    }
                )

    pd.DataFrame(assignment_entries).to_csv(
        os.path.join(args.out, "outer_cv_assignments.csv"),
        index=False,
    )

    # --------------------------------------------------------
    # Nested CV model survey.
    # --------------------------------------------------------
    all_fold_results = []
    all_best_params = []
    model_summaries = []

    for model_name in args.models:
        print("\n" + "-" * 72)
        print(f"Running model: {model_name}")
        print("-" * 72)

        model_template = models[model_name]
        model_fold_results = []
        model_oof_entries = []
        model_trajectory_frames = []

        for repeat in range(args.repeats):
            outer_splits = outer_splits_by_repeat[repeat]

            for fold_id, (train_idx, test_idx) in enumerate(
                outer_splits,
                start=1,
            ):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]
                group_train = groups[train_idx]

                inner_splits = make_group_splits(
                    X=X_train,
                    y=y_train,
                    groups=group_train,
                    n_splits=args.inner_folds,
                    random_state=(
                        args.seed
                        + 10_000
                        + repeat * args.outer_folds
                        + fold_id
                    ),
                )
                validate_splits(
                    inner_splits,
                    y_train,
                    group_train,
                    context=(
                        f"{model_name}, repeat {repeat + 1}, "
                        f"outer fold {fold_id}, inner CV"
                    ),
                )

                grid = GridSearchCV(
                    estimator=clone(model_template),
                    param_grid=param_grids[model_name],
                    scoring="roc_auc",
                    cv=inner_splits,
                    n_jobs=args.jobs,
                    refit=True,
                    return_train_score=False,
                    error_score="raise",
                    verbose=0,
                )
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_

                probability = best_model.predict_proba(X_test)[:, 1]
                prediction = (probability >= 0.5).astype(int)
                metrics = calculate_metrics(
                    y_test,
                    probability,
                    prediction,
                )

                fold_record = {
                    "Model": model_name,
                    "Repeat": repeat + 1,
                    "Fold": fold_id,
                    "Train_samples": len(train_idx),
                    "Test_samples": len(test_idx),
                    "Train_publications": pd.Series(
                        groups[train_idx]
                    ).nunique(),
                    "Test_publications": pd.Series(
                        groups[test_idx]
                    ).nunique(),
                    "Train_positive_rate": float(y_train.mean()),
                    "Test_positive_rate": float(y_test.mean()),
                    **metrics,
                }
                model_fold_results.append(fold_record)
                all_fold_results.append(fold_record)

                parameter_record = {
                    "Model": model_name,
                    "Repeat": repeat + 1,
                    "Fold": fold_id,
                    "Best_params": serialize_params(
                        grid.best_params_
                    ),
                    "InnerCV_score": float(grid.best_score_),
                }
                all_best_params.append(parameter_record)

                trajectory = pd.DataFrame(grid.cv_results_)
                trajectory.insert(0, "Fold", fold_id)
                trajectory.insert(0, "Repeat", repeat + 1)
                trajectory.insert(0, "Model", model_name)
                model_trajectory_frames.append(trajectory)

                for local_position, sample_index in enumerate(test_idx):
                    model_oof_entries.append(
                        {
                            "Model": model_name,
                            "Repeat": repeat + 1,
                            "Fold": fold_id,
                            "Sample_index": int(sample_index),
                            "CSV_row_number": int(sample_index) + 2,
                            "Name": groups[sample_index],
                            "True_label": int(y_test[local_position]),
                            "Predicted_probability": float(
                                probability[local_position]
                            ),
                            "Predicted_label": int(
                                prediction[local_position]
                            ),
                        }
                    )

                print(
                    f"Repeat {repeat + 1}/{args.repeats}, "
                    f"fold {fold_id}/{args.outer_folds}: "
                    f"AUC={metrics['roc_auc']:.4f}, "
                    f"best={grid.best_params_}"
                )

        fold_df = pd.DataFrame(model_fold_results)
        oof_df = pd.DataFrame(model_oof_entries)
        trajectory_df = pd.concat(
            model_trajectory_frames,
            ignore_index=True,
        )

        fold_df.to_csv(
            os.path.join(
                args.out,
                f"{model_name}_nested_groupCV_results.csv",
            ),
            index=False,
        )
        trajectory_df.to_csv(
            os.path.join(
                args.out,
                f"{model_name}_traj.csv",
            ),
            index=False,
        )

        # Save traceable out-of-fold predictions in long format.
        oof_df.to_csv(
            os.path.join(
                args.out,
                f"{model_name}_output.csv",
            ),
            index=False,
        )

        model_summaries.append(
            {
                "Model": model_name,
                "roc_auc_mean": fold_df["roc_auc"].mean(),
                "roc_auc_std": fold_df["roc_auc"].std(ddof=1),
                "balanced_accuracy_mean": fold_df[
                    "balanced_accuracy"
                ].mean(),
                "balanced_accuracy_std": fold_df[
                    "balanced_accuracy"
                ].std(ddof=1),
                "f1_mean": fold_df["f1_score"].mean(),
                "f1_std": fold_df["f1_score"].std(ddof=1),
                "mcc_mean": fold_df["mcc_score"].mean(),
                "mcc_std": fold_df["mcc_score"].std(ddof=1),
                "pr_auc_mean": fold_df["pr_auc_score"].mean(),
                "pr_auc_std": fold_df["pr_auc_score"].std(ddof=1),
                "cohen_kappa_mean": fold_df["cohen_kappa"].mean(),
                "cohen_kappa_std": fold_df["cohen_kappa"].std(
                    ddof=1
                ),
            }
        )

    # --------------------------------------------------------
    # Global result files.
    # --------------------------------------------------------
    all_fold_df = pd.DataFrame(all_fold_results)
    all_params_df = pd.DataFrame(all_best_params)
    summary_df = (
        pd.DataFrame(model_summaries)
        .sort_values("roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )

    all_fold_df.to_csv(
        os.path.join(args.out, "all_fold_results.csv"),
        index=False,
    )
    all_params_df.to_csv(
        os.path.join(
            args.out,
            "best_parameters_all_models.csv",
        ),
        index=False,
    )
    summary_df.to_csv(
        os.path.join(
            args.out,
            "nested_groupCV_model_summary.csv",
        ),
        index=False,
    )

    print("\n" + "=" * 72)
    print("Final nested-CV ranking by mean outer-fold ROC-AUC")
    print("=" * 72)
    print(summary_df.round(4).to_string(index=False))

    best_model_name = str(summary_df.iloc[0]["Model"])
    print(f"\nWinning algorithm: {best_model_name}")

    # --------------------------------------------------------
    # Tune and fit one deployable final model on all development data.
    # This model must be evaluated on genuinely external publications.
    # --------------------------------------------------------
    if TRAIN_AND_SAVE_FINAL_MODEL and not args.no_final_model:
        print(
            "\nTuning the winning algorithm with grouped CV on all "
            "development data..."
        )

        final_splits = make_group_splits(
            X=X,
            y=y,
            groups=groups,
            n_splits=args.outer_folds,
            random_state=args.seed + 999_999,
        )
        validate_splits(
            final_splits,
            y,
            groups,
            context="Final grouped hyperparameter search",
        )

        final_grid = GridSearchCV(
            estimator=clone(models[best_model_name]),
            param_grid=param_grids[best_model_name],
            scoring="roc_auc",
            cv=final_splits,
            n_jobs=args.jobs,
            refit=True,
            return_train_score=False,
            error_score="raise",
            verbose=0,
        )
        final_grid.fit(X, y)
        final_model = final_grid.best_estimator_

        final_cv_results = pd.DataFrame(
            final_grid.cv_results_
        )
        final_cv_results.to_csv(
            os.path.join(
                args.out,
                f"{best_model_name}_final_groupCV_search.csv",
            ),
            index=False,
        )

        artifact = {
            "pipeline": final_model,
            "model_name": best_model_name,
            "best_params": final_grid.best_params_,
            "best_group_cv_auc": float(final_grid.best_score_),
            "feature_columns": X.columns.tolist(),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "group_column": args.group,
            "outcome_column": args.outcome,
            "target_column": args.target,
            "classification_threshold": 0.5,
            "random_seed": args.seed,
            "training_rows": len(X),
            "training_publications": int(
                pd.Series(groups).nunique()
            ),
        }

        model_path = os.path.join(
            args.out,
            f"{best_model_name}_final_model.joblib",
        )
        joblib.dump(artifact, model_path)

        print(f"Final best parameters: {final_grid.best_params_}")
        print(
            f"Final grouped-CV AUC: {final_grid.best_score_:.6f}"
        )
        print(f"Saved final model:\n{model_path}")

    print("\nAll analysis completed successfully.")
    print(f"Results saved to:\n{args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nAnalysis failed:", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise
