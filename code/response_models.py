"""Organic nanomaterial classification with nested publication-grouped validation."""
import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


EXCLUDED = {"paper_id", "binary_labels", "multiclass_labels", "Tumor_Vol_Reduction_(%)",
            "Tumor Inhibition Rate (%)", "Targeting rate", "Name", "DOI", "doi",
            "sample_id", "Unnamed: 0"}
MODEL_NAMES = ["forest", "svm", "neighbors", "tree", "bayes", "boost", "mlp", "xgboost"]


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


def group_splits(X, y, groups, folds, seed):
    if folds < 2 or len(np.unique(groups)) < folds:
        raise ValueError("Need at least as many publications as folds, with folds >= 2.")
    splits = list(GroupKFold(folds, shuffle=True, random_state=seed).split(X, y, groups))
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("A publication occurs in both training and testing.")
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError("A fold has one class only; revise the prespecified group design.")
    return splits


def model_spec(name, seed):
    specs = {
        "forest": (RandomForestClassifier(random_state=seed, n_jobs=1),
                   {"n_estimators": [200], "min_samples_leaf": [1, 3]}),
        "svm": (SVC(probability=True, random_state=seed), {"C": [0.1, 1, 10]}),
        "neighbors": (KNeighborsClassifier(), {"n_neighbors": [3, 7]}),
        "tree": (DecisionTreeClassifier(random_state=seed), {"max_depth": [4, None]}),
        "bayes": (GaussianNB(), {"var_smoothing": [1e-9, 1e-7]}),
        "boost": (AdaBoostClassifier(random_state=seed), {"n_estimators": [50, 200]}),
        "mlp": (MLPClassifier(random_state=seed, max_iter=1500),
                {"hidden_layer_sizes": [(32,), (64, 32)]}),
    }
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except (ImportError, OSError, ValueError) as exc:
            raise RuntimeError("Install optional XGBoost and its runtime, or omit xgboost.") from exc
        return XGBClassifier(random_state=seed, n_jobs=1, eval_metric="logloss"), {"max_depth": [3, 6]}
    return specs[name]


def tune(name, X, y, splits, seed, jobs):
    estimator, grid = model_spec(name, seed)
    if name == "neighbors":
        minimum = min(len(train) for train, _ in splits)
        grid["n_neighbors"] = [k for k in grid["n_neighbors"] if k <= minimum]
        if not grid["n_neighbors"]:
            raise ValueError("Too few training rows for the nearest-neighbor grid.")
    pipe = Pipeline([("prepare", prepare(X)), ("model", estimator)])
    return GridSearchCV(pipe, {f"model__{k}": v for k, v in grid.items()},
                        cv=splits, scoring="roc_auc", n_jobs=jobs,
                        error_score="raise").fit(X, y)


def metrics(y, probability):
    predicted = (probability >= 0.5).astype(int)
    return {"roc_auc": roc_auc_score(y, probability),
            "average_precision": average_precision_score(y, probability),
            "balanced_accuracy": balanced_accuracy_score(y, predicted),
            "f1": f1_score(y, predicted, zero_division=0),
            "mcc": matthews_corrcoef(y, predicted), "kappa": cohen_kappa_score(y, predicted)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/models"))
    parser.add_argument("--label", default="binary_labels")
    parser.add_argument("--group", default="paper_id")
    parser.add_argument("--features", nargs="+")
    parser.add_argument("--drop", nargs="+", default=[])
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES[:6])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    frame, X = read_data(args.data, args.label, args.group, args.features, args.drop)
    labels = pd.to_numeric(frame[args.label], errors="coerce")
    if labels.isna().any() or set(labels.unique()) != {0, 1}:
        raise ValueError("Machine-learning labels must be exactly 0 and 1.")
    if args.group not in frame:
        raise ValueError(f"Missing publication column: {args.group}")
    groups = frame[args.group].str.strip()
    if groups.isna().any() or groups.eq("").any():
        raise ValueError("Every formulation needs a publication ID.")
    groups, y = groups.to_numpy(dtype=str), labels.to_numpy(dtype=int)
    if args.repeats < 1 or len(set(args.models)) != len(args.models):
        raise ValueError("Need positive repeats and unique model names.")
    plan = []
    for repeat in range(1, args.repeats + 1):
        for fold, (train, test) in enumerate(group_splits(X, y, groups, args.folds, args.seed+repeat), 1):
            inner = group_splits(X.iloc[train], y[train], groups[train], args.inner_folds,
                                 args.seed + 100*repeat + fold)
            plan.append((repeat, fold, train, test, inner))
    final_splits = group_splits(X, y, groups, args.inner_folds, args.seed)
    for name in args.models:
        model_spec(name, args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    records, results = [], []
    for repeat, fold, train, test, inner in plan:
        for name in args.models:
            print(f"Repeat {repeat}, fold {fold}: {name}", flush=True)
            fitted = tune(name, X.iloc[train], y[train], inner, args.seed, args.jobs)
            probability = fitted.predict_proba(X.iloc[test])[:, 1]
            results.append({"model": name, "repeat": repeat, "fold": fold,
                            **metrics(y[test], probability)})
            records.extend({"model": name, "repeat": repeat, "fold": fold,
                            "row_id": int(i), "paper_id": groups[i], "observed": int(y[i]),
                            "probability": float(p)} for i, p in zip(test, probability))
        pd.DataFrame(records).to_csv(args.out / "heldout_predictions.csv", index=False)
        pd.DataFrame(results).to_csv(args.out / "fold_metrics.csv", index=False)
    summary = pd.DataFrame(results).groupby("model")[list(metrics(y, np.full(len(y), 0.5)))].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.sort_values("roc_auc_mean", ascending=False)
    summary.to_csv(args.out / "model_summary.csv")
    winner = summary.index[0]
    final = tune(winner, X, y, final_splits, args.seed, args.jobs)
    joblib.dump(final.best_estimator_, args.out / "selected_model.joblib", compress=3)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    info = {"settings": settings, "features": X.columns.tolist(), "selected_model": winner,
            "parameters": final.best_params_, "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib"]}}
    (args.out / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Selected {winner}. Results saved to {args.out}")


if __name__ == "__main__":
    main()
