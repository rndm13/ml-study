#!/usr/bin/env python3
"""
Lab 3 — Metric methods and Support Vector Machines
Hill-Valley binary classification.

Creates a single report-ready JSON, detailed CSV tables, and automatically
generated figures. The script keeps the individual test set untouched until
model selection is complete.

Input:
  CSV/TXT/DATA file with numeric signal features and a binary `class` column.
  If the file has no header, pass --no-header; the last column is the target.

Example:
  python hill_valley_lab3.py --data Hill-Valley.csv --target class --output-dir hill_valley_lab3_output

When official training and test files are available:
  python hill_valley_lab3.py --data Hill_Valley_Training.data --test-data Hill_Valley_Testing.data --no-header --target class

The script also runs the controlled make_moons geometry experiment from the
lab instructions, plus the noise-feature/dimensionality experiment on the
Hill-Valley training split.

Dependencies: numpy, pandas, matplotlib, scikit-learn, joblib.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.datasets import make_moons
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


SCORING = {
    "macro_f1": "f1_macro",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
}
SCALERS = ("none", "standard", "minmax")
STABILITY_NOISE_COUNTS = [0, 64, 256, 512, 1024]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Lab 3 Hill-Valley experiments and export report data to JSON."
    )
    parser.add_argument("--data", required=True, help="Path to the individual dataset file.")
    parser.add_argument("--target", default="class", help="Target/label column (default: class).")
    parser.add_argument("--test-data", default=None, help="Optional separate official test file.")
    parser.add_argument("--no-header", action="store_true", help="Read input as headerless; last column is target.")
    parser.add_argument("--sep", default=None, help="Optional delimiter, e.g. ',' or ';'. Default: detect automatically.")
    parser.add_argument("--features", default=None, help="Optional comma-separated predictor column list.")
    parser.add_argument("--output-dir", default="hill_valley_lab3_output", help="Output directory.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Holdout fraction if --test-data is not supplied.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed for split, CV and controlled experiments.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified CV folds.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel CV/grid-search jobs; use 1 if memory is limited.")
    parser.add_argument("--noise-counts", default="0,64,256,512,1024", help="Noise feature counts for dimensionality experiment.")
    parser.add_argument("--max-oof-disagreements", type=int, default=100, help="Maximum disagreement cases with full neighbor details in JSON.")
    parser.add_argument("--mesh-resolution", type=int, default=180, help="Resolution per axis for make_moons boundary plots.")
    parser.add_argument("--positive-label", default=None, help="Explicit positive label for binary metrics; default auto-detects Hill or numeric class 1.")
    return parser.parse_args()


def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return json_safe(obj.item())
        except Exception:
            pass
    return str(obj)


def read_dataset(path: Path, target: str, no_header: bool, sep: Optional[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    delimiter = sep if sep is not None else None
    header = None if no_header else "infer"
    try:
        df = pd.read_csv(path, sep=delimiter, engine="python", header=header)
    except Exception as exc:
        raise ValueError(f"Could not read dataset {path}: {exc}") from exc
    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")
    if no_header:
        if df.shape[1] < 2:
            raise ValueError("Headerless input must contain predictor columns and a final target column.")
        columns = [f"point_{i + 1:03d}" for i in range(df.shape[1] - 1)] + [target]
        df.columns = columns
    else:
        df.columns = [str(c).strip() for c in df.columns]
    if target not in df.columns:
        raise ValueError(
            f"Target '{target}' was not found in {path.name}. Available columns: {list(df.columns)}. "
            "For a headerless file, use --no-header."
        )
    return df


def identify_features(
    df: pd.DataFrame, target: str, requested: Optional[str]
) -> Tuple[List[str], List[str]]:
    if requested:
        features = [part.strip() for part in requested.split(",") if part.strip()]
        missing = [col for col in features if col not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns not found: {missing}")
        if target in features:
            raise ValueError("Target column cannot be included in --features (prevents target leakage).")
        return features, [c for c in df.columns if c != target and c not in features]

    features: List[str] = []
    excluded: List[str] = []
    for col in df.columns:
        if col == target:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        original_nonmissing = int(df[col].notna().sum())
        numeric_rate = (int(converted.notna().sum()) / original_nonmissing) if original_nonmissing else 0.0
        if numeric_rate >= 0.95:
            features.append(col)
        else:
            excluded.append(col)
    if not features:
        raise ValueError("No numeric predictor columns were detected.")
    return features, excluded


def coerce_features(df: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input is missing predictor columns: {missing_cols}")
    X = df.loc[:, list(feature_names)].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    # All-empty predictors cannot be median-imputed; remove them explicitly.
    all_empty = [c for c in X.columns if X[c].notna().sum() == 0]
    if all_empty:
        X = X.drop(columns=all_empty)
    if X.shape[1] == 0:
        raise ValueError("No usable numeric predictor columns remain.")
    return X


def clean_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    y0 = df[target]
    valid = y0.notna()
    dropped = int((~valid).sum())
    cleaned = df.loc[valid].copy()
    # String representation is used for consistent mapping across training and test files.
    y = cleaned[target].astype(str).str.strip()
    if y.nunique() != 2:
        raise ValueError(f"Hill-Valley must be binary; found {y.nunique()} target classes: {sorted(y.unique().tolist())}")
    return cleaned, y, {"rows_with_missing_target_dropped": dropped, "n_classes": int(y.nunique())}


def choose_positive_class(labels: Sequence[str], requested: Optional[str]) -> str:
    classes = [str(x) for x in labels]
    if requested is not None:
        req = requested.strip()
        matches = [x for x in classes if x == req]
        if not matches:
            raise ValueError(f"--positive-label '{requested}' is not one of the target labels: {classes}")
        return matches[0]
    by_lower = {x.lower(): x for x in classes}
    for label in classes:
        if "hill" in label.lower():
            return label
    numeric = {}
    for label in classes:
        try:
            numeric[float(label)] = label
        except ValueError:
            pass
    if 1.0 in numeric and 0.0 in numeric:
        return numeric[1.0]
    return sorted(classes)[-1]


def make_scaler(name: str):
    if name == "none":
        return "passthrough"
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaler: {name}")


def make_pipeline(model: Any, scaler_name: str) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", make_scaler(scaler_name)),
        ("model", model),
    ])


def safe_cv(X: pd.DataFrame, y: np.ndarray, folds: int, seed: int) -> StratifiedKFold:
    counts = pd.Series(y).value_counts()
    if int(counts.min()) < folds:
        raise ValueError(
            f"The smallest class has {int(counts.min())} objects, fewer than cv-folds={folds}. "
            "Reduce --cv-folds."
        )
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def metric_summary(scores: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
        test = np.asarray(scores[f"test_{metric}"], dtype=float)
        out[f"cv_{metric}_mean"] = float(test.mean())
        out[f"cv_{metric}_std"] = float(test.std(ddof=1)) if len(test) > 1 else 0.0
        out[f"cv_{metric}_folds"] = test.tolist()
    if "train_macro_f1" in scores:
        out["train_macro_f1_mean"] = float(np.mean(scores["train_macro_f1"]))
        out["train_macro_f1_std"] = float(np.std(scores["train_macro_f1"], ddof=1))
    out["fit_time_sec_mean"] = float(np.mean(scores["fit_time"]))
    out["fit_time_sec_std"] = float(np.std(scores["fit_time"], ddof=1))
    out["score_time_sec_mean"] = float(np.mean(scores["score_time"]))
    out["score_time_sec_std"] = float(np.std(scores["score_time"], ddof=1))
    return out


def evaluate_cv(model: Pipeline, X: pd.DataFrame, y: np.ndarray, cv, n_jobs: int) -> Dict[str, Any]:
    scores = cross_validate(
        model, X, y, cv=cv, scoring=SCORING,
        return_train_score=True, n_jobs=n_jobs, error_score="raise"
    )
    return metric_summary(scores)


def scaler_experiment(
    X_train: pd.DataFrame, y_train: np.ndarray, cv, n_jobs: int
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    fixed_models = {
        "KNN_fixed": KNeighborsClassifier(n_neighbors=5, weights="distance", p=2, n_jobs=1),
        "Linear_SVM_fixed": SVC(kernel="linear", C=1.0),
        "RBF_SVM_fixed": SVC(kernel="rbf", C=10.0, gamma="scale"),
        "Nearest_Centroid": NearestCentroid(),
    }
    rows = []
    for model_name, base_model in fixed_models.items():
        for scaler_name in SCALERS:
            model = make_pipeline(clone(base_model), scaler_name)
            summary = evaluate_cv(model, X_train, y_train, cv, n_jobs)
            rows.append({
                "model_family": model_name,
                "scaler": scaler_name,
                **summary,
            })
    table = pd.DataFrame(rows)
    choices = {}
    for model_name in fixed_models:
        part = table[table.model_family == model_name].sort_values(
            ["cv_macro_f1_mean", "cv_macro_f1_std"],
            ascending=[False, True]
        )
        choices[model_name] = str(part.iloc[0]["scaler"])
    return table, choices


def search_model(
    model: Any,
    scaler_name: str,
    param_grid: Dict[str, List[Any]],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv,
    n_jobs: int,
):
    search = GridSearchCV(
        make_pipeline(model, scaler_name),
        param_grid=param_grid,
        scoring=SCORING,
        refit="macro_f1",
        cv=cv,
        return_train_score=True,
        n_jobs=n_jobs,
        error_score="raise",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        search.fit(X_train, y_train)
    warning_items = [
        {"category": type(w.message).__name__, "message": str(w.message)}
        for w in caught
    ]
    raw = pd.DataFrame(search.cv_results_)
    export_cols = [
        c for c in raw.columns
        if c == "params" or c == "rank_test_macro_f1"
        or c.startswith("param_")
        or c.startswith("mean_test_") or c.startswith("std_test_")
        or c.startswith("mean_train_") or c.startswith("std_train_")
        or c in ("mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time")
        or c.startswith("split") and "_test_" in c
    ]
    result_table = raw[export_cols].copy()
    result_table.insert(0, "model_family", type(model).__name__)
    return search, result_table, warning_items


def summarize_search(search: GridSearchCV) -> Dict[str, Any]:
    idx = int(search.best_index_)
    cvres = search.cv_results_
    return {
        "best_params": json_safe(search.best_params_),
        "best_cv_macro_f1": float(search.best_score_),
        "cv_macro_f1_std": float(cvres["std_test_macro_f1"][idx]),
        "cv_accuracy_mean": float(cvres["mean_test_accuracy"][idx]),
        "cv_accuracy_std": float(cvres["std_test_accuracy"][idx]),
        "cv_balanced_accuracy_mean": float(cvres["mean_test_balanced_accuracy"][idx]),
        "cv_balanced_accuracy_std": float(cvres["std_test_balanced_accuracy"][idx]),
        "cv_fit_time_sec_mean": float(cvres["mean_fit_time"][idx]),
        "cv_score_time_sec_mean": float(cvres["mean_score_time"][idx]),
        "train_macro_f1_mean": float(cvres["mean_train_macro_f1"][idx]),
        "train_macro_f1_std": float(cvres["std_train_macro_f1"][idx]),
        "n_configurations": int(len(search.cv_results_["params"])),
    }


def measure_model_cost(model_template: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[Pipeline, Dict[str, Any]]:
    fitted = clone(model_template)
    started = time.perf_counter()
    fitted.fit(X_train, y_train)
    fit_time = time.perf_counter() - started

    # Measure prediction latency only on training rows: test data remains locked.
    X_profile = X_train.iloc[:min(100, len(X_train))]
    fitted.predict(X_profile)  # warm-up
    repeats = 15
    elapsed = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fitted.predict(X_profile)
        elapsed.append(time.perf_counter() - t0)
    predict_batch_ms = float(np.mean(elapsed) * 1000.0)
    predict_one_ms = float(predict_batch_ms / max(1, len(X_profile)))
    try:
        serialized = pickle.dumps(fitted, protocol=pickle.HIGHEST_PROTOCOL)
        model_size_bytes = len(serialized)
    except Exception:
        model_size_bytes = None

    estimator = fitted.named_steps["model"]
    stored_objects: Dict[str, Any] = {"kind": type(estimator).__name__, "count": None}
    if isinstance(estimator, KNeighborsClassifier):
        stored_objects = {"kind": "training examples", "count": int(len(estimator._fit_X))}
    elif isinstance(estimator, NearestCentroid):
        stored_objects = {"kind": "class centroids", "count": int(len(estimator.centroids_))}
    elif isinstance(estimator, SVC):
        stored_objects = {"kind": "support vectors", "count": int(np.sum(estimator.n_support_))}
        stored_objects["support_vectors_per_class"] = estimator.n_support_.astype(int).tolist()
    elif isinstance(estimator, LogisticRegression):
        stored_objects = {"kind": "stored coefficient vectors", "count": int(estimator.coef_.shape[0])}

    return fitted, {
        "fit_time_sec_one_refit": float(fit_time),
        "prediction_batch_size": int(len(X_profile)),
        "prediction_batch_ms_mean": predict_batch_ms,
        "prediction_per_object_ms_mean": predict_one_ms,
        "serialized_size_bytes": model_size_bytes,
        "serialized_size_kb": None if model_size_bytes is None else float(model_size_bytes / 1024.0),
        "stored_objects": stored_objects,
        "profile_repeats": repeats,
        "profile_sample_source": "training rows only; the individual test set was not used for profiling",
    }


def candidate_cv_from_search(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cv_macro_f1_mean": summary["best_cv_macro_f1"],
        "cv_macro_f1_std": summary["cv_macro_f1_std"],
        "cv_accuracy_mean": summary["cv_accuracy_mean"],
        "cv_accuracy_std": summary["cv_accuracy_std"],
        "cv_balanced_accuracy_mean": summary["cv_balanced_accuracy_mean"],
        "cv_balanced_accuracy_std": summary["cv_balanced_accuracy_std"],
        "cv_fit_time_sec_mean": summary["cv_fit_time_sec_mean"],
        "cv_score_time_sec_mean": summary["cv_score_time_sec_mean"],
        "train_macro_f1_mean": summary["train_macro_f1_mean"],
        "train_macro_f1_std": summary["train_macro_f1_std"],
    }


def select_final_model(candidates: List[Dict[str, Any]], rule: str = "one_se") -> Dict[str, Any]:
    best = max(candidates, key=lambda r: r["cv_macro_f1_mean"])
    se = float(best["cv_macro_f1_std"] / math.sqrt(max(1, best["cv_folds"])))
    cutoff = float(best["cv_macro_f1_mean"] - se)
    if rule == "best_cv":
        eligible = [best]
    else:
        eligible = [c for c in candidates if c["cv_macro_f1_mean"] >= cutoff - 1e-12]
    # One-SE rule: among models statistically close to the maximum, prefer smaller
    # serialized model and then lower training-only prediction latency.
    chosen = sorted(
        eligible,
        key=lambda c: (
            float("inf") if c.get("serialized_size_bytes") is None else c["serialized_size_bytes"],
            c["prediction_per_object_ms_mean"],
            -c["cv_macro_f1_mean"],
        )
    )[0]
    return {
        "selection_rule": rule,
        "best_mean_cv_candidate": best["model"],
        "best_mean_cv_macro_f1": float(best["cv_macro_f1_mean"]),
        "best_candidate_cv_standard_error": se,
        "one_se_cutoff": cutoff,
        "eligible_within_one_se": [c["model"] for c in eligible],
        "selected_model": chosen["model"],
        "selected_cv_macro_f1": float(chosen["cv_macro_f1_mean"]),
        "rationale": (
            "Predeclared one-standard-error engineering rule: keep candidates within one SE of the "
            "best mean CV macro-F1, then prefer smaller serialized model size, lower training-only "
            "prediction latency, and finally higher CV macro-F1."
            if rule == "one_se" else
            "Choose the highest mean CV macro-F1; break exact ties by smaller model size and lower latency."
        ),
    }


def get_positive_scores(model: Pipeline, X: pd.DataFrame, positive_code: int) -> Optional[np.ndarray]:
    estimator = model.named_steps["model"]
    model_classes = np.asarray(getattr(estimator, "classes_", [0, 1]))
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            pos_idx = int(np.flatnonzero(model_classes == positive_code)[0])
            return np.asarray(probs[:, pos_idx], dtype=float)
        except Exception:
            return None
    if hasattr(model, "decision_function"):
        try:
            scores = np.asarray(model.decision_function(X), dtype=float)
            if scores.ndim == 1 and len(model_classes) == 2:
                return scores if positive_code == model_classes[1] else -scores
        except Exception:
            return None
    return None


def final_test_evaluation(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    classes: Sequence[str],
    positive_label: str,
    original_test_indices: Sequence[Any],
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    # This function is intentionally called only after final model selection.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    positive_code = list(classes).index(positive_label)
    negative_code = [i for i in range(len(classes)) if i != positive_code][0]

    cm = confusion_matrix(y_test, y_pred, labels=[negative_code, positive_code])
    scores = get_positive_scores(model, X_test, positive_code)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "precision_positive": float(precision_score(y_test, y_pred, pos_label=positive_code, zero_division=0)),
        "recall_positive": float(recall_score(y_test, y_pred, pos_label=positive_code, zero_division=0)),
        "f1_positive": float(f1_score(y_test, y_pred, pos_label=positive_code, zero_division=0)),
    }
    if scores is not None:
        y_binary = (y_test == positive_code).astype(int)
        metrics["roc_auc_positive"] = float(roc_auc_score(y_binary, scores))
        fpr, tpr, thresholds = roc_curve(y_binary, scores)
        roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}
    else:
        metrics["roc_auc_positive"] = None
        roc = None

    report = classification_report(
        y_test, y_pred, labels=list(range(len(classes))),
        target_names=[str(c) for c in classes],
        output_dict=True, zero_division=0
    )
    pred_rows = []
    for row_pos, (idx, yt, yp) in enumerate(zip(original_test_indices, y_test, y_pred)):
        pred_row = {
            "row_index": json_safe(idx),
            "y_true_code": int(yt),
            "y_true_label": str(classes[int(yt)]),
            "y_pred_code": int(yp),
            "y_pred_label": str(classes[int(yp)]),
            "correct": bool(yt == yp),
        }
        if scores is not None:
            pred_row["positive_class_score"] = float(scores[row_pos])
        # Include original feature values to support case analysis in the report.
        pred_row["features"] = {
            str(name): json_safe(value)
            for name, value in zip(feature_names, X_test.iloc[row_pos].tolist())
        }
        pred_rows.append(pred_row)

    # Confusion-matrix order is explicitly negative then positive.
    return {
        "evaluated_after_model_selection": True,
        "test_sample_count": int(len(y_test)),
        "positive_label": positive_label,
        "negative_label": str(classes[negative_code]),
        "class_order_in_confusion_matrix": [str(classes[negative_code]), str(classes[positive_code])],
        "metrics": metrics,
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_counts": {
            "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
        },
        "classification_report": json_safe(report),
        "roc_curve": roc,
        "predictions": pred_rows,
    }


def manual_oof_disagreement_analysis(
    knn_template: Pipeline,
    svm_template: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv,
    classes: Sequence[str],
    positive_label: str,
    max_cases: int,
) -> Dict[str, Any]:
    n = len(X_train)
    knn_oof = np.full(n, -1, dtype=int)
    svm_oof = np.full(n, -1, dtype=int)
    fold_of = np.full(n, -1, dtype=int)
    svm_margin = np.full(n, np.nan, dtype=float)
    neighbor_debug: Dict[int, Any] = {}
    label_codes = list(range(len(classes)))

    for fold_id, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fit, X_val = X_train.iloc[fit_idx], X_train.iloc[val_idx]
        y_fit = y_train[fit_idx]

        knn = clone(knn_template)
        svm = clone(svm_template)
        knn.fit(X_fit, y_fit)
        svm.fit(X_fit, y_fit)
        kp = knn.predict(X_val).astype(int)
        sp = svm.predict(X_val).astype(int)
        knn_oof[val_idx] = kp
        svm_oof[val_idx] = sp
        fold_of[val_idx] = fold_id

        raw_scores = None
        try:
            raw_scores = np.asarray(svm.decision_function(X_val), dtype=float)
        except Exception:
            pass
        if raw_scores is not None and raw_scores.ndim == 1:
            svm_est = svm.named_steps["model"]
            pos_idx = list(classes).index(positive_label)
            svm_margin[val_idx] = raw_scores if pos_idx == svm_est.classes_[1] else -raw_scores

        # Transform validation and fold-training rows with the fold-fitted pipeline.
        kprep = knn[:-1]
        Xfit_t = kprep.transform(X_fit)
        Xval_t = kprep.transform(X_val)
        kmodel = knn.named_steps["model"]
        k_neighbors = min(int(kmodel.n_neighbors), len(fit_idx))
        distances, local_neighbors = kmodel.kneighbors(
            Xval_t, n_neighbors=k_neighbors, return_distance=True
        )
        for i, original_local_idx in enumerate(val_idx):
            neighbor_debug[int(original_local_idx)] = {
                "fold": int(fold_id),
                "neighbor_indices": fit_idx[local_neighbors[i]].astype(int).tolist(),
                "neighbor_distances": distances[i].astype(float).tolist(),
                "neighbor_labels": y_train[fit_idx[local_neighbors[i]]].astype(int).tolist(),
            }

    records = []
    mismatches = []
    for i in range(n):
        item = {
            "row_index": json_safe(X_train.index[i]),
            "fold": int(fold_of[i]),
            "y_true_code": int(y_train[i]),
            "y_true_label": str(classes[int(y_train[i])]),
            "knn_oof_code": int(knn_oof[i]),
            "knn_oof_label": str(classes[int(knn_oof[i])]),
            "svm_oof_code": int(svm_oof[i]),
            "svm_oof_label": str(classes[int(svm_oof[i])]),
            "knn_correct": bool(knn_oof[i] == y_train[i]),
            "svm_correct": bool(svm_oof[i] == y_train[i]),
            "models_disagree": bool(knn_oof[i] != svm_oof[i]),
            "svm_positive_oriented_margin": None if not math.isfinite(svm_margin[i]) else float(svm_margin[i]),
        }
        records.append(item)
        if knn_oof[i] != svm_oof[i]:
            dbg = neighbor_debug.get(i, {})
            neighbor_rows = []
            for ni, dist, lab in zip(
                dbg.get("neighbor_indices", []),
                dbg.get("neighbor_distances", []),
                dbg.get("neighbor_labels", []),
            ):
                neighbor_rows.append({
                    "row_index": json_safe(X_train.index[int(ni)]),
                    "distance_after_preprocessing": float(dist),
                    "class_code": int(lab),
                    "class_label": str(classes[int(lab)]),
                    "feature_values": {
                        str(c): json_safe(v)
                        for c, v in X_train.iloc[int(ni)].to_dict().items()
                    },
                })
            mismatch = dict(item)
            mismatch.update({
                "features": {
                    str(c): json_safe(v)
                    for c, v in X_train.iloc[i].to_dict().items()
                },
                "nearest_neighbors_from_oof_knn_fold": neighbor_rows,
            })
            mismatches.append(mismatch)

    # Prefer cases where exactly one model is correct, then cases where both fail.
    mismatches.sort(
        key=lambda r: (
            0 if (r["knn_correct"] != r["svm_correct"]) else 1,
            int(r["fold"]),
            str(r["row_index"]),
        )
    )
    positives = list(classes).index(positive_label)
    oof_summary = {
        "knn_oof_macro_f1": float(f1_score(y_train, knn_oof, average="macro", zero_division=0)),
        "knn_oof_accuracy": float(accuracy_score(y_train, knn_oof)),
        "svm_oof_macro_f1": float(f1_score(y_train, svm_oof, average="macro", zero_division=0)),
        "svm_oof_accuracy": float(accuracy_score(y_train, svm_oof)),
        "n_oof_predictions": int(n),
        "n_disagreements": int(np.sum(knn_oof != svm_oof)),
        "disagreement_rate": float(np.mean(knn_oof != svm_oof)),
        "n_disagreements_knn_only_correct": int(sum(r["knn_correct"] and not r["svm_correct"] for r in mismatches)),
        "n_disagreements_svm_only_correct": int(sum(r["svm_correct"] and not r["knn_correct"] for r in mismatches)),
        "n_disagreements_both_wrong": int(sum(not r["knn_correct"] and not r["svm_correct"] for r in mismatches)),
        "positive_label_for_svm_margin": positive_label,
    }
    return {
        "method": "manual stratified out-of-fold predictions; each row is predicted by models not fitted on that row",
        "caveat": (
            "The selected hyperparameters were chosen using CV on this same training partition. "
            "OOF predictions are out-of-fold for fitting, but this disagreement analysis is diagnostic "
            "rather than a fully nested, unbiased estimate of the entire tuning procedure."
        ),
        "summary": oof_summary,
        "all_oof_predictions": records,
        "disagreement_cases_total": mismatches,
        "disagreement_cases_with_full_neighbor_details": mismatches[:max_cases],
    }


def ratio_min_max_distance(X: np.ndarray, max_points: int = 250, seed: int = 42) -> Dict[str, Any]:
    if len(X) > max_points:
        rng = np.random.default_rng(seed)
        ids = np.sort(rng.choice(len(X), size=max_points, replace=False))
        sample = X[ids]
    else:
        sample = X
    if len(sample) < 3:
        return {"median_dmin_dmax": None, "n_points": int(len(sample))}
    d = pairwise_distances(sample, metric="euclidean")
    np.fill_diagonal(d, np.inf)
    dmin = np.min(d, axis=1)
    dmax = np.max(d, axis=1)
    ratio = np.divide(dmin, dmax, out=np.zeros_like(dmin), where=dmax > 0)
    return {
        "median_dmin_dmax": float(np.median(ratio)),
        "mean_dmin_dmax": float(np.mean(ratio)),
        "min_ratio": float(np.min(ratio)),
        "max_ratio": float(np.max(ratio)),
        "n_points": int(len(sample)),
        "formula": "median(min distance to another point / max distance to another point)",
    }


def run_noise_dimension_experiment(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv,
    best_knn_params: Dict[str, Any],
    noise_counts: Sequence[int],
    random_state: int,
    n_jobs: int,
) -> Dict[str, Any]:
    rows = []
    detailed_columns: Dict[str, List[str]] = {}
    rng = np.random.default_rng(random_state)
    requested_counts = sorted(set(int(x) for x in noise_counts if int(x) >= 0))
    max_noise_count = max(requested_counts, default=0)
    # Use one max-width random matrix so each smaller experiment uses a prefix
    # of the same fixed-seed noise features (a controlled, cumulative comparison).
    all_noise = rng.uniform(0.0, 1.0, size=(len(X_train), max_noise_count))
    noise_cache = {count: all_noise[:, :count] for count in requested_counts}
    for count in sorted(noise_cache):
        noise = noise_cache[count]
        # Keep raw predictors in X_aug. The imputer and scaler used for CV remain
        # inside the Pipeline and are fitted independently in each fold.
        X_aug = X_train.copy()
        new_cols = []
        for j in range(count):
            name = f"__noise_{j + 1:04d}"
            new_cols.append(name)
            X_aug[name] = noise[:, j]
        # The distance-ratio diagnostic is descriptive and uses training points only.
        # Its own preprocessing fit is kept separate from CV scoring.
        diag_imputer = SimpleImputer(strategy="median")
        diag_scaler = MinMaxScaler()
        X_diag = diag_scaler.fit_transform(diag_imputer.fit_transform(X_aug))
        dist_info = ratio_min_max_distance(X_diag, seed=random_state)
        k = int(best_knn_params.get("model__n_neighbors", 5))
        p = int(best_knn_params.get("model__p", 2))
        weights = str(best_knn_params.get("model__weights", "distance"))
        model = make_pipeline(
            KNeighborsClassifier(n_neighbors=k, p=p, weights=weights, n_jobs=1),
            "minmax",
        )
        scores = evaluate_cv(model, X_aug, y_train, cv, n_jobs)
        rows.append({
            "noise_features": int(count),
            "total_features": int(X_aug.shape[1]),
            "knn_k": k, "knn_p": p, "knn_weights": weights,
            **scores,
            **dist_info,
        })
        detailed_columns[str(count)] = new_cols
    return {
        "purpose": "Controlled test of how independent, non-informative dimensions affect distances and KNN.",
        "noise_values_range": [0.0, 1.0],
        "scaling": "noise is generated on raw training rows; MinMaxScaler and imputation remain inside each CV Pipeline fold; distance-ratio is a separate training-only descriptive calculation",
        "counts": rows,
        "added_feature_names_by_noise_count": detailed_columns,
        "not_used_for_final_model_selection": True,
        "note": "CV metrics are diagnostic; synthetic noise features are not retained in the final classifier.",
    }


def plot_scaling(table: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.3))
    families = table["model_family"].unique().tolist()
    x = np.arange(len(SCALERS))
    width = 0.8 / max(1, len(families))
    for i, family in enumerate(families):
        part = table[table.model_family == family].set_index("scaler").reindex(SCALERS)
        offsets = x - 0.4 + width / 2 + i * width
        ax.bar(offsets, part["cv_macro_f1_mean"], width=width, label=family,
               yerr=part["cv_macro_f1_std"], capsize=3)
    ax.set_xticks(x, ["No scaler", "StandardScaler", "MinMaxScaler"])
    ax.set_ylabel("CV macro F1 (mean ± SD)")
    ax.set_title("Influence of scaling on fixed metric/SVM configurations")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "01_scaling_comparison.png", dpi=160)
    plt.close(fig)


def plot_knn_grid(knn_table: pd.DataFrame, figures: Path) -> None:
    if knn_table.empty:
        return
    rows = []
    for _, r in knn_table.iterrows():
        p = r.get("param_model__p")
        weights = r.get("param_model__weights")
        k = r.get("param_model__n_neighbors")
        rows.append({"k": int(k), "p": int(p), "weights": str(weights),
                     "f1": float(r["mean_test_macro_f1"]), "sd": float(r["std_test_macro_f1"])})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, p in zip(axes, [1, 2]):
        sub = df[df.p == p]
        for weights in ("uniform", "distance"):
            w = sub[sub.weights == weights].sort_values("k")
            ax.errorbar(w.k, w.f1, yerr=w.sd, marker="o", capsize=3, label=weights)
        ax.set_title(f"kNN with Minkowski p={p}")
        ax.set_xlabel("Number of neighbors k")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("CV macro F1 (mean ± SD)")
    axes[1].legend(title="Weights")
    fig.suptitle("kNN hyperparameter search")
    fig.tight_layout()
    fig.savefig(figures / "02_knn_grid.png", dpi=160)
    plt.close(fig)


def plot_rbf_heatmap(rbf_table: pd.DataFrame, figures: Path) -> None:
    if rbf_table.empty:
        return
    df = rbf_table.copy()
    df["C"] = df["param_model__C"].astype(float)
    df["gamma"] = df["param_model__gamma"].astype(str)
    # Keep categories in the searched parameter order.
    c_values = sorted(df.C.unique())
    gamma_order = []
    for v in ["scale", "0.001", "0.01", "0.1", "1"]:
        if v in set(df.gamma):
            gamma_order.append(v)
    for v in sorted(set(df.gamma) - set(gamma_order)):
        gamma_order.append(v)
    pivot = df.pivot(index="gamma", columns="C", values="mean_test_macro_f1").reindex(gamma_order)
    fig, ax = plt.subplots(figsize=(8.2, 5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index.tolist())
    ax.set_xlabel("C")
    ax.set_ylabel("Gamma")
    ax.set_title("RBF SVM CV macro F1 heatmap")
    fig.colorbar(im, ax=ax, label="Mean CV macro F1")
    fig.tight_layout()
    fig.savefig(figures / "03_rbf_svm_heatmap.png", dpi=160)
    plt.close(fig)


def plot_quality_cost(table: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    for _, row in table.iterrows():
        x = max(float(row["serialized_size_kb"] or 0.001), 0.001)
        y = float(row["cv_macro_f1_mean"])
        ax.scatter(x, y, s=45)
        ax.annotate(str(row["model"]), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Serialized model size (KB, log scale)")
    ax.set_ylabel("CV macro F1")
    ax.set_title("Predictive quality versus model size")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "04_quality_vs_model_size.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    for _, row in table.iterrows():
        x = max(float(row["prediction_per_object_ms_mean"]), 1e-6)
        y = float(row["cv_macro_f1_mean"])
        ax.scatter(x, y, s=45)
        ax.annotate(str(row["model"]), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Prediction latency per training-profile object (ms, log scale)")
    ax.set_ylabel("CV macro F1")
    ax.set_title("Predictive quality versus prediction latency")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "05_quality_vs_latency.png", dpi=160)
    plt.close(fig)


def plot_confusion_and_roc(test_result: Dict[str, Any], figures: Path) -> None:
    cm = np.asarray(test_result["confusion_matrix"], dtype=int)
    labels = test_result["class_order_in_confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(cm)
    ax.set_xticks([0, 1], [f"Pred {labels[0]}", f"Pred {labels[1]}",])
    ax.set_yticks([0, 1], [f"Actual {labels[0]}", f"Actual {labels[1]}"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_title("Final model — test confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figures / "06_final_confusion_matrix.png", dpi=160)
    plt.close(fig)

    roc = test_result.get("roc_curve")
    if roc:
        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        ax.plot(roc["fpr"], roc["tpr"], label=f"AUC = {test_result['metrics'].get('roc_auc_positive', float('nan')):.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Final model — test ROC curve")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / "07_final_roc_curve.png", dpi=160)
        plt.close(fig)


def plot_noise(noise_data: Dict[str, Any], figures: Path) -> None:
    df = pd.DataFrame(noise_data["counts"])
    fig, ax1 = plt.subplots(figsize=(8.4, 5.2))
    ax1.errorbar(df.noise_features, df.cv_macro_f1_mean, yerr=df.cv_macro_f1_std,
                 marker="o", capsize=3, label="KNN CV macro F1")
    ax1.set_xlabel("Number of random noise features")
    ax1.set_ylabel("CV macro F1 (mean ± SD)")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(df.noise_features, df.median_dmin_dmax, marker="s", linestyle="--", label="Median dmin/dmax")
    ax2.set_ylabel("Median dmin / dmax")
    ax1.set_title("Effect of irrelevant dimensions on KNN and distance contrast")
    fig.tight_layout()
    fig.savefig(figures / "08_noise_dimension_experiment.png", dpi=160)
    plt.close(fig)


def plot_disagreements(oof: Dict[str, Any], figures: Path) -> None:
    records = oof["all_oof_predictions"]
    folds = sorted(set(r["fold"] for r in records))
    dis = {f: 0 for f in folds}
    total = {f: 0 for f in folds}
    for r in records:
        total[r["fold"]] += 1
        if r["models_disagree"]:
            dis[r["fold"]] += 1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rates = [dis[f] / max(total[f], 1) for f in folds]
    ax.bar([str(f + 1) for f in folds], rates)
    ax.set_xlabel("CV fold")
    ax.set_ylabel("KNN–SVM OOF disagreement rate")
    ax.set_title(f"Out-of-fold disagreements: {oof['summary']['n_disagreements']} total")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "09_oof_disagreement_by_fold.png", dpi=160)
    plt.close(fig)


def run_geometry_experiment(output_figures: Path, random_state: int, mesh_resolution: int) -> Dict[str, Any]:
    X, y = make_moons(n_samples=520, noise=0.22, random_state=random_state)
    X[:, 1] *= 18.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.28, stratify=y, random_state=random_state
    )
    model_factories = {
        "kNN (k=9)": lambda: KNeighborsClassifier(n_neighbors=9, p=2, weights="uniform"),
        "Linear SVM": lambda: SVC(kernel="linear", C=1.0),
        "RBF SVM": lambda: SVC(kernel="rbf", C=10.0, gamma="scale"),
    }
    scale_options = [("none", "No scaling"), ("standard", "StandardScaler")]
    all_rows = []
    fitted = {}
    for model_name, factory in model_factories.items():
        for scaler_name, scaler_label in scale_options:
            pipe = make_pipeline(factory(), scaler_name)
            pipe.fit(pd.DataFrame(X_train, columns=["x1", "x2"]), y_train)
            pred = pipe.predict(pd.DataFrame(X_test, columns=["x1", "x2"]))
            row = {
                "model": model_name,
                "scaler": scaler_name,
                "scaler_label": scaler_label,
                "test_macro_f1": float(f1_score(y_test, pred, average="macro")),
                "test_accuracy": float(accuracy_score(y_test, pred)),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
            }
            estimator = pipe.named_steps["model"]
            if isinstance(estimator, SVC):
                row["support_vectors"] = int(np.sum(estimator.n_support_))
            else:
                row["support_vectors"] = None
            all_rows.append(row)
            fitted[(model_name, scaler_name)] = pipe

    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad * 18, X[:, 1].max() + pad * 18
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, mesh_resolution),
        np.linspace(y_min, y_max, mesh_resolution),
    )
    mesh_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=["x1", "x2"])
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    for row_idx, model_name in enumerate(model_factories):
        for col_idx, (scaler_name, scaler_label) in enumerate(scale_options):
            ax = axes[row_idx, col_idx]
            pipe = fitted[(model_name, scaler_name)]
            z = pipe.predict(mesh_df).reshape(xx.shape)
            ax.contourf(xx, yy, z, alpha=0.25)
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=15, marker="o", label="train")
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=22, marker="x", label="test")
            est = pipe.named_steps["model"]
            if isinstance(est, SVC):
                support = est.support_vectors_
                # Display support vectors in the coordinates used by the fitted classifier.
                # Convert them back through inverse scaler for a correctly placed marker.
                scaler = pipe.named_steps["scaler"]
                if scaler_name != "none":
                    try:
                        support_raw = scaler.inverse_transform(support)
                    except Exception:
                        support_raw = support
                else:
                    support_raw = support
                ax.scatter(support_raw[:, 0], support_raw[:, 1], s=45, facecolors="none",
                           linewidths=0.8, label="support vectors")
            ax.set_title(f"{model_name} — {scaler_label}")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2 (scaled 18×)")
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("Controlled geometry experiment: make_moons with one feature scaled 18×", y=1.0)
    fig.tight_layout()
    fig.savefig(output_figures / "10_moons_decision_regions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {
        "dataset": "make_moons",
        "generation": {"n_samples": 520, "noise": 0.22, "random_state": random_state, "second_feature_scale_factor": 18.0},
        "split": {"test_size": 0.28, "stratified": True, "n_train": int(len(X_train)), "n_test": int(len(X_test))},
        "metrics": all_rows,
        "explanation": (
            "This controlled geometric experiment is separate from the Hill-Valley holdout. "
            "It isolates how a 18× scale distortion changes distance-based and margin-based decision boundaries."
        ),
    }


def save_table(frame: pd.DataFrame, path: Path) -> None:
    clean = frame.copy()
    for col in clean.columns:
        clean[col] = clean[col].map(
            lambda v: json.dumps(json_safe(v), ensure_ascii=False) if isinstance(v, (dict, list, tuple, np.ndarray)) else v
        )
    clean.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    if not (0 < args.test_size < 1):
        raise ValueError("--test-size must be between 0 and 1.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >=2.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data).expanduser().resolve()
    raw = read_dataset(data_path, args.target, args.no_header, args.sep)
    raw_clean, y_text, target_info = clean_target(raw, args.target)
    feature_names, excluded = identify_features(raw_clean, args.target, args.features)
    X_all = coerce_features(raw_clean, feature_names)
    # If coercion removed all-empty columns, respect the final actual feature list.
    feature_names = list(X_all.columns)
    y_text = y_text.loc[X_all.index].reset_index(drop=True)
    X_all = X_all.reset_index(drop=True)
    original_indices = raw_clean.index.to_list()
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(y_text.to_numpy())
    classes = [str(c) for c in label_encoder.classes_]
    if len(classes) != 2:
        raise ValueError(f"Hill-Valley requires exactly 2 target classes; got {classes}")
    positive_label = choose_positive_class(classes, args.positive_label)
    label_to_code = {str(label): int(i) for i, label in enumerate(classes)}
    positive_code = label_to_code[positive_label]

    missing_values = {str(c): int(X_all[c].isna().sum()) for c in feature_names}
    duplicate_rows = int(raw_clean.duplicated().sum())
    class_counts = {str(classes[i]): int(np.sum(y_all == i)) for i in range(len(classes))}
    if args.test_data:
        test_path = Path(args.test_data).expanduser().resolve()
        train_df = raw_clean.copy()
        # No random holdout in this mode: --data is the supplied training set and --test-data is final holdout.
        X_train = coerce_features(train_df, feature_names).reset_index(drop=True)
        y_train_text = train_df[args.target].astype(str).str.strip().reset_index(drop=True)
        y_train = label_encoder.transform(y_train_text.to_numpy())
        test_raw = read_dataset(test_path, args.target, args.no_header, args.sep)
        test_clean, test_y_text, test_info = clean_target(test_raw, args.target)
        X_test = coerce_features(test_clean, feature_names).reset_index(drop=True)
        y_test = label_encoder.transform(test_y_text.to_numpy())
        test_original_indices = test_clean.index.to_list()
        split_info = {
            "mode": "provided official test file",
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_size_fraction": float(len(X_test) / (len(X_train) + len(X_test))),
            "train_file": str(data_path),
            "test_file": str(test_path),
            "test_target_rows_dropped_missing": test_info["rows_with_missing_target_dropped"],
        }
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(X_all)),
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y_all,
            shuffle=True,
        )
        X_train = X_all.iloc[train_idx].copy().reset_index(drop=True)
        X_test = X_all.iloc[test_idx].copy().reset_index(drop=True)
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        test_original_indices = [original_indices[int(i)] for i in test_idx]
        split_info = {
            "mode": "stratified train_test_split",
            "test_size_fraction_requested": float(args.test_size),
            "random_state": int(args.random_state),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_original_row_indices": [int(original_indices[int(i)]) for i in train_idx],
            "test_original_row_indices": [int(original_indices[int(i)]) for i in test_idx],
        }

    cv = safe_cv(X_train, y_train, args.cv_folds, args.random_state)
    # Fix the protocol before model search / test-set evaluation.
    protocol = {
        "target_column": args.target,
        "positive_label": positive_label,
        "class_labels_encoding": {str(i): str(classes[i]) for i in range(len(classes))},
        "features_used": [str(x) for x in feature_names],
        "excluded_non_numeric_features": excluded,
        "class_counts_full_input": class_counts,
        "missing_values_per_feature_full_input": missing_values,
        "duplicate_rows_full_input": duplicate_rows,
        "split": split_info,
        "cross_validation": {
            "type": "StratifiedKFold",
            "n_splits": int(args.cv_folds),
            "shuffle": True,
            "random_state": int(args.random_state),
        },
        "primary_metric": "macro F1 (higher is better)",
        "additional_metrics": ["accuracy", "balanced accuracy", "per-class precision/recall/F1", "ROC-AUC for the selected model when a continuous score is available"],
        "selection_rule": "one_se",
        "test_set_used_for_selection": False,
        "pipeline": ["median imputation", "optional scaler inside pipeline", "classifier"],
        "scale_variants": list(SCALERS),
        "knn_grid": {"n_neighbors": [1, 3, 5, 7, 11], "p": [1, 2], "weights": ["uniform", "distance"]},
        "linear_svm_grid": {"C": [0.01, 0.1, 1, 10, 100]},
        "rbf_svm_grid": {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.01, 0.1, 1]},
        "regular_models": {
            "NearestCentroid": "metric reference model",
            "LogisticRegression": "fixed C=1, StandardScaler; control/reference from prior classification work, not re-tuned",
        },
        "search_budget": {
            "scaling_experiment": 4 * len(SCALERS),
            "knn_grid": 5 * 2 * 2,
            "linear_svm_grid": 5,
            "rbf_svm_grid": 4 * 5,
            "noise_counts": [int(x) for x in args.noise_counts.split(",") if x.strip()],
        },
    }

    print("[1/8] Running scaling experiment...", flush=True)
    scaling_table, scaler_choices = scaler_experiment(X_train, y_train, cv, args.n_jobs)
    save_table(scaling_table, tables_dir / "scaling_comparison.csv")

    print("[2/8] Tuning kNN, linear SVM and RBF SVM...", flush=True)
    knn_search, knn_grid_table, knn_warnings = search_model(
        KNeighborsClassifier(n_jobs=1),
        scaler_choices["KNN_fixed"],
        {
            "model__n_neighbors": [1, 3, 5, 7, 11],
            "model__p": [1, 2],
            "model__weights": ["uniform", "distance"],
        },
        X_train, y_train, cv, args.n_jobs
    )
    linear_search, linear_grid_table, linear_warnings = search_model(
        SVC(kernel="linear"),
        scaler_choices["Linear_SVM_fixed"],
        {"model__C": [0.01, 0.1, 1, 10, 100]},
        X_train, y_train, cv, args.n_jobs
    )
    rbf_search, rbf_grid_table, rbf_warnings = search_model(
        SVC(kernel="rbf"),
        scaler_choices["RBF_SVM_fixed"],
        {
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", 0.001, 0.01, 0.1, 1],
        },
        X_train, y_train, cv, args.n_jobs
    )
    save_table(knn_grid_table, tables_dir / "knn_full_cv_grid.csv")
    save_table(linear_grid_table, tables_dir / "linear_svm_full_cv_grid.csv")
    save_table(rbf_grid_table, tables_dir / "rbf_svm_full_cv_grid.csv")

    print("[3/8] Evaluating metric/control candidates...", flush=True)
    # NearestCentroid uses the best scaler from the fixed scaling comparison.
    centroid_pipe = make_pipeline(NearestCentroid(), scaler_choices["Nearest_Centroid"])
    centroid_cv = evaluate_cv(centroid_pipe, X_train, y_train, cv, args.n_jobs)
    lr_pipe = make_pipeline(LogisticRegression(C=1.0, max_iter=5000), "standard")
    lr_cv = evaluate_cv(lr_pipe, X_train, y_train, cv, args.n_jobs)

    candidates_spec = [
        ("NearestCentroid", centroid_pipe, centroid_cv, {"scaler": scaler_choices["Nearest_Centroid"]}, args.cv_folds),
        ("LogisticRegression_control", lr_pipe, lr_cv, {"C": 1.0, "scaler": "standard", "role": "fixed reference, not tuned"}, args.cv_folds),
        ("KNN", knn_search.best_estimator_, candidate_cv_from_search(summarize_search(knn_search)), summarize_search(knn_search)["best_params"], args.cv_folds),
        ("LinearSVM", linear_search.best_estimator_, candidate_cv_from_search(summarize_search(linear_search)), summarize_search(linear_search)["best_params"], args.cv_folds),
        ("RBFSVM", rbf_search.best_estimator_, candidate_cv_from_search(summarize_search(rbf_search)), summarize_search(rbf_search)["best_params"], args.cv_folds),
    ]

    candidate_rows = []
    fitted_models: Dict[str, Pipeline] = {}
    for name, template, cv_metrics, hyperparams, folds_count in candidates_spec:
        fitted, cost = measure_model_cost(template, X_train, y_train)
        fitted_models[name] = fitted
        # Candidate cv metrics have a common schema regardless of whether tuned or fixed.
        row = {
            "model": name,
            "hyperparameters": json_safe(hyperparams),
            "scaler": template.named_steps["scaler"] if isinstance(template.named_steps["scaler"], str) else type(template.named_steps["scaler"]).__name__,
            **{k: json_safe(v) for k, v in cv_metrics.items() if isinstance(v, (int, float, list))},
            **cost,
            "cv_folds": int(folds_count),
        }
        # Convert the pipeline scaler to a readable label if the scaler is an estimator.
        if not isinstance(template.named_steps["scaler"], str):
            row["scaler"] = type(template.named_steps["scaler"]).__name__
        candidate_rows.append(row)

    comparison = pd.DataFrame(candidate_rows)
    # Fixed reference rows use same schema; ensure there are mean/std metric fields.
    comparison.to_csv(tables_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    selection = select_final_model(candidate_rows, rule="one_se")
    chosen_name = selection["selected_model"]
    chosen_model = fitted_models[chosen_name]
    print(f"Selected model before test evaluation: {chosen_name}", flush=True)

    print("[4/8] Creating out-of-fold disagreement analysis...", flush=True)
    best_svm_name = "RBFSVM" if (
        next(r for r in candidate_rows if r["model"] == "RBFSVM")["cv_macro_f1_mean"]
        >= next(r for r in candidate_rows if r["model"] == "LinearSVM")["cv_macro_f1_mean"]
    ) else "LinearSVM"
    oof = manual_oof_disagreement_analysis(
        knn_search.best_estimator_,
        fitted_models[best_svm_name],
        X_train, y_train, cv, classes, positive_label, args.max_oof_disagreements
    )
    save_table(pd.DataFrame(oof["all_oof_predictions"]), tables_dir / "oof_predictions.csv")
    save_table(pd.DataFrame(oof["disagreement_cases_total"]), tables_dir / "oof_disagreements.csv")

    print("[5/8] Running controlled noise-feature experiment...", flush=True)
    noise_counts = [int(x.strip()) for x in args.noise_counts.split(",") if x.strip()]
    noise_experiment = run_noise_dimension_experiment(
        X_train,
        y_train,
        cv,
        summarize_search(knn_search)["best_params"],
        noise_counts,
        args.random_state,
        args.n_jobs,
    )
    save_table(pd.DataFrame(noise_experiment["counts"]), tables_dir / "noise_dimension_experiment.csv")

    print("[6/8] Evaluating selected model once on test set...", flush=True)
    # The test labels are first used only here, after model selection, OOF diagnostics,
    # and training-only inference cost measurements have finished.
    test_eval = final_test_evaluation(
        chosen_model,
        X_train, y_train,
        X_test, y_test,
        classes, positive_label,
        test_original_indices, feature_names
    )
    save_table(pd.DataFrame(test_eval["predictions"]), tables_dir / "final_test_predictions.csv")

    print("[7/8] Running controlled make_moons geometry experiment and drawing graphs...", flush=True)
    geometry = run_geometry_experiment(figures_dir, args.random_state, args.mesh_resolution)

    plot_scaling(scaling_table, figures_dir)
    plot_knn_grid(knn_grid_table, figures_dir)
    plot_rbf_heatmap(rbf_grid_table, figures_dir)
    plot_quality_cost(comparison, figures_dir)
    plot_confusion_and_roc(test_eval, figures_dir)
    plot_noise(noise_experiment, figures_dir)
    plot_disagreements(oof, figures_dir)

    # Detailed full grid result tables for JSON.
    def frame_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
        return json_safe(frame.to_dict(orient="records"))

    summaries = {
        "NearestCentroid": {"scaler": scaler_choices["Nearest_Centroid"], **centroid_cv},
        "LogisticRegression_control": {"scaler": "standard", **lr_cv},
        "KNN": {**summarize_search(knn_search), "scaler": scaler_choices["KNN_fixed"]},
        "LinearSVM": {**summarize_search(linear_search), "scaler": scaler_choices["Linear_SVM_fixed"]},
        "RBFSVM": {**summarize_search(rbf_search), "scaler": scaler_choices["RBF_SVM_fixed"]},
    }
    hypothesis_register = [
        {
            "id": "H1",
            "hypothesis": "Scaling affects distance-based and SVM classifiers because it changes the geometry of the feature space.",
            "experiment": "For fixed kNN, linear SVM, RBF SVM and NearestCentroid configurations, compare no scaling, StandardScaler and MinMaxScaler on the same StratifiedKFold.",
            "confirmation_criterion": "CV macro-F1 or its fold variability changes across scaler choices; interpret model families separately.",
            "evidence_location": "tables/scaling_comparison.csv; figures/01_scaling_comparison.png",
        },
        {
            "id": "H2",
            "hypothesis": "The choice of Minkowski distance, number of neighbors and voting weights changes kNN performance.",
            "experiment": "Grid search k in {1,3,5,7,11}, p in {1,2}, weights in {uniform,distance}, using a fixed scaler choice from the scaling experiment.",
            "confirmation_criterion": "At least one tested setting differs in CV macro-F1 beyond trivial fold noise, and best parameters are recorded.",
            "evidence_location": "tables/knn_full_cv_grid.csv; figures/02_knn_grid.png",
        },
        {
            "id": "H3",
            "hypothesis": "RBF SVM may better represent a non-linear boundary than linear SVM, at the cost of different latency/model size.",
            "experiment": "Compare linear SVM C grid with RBF SVM joint C/gamma grid using the same training split and folds.",
            "confirmation_criterion": "Compare mean CV macro-F1 and SD together with final candidate size and latency; do not assume RBF must win.",
            "evidence_location": "tables/linear_svm_full_cv_grid.csv; tables/rbf_svm_full_cv_grid.csv; figures/03_rbf_svm_heatmap.png",
        },
        {
            "id": "H4",
            "hypothesis": "Adding independent non-informative dimensions reduces the usefulness of Euclidean neighborhoods for kNN.",
            "experiment": "Add fixed-seed random features with comparable [0,1] scale and record median dmin/dmax and CV macro-F1.",
            "confirmation_criterion": "Distance ratio tends toward 1 while CV macro-F1 falls as noise-feature count rises.",
            "evidence_location": "tables/noise_dimension_experiment.csv; figures/08_noise_dimension_experiment.png",
        },
    ]
    final_summary = {
        "selected_model": chosen_name,
        "selection_details": selection,
        "best_cv_by_mean": max(candidate_rows, key=lambda r: r["cv_macro_f1_mean"])["model"],
        "best_cv_macro_f1": max(r["cv_macro_f1_mean"] for r in candidate_rows),
        "selected_cv_macro_f1": next(r for r in candidate_rows if r["model"] == chosen_name)["cv_macro_f1_mean"],
        "best_knn": summaries["KNN"],
        "best_linear_svm": summaries["LinearSVM"],
        "best_rbf_svm": summaries["RBFSVM"],
        "best_svm_for_oof_disagreement": best_svm_name,
        "test_metrics": test_eval["metrics"],
        "test_confusion_matrix_counts": test_eval["confusion_matrix_counts"],
        "oof_disagreement_summary": oof["summary"],
        "engineering_scenarios": {
            "small_dataset_needing_neighbor_explanations": {
                "recommended_model": "KNN",
                "reason": "Can explain a prediction by showing nearest training examples, their classes and distances; confirm performance and latency from this run before a real deployment.",
            },
            "high_query_volume_strict_latency": {
                "recommended_model": min(candidate_rows, key=lambda r: r["prediction_per_object_ms_mean"])["model"],
                "reason": "Chosen here as the lowest measured prediction latency on a training-only profile; compare the associated CV macro-F1 and serialized size before accepting.",
            },
        },
        "methodological_limitations": [
            "The test set is used only after selection, but a random holdout from the same dataset is not a new external population.",
            "For --test-data, results depend on the provenance and separation of the supplied test file.",
            "Scaler choice is selected on training CV scores; reported best scores may be optimistic due to finite search. A nested CV would estimate the full tuning procedure more conservatively.",
            "OOF model-disagreement predictions are out-of-fold for fitting, but hyperparameters were chosen using the same training CV, so the disagreement set is diagnostic rather than a completely unbiased performance estimate.",
            "The distance/noise experiment is controlled and diagnostic only; random synthetic dimensions are not used in the final classifier.",
            "Wall-clock latency and model size depend on hardware, software versions and process load; measured values are environment-specific.",
        ],
    }

    # Write structured AI-use template without inventing verification outcomes.
    ai_usage = """# AI_USAGE.md — Lab 3 (Hill-Valley)

Use this file to document actual decisions and verification performed by the student.

| Task given to AI | Proposal | What was accepted/changed/rejected | Verification performed | Evidence |
|---|---|---|---|---|
| Build a reproducible experiment script for metric methods and SVM on Hill-Valley | Compare scaling, tune kNN/linear SVM/RBF SVM, measure model costs, add a controlled random-feature experiment and out-of-fold disagreement analysis | Review and record your own decision here | Run the script on the chosen dataset; inspect warnings, generated JSON, CSV tables and figures; independently review the target/feature columns and class mapping | `report_data.json`, `tables/`, `figures/` |

Do not claim a result was verified unless you actually ran the script and checked its output.
"""
    (output_dir / "AI_USAGE.md").write_text(ai_usage, encoding="utf-8")

    result = {
        "metadata": {
            "script": "hill_valley_lab3.py",
            "lab": "Laboratory Work No. 3 — Comparison of metric methods and Support Vector Machines",
            "dataset": "Hill-Valley Dataset",
            "dataset_file": str(data_path),
            "test_file": str(Path(args.test_data).expanduser().resolve()) if args.test_data else None,
            "target": args.target,
            "generated_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "test_set_used_for_model_selection": False,
            "test_evaluated_once_after_selection": True,
            "output_dir": str(output_dir),
        },
        "experimental_protocol": protocol,
        "hypotheses": hypothesis_register,
        "data_audit": {
            "rows_after_target_cleaning": int(len(raw_clean)),
            "features_used_count": int(len(feature_names)),
            "features_used": [str(c) for c in feature_names],
            "non_numeric_features_excluded": excluded,
            "rows_dropped_missing_target": target_info["rows_with_missing_target_dropped"],
            "missing_values_per_feature": missing_values,
            "duplicate_rows": duplicate_rows,
            "class_counts": class_counts,
            "class_code_mapping": {str(i): str(classes[i]) for i in range(len(classes))},
            "positive_label_for_binary_metrics": positive_label,
            "target_excluded_from_features": bool(args.target not in feature_names),
        },
        "scaler_experiment": {
            "fixed_configurations": {
                "KNN_fixed": {"n_neighbors": 5, "weights": "distance", "p": 2},
                "Linear_SVM_fixed": {"kernel": "linear", "C": 1.0},
                "RBF_SVM_fixed": {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
                "Nearest_Centroid": {},
            },
            "best_scaler_by_family": scaler_choices,
            "table": frame_records(scaling_table),
        },
        "hyperparameter_search": {
            "KNN": {"summary": summarize_search(knn_search), "warnings": knn_warnings, "full_cv_results": frame_records(knn_grid_table)},
            "LinearSVM": {"summary": summarize_search(linear_search), "warnings": linear_warnings, "full_cv_results": frame_records(linear_grid_table)},
            "RBFSVM": {"summary": summarize_search(rbf_search), "warnings": rbf_warnings, "full_cv_results": frame_records(rbf_grid_table)},
        },
        "candidate_comparison": frame_records(comparison),
        "candidate_cv_summaries": json_safe(summaries),
        "final_model_selection": json_safe(selection),
        "final_test_evaluation": json_safe(test_eval),
        "out_of_fold_disagreement_analysis": json_safe(oof),
        "dimensionality_noise_experiment": json_safe(noise_experiment),
        "controlled_geometry_experiment": json_safe(geometry),
        "report_ready_summary": json_safe(final_summary),
        "output_files": {
            "main_json": "report_data.json",
            "csv_tables": [
                "tables/scaling_comparison.csv",
                "tables/knn_full_cv_grid.csv",
                "tables/linear_svm_full_cv_grid.csv",
                "tables/rbf_svm_full_cv_grid.csv",
                "tables/model_comparison.csv",
                "tables/oof_predictions.csv",
                "tables/oof_disagreements.csv",
                "tables/noise_dimension_experiment.csv",
                "tables/final_test_predictions.csv",
            ],
            "figures": [
                "figures/01_scaling_comparison.png",
                "figures/02_knn_grid.png",
                "figures/03_rbf_svm_heatmap.png",
                "figures/04_quality_vs_model_size.png",
                "figures/05_quality_vs_latency.png",
                "figures/06_final_confusion_matrix.png",
                "figures/07_final_roc_curve.png",
                "figures/08_noise_dimension_experiment.png",
                "figures/09_oof_disagreement_by_fold.png",
                "figures/10_moons_decision_regions.png",
            ],
            "ai_usage": "AI_USAGE.md",
            "final_model": "final_model.joblib",
        },
    }

    with (output_dir / "report_data.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, ensure_ascii=False, indent=2, allow_nan=False)

    # Persist only the selected final model.
    joblib.dump(chosen_model, output_dir / "final_model.joblib")

    # Contact sheet is convenient when compiling the report.
    figure_files = sorted(figures_dir.glob("*.png"))
    if figure_files:
        ncols = 2
        nrows = int(math.ceil(len(figure_files) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows))
        axes = np.asarray(axes).reshape(-1)
        for ax, path in zip(axes, figure_files):
            img = plt.imread(path)
            ax.imshow(img)
            ax.set_title(path.name, fontsize=9)
            ax.axis("off")
        for ax in axes[len(figure_files):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / "figures_contact_sheet.png", dpi=120)
        plt.close(fig)

    print("\nExperiment completed.", flush=True)
    print(f"JSON report data: {output_dir / 'report_data.json'}", flush=True)
    print(f"Figures: {figures_dir}", flush=True)
    print(f"Tables: {tables_dir}", flush=True)
    print(f"Selected model: {chosen_name}", flush=True)
    print(f"Test macro-F1: {test_eval['metrics']['f1_macro']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
