#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import io
import json
import os
import platform
import pickle
import random
import statistics
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

DATASET_NAME = "Hill-Valley (without noise, Training split)"
DATASET_PAGE = "https://doi.org/10.24432/C5JC8P"
DATASET_ZIP_URL = "https://archive.ics.uci.edu/static/public/166/hill+valley.zip"
SEED = 42
TEST_SIZE = 0.20
OUTER_SPLITS = 5
OUTER_REPEATS = 4
STABILITY_SEEDS = list(range(10, 20))  # 10 different split random_state values
BOOTSTRAP_REPEATS = 100
PERMUTATION_REPEATS = 10
N_JOBS = -1
TREE_GRID = {
    "model__max_depth": [2, 3, 4, 5, 8, None],
    "model__min_samples_leaf": [1, 2, 4, 8],
}


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas objects and non-finite numbers to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if np.isfinite(x) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]
    if isinstance(value, (pd.Series, pd.Index)):
        return [json_safe(x) for x in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return json_safe(value.to_dict(orient="records"))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(x) for x in value]
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _parse_data_bytes(raw: bytes, source_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Parse UCI comma-separated data with or without a header row."""
    try:
        table = pd.read_csv(io.BytesIO(raw), sep=",", header=0, skipinitialspace=True)
        table.columns = [str(c).strip() for c in table.columns]
        if "class" not in table.columns and "Class" not in table.columns:
            table = pd.read_csv(io.BytesIO(raw), sep=",", header=None, skipinitialspace=True)
    except Exception:
        table = pd.read_csv(io.BytesIO(raw), sep=",", header=None, skipinitialspace=True)
    table.columns = [str(c).strip() for c in table.columns]
    target = next((c for c in table.columns if c.lower() == "class"), None)
    if target is None:
        if table.shape[1] != 101:
            # Also tolerate arbitrary whitespace-delimited files.
            table = pd.read_csv(io.BytesIO(raw), sep=r"\s+", header=None, engine="python")
        if table.shape[1] != 101:
            raise ValueError(
                f"Очікується 101 колонка (100 ознак + class), отримано {table.shape[1]} "
                f"у файлі {source_name}. Перевірте формат .data."
            )
        table.columns = [f"X{i}" for i in range(1, 101)] + ["class"]
        target = "class"
    # Standardize feature order, independent of the source header format.
    features = [f"X{i}" for i in range(1, 101)]
    if not set(features).issubset(table.columns):
        # The first row may have been parsed as a malformed header; recover as headerless.
        table = pd.read_csv(io.BytesIO(raw), sep=",", header=None, skipinitialspace=True)
        if table.shape[1] != 101:
            raise ValueError(f"Не вдалося розпізнати 100 ознак у {source_name}.")
        table.columns = features + ["class"]
        target = "class"
    for col in features:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    y = pd.to_numeric(table[target], errors="coerce")
    X = table[features].copy()
    # Drop only rows with missing target; missing predictor values are handled inside Pipeline.
    keep = y.notna()
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].astype(int).reset_index(drop=True)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"Очікували двійкові класи 0/1, знайдено: {sorted(y.unique().tolist())}")
    if X.shape[1] != 100:
        raise ValueError(f"Очікується 100 ознак, знайдено {X.shape[1]}.")
    return X, y


def load_dataset(data_path: str | None, no_download: bool) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load local dataset or retrieve the original UCI archive."""
    if data_path:
        p = Path(data_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Не знайдено файл датасету: {p}")
        X, y = _parse_data_bytes(p.read_bytes(), str(p))
        return X, y, str(p)

    candidates = [
        Path("Hill_Valley_without_noise_Training.data"),
        Path("data/Hill_Valley_without_noise_Training.data"),
        Path("./Hill-Valley/Hill_Valley_without_noise_Training.data"),
    ]
    for p in candidates:
        if p.is_file():
            X, y = _parse_data_bytes(p.read_bytes(), str(p))
            return X, y, str(p.resolve())

    if no_download:
        raise FileNotFoundError(
            "Сирі дані не знайдено. Передайте --data /шлях/до/"
            "Hill_Valley_without_noise_Training.data або дозвольте автозавантаження."
        )
    request = urllib.request.Request(DATASET_ZIP_URL, headers={"User-Agent": "Mozilla/5.0 ML-lab-script"})
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            archive_bytes = response.read()
    except Exception as exc:
        raise RuntimeError(
            "Не вдалося завантажити Hill-Valley з UCI (перевірте мережу) і локального .data файлу немає. "
            "Завантажте архів вручну з офіційної сторінки UCI та передайте шлях через --data. "
            f"URL: {DATASET_ZIP_URL}. Причина: {exc}"
        ) from exc
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        members = zf.namelist()
        match = next(
            (name for name in members if name.lower().endswith("hill_valley_without_noise_training.data")),
            None,
        )
        if match is None:
            raise RuntimeError(f"В архіві UCI немає Training.data. Вміст: {members}")
        raw = zf.read(match)
        X, y = _parse_data_bytes(raw, match)
    return X, y, f"UCI archive: {DATASET_ZIP_URL} :: {match}"


def make_pipeline(estimator: Any) -> Pipeline:
    """Imputation is placed inside the model pipeline to prevent fold leakage."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])


def estimator_factories(controlled_params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = controlled_params or {"max_depth": 4, "min_samples_leaf": 2}
    return {
        "baseline": lambda: make_pipeline(DummyClassifier(strategy="most_frequent")),
        "unrestricted_tree": lambda: make_pipeline(DecisionTreeClassifier(random_state=SEED)),
        "controlled_tree": lambda: make_pipeline(DecisionTreeClassifier(random_state=SEED, **params)),
        "random_forest": lambda: make_pipeline(RandomForestClassifier(
            n_estimators=200, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", random_state=SEED, n_jobs=N_JOBS,
        )),
        "hist_gradient_boosting": lambda: make_pipeline(HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=0.5, random_state=SEED,
        )),
    }


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "std_sample": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "n": int(len(a)),
    }


def safe_feature_values(row: pd.Series, feature_names: list[str]) -> dict[str, Any]:
    return {name: float(row[name]) if pd.notna(row[name]) else None for name in feature_names}


def extract_tree_path(tree_estimator: DecisionTreeClassifier, sample: pd.DataFrame,
                      feature_names: list[str]) -> list[dict[str, Any]]:
    """Return human-readable rules along one fitted decision-tree path."""
    model = tree_estimator
    if isinstance(tree_estimator, Pipeline):
        model = tree_estimator.named_steps["model"]
        sample_arr = tree_estimator.named_steps["imputer"].transform(sample)
    else:
        sample_arr = sample.to_numpy()
    node_path = model.decision_path(sample_arr).indices
    result: list[dict[str, Any]] = []
    for node_id in node_path:
        feat_id = int(model.tree_.feature[node_id])
        if feat_id < 0:
            continue
        feature = feature_names[feat_id]
        threshold = float(model.tree_.threshold[node_id])
        value = float(sample.iloc[0][feature])
        goes_left = value <= threshold
        result.append({
            "node_id": int(node_id), "feature": feature, "threshold": threshold,
            "value": value, "condition": f"{feature} <= {threshold:.8g}",
            "value_satisfies_condition": bool(goes_left),
            "direction": "left" if goes_left else "right",
            "absolute_distance_to_threshold": abs(value - threshold),
        })
    leaf_id = int(model.apply(sample_arr)[0])
    leaf_counts = model.tree_.value[leaf_id][0]
    class_idx = int(np.argmax(leaf_counts))
    result.append({
        "leaf_id": leaf_id,
        "leaf_class_prediction": int(model.classes_[class_idx]),
        "leaf_training_class_counts": [float(v) for v in leaf_counts],
    })
    return result


def tree_summary(pipeline: Pipeline, feature_names: list[str], max_rules_depth: int = 3) -> dict[str, Any]:
    tree = pipeline.named_steps["model"]
    return {
        "depth": int(tree.get_depth()),
        "n_leaves": int(tree.get_n_leaves()),
        "n_nodes": int(tree.tree_.node_count),
        "root_feature": feature_names[int(tree.tree_.feature[0])] if tree.tree_.feature[0] >= 0 else None,
        "root_threshold": float(tree.tree_.threshold[0]) if tree.tree_.feature[0] >= 0 else None,
        "rules_first_levels": export_text(tree, feature_names=feature_names, max_depth=max_rules_depth),
        "impurity_feature_importance": {
            f: float(v) for f, v in zip(feature_names, tree.feature_importances_)
        },
    }


def classify_error_type(y_true: int, tree_pred: int, rf_pred: int, hgb_pred: int) -> list[str]:
    out: list[str] = []
    tree_ok, rf_ok, hgb_ok = tree_pred == y_true, rf_pred == y_true, hgb_pred == y_true
    if not tree_ok and rf_ok and hgb_ok:
        out.append("tree_wrong_both_ensembles_correct")
    if tree_ok and (not rf_ok or not hgb_ok):
        out.append("tree_correct_at_least_one_ensemble_wrong")
    if rf_pred != hgb_pred:
        out.append("ensembles_disagree")
    if not tree_ok and not rf_ok and not hgb_ok:
        out.append("all_three_wrong")
    return out


def basic_data_audit(X: pd.DataFrame, y: pd.Series, feature_names: list[str]) -> dict[str, Any]:
    full = pd.concat([X, y.rename("class")], axis=1)
    missing_by_feature = X.isna().sum().to_dict()
    nonzero_missing = {k: int(v) for k, v in missing_by_feature.items() if int(v) > 0}
    constant_features = [c for c in feature_names if X[c].nunique(dropna=False) <= 1]
    duplicate_features = [c for c in feature_names if X[c].notna().all() and X[c].equals(y.astype(float))]
    corr = X.corr(numeric_only=True)
    high_corr_pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val) and abs(float(val)) >= 0.98:
                high_corr_pairs.append({"feature_a": cols[i], "feature_b": cols[j], "correlation": float(val)})
    high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return {
        "rows": int(len(X)),
        "features_count": int(X.shape[1]),
        "feature_names": feature_names,
        "target": "class",
        "class_meaning_from_UCI": {"0": "valley", "1": "hill"},
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "class_proportions": {str(k): float(v / len(y)) for k, v in y.value_counts().sort_index().items()},
        "missing_target_rows_dropped": 0,
        "missing_values_per_feature": {k: int(v) for k, v in missing_by_feature.items()},
        "features_with_missing_values": nonzero_missing,
        "exact_duplicate_rows_features_and_target": int(full.duplicated().sum()),
        "duplicate_feature_vectors_ignoring_target": int(X.duplicated().sum()),
        "constant_features": constant_features,
        "features_identical_to_target_check": duplicate_features,
        "possible_target_leakage_notes": (
            "Ознаки X1–X100 є впорядкованими координатами профілю, клас є 0/1. "
            "Перевірка точної копії target не знаходить копій; це не замінює предметної перевірки витоку."
        ),
        "top_highly_correlated_feature_pairs_abs_ge_0_98": high_corr_pairs[:50],
        "interpretation_note": "Сусідні Xj — координати одного послідовного профілю; кореляція між ними очікувана і обмежує причинну інтерпретацію.",
    }


def main_cv_experiment(X_train: pd.DataFrame, y_train: pd.Series, feature_names: list[str]) -> dict[str, Any]:
    """Repeated common outer folds; tuned tree gets its own inner CV on each outer train fold."""
    splitter = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=SEED)
    splits = list(splitter.split(X_train, y_train))
    factories = estimator_factories()
    all_fold_results: list[dict[str, Any]] = []
    scores_by_model: dict[str, list[dict[str, float]]] = {k: [] for k in factories}
    tree_param_rows = []
    for fold_no, (tr, va) in enumerate(splits):
        Xtr, Xva = X_train.iloc[tr], X_train.iloc[va]
        ytr, yva = y_train.iloc[tr], y_train.iloc[va]
        repeat_no = fold_no // OUTER_SPLITS
        split_no = fold_no % OUTER_SPLITS
        for name, factory in factories.items():
            if name == "controlled_tree":
                estimator = GridSearchCV(
                    estimator=make_pipeline(DecisionTreeClassifier(random_state=SEED)),
                    param_grid=TREE_GRID,
                    scoring="f1_macro",
                    cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + repeat_no * 10 + split_no),
                    n_jobs=N_JOBS,
                    refit=True,
                    return_train_score=False,
                )
            else:
                estimator = factory()
            started = time.perf_counter()
            estimator.fit(Xtr, ytr)
            fit_seconds = time.perf_counter() - started
            pred_tr = estimator.predict(Xtr)
            pred_va = estimator.predict(Xva)
            train_m = fold_metrics(np.asarray(ytr), np.asarray(pred_tr))
            valid_m = fold_metrics(np.asarray(yva), np.asarray(pred_va))
            row = {
                "model": name, "outer_repeat": repeat_no + 1, "fold": split_no + 1,
                "n_train": int(len(tr)), "n_validation": int(len(va)),
                "train_macro_f1": train_m["macro_f1"], "validation_macro_f1": valid_m["macro_f1"],
                "validation_accuracy": valid_m["accuracy"], "validation_balanced_accuracy": valid_m["balanced_accuracy"],
                "train_validation_macro_f1_gap": train_m["macro_f1"] - valid_m["macro_f1"],
                "fit_seconds": fit_seconds,
            }
            if name == "controlled_tree":
                row["best_params_in_outer_fold"] = estimator.best_params_
                tree_param_rows.append({"outer_repeat": repeat_no + 1, "fold": split_no + 1, "best_params": estimator.best_params_, "best_inner_macro_f1": float(estimator.best_score_)})
            all_fold_results.append(row)
            scores_by_model[name].append({
                "validation_macro_f1": valid_m["macro_f1"],
                "train_macro_f1": train_m["macro_f1"],
                "validation_accuracy": valid_m["accuracy"],
                "validation_balanced_accuracy": valid_m["balanced_accuracy"],
                "gap": train_m["macro_f1"] - valid_m["macro_f1"],
            })
    summary: dict[str, Any] = {}
    for name, rows in scores_by_model.items():
        repeat_means = []
        for repeat_id in range(1, OUTER_REPEATS + 1):
            repeat_vals = [r["validation_macro_f1"] for r in all_fold_results
                           if r["model"] == name and r["outer_repeat"] == repeat_id]
            repeat_means.append(float(np.mean(repeat_vals)))
        summary[name] = {
            "validation_macro_f1": summarize_values([r["validation_macro_f1"] for r in rows]),
            "repeat_mean_validation_macro_f1": repeat_means,
            "repeat_mean_validation_macro_f1_summary": summarize_values(repeat_means),
            "train_macro_f1": summarize_values([r["train_macro_f1"] for r in rows]),
            "validation_accuracy": summarize_values([r["validation_accuracy"] for r in rows]),
            "validation_balanced_accuracy": summarize_values([r["validation_balanced_accuracy"] for r in rows]),
            "train_minus_validation_macro_f1_gap": summarize_values([r["gap"] for r in rows]),
            "fold_values": rows,
        }
    return {
        "protocol": {
            "outer_cv": f"RepeatedStratifiedKFold(n_splits={OUTER_SPLITS}, n_repeats={OUTER_REPEATS}, random_state={SEED})",
            "outer_folds_total": len(splits), "identical_outer_splits_for_all_models": True,
            "controlled_tree_tuning": "GridSearchCV inside each outer-training fold (nested CV)",
            "controlled_tree_grid": TREE_GRID,
            "scoring": "macro F1",
        },
        "fold_results": all_fold_results,
        "summaries": summary,
        "nested_tree_best_params_by_outer_fold": tree_param_rows,
    }


def tree_complexity_search(X_train: pd.DataFrame, y_train: pd.Series, feature_names: list[str]) -> dict[str, Any]:
    """Compact 5-fold complexity table for depth/min leaf; train and validation data only."""
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    configs = [(depth, leaf) for depth in [2, 3, 4, 5, 8, None] for leaf in [1, 2, 4, 8]]
    for idx, (depth, leaf) in enumerate(configs):
        fold_train, fold_val = [], []
        depth_actual, leaves_actual, nodes_actual = [], [], []
        for f, (tr, va) in enumerate(splitter.split(X_train, y_train)):
            model = make_pipeline(DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=SEED))
            model.fit(X_train.iloc[tr], y_train.iloc[tr])
            pred_tr = model.predict(X_train.iloc[tr])
            pred_va = model.predict(X_train.iloc[va])
            fold_train.append(float(f1_score(y_train.iloc[tr], pred_tr, average="macro", zero_division=0)))
            fold_val.append(float(f1_score(y_train.iloc[va], pred_va, average="macro", zero_division=0)))
            tree = model.named_steps["model"]
            depth_actual.append(int(tree.get_depth()))
            leaves_actual.append(int(tree.get_n_leaves()))
            nodes_actual.append(int(tree.tree_.node_count))
        rows.append({
            "config_id": idx + 1, "max_depth": depth, "min_samples_leaf": leaf,
            "train_macro_f1_mean": float(np.mean(fold_train)), "train_macro_f1_std": float(np.std(fold_train, ddof=1)),
            "validation_macro_f1_mean": float(np.mean(fold_val)), "validation_macro_f1_std": float(np.std(fold_val, ddof=1)),
            "generalization_gap": float(np.mean(fold_train) - np.mean(fold_val)),
            "depth_mean": float(np.mean(depth_actual)), "leaves_mean": float(np.mean(leaves_actual)),
            "nodes_mean": float(np.mean(nodes_actual)), "fold_train_macro_f1": fold_train, "fold_validation_macro_f1": fold_val,
        })
    return {
        "protocol": "5-fold StratifiedKFold on Dtrain only; no test labels used",
        "parameter_grid": {"max_depth": [2, 3, 4, 5, 8, None], "min_samples_leaf": [1, 2, 4, 8]},
        "rows": rows,
        "best_by_validation_macro_f1": max(rows, key=lambda x: x["validation_macro_f1_mean"]),
    }


def stability_experiment(X_train: pd.DataFrame, y_train: pd.Series, original_train_ids: np.ndarray,
                         controlled_params: dict[str, Any]) -> dict[str, Any]:
    """Ten split seeds; every repetition produces OOF prediction for every training row."""
    names = ["controlled_tree", "random_forest", "hist_gradient_boosting"]
    factories = estimator_factories(controlled_params)
    all_scores = {n: [] for n in names}
    all_oof = {n: [[] for _ in range(len(X_train))] for n in names}
    per_seed = []
    for seed in STABILITY_SEEDS:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_scores = {n: [] for n in names}
        for tr, va in splitter.split(X_train, y_train):
            for name in names:
                model = factories[name]()
                model.fit(X_train.iloc[tr], y_train.iloc[tr])
                pred = np.asarray(model.predict(X_train.iloc[va]), dtype=int)
                fold_scores[name].append(float(f1_score(y_train.iloc[va], pred, average="macro", zero_division=0)))
                for row_i, p in zip(va, pred):
                    all_oof[name][int(row_i)].append(int(p))
        row = {"random_state": seed, "models": {}}
        for name in names:
            q = float(np.mean(fold_scores[name]))
            all_scores[name].append(q)
            row["models"][name] = {"mean_fold_macro_f1": q, "fold_macro_f1": fold_scores[name]}
        per_seed.append(row)
    object_stability = {}
    for name in names:
        disagreements = []
        per_object_rates = []
        unstable = 0
        for i, values in enumerate(all_oof[name]):
            if not values:
                continue
            counts = Counter(values)
            consensus = min(counts, key=lambda k: (-counts[k], k))
            rate = float(np.mean(np.asarray(values) != consensus))
            per_object_rates.append(rate)
            if rate > 0.25:
                unstable += 1
            disagreements.append({
                "train_row_index": int(original_train_ids[i]), "y_true": int(y_train.iloc[i]),
                "consensus_prediction": int(consensus), "prediction_change_rate_vs_consensus": rate,
                "n_repeated_oof_predictions": int(len(values)), "prediction_counts": {str(k): int(v) for k, v in counts.items()},
                "unstable_gt_25_percent": bool(rate > 0.25),
            })
        object_stability[name] = {
            "per_repeat_macro_f1": all_scores[name], "macro_f1_across_random_states": summarize_values(all_scores[name]),
            "mean_object_prediction_change_rate_vs_consensus": float(np.mean(per_object_rates)),
            "max_object_prediction_change_rate": float(np.max(per_object_rates)),
            "objects_changed_in_more_than_25_percent_of_repeats": int(unstable),
            "per_object_results": disagreements,
        }
    return {
        "protocol": {"n_random_state_values": len(STABILITY_SEEDS), "random_state_values": STABILITY_SEEDS,
                     "folds_per_seed": 5, "hyperparameters_fixed": True,
                     "OOF_predictions_per_training_object": len(STABILITY_SEEDS),
                     "note": "Параметри дерева зафіксовані за попереднім пошуком на Dtrain; це діагностика стійкості конфігурації."},
        "per_seed_results": per_seed,
        "summaries": object_stability,
    }


def bootstrap_tree_structure(X_train: pd.DataFrame, y_train: pd.Series, feature_names: list[str],
                             controlled_params: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    roots, root_thresholds, depths, leaves, nodes = [], [], [], [], []
    top_level_counts = Counter()
    n = len(X_train)
    for b in range(BOOTSTRAP_REPEATS):
        indices = rng.integers(0, n, size=n)
        model = DecisionTreeClassifier(random_state=SEED + b, **controlled_params)
        model.fit(X_train.iloc[indices], y_train.iloc[indices])
        tree = model.tree_
        root_idx = int(tree.feature[0])
        root_name = feature_names[root_idx] if root_idx >= 0 else "leaf"
        roots.append(root_name)
        root_thresholds.append(float(tree.threshold[0]) if root_idx >= 0 else None)
        depths.append(int(model.get_depth()))
        leaves.append(int(model.get_n_leaves()))
        nodes.append(int(tree.node_count))
        # Record all splits at levels 0 and 1, providing a rough upper-structure stability view.
        for node in range(tree.node_count):
            if tree.children_left[node] == tree.children_right[node]:
                continue
            if int(tree.n_node_samples[node]) >= 0:
                level = 0
                parent = node
                # Find depth by walking parent links; tree sizes are small here.
                while parent != 0:
                    parent = next((j for j in range(tree.node_count) if tree.children_left[j] == parent or tree.children_right[j] == parent), 0)
                    level += 1
                    if level > 3:
                        break
                if level <= 1:
                    top_level_counts[f"level_{level}:{feature_names[int(tree.feature[node])]} "] += 1
    root_counts = Counter(roots)
    threshold_by_root = {}
    for feat in sorted(root_counts):
        vals = [v for r, v in zip(roots, root_thresholds) if r == feat and v is not None]
        threshold_by_root[feat] = summarize_values(vals) if vals else None
    return {
        "n_bootstrap_samples": BOOTSTRAP_REPEATS,
        "sampling": "bootstrap with replacement, sample size = |Dtrain|",
        "fixed_tree_parameters": controlled_params,
        "root_feature_frequency": [{"feature": f, "count": int(c), "frequency": float(c / BOOTSTRAP_REPEATS),
                                     "root_threshold_summary": threshold_by_root.get(f)}
                                    for f, c in root_counts.most_common()],
        "root_threshold_values_by_bootstrap": root_thresholds,
        "root_features_by_bootstrap": roots,
        "depth": summarize_values(depths), "leaves": summarize_values(leaves), "nodes": summarize_values(nodes),
        "top_two_levels_split_feature_counts": [{"key": k.strip(), "count": int(v)} for k, v in top_level_counts.most_common()],
        "interpretation_note": "Частота кореневої ознаки і розподіл її порогів показують структурну стійкість, але не причинність.",
    }


def fit_feature_importance(X_train: pd.DataFrame, y_train: pd.Series, feature_names: list[str],
                           final_name: str, final_factory: Any, random_forest_factory: Any,
                           controlled_params: dict[str, Any]) -> dict[str, Any]:
    """Impurity importance plus out-of-fold permutation importance aggregated across five folds."""
    fitted_tree = make_pipeline(DecisionTreeClassifier(random_state=SEED, **controlled_params))
    fitted_tree.fit(X_train, y_train)
    fitted_rf = random_forest_factory()
    fitted_rf.fit(X_train, y_train)
    tree_imp = fitted_tree.named_steps["model"].feature_importances_
    rf_imp = fitted_rf.named_steps["model"].feature_importances_

    # Use the final chosen model in OOF validation-fold permutation importance.
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 700)
    fold_importance = []
    fold_top5 = []
    fold_base_scores = []
    for fold_no, (tr, va) in enumerate(splitter.split(X_train, y_train), start=1):
        estimator = final_factory()
        estimator.fit(X_train.iloc[tr], y_train.iloc[tr])
        baseline_score = float(f1_score(y_train.iloc[va], estimator.predict(X_train.iloc[va]), average="macro", zero_division=0))
        pim = permutation_importance(
            estimator, X_train.iloc[va], y_train.iloc[va], scoring="f1_macro",
            n_repeats=PERMUTATION_REPEATS, random_state=SEED + 800 + fold_no,
            n_jobs=N_JOBS,
        )
        means = np.asarray(pim.importances_mean, dtype=float)
        fold_importance.append(means)
        fold_base_scores.append(baseline_score)
        ranks = np.argsort(-means)[:5]
        fold_top5.append(set(int(x) for x in ranks))
    imp_arr = np.vstack(fold_importance)
    top5_freq = {j: float(sum(j in ss for ss in fold_top5) / len(fold_top5)) for j in range(len(feature_names))}
    perm_mean = np.mean(imp_arr, axis=0)
    perm_std = np.std(imp_arr, axis=0, ddof=1)
    rows = []
    for i, feat in enumerate(feature_names):
        rows.append({
            "feature": feat,
            "permutation_importance_mean_across_folds": float(perm_mean[i]),
            "permutation_importance_std_between_folds": float(perm_std[i]),
            "top5_frequency": top5_freq[i],
            "tree_impurity_importance": float(tree_imp[i]),
            "random_forest_impurity_importance": float(rf_imp[i]),
        })
    rows.sort(key=lambda r: r["permutation_importance_mean_across_folds"], reverse=True)
    imp_rank = {r["feature"]: i + 1 for i, r in enumerate(rows)}
    impurity_rf_rank = {f: i + 1 for i, f in enumerate(sorted(feature_names, key=lambda f: rf_imp[feature_names.index(f)], reverse=True))}
    impurity_tree_rank = {f: i + 1 for i, f in enumerate(sorted(feature_names, key=lambda f: tree_imp[feature_names.index(f)], reverse=True))}
    for row in rows:
        f = row["feature"]
        row["permutation_rank"] = imp_rank[f]
        row["tree_impurity_rank"] = impurity_tree_rank[f]
        row["random_forest_impurity_rank"] = impurity_rf_rank[f]
        row["rank_gap_vs_random_forest_impurity"] = row["permutation_rank"] - row["random_forest_impurity_rank"]
    return {
        "evaluated_model_for_permutation": final_name,
        "permutation_protocol": {
            "validation_scheme": "5-fold StratifiedKFold; model fitted without each validation fold",
            "n_repeats_per_feature_per_fold": PERMUTATION_REPEATS,
            "scoring": "macro F1", "fold_baseline_macro_f1": fold_base_scores,
            "leakage_note": "Permutation importance обчислена лише на відкладених fold-частинах.",
        },
        "all_features": rows,
        "top_20_by_permutation": rows[:20],
        "largest_rank_disagreements_with_rf_impurity": sorted(rows, key=lambda r: abs(r["rank_gap_vs_random_forest_impurity"]), reverse=True)[:20],
        "important_top_features_caveat": "Значення різних importance-методів мають різні масштаби; ранги не означають причинний вплив.",
    }


def partial_dependence_payload(X_train: pd.DataFrame, feature_names: list[str], final_model: Any,
                               importances: dict[str, Any]) -> dict[str, Any]:
    top_features = [x["feature"] for x in importances["top_20_by_permutation"][:2]]
    curve_entries = []
    correlations = X_train.corr(numeric_only=True)
    for feature in top_features:
        # Robust support grid: use quantiles to avoid letting outliers dominate the visualization.
        grid = np.unique(np.quantile(X_train[feature].dropna().to_numpy(), np.linspace(0.05, 0.95, 20)))
        means, stds, local_counts = [], [], []
        arr = X_train.copy()
        values = X_train[feature].to_numpy()
        for z in grid:
            arr[feature] = float(z)
            probs = final_model.predict_proba(arr)[:, 1]
            means.append(float(np.mean(probs)))
            stds.append(float(np.std(probs, ddof=1)))
            # Observations within ±2.5% of the observed range around this grid point.
            span = float(X_train[feature].max() - X_train[feature].min())
            radius = span * 0.025
            local_counts.append(int(np.sum(np.abs(values - z) <= radius)))
        others = correlations[feature].drop(labels=[feature]).abs().sort_values(ascending=False).head(5)
        curve_entries.append({
            "feature": feature, "target_probability_class": 1,
            "grid_values": [float(x) for x in grid],
            "mean_predicted_probability": means,
            "prediction_std_across_rows": stds,
            "local_observation_counts_within_2_5_percent_range": local_counts,
            "feature_observed_min": float(X_train[feature].min()),
            "feature_observed_max": float(X_train[feature].max()),
            "strongest_abs_correlations": [{"feature": str(k), "correlation": float(correlations.loc[feature, k])} for k in others.index],
        })
    return {
        "method": "manual partial dependence, averaging predict_proba over Dtrain while setting one feature to grid value",
        "features": curve_entries,
        "target_class": 1,
        "interpretation_limits": [
            "PDP описує поведінку моделі, а не причинний закон.",
            "Для послідовного Hill-Valley сусідні координати можуть сильно корелювати; штучна зміна однієї координати створює нереалістичні профілі.",
            "Малі локальні кількості спостережень біля grid value означають менш надійний інтервал.",
        ],
    }


def common_oof_disagreement(X_train: pd.DataFrame, y_train: pd.Series, train_original_ids: np.ndarray,
                            feature_names: list[str], controlled_params: dict[str, Any]) -> dict[str, Any]:
    names = ["controlled_tree", "random_forest", "hist_gradient_boosting"]
    factories = estimator_factories(controlled_params)
    oof = {n: np.full(len(X_train), -1, dtype=int) for n in names}
    tree_paths: dict[int, list[dict[str, Any]]] = {}
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, va in split.split(X_train, y_train):
        for name in names:
            model = factories[name]()
            model.fit(X_train.iloc[tr], y_train.iloc[tr])
            pred = np.asarray(model.predict(X_train.iloc[va]), dtype=int)
            oof[name][va] = pred
            if name == "controlled_tree":
                for row_i in va:
                    tree_paths[int(row_i)] = extract_tree_path(model, X_train.iloc[[int(row_i)]], feature_names)
    rows = []
    summary = Counter()
    for i in range(len(X_train)):
        true = int(y_train.iloc[i])
        tree_pred, rf_pred, hgb_pred = int(oof["controlled_tree"][i]), int(oof["random_forest"][i]), int(oof["hist_gradient_boosting"][i])
        types = classify_error_type(true, tree_pred, rf_pred, hgb_pred)
        for t in types:
            summary[t] += 1
        if tree_pred == rf_pred == hgb_pred:
            continue
        rows.append({
            "train_row_index": int(train_original_ids[i]), "y_true": true,
            "controlled_tree_oof_prediction": tree_pred,
            "random_forest_oof_prediction": rf_pred,
            "hist_gradient_boosting_oof_prediction": hgb_pred,
            "controlled_tree_correct": bool(tree_pred == true),
            "random_forest_correct": bool(rf_pred == true),
            "hist_gradient_boosting_correct": bool(hgb_pred == true),
            "disagreement_categories": types,
            "controlled_tree_oof_path": tree_paths.get(i, []),
            "features": safe_feature_values(X_train.iloc[i], feature_names),
        })
    return {
        "protocol": "One shared 5-fold StratifiedKFold split on Dtrain; these predictions are OOF.",
        "summary": {"oof_objects": int(len(X_train)), "objects_with_disagreement": int(len(rows)),
                    "pairwise_disagreement_rates": {
                        "tree_vs_rf": float(np.mean(oof["controlled_tree"] != oof["random_forest"])),
                        "tree_vs_hist_gradient_boosting": float(np.mean(oof["controlled_tree"] != oof["hist_gradient_boosting"])),
                        "rf_vs_hist_gradient_boosting": float(np.mean(oof["random_forest"] != oof["hist_gradient_boosting"])),
                    },
                    "category_counts": dict(summary),
                    "all_three_wrong_count": int(sum(1 for i in range(len(X_train)) if all(oof[n][i] != int(y_train.iloc[i]) for n in names)))},
        "all_disagreement_rows": rows,
        "top_10_detailed_examples": rows[:10],
        "all_oof_predictions": [
            {"train_row_index": int(train_original_ids[i]), "y_true": int(y_train.iloc[i]),
             "controlled_tree": int(oof["controlled_tree"][i]), "random_forest": int(oof["random_forest"][i]),
             "hist_gradient_boosting": int(oof["hist_gradient_boosting"][i]),
             "models_agree": bool(oof["controlled_tree"][i] == oof["random_forest"][i] == oof["hist_gradient_boosting"][i])}
            for i in range(len(X_train))
        ],
    }


def model_complexity(estimator: Any, name: str) -> dict[str, Any]:
    model = estimator.named_steps["model"] if isinstance(estimator, Pipeline) else estimator
    payload: dict[str, Any] = {"serialized_pickle_size_bytes": int(len(pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)))}
    if isinstance(model, DecisionTreeClassifier):
        payload.update({"depth": int(model.get_depth()), "n_leaves": int(model.get_n_leaves()), "n_nodes": int(model.tree_.node_count), "n_estimators_or_iterations": None})
    elif isinstance(model, RandomForestClassifier):
        payload.update({"n_estimators": int(len(model.estimators_)), "total_nodes_across_trees": int(sum(t.tree_.node_count for t in model.estimators_)), "max_depth_across_trees": int(max(t.get_depth() for t in model.estimators_)), "n_estimators_or_iterations": int(len(model.estimators_))})
    elif isinstance(model, HistGradientBoostingClassifier):
        payload.update({"n_iter_fitted": int(getattr(model, "n_iter_", 0)), "max_leaf_nodes_parameter": int(model.max_leaf_nodes), "n_estimators_or_iterations": int(getattr(model, "n_iter_", 0))})
    else:
        payload.update({"n_estimators_or_iterations": None})
    payload["explanation_method"] = {
        "controlled_tree": "decision path and readable if/else rules",
        "unrestricted_tree": "decision path (potentially very long rules)",
        "random_forest": "permutation importance, PDP; one component tree path is not a full-ensemble explanation",
        "hist_gradient_boosting": "permutation importance and PDP; no compact single-tree rule path for full model",
        "baseline": "majority-class rule",
    }.get(name, "model-specific")
    return payload


def benchmark_model(factory: Any, X_train: pd.DataFrame, y_train: pd.Series,
                    X_bench: pd.DataFrame, repeats: int = 5) -> dict[str, Any]:
    # Warm-up fit/predict followed by measured repetitions; timings are environment-specific.
    warm = factory()
    warm.fit(X_train, y_train)
    warm.predict(X_bench)
    fit_times, predict_times = [], []
    last_model = warm
    for i in range(repeats):
        current = factory()
        t0 = time.perf_counter()
        current.fit(X_train, y_train)
        fit_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        current.predict(X_bench)
        predict_times.append(time.perf_counter() - t0)
        last_model = current
    prediction_milliseconds = [t * 1000.0 for t in predict_times]
    payload = {
        "warm_up_runs": 1, "timed_repeats": repeats,
        "fit_seconds_each_repeat": fit_times, "fit_seconds_median": float(statistics.median(fit_times)),
        "predict_seconds_each_repeat_for_n_objects": predict_times,
        "predict_ms_median_for_n_objects": float(statistics.median(prediction_milliseconds)),
        "n_benchmark_objects": int(len(X_bench)),
        "complexity": model_complexity(last_model, getattr(last_model, "_lr4_name", "model")),
    }
    return payload


def evaluate_final_test(final_model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series, test_original_ids: np.ndarray) -> dict[str, Any]:
    final_model.fit(X_train, y_train)
    predictions = np.asarray(final_model.predict(X_test), dtype=int)
    probabilities = None
    if hasattr(final_model, "predict_proba"):
        probabilities = np.asarray(final_model.predict_proba(X_test))[:, list(final_model.classes_).index(1)]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "precision_hill_class_1": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "recall_hill_class_1": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
    }
    if probabilities is not None and len(np.unique(y_test)) == 2:
        metrics["roc_auc_class_1"] = float(roc_auc_score(y_test, probabilities))
        fpr, tpr, roc_thresholds = roc_curve(y_test, probabilities, pos_label=1)
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, probabilities, pos_label=1)
        roc_payload = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist()}
        pr_payload = {"precision": precision_curve.tolist(), "recall": recall_curve.tolist(), "thresholds": pr_thresholds.tolist()}
    else:
        roc_payload, pr_payload = None, None
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    records = []
    for i, (yt, yp) in enumerate(zip(y_test.to_numpy(), predictions)):
        records.append({
            "row_index": int(test_original_ids[i]), "y_true": int(yt), "y_pred": int(yp),
            "correct": bool(yt == yp), "probability_class_1_hill": float(probabilities[i]) if probabilities is not None else None,
            "features": {k: float(v) if pd.notna(v) else None for k, v in X_test.iloc[i].to_dict().items()},
        })
    return {
        "model": model_name, "test_set_opened_after_model_selection": True,
        "test_sample_count": int(len(y_test)), "class_order_confusion_matrix": [0, 1],
        "class_labels": {"0": "valley", "1": "hill"}, "metrics": metrics,
        "confusion_matrix": cm.tolist(), "classification_report": classification_report(y_test, predictions, labels=[0, 1], target_names=["valley_0", "hill_1"], output_dict=True, zero_division=0),
        "roc_curve": roc_payload, "precision_recall_curve": pr_payload,
        "prediction_records": records,
        "errors": [r for r in records if not r["correct"]],
        "correct_predictions": int(np.sum(predictions == y_test.to_numpy())),
        "error_count": int(np.sum(predictions != y_test.to_numpy())),
        "caveat": "Test results are reported only once after the model and selection rule are fixed; they are not used to retune parameters.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LR4: Hill-Valley — trees, ensembles, stability, importance, PDP; one JSON output.")
    parser.add_argument("--data", default=None, help="Path to Hill_Valley_without_noise_Training.data")
    parser.add_argument("--out", default="report_data_lr4_hill_valley.json", help="Output JSON path")
    parser.add_argument("--no-download", action="store_true", help="Do not attempt automatic UCI download")
    args = parser.parse_args()
    t_all = time.perf_counter()

    X, y, dataset_source = load_dataset(args.data, args.no_download)
    feature_names = list(X.columns)
    # Canonical fixed split is created once. Test is not used by CV, search, stability, interpretation or selection.
    original_ids = np.arange(len(X), dtype=int)
    train_ids, test_ids = train_test_split(
        original_ids, test_size=TEST_SIZE, random_state=SEED, stratify=y.to_numpy()
    )
    X_train = X.iloc[train_ids].reset_index(drop=True)
    y_train = y.iloc[train_ids].reset_index(drop=True)
    X_test = X.iloc[test_ids].reset_index(drop=True)
    y_test = y.iloc[test_ids].reset_index(drop=True)

    audit = basic_data_audit(X, y, feature_names)
    audit["split_class_counts"] = {
        "train": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
    }
    audit["split_class_proportions"] = {
        "train": {str(k): float(v / len(y_train)) for k, v in y_train.value_counts().sort_index().items()},
        "test": {str(k): float(v / len(y_test)) for k, v in y_test.value_counts().sort_index().items()},
    }

    # Pre-declared hypotheses and criteria, stored to make the research question operational.
    hypotheses = [
        {"id": "H1", "statement": "Random Forest або Gradient Boosting перевищить контрольоване дерево за середнім macro F1.", "measure": "Repeated outer-CV macro F1 mean", "decision_rule": "порівняти парні оцінки на однакових outer folds; врахувати std та мінімум"},
        {"id": "H2", "statement": "Ансамблі матимуть меншу нестійкість прогнозів, ніж одне дерево.", "measure": "середня частка OOF прогнозів, що не збігаються з консенсусом; кількість об'єктів >25%", "decision_rule": "нижча середня нестійкість і менша кількість нестійких об'єктів підтримують H2"},
        {"id": "H3", "statement": "Коренева ознака дерева буде змінюватись між bootstrap-вибірками.", "measure": "частотний розподіл root feature та розкид root threshold", "decision_rule": "відсутність однієї майже домінантної ознаки означає структурну нестійкість"},
        {"id": "H4", "statement": "Ранги важливості ознак залежатимуть від способу оцінювання та валідаційного fold.", "measure": "rank gap між permutation та impurity-based importance; std між folds", "decision_rule": "помітні rank gaps / високий fold std свідчать, що точний ранг ненадійний"},
    ]
    protocol = {
        "research_question": "Чи виправданий перехід від окремого дерева до ансамблю на Hill-Valley з огляду на якість, стійкість, інтерпретованість і витрати?",
        "dataset_source": dataset_source,
        "dataset_citation": "Graham, L. & Oppacher, F. (2008). Hill-Valley. UCI Machine Learning Repository. DOI: 10.24432/C5JC8P.",
        "uci_dataset_page": DATASET_PAGE,
        "target": "class", "class_mapping": {"0": "valley", "1": "hill"},
        "unit_of_observation": "Один ряд — 100 послідовних числових координат одного синтетичного профілю hill/valley.",
        "primary_metric": "macro F1", "primary_metric_reason": "Враховує F1 обох класів незалежно від їхньої частоти.",
        "test_size_fraction": TEST_SIZE, "stratification": True, "random_state": SEED,
        "train_original_row_indices": train_ids.tolist(), "test_original_row_indices": test_ids.tolist(),
        "split_rows": {"train": int(len(train_ids)), "test": int(len(test_ids)), "full": int(len(X))},
        "preprocessing": "median SimpleImputer inside each Pipeline; no scaling for tree-based methods",
        "selection_rule": "Серед моделей (unrestricted tree, controlled tree, Random Forest, HistGradientBoosting) визначити найкращий mean CV macro F1; потім сформувати one-standard-error eligible set за SE найкращої моделі та обрати найменший serialized model серед eligible. Test set не використовується для вибору.",
        "outer_cv": {"type": "RepeatedStratifiedKFold", "n_splits": OUTER_SPLITS, "n_repeats": OUTER_REPEATS, "random_state": SEED},
        "stability_cv": {"n_random_state_values": len(STABILITY_SEEDS), "seeds": STABILITY_SEEDS, "n_splits": 5},
        "tree_complexity_grid": TREE_GRID,
        "candidate_models": {
            "baseline": "DummyClassifier(strategy='most_frequent')",
            "unrestricted_tree": "DecisionTreeClassifier(random_state=42)",
            "controlled_tree": "DecisionTreeClassifier; GridSearchCV over max_depth and min_samples_leaf inside each outer fold",
            "random_forest": {"n_estimators": 200, "max_features": "sqrt", "min_samples_leaf": 2, "class_weight": "balanced", "random_state": SEED},
            "hist_gradient_boosting": {"max_iter": 150, "learning_rate": 0.06, "max_leaf_nodes": 15, "l2_regularization": 0.5, "random_state": SEED},
        },
        "selection_leakage_protection": "All model comparison and feature analysis are confined to Dtrain. Dtest is opened for the final evaluation only.",
    }

    main_cv = main_cv_experiment(X_train, y_train, feature_names)
    complexity = tree_complexity_search(X_train, y_train, feature_names)

    # Tuned controlled tree configuration selected only inside Dtrain.
    tree_search = GridSearchCV(
        make_pipeline(DecisionTreeClassifier(random_state=SEED)),
        param_grid=TREE_GRID, scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        n_jobs=N_JOBS, refit=True, return_train_score=True,
    )
    tree_search.fit(X_train, y_train)
    controlled_params = {k.replace("model__", ""): v for k, v in tree_search.best_params_.items()}
    factories = estimator_factories(controlled_params)

    stability = stability_experiment(X_train, y_train, train_ids, controlled_params)
    bootstrap = bootstrap_tree_structure(X_train, y_train, feature_names, controlled_params)

    # Fit all model candidates on Dtrain for pre-test resource estimates and final one-SE selection.
    benchmark_rows = {}
    fitted_for_importance = {}
    for name in ["unrestricted_tree", "controlled_tree", "random_forest", "hist_gradient_boosting"]:
        factory = factories[name]
        sample_model = factory()
        sample_model.fit(X_train, y_train)
        fitted_for_importance[name] = sample_model
        # Warm-up and 5 timed repeats, each using the same training and prediction batch.
        benchmark_rows[name] = benchmark_model(factory, X_train, y_train, X_train.iloc[:min(100, len(X_train))], repeats=5)
        # Correct complexity label in metadata.
        benchmark_rows[name]["complexity"] = model_complexity(sample_model, name)

    # One-standard-error model selection based only on outer-CV scores and Dtrain resource estimates.
    cv_summaries = main_cv["summaries"]
    eligible_candidates = ["unrestricted_tree", "controlled_tree", "random_forest", "hist_gradient_boosting"]
    means = {n: cv_summaries[n]["validation_macro_f1"]["mean"] for n in eligible_candidates}
    best_cv_name = max(means, key=means.get)
    # Repeated folds overlap, so estimate the selection SE from the repeat-level means,
    # not by treating all 20 fold scores as independent observations.
    repeat_stats = cv_summaries[best_cv_name]["repeat_mean_validation_macro_f1_summary"]
    best_cv_std = repeat_stats["std_sample"]
    best_cv_n = repeat_stats["n"]
    one_se = best_cv_std / np.sqrt(best_cv_n) if best_cv_n else 0.0
    cutoff = means[best_cv_name] - one_se
    eligible = [n for n in eligible_candidates if means[n] >= cutoff]
    selected_name = min(eligible, key=lambda n: benchmark_rows[n]["complexity"]["serialized_pickle_size_bytes"])
    selection_table = []
    for name in eligible_candidates:
        selection_table.append({
            "model": name, "cv_macro_f1_mean": means[name],
            "cv_macro_f1_std": cv_summaries[name]["validation_macro_f1"]["std_sample"],
            "cv_macro_f1_min": cv_summaries[name]["validation_macro_f1"]["min"],
            "within_one_standard_error": bool(name in eligible),
            "serialized_size_bytes_on_Dtrain": benchmark_rows[name]["complexity"]["serialized_pickle_size_bytes"],
            "fit_seconds_median": benchmark_rows[name]["fit_seconds_median"],
            "predict_ms_median_for_100_objects": benchmark_rows[name]["predict_ms_median_for_n_objects"],
            "tree_depth": benchmark_rows[name]["complexity"].get("depth"),
            "tree_nodes": benchmark_rows[name]["complexity"].get("n_nodes"),
            "n_estimators_or_iterations": benchmark_rows[name]["complexity"].get("n_estimators_or_iterations"),
        })
    selection = {
        "rule": protocol["selection_rule"], "candidate_cv_macro_f1_means": means,
        "best_mean_cv_model": best_cv_name, "best_mean_cv_macro_f1": float(means[best_cv_name]),
        "best_model_repeat_mean_cv_std": float(best_cv_std), "repeat_count_for_standard_error": int(best_cv_n), "one_standard_error": float(one_se),
        "one_se_note": "Standard error is estimated from the repeat-level mean scores (not by assuming all overlapping folds are independent). With four repeats this remains an approximate selection heuristic.",
        "one_se_cutoff": float(cutoff), "eligible_models_within_one_se": eligible,
        "selected_model": selected_name, "decision_table": selection_table,
        "rationale": (
            f"Спочатку визначено найвище mean outer-CV macro F1 ({best_cv_name}); потім застосовано правило one-standard-error. "
            f"Серед моделей у межах cutoff обрано найменшу серіалізовану модель ({selected_name}); test set до вибору не залучався."
        ),
    }

    # Use final chosen fitted candidate for permutation importance and PDP; both use Dtrain folds only.
    final_factory = factories[selected_name]
    importance = fit_feature_importance(X_train, y_train, feature_names, selected_name, final_factory, factories["random_forest"], controlled_params)
    final_model_for_pdp = final_factory()
    final_model_for_pdp.fit(X_train, y_train)
    pdp = partial_dependence_payload(X_train, feature_names, final_model_for_pdp, importance)

    disagreement = common_oof_disagreement(X_train, y_train, train_ids, feature_names, controlled_params)

    # Tree interpretability: final controlled tree structure and one actual training observation path.
    controlled_final = factories["controlled_tree"]()
    controlled_final.fit(X_train, y_train)
    sample_path = extract_tree_path(controlled_final, X_train.iloc[[0]], feature_names)
    tree_interpretation = {
        "controlled_tree_parameters_selected_by_5fold_cv_on_Dtrain": tree_search.best_params_,
        "controlled_tree_inner_cv_best_macro_f1": float(tree_search.best_score_),
        "controlled_tree_structure": tree_summary(controlled_final, feature_names, max_rules_depth=3),
        "example_training_row_index": int(train_ids[0]),
        "example_true_class": int(y_train.iloc[0]),
        "example_predicted_class": int(controlled_final.predict(X_train.iloc[[0]])[0]),
        "example_decision_path": sample_path,
    }

    # Final test evaluation occurs once after the selection object is fixed.
    final_model = factories[selected_name]()
    final_test = evaluate_final_test(final_model, selected_name, X_train, y_train, X_test, y_test, test_ids)

    # Test score is not used in any earlier selection or explanation choice.
    report_summary = {
        "dataset_rows": int(len(X)), "feature_count": int(len(feature_names)),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "baseline_cv_macro_f1": cv_summaries["baseline"]["validation_macro_f1"]["mean"],
        "unrestricted_tree_cv_macro_f1": means["unrestricted_tree"],
        "controlled_tree_cv_macro_f1": means["controlled_tree"],
        "random_forest_cv_macro_f1": means["random_forest"],
        "hist_gradient_boosting_cv_macro_f1": means["hist_gradient_boosting"],
        "selected_model": selected_name,
        "test_macro_f1": final_test["metrics"]["macro_f1"],
        "test_accuracy": final_test["metrics"]["accuracy"],
        "test_error_count": final_test["error_count"],
        "test_evaluation_caveat": "test metrics are final diagnostics only and must not trigger model retuning",
    }

    output = {
        "metadata": {
            "script": Path(__file__).name, "lab": "Лабораторна робота №4 — Дерева рішень та ансамблі: якість, стійкість та інтерпретація",
            "dataset": DATASET_NAME, "dataset_source": dataset_source,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            "numpy_version": np.__version__, "pandas_version": pd.__version__, "scikit_learn_version": sklearn.__version__,
            "seed": SEED, "elapsed_seconds": None,
        },
        "research_question_and_hypotheses": {"protocol": protocol, "hypotheses": hypotheses},
        "data_audit": audit,
        "tree_complexity_search": complexity,
        "nested_cross_validation_model_comparison": main_cv,
        "controlled_tree_final_search": {
            "best_params": tree_search.best_params_, "best_cv_macro_f1": float(tree_search.best_score_),
            "candidate_rows": [
                {"params": p, "mean_test_macro_f1": float(m), "std_test_macro_f1": float(s),
                 "mean_train_macro_f1": float(tr)}
                for p, m, s, tr in zip(tree_search.cv_results_["params"], tree_search.cv_results_["mean_test_score"],
                                       tree_search.cv_results_["std_test_score"], tree_search.cv_results_["mean_train_score"])
            ],
        },
        "stability_across_random_states": stability,
        "bootstrap_tree_structure_stability": bootstrap,
        "feature_importance_and_variability": importance,
        "partial_dependence_payload": pdp,
        "oof_prediction_disagreements": disagreement,
        "engineering_benchmarks": {
            "environment": {"python_version": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                             "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
            "benchmark_method": "one warm-up run then 5 timed fit/predict repeats; median reported; prediction batch up to 100 Dtrain rows",
            "models": benchmark_rows,
        },
        "model_selection_before_test": selection,
        "tree_interpretability": tree_interpretation,
        "final_test_evaluation": final_test,
        "report_ready_summary": report_summary,
        "limitations_and_notes": [
            "Усі оцінки залежать від фіксованого random_state і заданих гіперпараметрів; необхідно надавати розкид, а не лише середнє.",
            "Профіль Hill-Valley має 100 послідовних координат; окремі Xj — позиції кривої, а не незалежні предметні фактори.",
            "Impurity importance може бути biased до ознак із багатьма точками розбиття; permutation importance залежить від вибірки та кореляцій між ознаками.",
            "Partial dependence описує поведінку навченої моделі, не доводить причинного впливу і може створювати нерепрезентативні комбінації корельованих координат.",
            "Часи й розмір pickle є характеристиками поточного середовища і не є універсальними константами.",
        ],
        "output_contract": {
            "single_json_contains_data_for_tables_and_plots": True,
            "plot_payload_sections": ["tree_complexity_search.rows", "nested_cross_validation_model_comparison.fold_results",
                                      "stability_across_random_states", "bootstrap_tree_structure_stability.root_feature_frequency",
                                      "feature_importance_and_variability.all_features", "partial_dependence_payload.features",
                                      "oof_prediction_disagreements.summary", "final_test_evaluation.confusion_matrix",
                                      "final_test_evaluation.roc_curve", "final_test_evaluation.precision_recall_curve"],
        },
    }
    output["metadata"]["elapsed_seconds"] = float(time.perf_counter() - t_all)
    output_path = Path(args.out).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(output), f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"JSON записано: {output_path}")
    print(f"Рядків: {len(X)}; ознак: {X.shape[1]}; train/test: {len(train_ids)}/{len(test_ids)}")
    print(f"Обрана модель (до test): {selected_name}")
    print(f"Фінальний test macro F1: {final_test['metrics']['macro_f1']:.4f}; accuracy: {final_test['metrics']['accuracy']:.4f}")
    print(f"Тривалість: {output['metadata']['elapsed_seconds']:.1f} с; JSON: {output_path.stat().st_size / 1024:.1f} KiB")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Виконання перервано користувачем.", file=sys.stderr)
        raise SystemExit(130)
