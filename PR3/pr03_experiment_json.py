#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.datasets import load_iris

DEFAULT_DEGREES = [1, 2, 3, 5, 8, 12, 16]
DEFAULT_ALPHAS = np.logspace(-6, 3, 19)
DEFAULT_L1_RATIOS = [0.2, 0.5, 0.8]
STABILITY_SEEDS = [3, 11, 21, 42, 77]


@dataclass
class CVStats:
    train_rmse: float
    cv_rmse: float
    cv_std: float
    cv_se: float
    gap: float
    cv_time_sec: float
    convergence_warnings: int


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PR #3 and save all report data to JSON.")
    p.add_argument("--target", required=True, help="Regression target column")
    p.add_argument("--features", default=None, help="Comma-separated numeric predictor columns")
    p.add_argument("--output", default="pr03_report_data.json")
    p.add_argument("--test-size", type=float, default=0.20,
                   help="Use the same value as in previous work")
    p.add_argument("--random-state", type=int, default=42,
                   help="Use the same value as in previous work")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--degrees", default="1,2,3,5,8,12,16")
    p.add_argument("--max-poly-features", type=int, default=20000)
    p.add_argument("--max-estimated-mb", type=float, default=1024.0)
    p.add_argument("--selection-rule", choices=["min_cv_rmse", "one_se"], default="min_cv_rmse")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--lasso-max-iter", type=int, default=200000)
    p.add_argument("--elasticnet-max-iter", type=int, default=200000)
    p.add_argument("--tol", type=float, default=1e-7)
    return p.parse_args()


def safe(o: Any) -> Any:
    print("safe(o: Any) -> Any")

    if o is None or isinstance(o, (str, bool, int)):
        return o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return [safe(v) for v in o.tolist()]
    if isinstance(o, (list, tuple)):
        return [safe(v) for v in o]
    if isinstance(o, dict):
        return {str(k): safe(v) for k, v in o.items()}
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


def load_table(path: Path) -> pd.DataFrame:
    print("load_table(path: Path) -> pd.DataFrame")
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported format {ext}; use CSV/XLSX/XLS/Parquet/JSON.")


def prepare_data(df: pd.DataFrame, target: str, requested: Optional[str]):
    print("prepare_data(df: pd.DataFrame, target: str, requested: Optional[str])")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Available columns: {list(df.columns)}")

    y = pd.to_numeric(df[target], errors="coerce")
    target_missing = int(y.isna().sum())
    keep = y.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep].astype(float)

    if requested:
        features = [x.strip() for x in requested.split(",") if x.strip()]
        absent = [x for x in features if x not in df.columns]
        if absent:
            raise ValueError(f"Requested features not found: {absent}")
        for c in features:
            if not pd.api.types.is_numeric_dtype(df[c]):
                converted = pd.to_numeric(df[c], errors="coerce")
                if float(converted.notna().mean()) < 0.95:
                    raise ValueError(f"Feature '{c}' is not safely numeric.")
                df[c] = converted
    else:
        features = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]

    if not features:
        raise ValueError("No numeric predictor features available.")

    non_numeric_excluded = [
        c for c in df.columns if c != target and c not in features
        and not pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[features].copy()
    info = {
        "original_rows": int(len(keep)),
        "rows_with_missing_target": target_missing,
        "rows_after_target_filter": int(len(df)),
        "features_used": features,
        "n_numeric_features_used": int(len(features)),
        "non_numeric_features_excluded": non_numeric_excluded,
        "missing_values_per_feature": {c: int(X[c].isna().sum()) for c in features},
    }
    return df, X, y, features, info


def poly_count(p: int, degree: int) -> int:
    print("poly_count(p: int, degree: int) -> int")
    return math.comb(p + degree, degree) - 1


def est_mb(n_train: int, n_features: int) -> float:
    print("est_mb(n_train: int, n_features: int) -> float")
    return n_train * n_features * 8 * 3 / (1024 ** 2)


def select_feasible_degrees(p: int, n_train: int, requested: Sequence[int], max_features: int, max_mb: float):
    print("select_feasible_degrees(p: int, n_train: int, requested: Sequence[int], max_features: int, max_mb: float)")
    checks = []
    for d in sorted(set(int(x) for x in requested if int(x) >= 1)):
        n = poly_count(p, d)
        mb = est_mb(n_train, n)
        ok = n <= max_features and mb <= max_mb
        reason = None
        if n > max_features:
            reason = f"{n} polynomial features > cap {max_features}"
        elif mb > max_mb:
            reason = f"estimated {mb:.1f} MB > cap {max_mb:.1f} MB"
        checks.append({"degree": d, "n_features": n, "estimated_training_memory_mb": mb,
                       "feasible": ok, "skip_reason": reason})
    feasible = [r["degree"] for r in checks if r["feasible"]]
    if len(feasible) < 5:
        raise ValueError(
            "Fewer than five requested degree levels fit the safety limits. "
            "Use --features to select a smaller justified numeric subset, or adjust the caps after checking memory."
        )
    return feasible, checks


def make_pipeline(degree: int, estimator: Any) -> Pipeline:
    print("make_pipeline(degree: int, estimator: Any) -> Pipeline")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def estimator(name: str, alpha=None, l1_ratio=None, lasso_max_iter=200000, elasticnet_max_iter=200000, tol=1e-7):
    print("estimator(name: str, alpha=None, l1_ratio=None, lasso_max_iter=200000, elasticnet_max_iter=200000, tol=1e-7)")
    if name == "LinearRegression":
        return LinearRegression()
    if name == "Ridge":
        return Ridge(alpha=float(alpha))
    if name == "Lasso":
        return Lasso(alpha=float(alpha), max_iter=lasso_max_iter, tol=tol)
    if name == "ElasticNet":
        return ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio),
                          max_iter=elasticnet_max_iter, tol=tol)
    raise ValueError(name)


def cv_stats(model, X, y, cv, n_jobs=1) -> CVStats:
    print("cv_stats(model, X, y, cv, n_jobs=1) -> CVStats")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        scores = cross_validate(
            model, X, y, cv=cv, scoring="neg_root_mean_squared_error",
            return_train_score=True, n_jobs=n_jobs, error_score="raise"
        )
        elapsed = time.perf_counter() - t0
    tr = -scores["train_score"].astype(float)
    va = -scores["test_score"].astype(float)
    mean_tr = float(np.mean(tr))
    mean_va = float(np.mean(va))
    sd = float(np.std(va, ddof=1)) if len(va) > 1 else 0.0
    return CVStats(
        train_rmse=mean_tr,
        cv_rmse=mean_va,
        cv_std=sd,
        cv_se=sd / math.sqrt(len(va)),
        gap=mean_va - mean_tr,
        cv_time_sec=float(elapsed),
        convergence_warnings=sum(isinstance(w.message, ConvergenceWarning) for w in caught),
    )


def complexity_experiment(X, y, degrees, cv, n_jobs):
    print("complexity_experiment(X, y, degrees, cv, n_jobs)")
    rows = []
    for d in degrees:
        m = make_pipeline(d, LinearRegression())
        s = cv_stats(m, X, y, cv, n_jobs)
        rows.append({"degree": d, "n_features": poly_count(X.shape[1], d), **asdict(s)})
    return pd.DataFrame(rows).sort_values("degree").reset_index(drop=True)


def degree_diagnosis(table: pd.DataFrame):
    print("degree_diagnosis(table: pd.DataFrame)")
    best = table.loc[table["cv_rmse"].idxmin()]
    threshold = float(best["cv_rmse"] + best["cv_se"])
    eligible = table[table["cv_rmse"] <= threshold + 1e-15]
    one_se = eligible.sort_values(["n_features", "degree"]).iloc[0]
    under = table[
        (table.degree < best.degree) &
        (table.train_rmse >= table.train_rmse.median()) &
        (table.cv_rmse >= table.cv_rmse.median())
    ]["degree"].astype(int).tolist()
    over = table[
        (table.degree > best.degree) &
        (table.cv_rmse > best.cv_rmse) &
        (table.gap > best.gap)
    ]["degree"].astype(int).tolist()
    return {
        "best_degree_by_min_cv_rmse": int(best.degree),
        "best_cv_rmse": float(best.cv_rmse),
        "best_cv_se": float(best.cv_se),
        "one_se_threshold": threshold,
        "one_se_simplest_degree": int(one_se.degree),
        "possible_underfitting_degrees_heuristic": under,
        "possible_overfitting_degrees_heuristic": over,
        "diagnostic_note": "These labels are heuristic evidence flags; interpret the full CV curve and fold variability in the report."
    }


def learning_data(X, y, degree, cv, seed, n_jobs):
    print("learning_data(X, y, degree, cv, seed, n_jobs)")
    m = make_pipeline(degree, LinearRegression())
    sizes, tr, va = learning_curve(
        m, X, y, train_sizes=np.linspace(0.25, 1.0, 6), cv=cv,
        scoring="neg_root_mean_squared_error", shuffle=True,
        random_state=seed, n_jobs=n_jobs
    )
    tr, va = -tr, -va
    return {
        "degree": int(degree),
        "train_sizes": sizes.astype(int).tolist(),
        "train_rmse_mean": np.mean(tr, axis=1).tolist(),
        "train_rmse_std": np.std(tr, axis=1, ddof=1).tolist(),
        "cv_rmse_mean": np.mean(va, axis=1).tolist(),
        "cv_rmse_std": np.std(va, axis=1, ddof=1).tolist(),
        "final_gap_at_largest_size": float(np.mean(va[-1]) - np.mean(tr[-1])),
    }


def regularization_experiment(X, y, degree, cv, alphas, l1_ratios, lm_iter, en_iter, tol, n_jobs):
    print("regularization_experiment(X, y, degree, cv, alphas, l1_ratios, lm_iter, en_iter, tol, n_jobs)")
    rows = []
    for name, a, r in [("LinearRegression", None, None)]:
        s = cv_stats(make_pipeline(degree, estimator(name)), X, y, cv, n_jobs)
        rows.append({"model": name, "degree": degree, "alpha": a, "l1_ratio": r,
                     "n_features": poly_count(X.shape[1], degree), **asdict(s)})
    for name in ("Ridge", "Lasso"):
        for a in alphas:
            s = cv_stats(make_pipeline(degree, estimator(name, a, None, lm_iter, en_iter, tol)), X, y, cv, n_jobs)
            rows.append({"model": name, "degree": degree, "alpha": float(a), "l1_ratio": None,
                         "n_features": poly_count(X.shape[1], degree), **asdict(s)})
    for a in alphas:
        for r in l1_ratios:
            s = cv_stats(make_pipeline(degree, estimator("ElasticNet", a, r, lm_iter, en_iter, tol)), X, y, cv, n_jobs)
            rows.append({"model": "ElasticNet", "degree": degree, "alpha": float(a), "l1_ratio": float(r),
                         "n_features": poly_count(X.shape[1], degree), **asdict(s)})
    tab = pd.DataFrame(rows)
    best = {}
    for name in ("Ridge", "Lasso", "ElasticNet"):
        sub = tab[tab.model == name].sort_values(["cv_rmse", "cv_std", "alpha"])
        best[name] = safe(sub.iloc[0].to_dict())
    return tab, best


def coefficient_summary(model: Pipeline):
    print("coefficient_summary(model: Pipeline)")
    coef = np.asarray(model.named_steps["model"].coef_, dtype=float).ravel()
    names = [str(x) for x in model.named_steps["poly"].get_feature_names_out()]
    nz = np.flatnonzero(np.abs(coef) > 1e-12)
    return {
        "l1_norm": float(np.sum(np.abs(coef))),
        "l2_norm": float(np.linalg.norm(coef, ord=2)),
        "n_nonzero": int(len(nz)),
        "n_coefficients": int(len(coef)),
        "nonzero_feature_indices": nz.astype(int).tolist(),
        "nonzero_feature_names": [names[i] for i in nz],
        "feature_names": names,
        "coefficients": coef.tolist(),
    }


def fit_selected_coeffs(X, y, degree, best_cfg, lm_iter, en_iter, tol):
    print("fit_selected_coeffs(X, y, degree, best_cfg, lm_iter, en_iter, tol)")
    out = {}
    for name, cfg in best_cfg.items():
        m = make_pipeline(degree, estimator(name, cfg["alpha"], cfg.get("l1_ratio"), lm_iter, en_iter, tol))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m.fit(X, y)
        out[name] = coefficient_summary(m)
    return out


def regularization_path(X, y, degree, name, alphas, lm_iter, en_iter, tol, top_n=100):
    print("regularization_path(X, y, degree, name, alphas, lm_iter, en_iter, tol, top_n=100)")
    paths, coef_rows = [], []
    names = None
    for a in alphas:
        m = make_pipeline(degree, estimator(name, a, None, lm_iter, en_iter, tol))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m.fit(X, y)
        c = coefficient_summary(m)
        names = c["feature_names"]
        coef_rows.append(c["coefficients"])
        paths.append({"alpha": float(a), "l1_norm": c["l1_norm"], "l2_norm": c["l2_norm"], "n_nonzero": c["n_nonzero"]})
    cm = np.asarray(coef_rows, dtype=float)
    if cm.shape[1] <= top_n:
        idx = list(range(cm.shape[1]))
    else:
        idx = np.argsort(np.max(np.abs(cm), axis=0))[::-1][:top_n]
        idx = sorted(int(i) for i in idx)
    return {
        "model": name, "degree": degree, "alpha_grid": [float(a) for a in alphas],
        "summary_by_alpha": paths, "n_coefficients": int(cm.shape[1]),
        "visualized_feature_indices": idx,
        "visualized_feature_names": [names[i] for i in idx],
        "coefficient_paths": cm[:, idx].tolist(),
        "note": "If there are more than 100 coefficients, the path keeps the 100 with largest maximum absolute coefficient."
    }


def best_seed_config(X, y, degree, seed, name, alphas, l1_ratios, lm_iter, en_iter, tol, n_jobs):
    print("best_seed_config(X, y, degree, seed, name, alphas, l1_ratios, lm_iter, en_iter, tol, n_jobs)")
    cv = KFold(5, shuffle=True, random_state=seed)
    candidates = []
    grid = [(float(a), None) for a in alphas] if name != "ElasticNet" else [(float(a), float(r)) for a in alphas for r in l1_ratios]
    for a, r in grid:
        m = make_pipeline(degree, estimator(name, a, r, lm_iter, en_iter, tol))
        s = cv_stats(m, X, y, cv, n_jobs)
        candidates.append({"alpha": a, "l1_ratio": r, **asdict(s)})
    best = min(candidates, key=lambda z: (z["cv_rmse"], z["cv_std"], z["alpha"], z["l1_ratio"] or 0))
    m = make_pipeline(degree, estimator(name, best["alpha"], best["l1_ratio"], lm_iter, en_iter, tol))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m.fit(X, y)
    c = coefficient_summary(m)
    return {"seed": seed, "model": name, "degree": degree, **best,
            "l1_norm": c["l1_norm"], "l2_norm": c["l2_norm"],
            "n_nonzero": c["n_nonzero"],
            "nonzero_feature_indices": c["nonzero_feature_indices"],
            "nonzero_feature_names": c["nonzero_feature_names"]}


def jaccard(sets):
    print("jaccard(sets)")
    if len(sets) < 2:
        return {"mean": None, "min": None, "pairwise": []}
    vals, pairs = [], []
    for i, j in combinations(range(len(sets)), 2):
        u = sets[i] | sets[j]
        score = 1.0 if not u else len(sets[i] & sets[j]) / len(u)
        vals.append(score)
        pairs.append({"i": i, "j": j, "jaccard": float(score)})
    return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "pairwise": pairs}


def stability(X, y, degree, alphas, l1_ratios, lm_iter, en_iter, tol, n_jobs):
    print("stability(X, y, degree, alphas, l1_ratios, lm_iter, en_iter, tol, n_jobs)")
    by_model, summary = [], {}
    for name in ("Ridge", "Lasso", "ElasticNet"):
        rows, supports = [], []
        for seed in STABILITY_SEEDS:
            r = best_seed_config(X, y, degree, seed, name, alphas, l1_ratios, lm_iter, en_iter, tol, n_jobs)
            rows.append(r)
            if name != "Ridge":
                supports.append(set(r["nonzero_feature_indices"]))
        by_model.extend(rows)
        av = [r["alpha"] for r in rows]
        cv = [r["cv_rmse"] for r in rows]
        l1 = [r["l1_norm"] for r in rows]
        l2 = [r["l2_norm"] for r in rows]
        nnz = [r["n_nonzero"] for r in rows]
        item = {
            "model": name,
            "alpha_min": float(min(av)), "alpha_max": float(max(av)),
            "mean_cv_rmse": float(np.mean(cv)), "sd_cv_rmse": float(np.std(cv, ddof=1)),
            "mean_l1_norm": float(np.mean(l1)), "sd_l1_norm": float(np.std(l1, ddof=1)),
            "mean_l2_norm": float(np.mean(l2)), "sd_l2_norm": float(np.std(l2, ddof=1)),
            "n_nonzero_min": int(min(nnz)), "n_nonzero_max": int(max(nnz)),
            "per_seed": rows,
        }
        if supports:
            item["support_jaccard"] = jaccard(supports)
            freq = {}
            for r in rows:
                for f in r["nonzero_feature_names"]:
                    freq[f] = freq.get(f, 0) + 1
            item["support_frequency"] = dict(sorted(freq.items(), key=lambda x: (-x[1], x[0])))
        summary[name] = item
    return {"seeds": STABILITY_SEEDS, "results_by_seed": by_model, "summary_by_model": summary,
            "note": "Predictive stability and coefficient/support stability are separate properties."}


def final_selection(complexity, diagnosis, best_cfg, rule):
    print("final_selection(complexity, diagnosis, best_cfg, rule)")
    degree = diagnosis["one_se_simplest_degree"] if rule == "one_se" else diagnosis["best_degree_by_min_cv_rmse"]
    r = complexity[complexity.degree == degree].iloc[0]
    candidates = [{
        "candidate": f"LinearRegression_d{degree}", "model": "LinearRegression", "degree": int(degree),
        "n_features": int(r.n_features), "alpha": None, "l1_ratio": None,
        "train_rmse": float(r.train_rmse), "cv_rmse": float(r.cv_rmse), "cv_std": float(r.cv_std),
        "cv_se": float(r.cv_se), "gap": float(r.gap), "formal_complexity": int(r.n_features),
    }]
    for name in ("Ridge", "Lasso", "ElasticNet"):
        b = best_cfg[name]
        candidates.append({
            "candidate": name, "model": name, "degree": int(b["degree"]),
            "n_features": int(b["n_features"]), "alpha": float(b["alpha"]),
            "l1_ratio": None if b.get("l1_ratio") is None else float(b["l1_ratio"]),
            "train_rmse": float(b["train_rmse"]), "cv_rmse": float(b["cv_rmse"]),
            "cv_std": float(b["cv_std"]), "cv_se": float(b["cv_se"]), "gap": float(b["gap"]),
            "formal_complexity": int(b["n_features"]),
        })
    min_c = min(candidates, key=lambda c: (c["cv_rmse"], c["cv_std"]))
    threshold = min_c["cv_rmse"] + min_c["cv_se"]
    for c in candidates:
        c["within_one_se"] = bool(c["cv_rmse"] <= threshold + 1e-15)
    if rule == "min_cv_rmse":
        chosen = min(candidates, key=lambda c: (c["cv_rmse"], c["cv_std"], c["formal_complexity"]))
    else:
        eligible = [c for c in candidates if c["within_one_se"]]
        chosen = min(eligible, key=lambda c: (c["formal_complexity"], c["cv_rmse"], c["cv_std"]))
    return {"selection_rule": rule, "one_se_threshold_across_final_candidates": float(threshold),
            "chosen_candidate": chosen, "candidates": candidates}


def test_eval(model, Xtr, ytr, Xte, yte, indices):
    print("test_eval(model, Xtr, ytr, Xte, yte, indices)")
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    rows = [{"index": safe(i), "y_true": float(a), "y_pred": float(b),
             "residual": float(a-b), "absolute_error": float(abs(a-b))}
            for i, a, b in zip(indices, yte.to_numpy(), pred)]
    return {
        "MAE": float(mean_absolute_error(yte, pred)),
        "RMSE": float(math.sqrt(mean_squared_error(yte, pred))),
        "R2": float(r2_score(yte, pred)),
        "n_test": int(len(yte)), "predictions": rows,
    }


def main() -> int:
    print("main() -> int")
    a = args_parser()
    out_path = Path(a.output).expanduser().resolve()
    if not 0 < a.test_size < 1:
        raise ValueError("--test-size must be in (0,1)")
    if a.cv_folds < 2:
        raise ValueError("--cv-folds must be >=2")

    iris = load_iris(as_frame=True)
    df, X, y, features, data_info = prepare_data(iris.frame, a.target, a.features)
    if len(X) < max(30, a.cv_folds * 5):
        raise ValueError("Dataset is too small for this experiment.")

    idx_train, idx_test = train_test_split(np.arange(len(X)), test_size=a.test_size,
                                            random_state=a.random_state, shuffle=True)
    Xtr, Xte = X.iloc[idx_train].copy(), X.iloc[idx_test].copy()
    ytr, yte = y.iloc[idx_train].copy(), y.iloc[idx_test].copy()

    requested = [int(x.strip()) for x in a.degrees.split(",") if x.strip()]
    degrees, safety = select_feasible_degrees(
        Xtr.shape[1], len(Xtr), requested, a.max_poly_features, a.max_estimated_mb
    )
    cv = KFold(a.cv_folds, shuffle=True, random_state=a.random_state)

    complexity = complexity_experiment(Xtr, ytr, degrees, cv, a.n_jobs)
    diagnosis = degree_diagnosis(complexity)
    under_degree = min(degrees)
    over_degree = int(max(degrees, key=lambda d: float(complexity.loc[complexity.degree == d, "gap"].iloc[0])))
    lc = {
        f"degree_{under_degree}": learning_data(Xtr, ytr, under_degree, cv, a.random_state, a.n_jobs),
        f"degree_{over_degree}": learning_data(Xtr, ytr, over_degree, cv, a.random_state, a.n_jobs),
    }

    reg_degree = max(degrees)
    reg, best = regularization_experiment(
        Xtr, ytr, reg_degree, cv, DEFAULT_ALPHAS, DEFAULT_L1_RATIOS,
        a.lasso_max_iter, a.elasticnet_max_iter, a.tol, a.n_jobs
    )
    coefs = fit_selected_coeffs(Xtr, ytr, reg_degree, best,
                                a.lasso_max_iter, a.elasticnet_max_iter, a.tol)
    paths = {
        "Ridge": regularization_path(Xtr, ytr, reg_degree, "Ridge", DEFAULT_ALPHAS,
                                      a.lasso_max_iter, a.elasticnet_max_iter, a.tol),
        "Lasso": regularization_path(Xtr, ytr, reg_degree, "Lasso", DEFAULT_ALPHAS,
                                      a.lasso_max_iter, a.elasticnet_max_iter, a.tol),
    }
    stab = stability(Xtr, ytr, reg_degree, DEFAULT_ALPHAS, DEFAULT_L1_RATIOS,
                     a.lasso_max_iter, a.elasticnet_max_iter, a.tol, a.n_jobs)

    selection = final_selection(complexity, diagnosis, best, a.selection_rule)
    chosen = selection["chosen_candidate"]
    final_model = make_pipeline(
        int(chosen["degree"]),
        estimator(chosen["model"], chosen.get("alpha"), chosen.get("l1_ratio"),
                  a.lasso_max_iter, a.elasticnet_max_iter, a.tol)
    )
    test = test_eval(final_model, Xtr, ytr, Xte, yte, df.index[idx_test].tolist())
    cv_test = {
        "selected_model_cv_rmse": float(chosen["cv_rmse"]),
        "test_rmse": float(test["RMSE"]),
        "test_minus_cv_rmse": float(test["RMSE"] - chosen["cv_rmse"]),
        "absolute_difference": float(abs(test["RMSE"] - chosen["cv_rmse"])),
        "relative_difference": float(abs(test["RMSE"] - chosen["cv_rmse"]) / max(abs(chosen["cv_rmse"]), 1e-12)),
    }

    result = {
        "metadata": {
            "script": "pr03_experiment_json.py",
            "test_set_used_for_selection": False,
            "test_evaluated_only_after_selection": True,
            "selection_rule": a.selection_rule,
        },
        "hypotheses": [
            {"id": "H1", "statement": "Increasing polynomial degree should not increase training RMSE, while CV RMSE may eventually stop improving.",
             "measures": ["train_rmse", "cv_rmse", "gap", "cv_std", "cv_se"]},
            {"id": "H2", "statement": "At sufficiently high complexity, training RMSE may keep decreasing while CV RMSE and the generalization gap increase.",
             "measures": ["train_rmse", "cv_rmse", "gap", "cv_std"]},
            {"id": "H3", "statement": "Moderate regularization should reduce instability of a complex polynomial model and may improve CV RMSE and the generalization gap.",
             "measures": ["cv_rmse", "cv_std", "gap", "l1_norm", "l2_norm", "n_nonzero"]},
            {"id": "H4", "statement": "Excessively large alpha should eventually increase both training and CV error, indicating underfitting from over-regularization.",
             "measures": ["train_rmse", "cv_rmse", "gap"]},
        ],
        "experimental_protocol": {
            "test_size": float(a.test_size), "random_state": int(a.random_state),
            "main_cv": {"type": "KFold", "n_splits": a.cv_folds, "shuffle": True, "random_state": a.random_state},
            "primary_metric": "RMSE (minimize)",
            "pipeline": ["SimpleImputer(strategy='median')", "PolynomialFeatures(degree, include_bias=False)", "StandardScaler()", "model"],
            "degree_grid_requested": requested, "degree_grid_used": degrees, "regularization_degree": reg_degree,
            "alpha_grid": [float(x) for x in DEFAULT_ALPHAS], "l1_ratio_grid": DEFAULT_L1_RATIOS,
            "search_budget": {
                "complexity_configs": len(degrees), "Ridge_configs": len(DEFAULT_ALPHAS),
                "Lasso_configs": len(DEFAULT_ALPHAS), "ElasticNet_configs": len(DEFAULT_ALPHAS) * len(DEFAULT_L1_RATIOS),
                "stability_seeds": STABILITY_SEEDS,
            },
        },
        "dataset": {
            **data_info, "n_rows": int(len(X)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
            "n_features": int(Xtr.shape[1]), "target_train_mean": float(ytr.mean()), "target_train_std": float(ytr.std(ddof=1)),
            "target_test_mean": float(yte.mean()), "target_test_std": float(yte.std(ddof=1)),
            "train_row_indices": safe(df.index[idx_train].tolist()), "test_row_indices": safe(df.index[idx_test].tolist()),
        },
        "polynomial_feature_safety": {"candidate_levels": safety, "formula": "C(p+d,d)-1",
                                      "max_poly_features": a.max_poly_features, "max_estimated_mb": a.max_estimated_mb},
        "complexity_results": complexity.to_dict(orient="records"),
        "complexity_selection_and_diagnosis": diagnosis,
        "learning_curves": lc,
        "regularization_results": reg.to_dict(orient="records"),
        "best_regularized_configurations": best,
        "selected_coefficients": coefs,
        "regularization_paths": paths,
        "stability": stab,
        "final_model_comparison": selection["candidates"],
        "final_selection": selection,
        "final_test_evaluation": test,
        "cv_vs_test": cv_test,
        "report_answer_data": {
            "hypotheses_tested": ["H1", "H2", "H3", "H4"],
            "best_unregularized_degree": diagnosis["best_degree_by_min_cv_rmse"],
            "one_se_simplest_degree": diagnosis["one_se_simplest_degree"],
            "regularization_degree": reg_degree,
            "best_Ridge": best["Ridge"], "best_Lasso": best["Lasso"], "best_ElasticNet": best["ElasticNet"],
            "final_model": chosen, "test_metrics": test, "cv_vs_test": cv_test,
            "methodological_limitations": [
                "The same held-out test set from the individual dataset is used for final control evaluation; it is not a completely new independent sample.",
                "CV stability is checked on five shuffled KFold schemes, but the exact selected alpha can still vary between splits.",
                "Polynomial feature count grows combinatorially, so an explicit safety cap is enforced.",
                "Automatic under/overfitting flags are heuristics and must be interpreted using the full curves and CV variability.",
                "Non-numeric predictors are excluded by default; use --features and a justified preprocessing design when categorical predictors are essential.",
                "Alpha is meaningful only within this preprocessing/model pipeline and should not be interpreted independently of scaling and degree.",
            ],
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(safe(result), f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")
    print(f"Final model: {chosen['candidate']}")
    print(f"Test MAE={test['MAE']:.6f}, RMSE={test['RMSE']:.6f}, R2={test['R2']:.6f}")
    return 0


if __name__ == "__main__":
    main()
