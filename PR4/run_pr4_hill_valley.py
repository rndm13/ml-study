#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from policy import DecisionPolicy

SEED = 42
N_SPLITS = 5
CAPACITY = 0.15
BOOTSTRAP_REPEATS = 250
MAIN_SCENARIO_NAME = "release_candidate"
SCENARIOS = [
    {"name": "symmetric", "label": "Symmetric errors", "c_fp": 1.0, "c_fn": 1.0,
     "interpretation": "Control scenario: both error types have equal relative cost."},
    {"name": "release_candidate", "label": "Release candidate", "c_fp": 1.0, "c_fn": 6.0,
     "interpretation": "Scenario assumption: missing a Hill is six times as costly as reviewing a Valley."},
    {"name": "critical", "label": "Critical component", "c_fp": 2.0, "c_fn": 20.0,
     "interpretation": "Stress scenario: FN has severe consequences; costs are relative penalty units."},
]


def to_builtin(value: Any) -> Any:
    """Convert NumPy/pandas objects and non-finite floats to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int)):
        return value
    return value


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def compute_threshold_grid(y_true: np.ndarray, scores: np.ndarray,
                           thresholds: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized binary threshold metrics for prediction rule score >= threshold."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    t = np.asarray(thresholds, dtype=float)
    pred = s[:, None] >= t[None, :]
    pos = y == 1
    neg = ~pos
    tp = (pred & pos[:, None]).sum(axis=0).astype(int)
    fp = (pred & neg[:, None]).sum(axis=0).astype(int)
    fn = ((~pred) & pos[:, None]).sum(axis=0).astype(int)
    tn = ((~pred) & neg[:, None]).sum(axis=0).astype(int)
    n = len(y)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    q = (tp + fp) / n
    accuracy = (tp + tn) / n
    return {"threshold": t, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "precision": precision, "recall": recall, "f1": f1,
            "alert_rate": q, "actions_per_1000": q * 1000,
            "accuracy": accuracy}


def cost_array(metrics: dict[str, np.ndarray], c_fp: float, c_fn: float) -> np.ndarray:
    return (c_fp * metrics["fp"] + c_fn * metrics["fn"]) / (metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"])


def grid_records(y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    m = compute_threshold_grid(y, scores, thresholds)
    df = pd.DataFrame({k: v for k, v in m.items()})
    scenario_best: dict[str, dict[str, Any]] = {}
    for s in SCENARIOS:
        c = cost_array(m, s["c_fp"], s["c_fn"])
        df[f"cost_{s['name']}"] = c
        free_ix = min(range(len(thresholds)), key=lambda i: (float(c[i]), float(m['alert_rate'][i]), -float(thresholds[i])))
        feasible = np.flatnonzero(m["alert_rate"] <= CAPACITY + 1e-12)
        if len(feasible) == 0:
            cap_ix = None
        else:
            cap_ix = min(feasible, key=lambda i: (float(c[i]), float(m['alert_rate'][i]), -float(thresholds[i])))
        theoretical_probability = s["c_fp"] / (s["c_fp"] + s["c_fn"])
        scenario_best[s["name"]] = {
            "scenario": s,
            "theoretical_probability_threshold_if_calibrated": theoretical_probability,
            "theoretical_threshold_on_calibrated_logit": float(math.log(s["c_fp"] / s["c_fn"])),
            "unconstrained_empirical_minimum": record_at_index(m, c, free_ix),
            "capacity_constrained_minimum": record_at_index(m, c, cap_ix) if cap_ix is not None else None,
        }
    return df, scenario_best


def record_at_index(m: dict[str, np.ndarray], costs: np.ndarray, i: int | None) -> dict[str, Any] | None:
    if i is None:
        return None
    return {"threshold_on_calibrated_logit": float(m["threshold"][i]),
            "equivalent_probability_threshold": float(expit(m["threshold"][i])),
            "TN": int(m["tn"][i]), "FP": int(m["fp"][i]),
            "FN": int(m["fn"][i]), "TP": int(m["tp"][i]), "precision": float(m["precision"][i]),
            "recall": float(m["recall"][i]), "F1": float(m["f1"][i]),
            "accuracy": float(m["accuracy"][i]), "alert_rate": float(m["alert_rate"][i]),
            "actions_per_1000": float(m["actions_per_1000"][i]), "mean_cost_per_object": float(costs[i])}


def evaluate_one(y_true: np.ndarray, scores: np.ndarray, threshold: float,
                 c_fp: float, c_fn: float) -> dict[str, Any]:
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = len(y_true)
    return {"threshold_on_calibrated_logit": float(threshold),
            "equivalent_probability_threshold": float(expit(threshold)),
            "threshold": float(threshold), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "F1": float(f1_score(y_true, pred, zero_division=0)),
            "alert_rate": float(pred.mean()), "actions_per_1000": float(pred.mean() * 1000),
            "total_cost": float(c_fp * fp + c_fn * fn),
            "mean_cost_per_object": float((c_fp * fp + c_fn * fn) / n),
            "c_fp": float(c_fp), "c_fn": float(c_fn), "n": int(n)}


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    """Quantile bins; deduplicate boundaries and keep counts for transparency."""
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return [{"count": int(len(y)), "mean_predicted_probability": float(p.mean()),
                 "observed_positive_rate": float(y.mean())}]
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if mask.any():
            out.append({"count": int(mask.sum()), "bin_lower": float(lo), "bin_upper": float(hi),
                        "mean_predicted_probability": float(p[mask].mean()),
                        "observed_positive_rate": float(y[mask].mean())})
    return out


def plot_calibration(y: np.ndarray, p_before: np.ndarray, p_after: np.ndarray, out: Path) -> dict[str, Any]:
    before_bins = calibration_bins(y, p_before)
    after_bins = calibration_bins(y, p_after)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal calibration")
    for label, bins in [("Naive sigmoid of margin (not a probability model)", before_bins),
                         ("Cross-fitted sigmoid calibration", after_bins)]:
        x = [b["mean_predicted_probability"] for b in bins]
        z = [b["observed_positive_rate"] for b in bins]
        ax.plot(x, z, marker="o", label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed positive rate",
           title="Calibration diagnostic on training OOF scores")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return {"naive_sigmoid_before_calibration": before_bins, "cross_fitted_sigmoid_after_calibration": after_bins}


def make_plots(y: np.ndarray, p_before: np.ndarray, p_after: np.ndarray, threshold_df: pd.DataFrame,
               scenarios: dict[str, dict[str, Any]], sensitivity_df: pd.DataFrame,
               zone_df: pd.DataFrame, selected_threshold: float, output_dir: Path) -> dict[str, str]:
    figs = output_dir / "figures"; figs.mkdir(parents=True, exist_ok=True)
    plot_calibration(y, p_before, p_after, figs / "01_calibration.png")

    # Log-odds margins span several orders of magnitude in Hill-Valley. Plot threshold
    # rank instead: 0% means every object is actioned; 100% means nearly none is.
    base = threshold_df.sort_values("threshold")
    selected_match = int(np.argmin(np.abs(base["threshold"].to_numpy() - selected_threshold)))
    selected_percentile = float(base["threshold_percentile"].iloc[selected_match])
    x = base["threshold_percentile"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, base["precision"], label="Precision")
    ax.plot(x, base["recall"], label="Recall")
    ax.axvline(selected_percentile, linestyle="--", label=f"Selected rank={selected_percentile:.1f}%")
    ax.set(xlabel="Threshold percentile in OOF score distribution (higher = fewer actions)", ylabel="Metric",
           title="Precision and Recall vs threshold percentile", xlim=(0, 100), ylim=(0, 1.02))
    ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figs / "02_precision_recall_threshold.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, base["alert_rate"] * 100, label="Objects routed to action (%)")
    ax.axhline(CAPACITY * 100, linestyle=":", label=f"Capacity {CAPACITY:.0%}")
    ax.axvline(selected_percentile, linestyle="--", label=f"Selected rank={selected_percentile:.1f}%")
    ax.set(xlabel="Threshold percentile in OOF score distribution", ylabel="Alert rate (%)",
           title="Operational load vs threshold percentile", xlim=(0, 100))
    ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figs / "03_alert_rate_threshold.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for scenario in SCENARIOS:
        ax.plot(x, base[f"cost_{scenario['name']}"], label=scenario["label"])
        point = scenarios[scenario["name"]]["capacity_constrained_minimum"]
        if point:
            point_x = 100.0 * (1.0 - point["alert_rate"])
            ax.scatter([point_x], [point["mean_cost_per_object"]], s=30)
    ax.axvline(selected_percentile, linestyle="--", label=f"Selected rank={selected_percentile:.1f}%")
    ax.set(xlabel="Threshold percentile in OOF score distribution", ylabel="Mean relative cost per object",
           title="Cost curves across scenarios", xlim=(0, 100))
    ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figs / "04_scenario_costs.png", dpi=160); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sensitivity_df["fn_fp_ratio"], sensitivity_df["threshold_percentile_unconstrained"],
             marker="o", label="Unconstrained threshold percentile")
    ax1.plot(sensitivity_df["fn_fp_ratio"], sensitivity_df["threshold_percentile_capacity"],
             marker="s", linestyle="--", label="Threshold percentile under 15% cap")
    ax1.set_xlabel("Cost ratio cFN / cFP"); ax1.set_ylabel("Threshold percentile (higher = fewer actions)"); ax1.set_ylim(0, 100)
    ax2 = ax1.twinx()
    ax2.plot(sensitivity_df["fn_fp_ratio"], sensitivity_df["alert_rate_capacity"] * 100,
             marker="^", linestyle=":", label="Alert rate under capacity")
    ax2.set_ylabel("Alert rate under capacity (%)")
    ax1.set_title("Sensitivity to relative error cost")
    ax1.grid(True, alpha=.25)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax1.legend(h1+h2, l1+l2, loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(figs / "05_cost_sensitivity.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(zone_df["zone"], zone_df["count"])
    ax.set(ylabel="Number of objects", title="OOF objects by three-zone decision policy")
    ax.grid(True, axis="y", alpha=.25); fig.tight_layout(); fig.savefig(figs / "06_three_zone_policy.png", dpi=160); plt.close(fig)

    return {"calibration": "figures/01_calibration.png",
            "precision_recall_vs_threshold": "figures/02_precision_recall_threshold.png",
            "alert_rate_vs_threshold": "figures/03_alert_rate_threshold.png",
            "cost_curves": "figures/04_scenario_costs.png",
            "cost_sensitivity": "figures/05_cost_sensitivity.png",
            "three_zone_policy": "figures/06_three_zone_policy.png"}
def run_policy_tests(output_dir: Path) -> dict[str, Any]:
    import unittest

    low, high = 0.30, 0.70
    class PolicyTests(unittest.TestCase):
        def setUp(self): self.policy = DecisionPolicy(low, high)
        def test_below_low(self): self.assertEqual(self.policy.decide(0.29), "standard_testing")
        def test_equal_low(self): self.assertEqual(self.policy.decide(0.30), "manual_review")
        def test_between_thresholds(self): self.assertEqual(self.policy.decide(0.50), "manual_review")
        def test_equal_high(self): self.assertEqual(self.policy.decide(0.70), "additional_testing")
        def test_above_high(self): self.assertEqual(self.policy.decide(0.91), "additional_testing")
        def test_invalid_score(self):
            for x in [float("nan"), float("inf"), -float("inf")]:
                with self.subTest(score=x), self.assertRaises(ValueError): self.policy.decide(x)
        def test_reversed_thresholds(self):
            with self.assertRaises(ValueError): DecisionPolicy(0.8, 0.2)
        def test_non_finite_threshold(self):
            with self.assertRaises(ValueError): DecisionPolicy(float("nan"), 0.5)
        def test_batch_order_preserved(self):
            self.assertEqual(self.policy.decide_many([0.8, 0.2, 0.5]),
                             ["additional_testing", "standard_testing", "manual_review"])
        def test_equal_thresholds_go_to_high_action(self):
            self.assertEqual(DecisionPolicy(0.5, 0.5).decide(0.5), "additional_testing")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PolicyTests)
    result_stream = __import__('io').StringIO()
    result = unittest.TextTestRunner(stream=result_stream, verbosity=2).run(suite)
    log = result_stream.getvalue()
    (output_dir / "policy_tests.log").write_text(log, encoding="utf-8")
    return {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "successful": result.wasSuccessful(), "log_file": "policy_tests.log", "log": log}


def load_source(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    oof = sorted(d["out_of_fold_disagreement_analysis"]["all_oof_predictions"], key=lambda r: r["row_index"])
    test = d["final_test_evaluation"]["predictions"]
    y_oof = np.array([int(r["y_true_code"]) for r in oof], dtype=int)
    margin_oof = np.array([float(r["svm_positive_oriented_margin"]) for r in oof], dtype=float)
    y_test = np.array([int(r["y_true_code"]) for r in test], dtype=int)
    margin_test = np.array([float(r["positive_class_score"]) for r in test], dtype=float)
    if len(y_oof) == 0 or len(y_test) == 0:
        raise ValueError("Source JSON must include OOF and final test predictions.")
    if not np.isfinite(margin_oof).all() or not np.isfinite(margin_test).all():
        raise ValueError("Non-finite decision margins found in source JSON.")
    if set(np.unique(y_oof)) != {0, 1} or set(np.unique(y_test)) != {0, 1}:
        raise ValueError("Expected binary class codes 0 and 1 in both OOF and test rows.")
    return y_oof, margin_oof, y_test, margin_test, oof, test, d


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lr3-json", default="input/lr3_report_data.json", help="JSON produced by prior Hill-Valley lab")
    parser.add_argument("--output-dir", default="outputs", help="Directory for JSON, CSV, figures, and policy config")
    args = parser.parse_args()
    source_path = Path(args.source_lr3_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not source_path.exists():
        parser.error(f"Source JSON not found: {source_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    started = time.perf_counter()

    y_oof, margin_oof, y_test, margin_test, oof_rows, test_rows, lr3 = load_source(source_path)
    # Explicitly cross-fit the calibrator on saved OOF margins: each logit/probability
    # is predicted by a calibrator that did not use that row's target.
    calibrator_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    calibrator_template = LogisticRegression(C=1.0, max_iter=3000, random_state=SEED)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibrated_oof_logits = cross_val_predict(calibrator_template, margin_oof.reshape(-1, 1), y_oof,
                                                  cv=calibrator_cv, method="decision_function", n_jobs=None)
        calibrated_oof_probabilities = expit(calibrated_oof_logits)
        calibrator_final = LogisticRegression(C=1.0, max_iter=3000, random_state=SEED)
        calibrator_final.fit(margin_oof.reshape(-1, 1), y_oof)
        calibrated_test_logits = calibrator_final.decision_function(margin_test.reshape(-1, 1))
        calibrated_test_probabilities = expit(calibrated_test_logits)
        warning_messages = [str(w.message) for w in caught]

    # This is an explicitly labelled diagnostic transform only. decision_function
    # margins are NOT probabilities; Brier score is only descriptive for this map.
    naive_oof = expit(margin_oof)
    naive_test = expit(margin_test)
    brier_naive_oof = float(brier_score_loss(y_oof, naive_oof))
    brier_cal_oof = float(brier_score_loss(y_oof, calibrated_oof_probabilities))
    brier_naive_test = float(brier_score_loss(y_test, naive_test))
    brier_cal_test = float(brier_score_loss(y_test, calibrated_test_probabilities))

    # The threshold grid covers the observed log-odds range and adds every training
    # OOF score so exact-boundary candidates are explicit. Test scores are not used.
    # Decision thresholds operate on the calibrator's continuous log-odds output.
    # This is mathematically equivalent to thresholds on sigmoid probabilities, but
    # avoids floating-point saturation (many probabilities can round to exactly 1.0).
    score_quantiles = np.quantile(calibrated_oof_logits, np.linspace(0.0, 1.0, 101))
    thresholds = np.unique(np.concatenate([score_quantiles, calibrated_oof_logits, np.array([0.0])]))
    threshold_df, scenario_best = grid_records(y_oof, calibrated_oof_logits, thresholds)
    threshold_df["equivalent_probability_threshold"] = expit(threshold_df["threshold"].to_numpy())
    threshold_df["threshold_percentile"] = 100.0 * (1.0 - threshold_df["alert_rate"])
    threshold_df.to_csv(output_dir / "tables" / "threshold_grid.csv", index=False, float_format="%.10g")

    # Main scenario threshold is frozen before computing any test metric.
    main_best = scenario_best[MAIN_SCENARIO_NAME]["capacity_constrained_minimum"]
    if main_best is None:
        raise RuntimeError("No threshold satisfies the chosen alert capacity; inspect the tie handling and score grid.")
    selected_threshold = float(main_best["threshold_on_calibrated_logit"])
    baseline_oof = evaluate_one(y_oof, calibrated_oof_logits, 0.0, 1.0, 6.0)
    selected_oof = evaluate_one(y_oof, calibrated_oof_logits, selected_threshold, 1.0, 6.0)
    unconstrained_oof = scenario_best[MAIN_SCENARIO_NAME]["unconstrained_empirical_minimum"]

    # Policy: 15% prioritized review, subdivided into 7% manual and 8% additional
    # testing. Quantiles are based only on OOF scores; ties are handled by >=.
    t_low = float(np.quantile(calibrated_oof_logits, 0.85))
    t_high = float(np.quantile(calibrated_oof_logits, 0.92))
    if t_low > t_high:
        t_low, t_high = t_high, t_low
    zone_policy = DecisionPolicy(t_low, t_high)
    zone_actions = np.array(zone_policy.decide_many(calibrated_oof_logits), dtype=object)
    zone_records = []
    for label, act in [("standard_testing", "standard_testing"), ("manual_review", "manual_review"), ("additional_testing", "additional_testing")]:
        mask = zone_actions == act
        zone_records.append({"zone": label, "count": int(mask.sum()), "share": float(mask.mean()),
                             "mean_score_logit": float(calibrated_oof_logits[mask].mean()) if mask.any() else None,
                             "mean_predicted_probability": float(calibrated_oof_probabilities[mask].mean()) if mask.any() else None,
                             "observed_positive_rate": float(y_oof[mask].mean()) if mask.any() else None,
                             "positive_count": int(y_oof[mask].sum()), "negative_count": int((1-y_oof[mask]).sum()),
                             "operational_interpretation": {"standard_testing": "No additional action; default path.",
                                "manual_review": "Human/secondary review; effectiveness and cost are not assumed.",
                                "additional_testing": "Highest-priority additional inspection."}[label]})
    zone_df = pd.DataFrame(zone_records)
    zone_df.to_csv(output_dir / "tables" / "three_zone_policy.csv", index=False)
    # Overall priority action for three-zone policy is score >= t_low; no credit is
    # assumed for what manual review may resolve.
    tri_binary = evaluate_one(y_oof, calibrated_oof_logits, t_low, 1.0, 6.0)

    # Main sensitivity across cost ratios. For capacity-limited thresholds, choose
    # the min cost among thresholds with OOF alert rate <=15%.
    sensitivity = []
    m_base = compute_threshold_grid(y_oof, calibrated_oof_logits, thresholds)
    for ratio in [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0, 24.0]:
        costs = cost_array(m_base, 1.0, ratio)
        feasible = np.flatnonzero(m_base["alert_rate"] <= CAPACITY + 1e-12)
        idx = min(feasible, key=lambda i: (float(costs[i]), float(m_base["alert_rate"][i]), -float(thresholds[i])))
        idx_free = min(range(len(thresholds)), key=lambda i: (float(costs[i]), float(m_base["alert_rate"][i]), -float(thresholds[i])))
        sensitivity.append({"fn_fp_ratio": ratio, "theoretical_probability_threshold": 1.0/(1.0+ratio),
                            "theoretical_logit_threshold": float(math.log(1.0/ratio)),
                            "threshold_unconstrained_logit": float(thresholds[idx_free]),
                            "threshold_unconstrained_probability_equivalent": float(expit(thresholds[idx_free])),
                            "threshold_percentile_unconstrained": float(100.0 * (1.0 - m_base["alert_rate"][idx_free])),
                            "alert_rate_unconstrained": float(m_base["alert_rate"][idx_free]),
                            "mean_cost_unconstrained": float(costs[idx_free]),
                            "selected_threshold_capacity_logit": float(thresholds[idx]),
                            "selected_threshold_capacity_probability_equivalent": float(expit(thresholds[idx])),
                            "threshold_percentile_capacity": float(100.0 * (1.0 - m_base["alert_rate"][idx])),
                            "alert_rate_capacity": float(m_base["alert_rate"][idx]),
                            "mean_cost_capacity": float(costs[idx]),
                            "precision_capacity": float(m_base["precision"][idx]),
                            "recall_capacity": float(m_base["recall"][idx])})
    sensitivity_df = pd.DataFrame(sensitivity)
    sensitivity_df.to_csv(output_dir / "tables" / "cost_sensitivity.csv", index=False, float_format="%.10g")

    # Bootstrap stability on training OOF rows only.
    rng = np.random.default_rng(SEED)
    boot_thresholds, boot_rates, boot_costs = [], [], []
    n_oof = len(y_oof)
    for _ in range(BOOTSTRAP_REPEATS):
        ix = rng.integers(0, n_oof, size=n_oof)
        yb, pb = y_oof[ix], calibrated_oof_logits[ix]
        mb = compute_threshold_grid(yb, pb, thresholds)
        cb = cost_array(mb, 1.0, 6.0)
        feasible = np.flatnonzero(mb["alert_rate"] <= CAPACITY + 1e-12)
        if len(feasible) == 0:
            continue
        bi = min(feasible, key=lambda i: (float(cb[i]), float(mb["alert_rate"][i]), -float(thresholds[i])))
        boot_thresholds.append(float(thresholds[bi])); boot_rates.append(float(mb["alert_rate"][bi])); boot_costs.append(float(cb[bi]))
    bootstrap = {"method": "nonparametric bootstrap over fixed training OOF (y, calibrated score) pairs",
                 "repeats_requested": BOOTSTRAP_REPEATS, "repeats_successful": len(boot_thresholds),
                 "random_state": SEED,
                 "selected_threshold_percentiles_5_50_95": [float(x) for x in np.percentile(boot_thresholds, [5,50,95])] if boot_thresholds else [],
                 "selected_threshold_median": float(np.median(boot_thresholds)) if boot_thresholds else None,
                 "selected_threshold_probability_equivalent_percentiles_5_50_95": [float(x) for x in expit(np.percentile(boot_thresholds, [5,50,95]))] if boot_thresholds else [],
                 "alert_rate_percentiles_5_50_95": [float(x) for x in np.percentile(boot_rates, [5,50,95])] if boot_rates else [],
                 "mean_cost_percentiles_5_50_95": [float(x) for x in np.percentile(boot_costs, [5,50,95])] if boot_costs else [],
                 "threshold_samples": boot_thresholds}

    # Freeze policy file before the code computes any test-set metrics.
    policy_config = {
        "policy_id": "hill_valley_linear_svm_cost_policy",
        "version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Hill-Valley without noise; fixed training/test split from LR3 artifact",
        "model": {"name": "LinearSVM", "C": 100, "scaler": "none", "score_semantics": "positive-class decision margin from frozen LR3 model; transformed by calibration model to log-odds"},
        "calibration": {"method": "sigmoid calibration by LogisticRegression on saved OOF margins",
                        "calibration_fold_count": N_SPLITS, "calibration_random_state": SEED,
                        "mapping": {"formula": "calibrated_logit = coefficient * decision_margin + intercept",
                                    "coefficient": float(calibrator_final.coef_[0, 0]),
                                    "intercept": float(calibrator_final.intercept_[0])},
                        "score_for_policy": "calibrated log-odds score; probability is expit(logit)",
                        "implementation_note": "Apply the policy threshold to calibrated_logit, not its float probability, which can round to exactly 1.0 for large margins."},
        "positive_class": {"code": 1, "meaning": "Hill"},
        "negative_class": {"code": 0, "meaning": "Valley"},
        "chosen_policy": {"type": "single_threshold", "threshold_on_calibrated_logit": selected_threshold,
                           "equivalent_probability_threshold_float": float(expit(selected_threshold)),
                           "threshold_probability_numerical_note": "Probability equivalent may round to 1.0 in IEEE float; apply threshold to calibrated_logit instead.",
                           "rule": "calibrated_logit >= threshold_on_calibrated_logit means prioritize for additional review/testing",
                           "cost_scenario": {"c_fp": 1.0, "c_fn": 6.0, "unit": "hypothetical relative penalty units"},
                           "capacity_target_share": CAPACITY, "capacity_is_soft_planning_target": True,
                           "selection_source": "OOF scores only"},
        "three_zone_alternative": {"type": "three_zone", "threshold_low_logit": t_low, "threshold_high_logit": t_high,
                                   "threshold_low_probability_equivalent": float(expit(t_low)),
                                   "threshold_high_probability_equivalent": float(expit(t_high)),
                                   "rule": "logit < low => standard_testing; low <= logit < high => manual_review; logit >= high => additional_testing",
                                   "target_prioritized_share": CAPACITY,
                                   "manual_review_effectiveness_assumed": False},
        "reproducibility": {"random_state": SEED, "source_lr3_json": str(source_path),
                            "test_data_used_for_threshold_selection": False}
    }
    (output_dir / "policy_config.json").write_text(json.dumps(to_builtin(policy_config), ensure_ascii=False, indent=2), encoding="utf-8")

    # FINAL evaluation after policy selection/freeze: use test scores now.
    test_single = evaluate_one(y_test, calibrated_test_logits, selected_threshold, 1.0, 6.0)
    test_baseline = evaluate_one(y_test, calibrated_test_logits, 0.0, 1.0, 6.0)
    test_auc = float(roc_auc_score(y_test, calibrated_test_probabilities))
    test_calibration = {"brier_naive_sigmoid_of_margin_descriptive_only": brier_naive_test,
                        "brier_cross_fitted_calibrated_oof": brier_cal_oof,
                        "brier_test_calibrated": brier_cal_test,
                        "roc_auc_test_calibrated_scores": test_auc}
    # A hard capacity is applied by ranking, with stable tie handling; y_test is not
    # used to construct this list, only to evaluate it after ranking.
    k = int(math.floor(CAPACITY * len(y_test)))
    order = np.argsort(-calibrated_test_logits, kind="stable")
    topk_pred = np.zeros(len(y_test), dtype=int)
    if k > 0: topk_pred[order[:k]] = 1
    tn, fp, fn, tp = confusion_matrix(y_test, topk_pred, labels=[0,1]).ravel()
    topk_result = {"capacity_fraction": CAPACITY, "k": k, "test_count": int(len(y_test)),
                   "actual_alert_rate": float(topk_pred.mean()), "tie_break": "stable input order for equal scores",
                   "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                   "precision": float(precision_score(y_test, topk_pred, zero_division=0)),
                   "recall": float(recall_score(y_test, topk_pred, zero_division=0)),
                   "F1": float(f1_score(y_test, topk_pred, zero_division=0)),
                   "mean_cost_per_object": float((fp + 6*fn)/len(y_test)),
                   "note": "Use this when 15% is a hard quota; a fixed threshold does not guarantee the same fraction on new batches."}

    # Detailed row files for reconstructing every plot/table in the report.
    oof_df = pd.DataFrame({"object_id": [int(r["row_index"]) for r in oof_rows],
                           "y_true": y_oof, "fold": [int(r["fold"]) for r in oof_rows],
                           "decision_margin": margin_oof, "naive_sigmoid_diagnostic": naive_oof,
                           "calibrated_logit_oof": calibrated_oof_logits,
                           "calibrated_probability_oof": calibrated_oof_probabilities,
                           "prediction_t_0_5": (calibrated_oof_logits >= 0.0).astype(int),
                           "prediction_selected_threshold": (calibrated_oof_logits >= selected_threshold).astype(int),
                           "three_zone_action": zone_actions})
    oof_df.to_csv(output_dir / "tables" / "oof_scores.csv", index=False, float_format="%.10g")
    test_df = pd.DataFrame({"object_id": [int(r.get("row_index", i)) for i, r in enumerate(test_rows)],
                            "y_true": y_test, "decision_margin": margin_test,
                            "naive_sigmoid_diagnostic": naive_test, "calibrated_logit": calibrated_test_logits,
                            "calibrated_probability": calibrated_test_probabilities,
                            "prediction_t_0_5": (calibrated_test_logits >= 0.0).astype(int),
                            "prediction_selected_threshold": (calibrated_test_logits >= selected_threshold).astype(int),
                            "prediction_top_k_15pct": topk_pred})
    test_df.to_csv(output_dir / "tables" / "test_scores.csv", index=False, float_format="%.10g")

    calibration_plots = make_plots(y_oof, naive_oof, calibrated_oof_probabilities, threshold_df, scenario_best,
                                   sensitivity_df, zone_df, selected_threshold, output_dir)
    policy_tests = run_policy_tests(output_dir)

    # Source metrics and methodological evidence.
    class_counts_train = {str(int(k)): int(v) for k, v in zip(*np.unique(y_oof, return_counts=True))}
    class_counts_test = {str(int(k)): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))}
    main_scenario = next(s for s in SCENARIOS if s["name"] == MAIN_SCENARIO_NAME)
    report_data = {
        "metadata": {"script": "run_pr4_hill_valley.py", "practical_work": "Practical Work No. 4 - Thresholds, error costs and decision-making in classification",
                     "generated_at_utc": datetime.now(timezone.utc).isoformat(), "python_version": sys.version,
                     "platform": platform.platform(), "numpy_version": np.__version__, "pandas_version": pd.__version__,
                     "scikit_learn_version": sklearn.__version__, "source_lr3_json": str(source_path),
                     "output_dir": str(output_dir), "elapsed_seconds": float(time.perf_counter()-started),
                     "warnings": warning_messages},
        "dataset": {"name": "Hill-Valley without noise", "source_dataset_split": "training file from previous LR3 workflow, then stratified 80/20 split",
                    "row_count": int(len(y_oof)+len(y_test)), "feature_count": 100,
                    "target": "class", "class_mapping": {"0": "Valley", "1": "Hill"},
                    "positive_class": 1, "train_oof_count": int(len(y_oof)), "test_count": int(len(y_test)),
                    "train_oof_class_counts": class_counts_train, "test_class_counts": class_counts_test,
                    "train_oof_fold_counts": {str(k): int(v) for k,v in pd.Series([int(r["fold"]) for r in oof_rows]).value_counts().sort_index().items()},
                    "missing_values": 0, "data_source_note": "Hill-Valley record encodes 100 ordered numeric coordinates; 0=Valley and 1=Hill."},
        "decision_context": {"positive_action": "prioritize a sequence predicted as Hill for an additional pattern review/inspection",
                             "negative_action": "leave a sequence predicted as Valley in the standard processing path",
                             "false_positive_consequence": "a Valley is unnecessarily routed to review; consumes review capacity (hypothetical scenario cost)",
                             "false_negative_consequence": "a Hill is not prioritized; a relevant hill pattern may be missed (hypothetical scenario cost)",
                             "responsible_component": "classification decision-policy component; a human/secondary checker executes manual review",
                             "feature_availability": "100 ordered numeric points are available when the sequence is classified",
                             "ground_truth_delay": "not supplied by dataset; assumed available after independent review/annotation",
                             "cost_units": "relative penalty units, not monetary or observed real-world costs",
                             "cost_assumption_source": "scenario assumptions for educational analysis; must be replaced by domain-owner estimates for deployment"},
        "fixed_model_and_protocol": {"model": "LinearSVM", "C": 100, "scaler": "none",
                                     "score_source": "saved positive-class-oriented decision margins from LR3 JSON",
                                     "hyperparameters_retuned": False, "random_state": SEED, "outer_oof_folds": N_SPLITS,
                                     "calibrator": "LogisticRegression(C=1) sigmoid mapping applied to one-dimensional decision margin; use its decision_function logit to avoid probability saturation",
                                     "calibrator_oof_strategy": "cross_val_predict(method=decision_function) over saved training OOF margins, 5 StratifiedKFold splits with shuffle=True and random_state=42; sigmoid probabilities computed with expit",
                                     "fitted_calibrator_mapping": {"formula": "logit = coefficient * decision_margin + intercept",
                                                                    "coefficient": float(calibrator_final.coef_[0, 0]),
                                                                    "intercept": float(calibrator_final.intercept_[0])},
                                     "test_set_used_for_threshold_selection": False,
                                     "test_set_history_caveat": "These 122 held-out examples were already evaluated in the prior LR3 output. This script does not tune policy using their labels, but results are not a pristine never-before-seen holdout across both assignments."},
        "calibration": {"oof_brier_naive_sigmoid_margin_diagnostic_only": brier_naive_oof,
                        "oof_brier_cross_fitted_sigmoid_calibration": brier_cal_oof,
                        "test_brier_naive_sigmoid_margin_diagnostic_only": brier_naive_test,
                        "test_brier_calibrated": brier_cal_test,
                        "test_roc_auc_calibrated": test_auc,
                        "note": "Linear SVM decision_function margins are not probabilities. expit(margin) is included only as a naive, unfitted diagnostic baseline; it is not native predict_proba. Calibrated probabilities are expit(calibrated_logit), but threshold selection and policy use the continuous calibrated_logit to avoid floating-point ties near 0/1.",
                        "oof_calibration_bins": {"naive_sigmoid_diagnostic": calibration_bins(y_oof, naive_oof),
                                                 "cross_fitted_calibration": calibration_bins(y_oof, calibrated_oof_probabilities)}},
        "threshold_analysis": {"rule": "positive if calibrated_logit >= threshold; p=expit(calibrated_logit)",
                                "numerical_note": "Thresholds are optimized and applied in log-odds space because calibrated probabilities can round to exact 0 or 1 for extreme SVM margins.",
                                "score_semantics": "continuous calibrated log-odds, not bounded probability",
                                "threshold_grid_size": int(len(thresholds)),
                                "threshold_range_logit": [float(thresholds.min()), float(thresholds.max())],
                                "threshold_range_equivalent_probability": [float(expit(thresholds.min())), float(expit(thresholds.max()))],
                                "grid_includes_all_oof_scores": True, "capacity_target_share": CAPACITY,
                                "baseline_probability_0_5_equivalent_oof": baseline_oof, "main_scenario": main_scenario,
                                "unconstrained_minimum_main_scenario_oof": unconstrained_oof,
                                "selected_capacity_constrained_oof": selected_oof,
                                "scenarios": scenario_best,
                                "five_seven_working_points": [baseline_oof,
                                     evaluate_one(y_oof, calibrated_oof_logits, float(np.quantile(calibrated_oof_logits, .50)),1,6),
                                     evaluate_one(y_oof, calibrated_oof_logits, float(np.quantile(calibrated_oof_logits, .70)),1,6),
                                     evaluate_one(y_oof, calibrated_oof_logits, float(np.quantile(calibrated_oof_logits, .85)),1,6),
                                     evaluate_one(y_oof, calibrated_oof_logits, float(np.quantile(calibrated_oof_logits, .92)),1,6),
                                     selected_oof,
                                     evaluate_one(y_oof, calibrated_oof_logits, float(np.max(calibrated_oof_logits)),1,6)]},
        "sensitivity_analysis": {"cost_ratio_table": sensitivity,
                                  "bootstrap_stability": bootstrap,
                                  "interpretation": "When cFN/cFP increases, the unconstrained solution tends to lower the threshold and grow the queue; a capacity constraint can prevent that response."},
        "resource_constraint": {"capacity_target": CAPACITY, "interpretation": "A 15% soft planning cap during threshold selection; fixed threshold does not guarantee a hard quota on future batches.",
                                "test_fixed_threshold": test_single,
                                "test_top_k_hard_quota": topk_result},
        "three_zone_policy": {"threshold_low_logit": t_low, "threshold_high_logit": t_high,
                              "threshold_low_probability_equivalent": float(expit(t_low)),
                              "threshold_high_probability_equivalent": float(expit(t_high)),
                              "zone_metrics_oof": zone_records,
                              "positive_action_equivalent": tri_binary,
                              "comparison_note": "Manual review effectiveness/cost are unknown and are not credited as correcting labels; three-zone policy changes routing priority, not the measured classifier ground truth."},
        "final_test_evaluation_after_policy_freeze": {"selected_threshold": test_single, "baseline_threshold_0_5": test_baseline,
                                                       "probability_metrics": test_calibration,
                                                       "test_confusion_matrix_selected_threshold": {"TN": test_single["TN"], "FP": test_single["FP"], "FN": test_single["FN"], "TP": test_single["TP"]},
                                                       "comparison_to_oof": {"oof_mean_cost": selected_oof["mean_cost_per_object"], "test_mean_cost": test_single["mean_cost_per_object"],
                                                                             "oof_alert_rate": selected_oof["alert_rate"], "test_alert_rate": test_single["alert_rate"]}},
        "policy_configuration": policy_config,
        "automated_tests": policy_tests,
        "figures": calibration_plots,
        "output_files": {"main_json": "report_data_pr4.json", "policy_config": "policy_config.json",
                         "oof_scores_csv": "tables/oof_scores.csv", "test_scores_csv": "tables/test_scores.csv",
                         "threshold_grid_csv": "tables/threshold_grid.csv", "sensitivity_csv": "tables/cost_sensitivity.csv",
                         "three_zone_csv": "tables/three_zone_policy.csv", "policy_tests_log": "policy_tests.log",
                         "figures": list(calibration_plots.values())},
        "limitations": [
            "Costs and the 15% capacity target are explicit educational assumptions, not measured business costs or validated operational capacity.",
            "The raw Hill-Valley data file was not attached in this chat; the script processes the saved OOF/test scores in the previous LR3 report_data.json, not raw coordinates.",
            "The held-out predictions and labels are reused from LR3; policy thresholds are selected only on OOF outputs, but the test evaluation is not a pristine unseen holdout across both assignments.",
            "The naive sigmoid(margin) is not a probability model and its Brier score is only a diagnostic contrast; calibrated probabilities come from the cross-fitted LogisticRegression mapping. Operational thresholds are applied to calibrated log-odds scores to avoid floating-point ties in probabilities near 0 or 1.",
            "A fixed threshold may breach a hard per-batch capacity constraint; use top-k ranking with deterministic tie handling when the quota is strict.",
            "Manual review effectiveness and labor cost are unknown; no improvement is imputed for routed objects without outcome data."
        ],
        "reproduction": {"command": "python run_pr4_hill_valley.py --source-lr3-json input/lr3_report_data.json --output-dir outputs",
                         "dependencies_file": "requirements.txt", "policy_tests_command": "python -m unittest -v test_decision_policy.py",
                         "note": "The main script also runs and records 10 inline policy tests; the separate test file is provided for external validation."}
    }
    report_path = output_dir / "report_data_pr4.json"
    report_path.write_text(json.dumps(to_builtin(report_data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    # Separate portable policy module/test file are provided in the project root;
    # policy tests are also recorded above to keep the JSON self-contained.
    print(json.dumps({"status": "ok", "report_json": str(report_path), "threshold_on_calibrated_logit": selected_threshold,
                      "equivalent_probability_threshold": float(expit(selected_threshold)), "oof_n": len(y_oof), "test_n": len(y_test), "main_oof_cost": selected_oof["mean_cost_per_object"],
                      "main_test_cost": test_single["mean_cost_per_object"], "policy_tests": policy_tests["tests_run"],
                      "policy_tests_passed": policy_tests["successful"], "elapsed_seconds": time.perf_counter()-started}, ensure_ascii=False, indent=2))
    return 0 if policy_tests["successful"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
