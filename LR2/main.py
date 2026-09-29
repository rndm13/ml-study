from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import load_hill_valley

CLASS_NAMES = {0: "Valley", 1: "Hill"}
RANDOM_STATE = 42
THRESHOLDS = [0.3, 0.5, 0.7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Навчання та повний аналіз бінарних класифікаторів Hill-Valley."
    )
    parser.add_argument("--data-file", default=None, help="Шлях до Hill_Valley_without_noise_Training.data.")
    parser.add_argument("--output-dir", default="results", help="Каталог для результатів.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Частка test set.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def scoring() -> dict[str, object]:
    return {
        "Accuracy": "accuracy",
        "Precision": make_scorer(precision_score, zero_division=0),
        "Recall": make_scorer(recall_score, zero_division=0),
        "F1": make_scorer(f1_score, zero_division=0),
        "ROC-AUC": "roc_auc",
    }


def evaluate_cv(models: dict[str, object], X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> tuple[pd.DataFrame, dict[str, object]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scorers = scoring()
    rows: list[dict[str, object]] = []
    raw_results: dict[str, object] = {}

    for name, model in models.items():
        result = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scorers,
            return_train_score=False,
            error_score="raise",
        )
        raw_results[name] = result
        row: dict[str, object] = {"Model": name}
        for metric in scorers:
            values = result[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        rows.append(row)

    return pd.DataFrame(rows), raw_results


def choose_model(cv_table: pd.DataFrame) -> str:
    # Primary criterion: F1, then ROC-AUC, then Accuracy.
    ranked = cv_table.sort_values(
        by=["F1_mean", "ROC-AUC_mean", "Accuracy_mean"],
        ascending=False,
        kind="mergesort",
    )
    return str(ranked.iloc[0]["Model"])


def compute_test_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_proba)),
    }


def threshold_experiment(y_true: pd.Series, y_proba: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        y_pred_t = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t, labels=[0, 1]).ravel()
        rows.append({
            "Threshold": threshold,
            "Precision": precision_score(y_true, y_pred_t, zero_division=0),
            "Recall": recall_score(y_true, y_pred_t, zero_division=0),
            "F1": f1_score(y_true, y_pred_t, zero_division=0),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "TN": int(tn),
        })
    return pd.DataFrame(rows)


def save_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_hill_valley(args.data_file)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    baseline = DummyClassifier(strategy="most_frequent")
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=args.random_state)),
    ])
    gaussian_nb = GaussianNB()

    models = {
        "Baseline": baseline,
        "Logistic Regression": logistic,
        "GaussianNB": gaussian_nb,
    }

    cv_table, _ = evaluate_cv(models, X_train, y_train, args.random_state)
    cv_table.to_csv(output_dir / "cv_results.csv", index=False)

    selected_name = choose_model(cv_table.loc[cv_table["Model"] != "Baseline"].reset_index(drop=True))
    selected_model = models[selected_name]
    selected_model.fit(X_train, y_train)

    y_pred = selected_model.predict(X_test)
    y_proba = selected_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_test_metrics(y_test, y_pred, y_proba)

    baseline_fitted = baseline.fit(X_train, y_train)
    baseline_pred = baseline_fitted.predict(X_test)
    baseline_metrics = {
        "Accuracy": float(accuracy_score(y_test, baseline_pred)),
        "Precision": float(precision_score(y_test, baseline_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, baseline_pred, zero_division=0)),
        "F1": float(f1_score(y_test, baseline_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, baseline_fitted.predict_proba(X_test)[:, 1])),
    }

    save_json(output_dir / "test_metrics.json", {
        "selected_model": selected_name,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_metrics,
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
        "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "random_state": args.random_state,
        "test_size": args.test_size,
    })

    # Confusion matrix.
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    save_json(output_dir / "confusion_matrix.json", {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    })

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Valley (0)", "Hill (1)"]).plot(ax=ax, values_format="d")
    ax.set_title(f"Confusion matrix: {selected_name}")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Misclassified objects.
    test_results = X_test.copy()
    test_results["actual"] = y_test
    test_results["predicted"] = y_pred
    test_results["probability"] = y_proba
    test_results["error_type"] = np.where(
        (y_test.to_numpy() == 0) & (y_pred == 1), "FP",
        np.where((y_test.to_numpy() == 1) & (y_pred == 0), "FN", "Correct"),
    )
    test_results["distance_to_0_5"] = np.abs(y_proba - 0.5)
    errors = test_results.loc[test_results["error_type"] != "Correct"].copy()
    errors.index.name = "Index"
    errors.to_csv(output_dir / "misclassified_objects.csv")

    near_threshold = errors.loc[errors["distance_to_0_5"] <= 0.05].copy()
    near_threshold.to_csv(output_dir / "near_threshold_errors.csv")

    high_confidence = errors.loc[
        ((errors["error_type"] == "FP") & (errors["probability"] >= 0.95))
        | ((errors["error_type"] == "FN") & (errors["probability"] <= 0.05))
    ].copy()
    high_confidence.to_csv(output_dir / "high_confidence_errors.csv")

    # Representative errors: three closest to threshold and three most confident of each type when available.
    reps = []
    for error_type in ["FP", "FN"]:
        subset = errors[errors["error_type"] == error_type]
        reps.append(subset.nsmallest(3, "distance_to_0_5"))
        reps.append(subset.assign(_confidence_distance=np.abs(subset["probability"] - (0 if error_type == "FN" else 1))).nlargest(3, "_confidence_distance").drop(columns="_confidence_distance"))
    if reps:
        representative = pd.concat(reps).loc[lambda d: ~d.index.duplicated(keep="first")]
        representative.to_csv(output_dir / "representative_errors.csv")

    # Error type counts.
    error_counts = errors["error_type"].value_counts().reindex(["FP", "FN"], fill_value=0)
    error_counts.rename("count").to_csv(output_dir / "error_counts.csv")

    # Simple feature-level summary by error type for quick inspection.
    feature_summary_rows = []
    train_stats = X_train.agg(["mean", "std"]).T.replace(0, np.nan)
    for error_type in ["FP", "FN"]:
        subset = errors[errors["error_type"] == error_type]
        if subset.empty:
            continue
        mean_abs_z = ((subset[X_train.columns] - train_stats["mean"]) / train_stats["std"]).abs().mean().sort_values(ascending=False)
        for feature in mean_abs_z.head(10).index:
            feature_summary_rows.append({
                "error_type": error_type,
                "feature": feature,
                "mean_abs_z_vs_train": float(mean_abs_z.loc[feature]),
                "mean_feature_value": float(subset[feature].mean()),
            })
    pd.DataFrame(feature_summary_rows).to_csv(output_dir / "error_feature_summary.csv", index=False)

    # Threshold analysis.
    threshold_table = threshold_experiment(y_test, y_proba)
    threshold_table.to_csv(output_dir / "threshold_metrics.csv", index=False)

    threshold_grid = np.linspace(0.01, 0.99, 99)
    grid_rows = []
    for threshold in threshold_grid:
        y_grid = (y_proba >= threshold).astype(int)
        grid_rows.append({
            "threshold": threshold,
            "Precision": precision_score(y_test, y_grid, zero_division=0),
            "Recall": recall_score(y_test, y_grid, zero_division=0),
            "F1": f1_score(y_test, y_grid, zero_division=0),
        })
    grid = pd.DataFrame(grid_rows)

    plt.figure(figsize=(9, 5.5))
    plt.plot(grid["threshold"], grid["Precision"], label="Precision")
    plt.plot(grid["threshold"], grid["Recall"], label="Recall")
    plt.plot(grid["threshold"], grid["F1"], label="F1")
    for threshold in THRESHOLDS:
        row = threshold_table.loc[np.isclose(threshold_table["Threshold"], threshold)].iloc[0]
        plt.scatter([threshold], [row["Precision"]])
        plt.scatter([threshold], [row["Recall"]])
        plt.scatter([threshold], [row["F1"]])
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric value")
    plt.title("Metrics vs decision threshold")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_metrics.png", dpi=160)
    plt.close()

    # ROC curve.
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr, tpr, label=f"{selected_name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", label="Random ranking")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    # Human-readable summary.
    summary_lines = [
        "Hill-Valley — classification experiment",
        "",
        f"Selected model: {selected_name}",
        f"Train shape: {X_train.shape}",
        f"Test shape: {X_test.shape}",
        "",
        "Test metrics:",
    ]
    summary_lines.extend([f"  {metric}: {value:.4f}" for metric, value in test_metrics.items()])
    summary_lines += [
        "",
        f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        f"Misclassified objects: {len(errors)}",
        f"Near threshold (|p-0.5| <= 0.05): {len(near_threshold)}",
        f"High-confidence errors: {len(high_confidence)}",
        "",
        "Thresholds 0.3 / 0.5 / 0.7:",
    ]
    for _, row in threshold_table.iterrows():
        summary_lines.append(
            f"  t={row['Threshold']:.1f}: Precision={row['Precision']:.4f}, "
            f"Recall={row['Recall']:.4f}, F1={row['F1']:.4f}, FP={int(row['FP'])}, FN={int(row['FN'])}"
        )
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Compact comparison table for the report.
    comparison_rows = []
    for _, row in cv_table.iterrows():
        comparison_rows.append({
            "Model": row["Model"],
            "Accuracy": row["Accuracy_mean"],
            "Precision": row["Precision_mean"],
            "Recall": row["Recall_mean"],
            "F1": row["F1_mean"],
            "ROC-AUC": row["ROC-AUC_mean"],
        })
    pd.DataFrame(comparison_rows).to_csv(output_dir / "model_comparison.csv", index=False)

    print("=== Hill-Valley: classifier training and analysis ===")
    print(f"Selected model: {selected_name}")
    print("\n5-fold stratified cross-validation:")
    print(cv_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nTest set metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric:10s}: {value:.4f}")
    print(f"\nConfusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
