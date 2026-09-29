from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data_utils import EXPECTED_FEATURES, load_hill_valley


CLASS_NAMES = {0: "Valley", 1: "Hill"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Первинний аналіз Hill-Valley Dataset."
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Шлях до Hill_Valley_without_noise_Training.data. "
        "Якщо не задано, скрипт спробує завантажити набір з UCI.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="Каталог для таблиць і графіків аналізу.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_hill_valley(args.data_file)
    df = X.copy()
    df["class"] = y

    class_counts = y.value_counts().sort_index()
    class_share = (class_counts / len(y)).round(4)

    summary = {
        "rows": int(df.shape[0]),
        "features": int(X.shape[1]),
        "target": "class",
        "target_classes": {str(k): CLASS_NAMES.get(int(k), str(k)) for k in class_counts.index},
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {CLASS_NAMES[int(k)]: int(v) for k, v in class_counts.items()},
        "class_share": {CLASS_NAMES[int(k)]: float(v) for k, v in class_share.items()},
    }

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    stats = X.describe().T
    stats.insert(0, "feature", stats.index)
    stats.to_csv(output_dir / "feature_statistics.csv", index=False)

    class_table = pd.DataFrame({
        "class": class_counts.index.map(CLASS_NAMES),
        "count": class_counts.values,
        "share": class_share.values,
    })
    class_table.to_csv(output_dir / "class_distribution.csv", index=False)

    # Plot 1: class distribution.
    plt.figure(figsize=(7, 4.5))
    plt.bar(class_table["class"], class_table["count"])
    plt.xlabel("Class")
    plt.ylabel("Number of objects")
    plt.title("Hill-Valley: class distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close()

    # Plot 2: several representative signals.
    plt.figure(figsize=(10, 5.5))
    shown = 0
    for class_value in [0, 1]:
        indices_arr = [[249, 399, 429], []] # y.index[y == class_value].tolist()[:3]
        indices = indices_arr[class_value]
        for idx in indices:
            plt.plot(range(1, 101), X.loc[idx, EXPECTED_FEATURES].to_numpy(), label=f"{CLASS_NAMES[class_value]} #{idx}")
            shown += 1
    if shown:
        plt.xlabel("Point")
        plt.ylabel("Value")
        plt.title("Representative Hill-Valley signals")
        plt.legend(ncol=2, fontsize=8)
        plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "representative_signals.png", dpi=160)
    plt.close()

    # Plot 3: average profile for each class.
    mean_profiles = df.groupby("class")[EXPECTED_FEATURES].mean()
    plt.figure(figsize=(10, 5.5))
    plt.plot(range(1, 101), mean_profiles.loc[0].to_numpy(), label="Valley")
    plt.plot(range(1, 101), mean_profiles.loc[1].to_numpy(), label="Hill")
    plt.xlabel("Point")
    plt.ylabel("Mean value")
    plt.title("Mean signal profile by class")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "mean_profiles.png", dpi=160)
    plt.close()

    print("=== Hill-Valley Dataset: input data analysis ===")
    print(f"Rows:              {df.shape[0]}")
    print(f"Features:           {X.shape[1]}")
    print(f"Missing values:     {summary['missing_values_total']}")
    print(f"Duplicate rows:     {summary['duplicate_rows']}")
    print("Class distribution:")
    for class_value, count in class_counts.items():
        print(f"  {class_value} ({CLASS_NAMES[int(class_value)]}): {count} ({class_share.loc[class_value]:.4f})")
    print(f"Saved analysis to:  {output_dir.resolve()}")


if __name__ == "__main__":
    main()
