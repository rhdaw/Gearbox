import argparse
import glob
import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(prediction_csv_dir: str, gates: list[str]) -> pd.DataFrame:
    files = sorted(
        glob.glob(os.path.join(prediction_csv_dir, "*_with_UNITO_predictions.csv"))
    )

    rows = []
    for file_path in files:
        df = pd.read_csv(file_path)
        file_name = os.path.basename(file_path)

        for gate in gates:
            pred_col = f"UNITO_{gate}"
            if gate not in df.columns or pred_col not in df.columns:
                continue

            y_true = df[gate]
            y_pred = df[pred_col]

            if not (
                y_true.dropna().isin([0, 1]).all()
                and y_pred.dropna().isin([0, 1]).all()
            ):
                continue

            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())

            rows.append(
                {
                    "section": "per_file",
                    "file": file_name,
                    "gate": gate,
                    "n_rows": int(len(df)),
                    "true_positives": int((y_true == 1).sum()),
                    "predicted_positives": int((y_pred == 1).sum()),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision": float(
                        precision_score(y_true, y_pred, zero_division=0)
                    ),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                    "stat": "",
                    "value": "",
                }
            )

    return pd.DataFrame(rows)


def build_summary(per_file_df: pd.DataFrame) -> pd.DataFrame:
    if per_file_df.empty:
        return pd.DataFrame(
            columns=[
                "section",
                "file",
                "gate",
                "n_rows",
                "true_positives",
                "predicted_positives",
                "tp",
                "fp",
                "fn",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "stat",
                "value",
            ]
        )

    summary_rows = []
    metrics = ["accuracy", "precision", "recall", "f1"]
    stats = ["mean", "median", "min"]

    grouped = per_file_df.groupby("gate", dropna=False)
    for gate, gate_df in grouped:
        for metric in metrics:
            for stat in stats:
                val = float(getattr(gate_df[metric], stat)())
                summary_rows.append(
                    {
                        "section": "summary",
                        "file": "",
                        "gate": gate,
                        "n_rows": "",
                        "true_positives": "",
                        "predicted_positives": "",
                        "tp": "",
                        "fp": "",
                        "fn": "",
                        "accuracy": "",
                        "precision": "",
                        "recall": "",
                        "f1": "",
                        "stat": f"{metric}_{stat}",
                        "value": val,
                    }
                )

    return pd.DataFrame(summary_rows)


def write_report(
    per_file_df: pd.DataFrame, summary_df: pd.DataFrame, output_csv: str
) -> None:
    cols = [
        "section",
        "file",
        "gate",
        "n_rows",
        "true_positives",
        "predicted_positives",
        "tp",
        "fp",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "stat",
        "value",
    ]

    per_file_df = per_file_df.reindex(columns=cols)
    summary_df = summary_df.reindex(columns=cols)

    with open(output_csv, "w", encoding="utf-8") as f:
        per_file_df.to_csv(f, index=False)
        f.write("\n")
        summary_df.to_csv(f, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build UNITO per-file gate metrics CSV with summary section appended."
    )
    parser.add_argument(
        "--prediction-csv-dir",
        default="/Users/user/Documents/UNITO_csv_conversion/",
        help="Directory containing *_with_UNITO_predictions.csv files.",
    )
    parser.add_argument(
        "--output-csv",
        default="/Users/user/Documents/GitHub/Gearbox/gate_metrics_report.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--gates",
        nargs="+",
        default=["Lymphocytes", "Single Cells", "Neutrophils"],
        help="Gate names to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_file_df = compute_metrics(args.prediction_csv_dir, args.gates)
    summary_df = build_summary(per_file_df)
    write_report(per_file_df, summary_df, args.output_csv)

    print(f"Per-file rows: {len(per_file_df)}")
    print(f"Summary rows: {len(summary_df)}")
    print(f"Saved report to: {args.output_csv}")


if __name__ == "__main__":
    main()
