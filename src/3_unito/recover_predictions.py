"""
Recover UNITO predictions from intermediate prediction folder and apply to original CSVs.
This script reads final predictions from the Neutrophils prediction folder
(which contains all sequential gate predictions) and writes them to the predict_dir CSVs.
"""

import os
import pandas as pd


def recover_predictions_from_intermediate(
    neutrophils_pred_dir: str,
    predict_dir: str,
    output_suffix: str = "_with_UNITO_predictions.csv",
):
    """
    Read predictions from intermediate prediction folder and apply to original CSVs.

    Args:
        neutrophils_pred_dir: Path to UNITO_train_data/prediction/Neutrophils/
        predict_dir: Path to original predict CSV directory
        output_suffix: Suffix for output files
    """
    pred_files = [f for f in os.listdir(neutrophils_pred_dir) if f.endswith(".csv")]

    if not pred_files:
        print(f"ERROR: No CSV files found in {neutrophils_pred_dir}")
        return False

    print(f"Found {len(pred_files)} prediction files in {neutrophils_pred_dir}")
    print(f"Reading from predict_dir: {predict_dir}")

    success_count = 0

    for pred_file in pred_files:
        pred_path = os.path.join(neutrophils_pred_dir, pred_file)
        pred_df = pd.read_csv(pred_path)

        # Extract prediction columns
        pred_cols = [col for col in pred_df.columns if col.endswith("_pred")]

        if not pred_cols:
            print(f"WARNING: No prediction columns found in {pred_file}")
            continue

        # Find matching original CSV
        # pred_file might be named like "sample123.csv" or "sample123.csv"
        # Original might be "sample123.csv"
        original_file = os.path.join(predict_dir, pred_file)

        if not os.path.exists(original_file):
            print(f"WARNING: Original file not found: {original_file}")
            continue

        # Read original CSV
        original_df = pd.read_csv(original_file)

        # Check row count match
        if len(original_df) != len(pred_df):
            print(
                f"ERROR: Row count mismatch for {pred_file}: {len(original_df)} vs {len(pred_df)}"
            )
            continue

        # Add prediction columns
        for col in pred_cols:
            original_df[col] = pred_df[col]

        # Save with output suffix
        output_file = pred_file.replace(".csv", output_suffix)
        output_path = os.path.join(predict_dir, output_file)
        original_df.to_csv(output_path, index=False)

        print(
            f"✓ {pred_file} -> {output_file} ({len(pred_cols)} prediction columns added)"
        )
        success_count += 1

    print(f"\n✓ Successfully processed {success_count}/{len(pred_files)} files")
    return success_count == len(pred_files)


if __name__ == "__main__":
    neutrophils_pred_dir = (
        "/Users/user/Documents/UNITO_train_data/prediction/Neutrophils"
    )
    predict_dir = "/Users/user/Documents/UNITO_csv_conversion_predict"

    success = recover_predictions_from_intermediate(neutrophils_pred_dir, predict_dir)
    exit(0 if success else 1)
