import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from UNITO_Train_Predict.Data_Preprocessing import fill_hull, matrix_plot, normalize


def build_masks(df: pd.DataFrame, x_axis: str, y_axis: str, gate: str):
    data_df_selected = df[[x_axis, y_axis, gate]].copy()

    data_df_selected[x_axis] = normalize(data_df_selected, x_axis) * 100
    data_df_selected[y_axis] = normalize(data_df_selected, y_axis) * 100

    data_df_masked = data_df_selected[data_df_selected[gate] == 1]

    df_plot = matrix_plot(data_df_masked, x_axis, y_axis, 0)
    raw_binary = (df_plot.to_numpy() != 0).astype(np.uint8)

    if np.sum(raw_binary) > 3:
        filled_nonconvex = fill_hull(raw_binary.copy(), convex=False).astype(np.uint8)
        filled_convex = fill_hull(raw_binary.copy(), convex=True).astype(np.uint8)
    else:
        filled_nonconvex = raw_binary.copy()
        filled_convex = raw_binary.copy()

    return raw_binary, filled_nonconvex, filled_convex


def save_outputs(raw_mask, nonconvex_mask, convex_mask, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, f"{prefix}_raw_mask.npy"), raw_mask)
    np.save(os.path.join(out_dir, f"{prefix}_filled_nonconvex.npy"), nonconvex_mask)
    np.save(os.path.join(out_dir, f"{prefix}_filled_convex.npy"), convex_mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(raw_mask, cmap="gray")
    axes[0].set_title("Raw mask (no fill)")
    axes[0].axis("off")

    axes[1].imshow(nonconvex_mask, cmap="gray")
    axes[1].set_title("Filled mask (convex=False)")
    axes[1].axis("off")

    axes[2].imshow(convex_mask, cmap="gray")
    axes[2].set_title("Filled mask (convex=True)")
    axes[2].axis("off")

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{prefix}_mask_compare.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return fig_path


def print_stats(raw_mask, nonconvex_mask, convex_mask):
    raw_count = int(raw_mask.sum())
    nonconvex_count = int(nonconvex_mask.sum())
    convex_count = int(convex_mask.sum())

    added_nonconvex = nonconvex_count - raw_count
    added_convex = convex_count - raw_count

    print("Mask pixel stats:")
    print(f"  Raw mask pixels: {raw_count}")
    print(f"  Filled non-convex pixels: {nonconvex_count} (added {added_nonconvex})")
    print(f"  Filled convex pixels: {convex_count} (added {added_convex})")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare raw vs filled gate masks for one CSV and gate."
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path to input CSV containing gate column.",
    )
    parser.add_argument("--gate", default="Neutrophils", help="Gate column name.")
    parser.add_argument("--x-axis", default="CD15", help="X axis column name.")
    parser.add_argument("--y-axis", default="SSC-A", help="Y axis column name.")
    parser.add_argument(
        "--out-dir",
        default="/Users/user/Documents/GitHub/Gearbox/src/3_unito/mask_fill_debug",
        help="Output directory for debug artifacts.",
    )
    parser.add_argument(
        "--prefix",
        default="debug",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv_path)
    missing = [c for c in [args.x_axis, args.y_axis, args.gate] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    raw_mask, nonconvex_mask, convex_mask = build_masks(
        df, args.x_axis, args.y_axis, args.gate
    )
    fig_path = save_outputs(
        raw_mask, nonconvex_mask, convex_mask, args.out_dir, args.prefix
    )
    print_stats(raw_mask, nonconvex_mask, convex_mask)
    print(f"Saved comparison image: {fig_path}")


if __name__ == "__main__":
    main()
