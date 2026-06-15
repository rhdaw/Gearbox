import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from UNITO_Train_Predict.Data_Preprocessing import normalize


def _to_binary_mask(
    df: pd.DataFrame, x_axis: str, y_axis: str, label_col: str
) -> np.ndarray:
    mask = np.zeros((101, 101), dtype=np.uint8)
    pos_df = df[df[label_col] == 1]
    if pos_df.empty:
        return mask

    xs = pos_df[x_axis + "_normalized"].to_numpy(dtype=int)
    ys = pos_df[y_axis + "_normalized"].to_numpy(dtype=int)

    xs = np.clip(xs, 0, 100)
    ys = np.clip(ys, 0, 100)

    # Match UNITO matrix orientation (invert x index)
    row_idx = 100 - xs
    col_idx = ys
    mask[row_idx, col_idx] = 1
    return mask


def _centroid(mask: np.ndarray):
    coords = np.argwhere(mask == 1)
    if len(coords) == 0:
        return None
    return coords.mean(axis=0)


def _compute_metrics(manual_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    tp = int(np.logical_and(manual_mask == 1, pred_mask == 1).sum())
    fp = int(np.logical_and(manual_mask == 0, pred_mask == 1).sum())
    fn = int(np.logical_and(manual_mask == 1, pred_mask == 0).sum())

    union = int(np.logical_or(manual_mask == 1, pred_mask == 1).sum())
    manual_count = int((manual_mask == 1).sum())
    pred_count = int((pred_mask == 1).sum())

    iou = tp / union if union > 0 else 0.0
    dice = (
        (2 * tp) / (manual_count + pred_count)
        if (manual_count + pred_count) > 0
        else 0.0
    )

    c_manual = _centroid(manual_mask)
    c_pred = _centroid(pred_mask)
    if c_manual is not None and c_pred is not None:
        centroid_shift_pixels = float(np.linalg.norm(c_manual - c_pred))
    else:
        centroid_shift_pixels = float("nan")

    return {
        "manual_pixels": manual_count,
        "pred_pixels": pred_count,
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "iou": iou,
        "dice": dice,
        "centroid_shift_pixels": centroid_shift_pixels,
    }


def _save_overlay(
    manual_mask: np.ndarray, pred_mask: np.ndarray, out_dir: str, prefix: str
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    tp = np.logical_and(manual_mask == 1, pred_mask == 1)
    fp = np.logical_and(manual_mask == 0, pred_mask == 1)
    fn = np.logical_and(manual_mask == 1, pred_mask == 0)

    overlay = np.zeros((101, 101, 3), dtype=np.float32)
    overlay[tp] = [0.2, 0.8, 0.2]  # green
    overlay[fp] = [0.9, 0.2, 0.2]  # red
    overlay[fn] = [1.0, 0.6, 0.0]  # orange

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(manual_mask, cmap="Blues")
    axes[0].set_title("Manual mask (101x101)")
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="Reds")
    axes[1].set_title("Predicted mask (101x101)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay: TP green / FP red / FN orange")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{prefix}_mask_alignment.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare manual vs predicted gate masks on UNITO 101x101 grid."
    )
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--gate", default="Neutrophils")
    parser.add_argument("--x-axis", default="CD15")
    parser.add_argument("--y-axis", default="SSC-A")
    parser.add_argument(
        "--parent-pred-col",
        default=None,
        help="Optional parent prediction column for sequential gates (e.g., 'Single Cells_pred').",
    )
    parser.add_argument(
        "--out-dir",
        default="/Users/user/Documents/GitHub/Gearbox/src/3_unito/mask_alignment_debug",
    )
    parser.add_argument("--prefix", default="debug")
    return parser.parse_args()


def main():
    args = parse_args()
    pred_col = f"UNITO_{args.gate}"

    df = pd.read_csv(args.csv_path)
    required_cols = [args.x_axis, args.y_axis, args.gate, pred_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work_df = df.copy()

    if args.parent_pred_col:
        parent_col = args.parent_pred_col
        if parent_col not in work_df.columns:
            alt_candidates = []
            if parent_col.endswith("_pred"):
                alt_candidates.append("UNITO_" + parent_col[: -len("_pred")])
            elif parent_col.startswith("UNITO_"):
                alt_candidates.append(parent_col.replace("UNITO_", "") + "_pred")
            else:
                alt_candidates.append(parent_col + "_pred")
                alt_candidates.append("UNITO_" + parent_col)

            found = [c for c in alt_candidates if c in work_df.columns]
            if found:
                parent_col = found[0]
            else:
                raise ValueError(
                    f"Parent prediction column not found: {args.parent_pred_col}. "
                    f"Tried alternatives: {alt_candidates}"
                )

        work_df = work_df[work_df[parent_col] == 1].copy()

    if work_df.empty:
        raise ValueError("No rows available after optional parent filtering.")

    work_df[args.x_axis + "_normalized"] = normalize(work_df, args.x_axis) * 100
    work_df[args.y_axis + "_normalized"] = normalize(work_df, args.y_axis) * 100
    work_df[args.x_axis + "_normalized"] = (
        work_df[args.x_axis + "_normalized"].round(0).astype(int)
    )
    work_df[args.y_axis + "_normalized"] = (
        work_df[args.y_axis + "_normalized"].round(0).astype(int)
    )

    manual_mask = _to_binary_mask(work_df, args.x_axis, args.y_axis, args.gate)
    pred_mask = _to_binary_mask(work_df, args.x_axis, args.y_axis, pred_col)

    metrics = _compute_metrics(manual_mask, pred_mask)
    img_path = _save_overlay(manual_mask, pred_mask, args.out_dir, args.prefix)

    print("Mask alignment metrics:")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    print(f"Saved overlay: {img_path}")

    stats_path = os.path.join(args.out_dir, f"{args.prefix}_mask_alignment_metrics.csv")
    pd.DataFrame([metrics]).to_csv(stats_path, index=False)
    print(f"Saved metrics CSV: {stats_path}")


if __name__ == "__main__":
    main()
