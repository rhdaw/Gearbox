"""
Gate comparison visualisation
-------------------------------
For each gate in the gating strategy, plots the manual label vs UNITO prediction side
by side across all predicted CSV files.

Outputs one PNG per file per gate into the output_dir.

Usage:
    python gate_comparison_plots.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
CSV_DIR = "/Users/user/Documents/UNITO_csv_conversion/"
GATING_STRATEGY_PATH = "/Users/user/Documents/GitHub/Gearbox/gating_strategy.csv"
OUTPUT_DIR = "/Users/user/Documents/GitHub/Gearbox/src/3_unito/gate_comparison_plots/"
MAX_FILES = 5  # Set to an int (e.g. 5) to limit how many files are plotted
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load gating strategy
gating_df = pd.read_csv(GATING_STRATEGY_PATH)
gates = gating_df[["Gate", "X_axis", "Y_axis"]].values.tolist()

# Find predicted files
pred_files = sorted(
    [f for f in os.listdir(CSV_DIR) if f.endswith("_with_UNITO_predictions.csv")]
)

if MAX_FILES:
    pred_files = pred_files[:MAX_FILES]

print(f"Found {len(pred_files)} predicted file(s). Plotting {len(gates)} gate(s) each.")


def scatter_gate(
    ax, df, x_col, y_col, gate_col, title, color_pos="tab:red", downsample=50_000
):
    """Scatter plot with all cells in grey, gate-positive cells highlighted."""
    if x_col not in df.columns or y_col not in df.columns:
        ax.set_title(f"{title}\n(missing columns)")
        return

    # Downsample background for speed
    neg = df[df[gate_col] == 0][[x_col, y_col]]
    pos = df[df[gate_col] == 1][[x_col, y_col]]

    if len(neg) > downsample:
        neg = neg.sample(downsample, random_state=42)
    if len(pos) > downsample:
        pos = pos.sample(downsample, random_state=42)

    ax.scatter(
        neg[x_col],
        neg[y_col],
        s=0.4,
        alpha=0.25,
        color="lightgrey",
        rasterized=True,
        label="Negative",
    )
    ax.scatter(
        pos[x_col],
        pos[y_col],
        s=0.4,
        alpha=0.35,
        color=color_pos,
        rasterized=True,
        label="Positive",
    )

    n_pos = (df[gate_col] == 1).sum()
    n_total = len(df)
    pct = 100 * n_pos / n_total if n_total > 0 else 0

    ax.set_xlabel(x_col, fontsize=8)
    ax.set_ylabel(y_col, fontsize=8)
    ax.set_title(f"{title}\n{n_pos:,} / {n_total:,} positive ({pct:.1f}%)", fontsize=9)
    ax.tick_params(labelsize=7)


def scatter_confusion(
    ax, df, x_col, y_col, manual_col, pred_col, title, downsample=50_000
):
    """Scatter plot showing TP/FP/FN categories for clearer error interpretation."""
    if (
        x_col not in df.columns
        or y_col not in df.columns
        or manual_col not in df.columns
        or pred_col not in df.columns
    ):
        ax.set_title(f"{title}\n(missing columns)")
        return

    tp = df[(df[manual_col] == 1) & (df[pred_col] == 1)][[x_col, y_col]]
    fp = df[(df[manual_col] == 0) & (df[pred_col] == 1)][[x_col, y_col]]
    fn = df[(df[manual_col] == 1) & (df[pred_col] == 0)][[x_col, y_col]]

    if len(tp) > downsample:
        tp = tp.sample(downsample, random_state=42)
    if len(fp) > downsample:
        fp = fp.sample(downsample, random_state=42)
    if len(fn) > downsample:
        fn = fn.sample(downsample, random_state=42)

    ax.scatter(
        tp[x_col],
        tp[y_col],
        s=0.4,
        alpha=0.35,
        color="tab:green",
        rasterized=True,
        label="TP",
    )
    ax.scatter(
        fp[x_col],
        fp[y_col],
        s=0.4,
        alpha=0.35,
        color="tab:red",
        rasterized=True,
        label="FP",
    )
    ax.scatter(
        fn[x_col],
        fn[y_col],
        s=0.4,
        alpha=0.35,
        color="tab:orange",
        rasterized=True,
        label="FN",
    )

    ax.set_xlabel(x_col, fontsize=8)
    ax.set_ylabel(y_col, fontsize=8)
    ax.set_title(
        f"{title}\nTP={len(tp):,} FP={len(fp):,} FN={len(fn):,} (sampled)",
        fontsize=9,
    )
    ax.tick_params(labelsize=7)
    ax.legend(loc="best", fontsize=7)


for fname in pred_files:
    fpath = os.path.join(CSV_DIR, fname)
    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        print(f"Could not read {fname}: {e}")
        continue

    sample_name = fname.replace("_with_UNITO_predictions.csv", "")

    for gate, x_col, y_col in gates:
        manual_col = gate
        pred_col = f"UNITO_{gate}"

        has_manual = manual_col in df.columns
        has_pred = pred_col in df.columns

        if not has_manual and not has_pred:
            print(
                f"  Skipping {gate} in {fname} — neither manual nor predicted column found"
            )
            continue

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f"{sample_name}  |  Gate: {gate}  ({x_col} vs {y_col})", fontsize=11
        )

        # Left: manual gate
        if has_manual:
            scatter_gate(
                axes[0],
                df,
                x_col,
                y_col,
                manual_col,
                "Manual gate",
                color_pos="tab:blue",
            )
        else:
            axes[0].set_title("Manual gate\n(not available)")
            axes[0].axis("off")

        # Right: UNITO prediction
        if has_pred:
            scatter_gate(
                axes[1],
                df,
                x_col,
                y_col,
                pred_col,
                "UNITO prediction",
                color_pos="tab:red",
            )
        else:
            axes[1].set_title("UNITO prediction\n(not available)")
            axes[1].axis("off")

        # Right: confusion categories
        if has_manual and has_pred:
            scatter_confusion(
                axes[2],
                df,
                x_col,
                y_col,
                manual_col,
                pred_col,
                "Error map (TP/FP/FN)",
            )
        else:
            axes[2].set_title("Error map\n(not available)")
            axes[2].axis("off")

        # Add precision/recall annotation if both columns present
        if has_manual and has_pred:
            tp = int(((df[manual_col] == 1) & (df[pred_col] == 1)).sum())
            fp = int(((df[manual_col] == 0) & (df[pred_col] == 1)).sum())
            fn = int(((df[manual_col] == 1) & (df[pred_col] == 0)).sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                (2 * precision * recall / (precision + recall))
                if (precision + recall) > 0
                else 0
            )

            fig.text(
                0.5,
                0.01,
                f"Precision: {precision:.3f}   Recall: {recall:.3f}   F1: {f1:.3f}",
                ha="center",
                fontsize=10,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="grey"
                ),
            )

        plt.tight_layout(rect=[0, 0.05, 1, 1])

        safe_gate = gate.replace(" ", "_").replace("/", "-")
        out_path = os.path.join(OUTPUT_DIR, f"{sample_name}__{safe_gate}.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

print(f"\nDone. Plots saved to: {OUTPUT_DIR}")
