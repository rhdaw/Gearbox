"""
Extract FMO thresholds from .fcs files.

Reads FMO_*.fcs files, extracts marker from filename using regex,
finds the maximum intensity value (rightmost edge of population),
applies arcsinh transform to match UNITO preprocessing, and returns
a dictionary of marker → threshold for use in FlowSOM.

Filename format: FMO_MARKER_FLUOROPHORE.fcs
(e.g., C8 FMO_CD19_PEFire810_FMOandSCplate-Copy1.fcs)
"""

import os
import re
import json
import argparse
from typing import Any, Callable, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fcsparser


_parse_fcs = getattr(fcsparser, "parse", None)
if _parse_fcs is None:
    _parse_fcs = getattr(fcsparser, "parse_fcs", None)
if _parse_fcs is None:
    raise ImportError("No compatible parser found in fcsparser module.")
parse_fcs: Callable[..., Any] = _parse_fcs


def _normalize_marker_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(name)).lower()


def _resolve_marker_column(marker: str, columns: pd.Index) -> Optional[str]:
    if marker in columns:
        return marker

    lower_map = {str(col).lower(): str(col) for col in columns}
    marker_lower = marker.lower()
    if marker_lower in lower_map:
        return lower_map[marker_lower]

    marker_norm = _normalize_marker_name(marker)
    norm_map = {}
    for col in columns:
        col_str = str(col)
        norm_map.setdefault(_normalize_marker_name(col_str), col_str)
    if marker_norm in norm_map:
        return norm_map[marker_norm]

    prefix_matches = []
    for col in columns:
        col_str = str(col)
        col_norm = _normalize_marker_name(col_str)
        if col_norm.startswith(marker_norm) or marker_norm.startswith(col_norm):
            prefix_matches.append(col_str)
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    return None


def _marker_candidates_from_filename(filename: str) -> list[str]:
    match = re.search(r"FMO_(.+?)\.fcs$", filename)
    if not match:
        return []

    tail = match.group(1)
    tokens = [token for token in tail.split("_") if token]
    candidates: list[str] = []
    if len(tokens) >= 1:
        candidates.append(tokens[0])
    if len(tokens) >= 2:
        candidates.append(f"{tokens[0]}_{tokens[1]}")
    return candidates


def _resolve_side_scatter_column(columns: pd.Index) -> Optional[str]:
    preferred = ["SSC-A", "SSC_A", "SSCA", "SSC-H", "SSC_H", "SSCH", "SSC"]
    lower_map = {str(col).lower(): str(col) for col in columns}

    for candidate in preferred:
        match = lower_map.get(candidate.lower())
        if match is not None:
            return match

    for col in columns:
        col_str = str(col)
        col_norm = _normalize_marker_name(col_str)
        if "ssc" in col_norm:
            return col_str

    return None


def _compute_threshold_raw(
    marker_values: np.ndarray,
    method: str = "hist_edge",
    quantile: float = 0.999,
    bins: int = 256,
    hist_min_fraction: float = 0.0015,
) -> float:
    if marker_values.size == 0:
        raise ValueError("No marker values available for threshold computation")

    if method == "max":
        return float(marker_values.max())

    if method == "quantile":
        if not 0.0 < quantile <= 1.0:
            raise ValueError("quantile must be in (0, 1]")
        return float(np.quantile(marker_values, quantile))

    if method == "hist_edge":
        counts, edges = np.histogram(marker_values, bins=bins)
        if counts.size == 0:
            return float(marker_values.max())

        if not 0.0 < hist_min_fraction <= 1.0:
            raise ValueError("hist_min_fraction must be in (0, 1]")

        min_count = max(2, int(np.ceil(hist_min_fraction * marker_values.size)))
        occupied_idx = np.where(counts >= min_count)[0]
        if occupied_idx.size == 0:
            occupied_idx = np.where(counts > 0)[0]
        if occupied_idx.size == 0:
            return float(marker_values.max())

        right_bin = int(occupied_idx[-1])
        return float(edges[right_bin + 1])

    raise ValueError(f"Unsupported threshold method: {method}")


def extract_fmo_thresholds(
    fmo_dir: str,
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
    cofactor: float = 200.0,
    plot_dir: Optional[str] = None,
    threshold_method: str = "hist_edge",
    quantile: float = 0.999,
    histogram_bins: int = 256,
    hist_min_fraction: float = 0.0015,
) -> dict:
    """
    Extract FMO thresholds from .fcs files.

    For each FMO file, extracts the marker name from filename (FMO_MARKER_),
    finds a robust right-edge intensity value in that marker column,
    applies arcsinh transform, and returns thresholds.

    Generates histograms with threshold lines for each marker.

    Args:
        fmo_dir: directory containing FMO_*.fcs files
        output_json: optional path to save thresholds as JSON
        output_csv: optional path to save thresholds as CSV
        cofactor: arcsinh cofactor (should match UNITO's 200.0)
        plot_dir: optional directory to save histogram PDFs
        threshold_method: one of {'hist_edge', 'quantile', 'max'}
        quantile: quantile used when threshold_method='quantile'
        histogram_bins: bin count used when threshold_method='hist_edge'
        hist_min_fraction: minimum event fraction required for a histogram bin to
            be considered occupied when threshold_method='hist_edge'

    Returns:
        dict: {marker_name: threshold_value} in arcsinh-transformed scale
    """
    thresholds = {}

    # Create plot directory if requested
    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    fmo_files = sorted(f for f in os.listdir(fmo_dir) if f.endswith(".fcs"))

    if not fmo_files:
        print(f"ERROR: No .fcs files found in {fmo_dir}")
        return thresholds

    print(f"Found {len(fmo_files)} FMO files in {fmo_dir}\n")

    for filename in fmo_files:
        marker_candidates = _marker_candidates_from_filename(filename)
        if not marker_candidates:
            print(f"⚠ Skipping {filename} — no FMO marker found")
            continue
        filepath = os.path.join(fmo_dir, filename)

        try:
            meta, data = parse_fcs(filepath)

            marker_column = None
            selected_marker_name = None
            for candidate in marker_candidates:
                resolved = _resolve_marker_column(candidate, data.columns)
                if resolved is not None:
                    marker_column = resolved
                    selected_marker_name = candidate
                    break

            if marker_column is None:
                marker_preview = ", ".join(marker_candidates)
                print(f"⚠ Marker candidates '{marker_preview}' not found in {filename}")
                continue
            marker = selected_marker_name
            if marker_column != marker:
                print(
                    f"ℹ Resolved marker '{marker}' to column '{marker_column}' "
                    f"in {filename}"
                )

            # Get raw marker values and estimate threshold
            marker_all = data[marker_column].to_numpy(dtype=float, na_value=np.nan)
            marker_values = marker_all[np.isfinite(marker_all)]
            if marker_values.size == 0:
                print(f"⚠ Marker '{marker_column}' in {filename} has no finite values")
                continue

            ssc_column = _resolve_side_scatter_column(data.columns)
            scatter_marker_values = None
            scatter_ssc_values = None
            if ssc_column is not None:
                ssc_all = data[ssc_column].to_numpy(dtype=float, na_value=np.nan)
                valid_mask = np.isfinite(marker_all) & np.isfinite(ssc_all)
                if np.any(valid_mask):
                    scatter_marker_values = marker_all[valid_mask]
                    scatter_ssc_values = ssc_all[valid_mask]

            threshold_raw = _compute_threshold_raw(
                marker_values,
                method=threshold_method,
                quantile=quantile,
                bins=histogram_bins,
                hist_min_fraction=hist_min_fraction,
            )

            # Apply arcsinh transform (match UNITO cofactor=200)
            threshold_transformed = np.arcsinh(threshold_raw / cofactor)

            thresholds[marker] = threshold_transformed
            print(
                f"✓ {marker:12s}: raw_threshold={threshold_raw:10.0f} "
                f"→ arcsinh={threshold_transformed:6.2f}"
            )

            # Generate histogram if plot_dir is specified
            if plot_dir:
                _generate_histogram(
                    marker,
                    marker_values,
                    threshold_raw,
                    plot_dir,
                    threshold_method,
                    cofactor,
                    scatter_marker_values=scatter_marker_values,
                    scatter_ssc_values=scatter_ssc_values,
                    ssc_label=ssc_column,
                )

        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")

    print(f"\n✓ Extracted {len(thresholds)} thresholds")

    # Save to JSON if requested
    if output_json:
        with open(output_json, "w") as f:
            json.dump(thresholds, f, indent=2)
        print(f"✓ Saved to {output_json}")

    # Save to CSV if requested
    if output_csv:
        df = pd.DataFrame(sorted(thresholds.items()), columns=["Marker", "Threshold"])
        df.to_csv(output_csv, index=False)
        print(f"✓ Saved to {output_csv}")

    if plot_dir:
        print(f"✓ Histograms saved to {plot_dir}")

    return thresholds


def _generate_histogram(
    marker: str,
    marker_values: np.ndarray,
    threshold_value: float,
    output_dir: str,
    threshold_method: str,
    cofactor: float,
    scatter_marker_values: Optional[np.ndarray] = None,
    scatter_ssc_values: Optional[np.ndarray] = None,
    ssc_label: Optional[str] = None,
):
    """Generate and save histogram + side-scatter panel as PDF."""
    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(14, 6))

    marker_transformed = np.arcsinh(marker_values / cofactor)
    threshold_transformed = np.arcsinh(threshold_value / cofactor)

    # Plot histogram
    ax_hist.hist(
        marker_transformed,
        bins=100,
        density=True,
        alpha=0.7,
        color="lightblue",
        edgecolor="black",
        linewidth=0.5,
    )

    # Draw vertical line at selected threshold
    ax_hist.axvline(
        threshold_transformed,
        color="black",
        linestyle="-",
        linewidth=2,
        label=f"Threshold ({threshold_method}): {threshold_transformed:.2f}",
    )

    # Labels and formatting
    ax_hist.set_xlabel("Expression (arcsinh transformed)", fontsize=12)
    ax_hist.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax_hist.set_title(f"{marker} FMO Histogram", fontsize=14, fontweight="bold")
    ax_hist.legend(fontsize=11)
    ax_hist.grid(True, alpha=0.3)

    stats_text = (
        f"Threshold: {threshold_transformed:.2f}\n"
        f"Mean: {np.mean(marker_transformed):.2f}\n"
        f"Median: {np.median(marker_transformed):.2f}\n"
        f"Std: {np.std(marker_transformed):.2f}"
    )
    ax_hist.text(
        0.02,
        0.98,
        stats_text,
        transform=ax_hist.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    if (
        scatter_marker_values is not None
        and scatter_ssc_values is not None
        and scatter_marker_values.size > 0
    ):
        max_points = 50000
        if scatter_marker_values.size > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(scatter_marker_values.size, size=max_points, replace=False)
            x_vals = scatter_marker_values[idx]
            y_vals = scatter_ssc_values[idx]
        else:
            x_vals = scatter_marker_values
            y_vals = scatter_ssc_values

        x_vals = np.arcsinh(x_vals / cofactor)
        ax_scatter.scatter(x_vals, y_vals, s=1, alpha=0.2, color="slateblue")
        ax_scatter.axvline(
            threshold_transformed,
            color="black",
            linestyle="-",
            linewidth=2.0,
            label=f"Threshold: {threshold_transformed:.2f}",
        )
        ax_scatter.set_xlabel(
            f"{marker} (arcsinh transformed)", fontsize=12, fontweight="bold"
        )
        ax_scatter.set_ylabel(
            ssc_label or "Side Scatter", fontsize=12, fontweight="bold"
        )
        ax_scatter.set_title(
            f"{marker} vs {ssc_label or 'SSC'}", fontsize=14, fontweight="bold"
        )
        ax_scatter.legend(fontsize=10)
        ax_scatter.grid(True, alpha=0.3)
    else:
        ax_scatter.text(
            0.5,
            0.5,
            "SSC channel not found\nor no valid paired values",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax_scatter.set_title("Marker vs Side Scatter", fontsize=14, fontweight="bold")
        ax_scatter.set_xticks([])
        ax_scatter.set_yticks([])
        ax_scatter.grid(False)

    # Save as PDF
    output_path = os.path.join(output_dir, f"{marker}_fmo_histogram.pdf")
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract FMO thresholds from FCS files"
    )
    parser.add_argument(
        "--fmo-dir",
        default="/Users/user/Documents/Gearbox/2_flow/fmo_fcs_files",
        help="Directory containing FMO .fcs files",
    )
    parser.add_argument(
        "--output-json",
        default="fmo_thresholds.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--output-csv",
        default="fmo_thresholds.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--plot-dir",
        default="fmo_histograms",
        help="Directory for histogram PDFs",
    )
    parser.add_argument(
        "--cofactor",
        type=float,
        default=200.0,
        help="Arcsinh cofactor",
    )
    parser.add_argument(
        "--threshold-method",
        choices=["hist_edge", "quantile", "max"],
        default="hist_edge",
        help="How to compute raw threshold from marker values",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.999,
        help="Quantile to use when --threshold-method quantile",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=256,
        help="Histogram bin count for --threshold-method hist_edge",
    )
    parser.add_argument(
        "--hist-min-fraction",
        type=float,
        default=0.0015,
        help=(
            "Minimum event fraction to treat a histogram bin as occupied in "
            "--threshold-method hist_edge"
        ),
    )
    args = parser.parse_args()

    fmo_dir = args.fmo_dir
    output_csv = args.output_csv
    output_json = args.output_json
    plot_dir = args.plot_dir
    cofactor = args.cofactor
    threshold_method = args.threshold_method
    quantile = args.quantile
    histogram_bins = args.histogram_bins
    hist_min_fraction = args.hist_min_fraction

    if not os.path.exists(fmo_dir):
        print(f"ERROR: Directory not found: {fmo_dir}")
        exit(1)

    thresholds = extract_fmo_thresholds(
        fmo_dir,
        output_json=output_json,
        output_csv=output_csv,
        cofactor=cofactor,
        plot_dir=plot_dir,
        threshold_method=threshold_method,
        quantile=quantile,
        histogram_bins=histogram_bins,
        hist_min_fraction=hist_min_fraction,
    )

    if thresholds:
        print("\nThresholds dict (for FlowSOM config):")
        print("{")
        for marker in sorted(thresholds.keys()):
            print(f"    '{marker}': {thresholds[marker]:.2f},")
        print("}")
    else:
        print("No thresholds extracted.")
        exit(1)
