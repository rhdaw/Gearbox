import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

csv_dir = "/Users/user/Documents/UNITO_csv_conversion/"

print("Checking transformed data ranges:")
for f in os.listdir(csv_dir):
    if f.endswith(".csv"):
        df = pd.read_csv(os.path.join(csv_dir, f))

        print(f"\n{f}:")
        print(
            f"  CD15 - min: {df['CD15'].min():.2f}, max: {df['CD15'].max():.2f}, median: {df['CD15'].median():.2f}"
        )
        print(
            f"  CD45 - min: {df['CD45'].min():.2f}, max: {df['CD45'].max():.2f}, median: {df['CD45'].median():.2f}"
        )

        # Check for NaN or inf values
        print(
            f"  CD15 NaN count: {df['CD15'].isna().sum()}, inf count: {np.isinf(df['CD15']).sum()}"
        )
        print(
            f"  CD45 NaN count: {df['CD45'].isna().sum()}, inf count: {np.isinf(df['CD45']).sum()}"
        )

        # Check Non-neutrophil Lymphocytes
        if "Non-neutrophil Lymphocytes" in df.columns:
            nnl = df[df["Non-neutrophil Lymphocytes"] == 1]
            if len(nnl) > 0:
                print(f"  Non-neutrophil Lymphocytes (n={len(nnl)}):")
                print(
                    f"    CD15 range: {nnl['CD15'].min():.2f} to {nnl['CD15'].max():.2f}"
                )
                print(
                    f"    CD45 range: {nnl['CD45'].min():.2f} to {nnl['CD45'].max():.2f}"
                )

        # Plot scatter plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"CD15 vs CD45 - {f}")

        # Plot 1: All cells
        axes[0, 0].scatter(df["CD15"], df["CD45"], alpha=0.1, s=1)
        axes[0, 0].set_title("All cells")
        axes[0, 0].set_xlabel("CD15")
        axes[0, 0].set_ylabel("CD45")

        # Plot 2: Neutrophils vs Non-neutrophils
        if "Neutrophils" in df.columns:
            neut = df[df["Neutrophils"] == 1]
            non_neut = df[df["Neutrophils"] == 0]
            axes[0, 1].scatter(
                non_neut["CD15"],
                non_neut["CD45"],
                alpha=0.1,
                s=1,
                c="blue",
                label="Non-neutrophils",
            )
            axes[0, 1].scatter(
                neut["CD15"], neut["CD45"], alpha=0.1, s=1, c="red", label="Neutrophils"
            )
            axes[0, 1].set_title("Neutrophils (red) vs Non-neutrophils (blue)")
            axes[0, 1].set_xlabel("CD15")
            axes[0, 1].set_ylabel("CD45")
            axes[0, 1].legend()

        # Plot 3: Non-neutrophil Lymphocytes
        if "Non-neutrophil Lymphocytes" in df.columns:
            nnl = df[df["Non-neutrophil Lymphocytes"] == 1]
            non_nnl = df[df["Non-neutrophil Lymphocytes"] == 0]
            axes[1, 0].scatter(
                non_nnl["CD15"],
                non_nnl["CD45"],
                alpha=0.1,
                s=1,
                c="gray",
                label="Other",
            )
            axes[1, 0].scatter(
                nnl["CD15"],
                nnl["CD45"],
                alpha=0.1,
                s=4,
                c="green",
                label="Non-neutrophil Lymphocytes",
            )
            axes[1, 0].set_title("Non-neutrophil Lymphocytes (green)")
            axes[1, 0].set_xlabel("CD15")
            axes[1, 0].set_ylabel("CD45")
            axes[1, 0].legend()

        # Plot 4: 2D Density (hexbin)
        axes[1, 1].hexbin(df["CD15"], df["CD45"], gridsize=50, cmap="viridis")
        axes[1, 1].set_title("Density (all cells)")
        axes[1, 1].set_xlabel("CD15")
        axes[1, 1].set_ylabel("CD45")

        plt.tight_layout()
        plt.savefig(
            f"/Users/user/Documents/GitHub/Gearbox/src/3_unito/density_plot_{f.replace('.csv', '.png')}"
        )
        print(f"Saved plot to density_plot_{f.replace('.csv', '.png')}")
        plt.close()

        break  # Just check first file
