#!/usr/bin/env python
"""
Journal-quality visualisation of model parameter counts and GFLOPs.

Instantiates all available model architectures, computes parameter
counts + GFLOPs using thop at multiple resolutions, and generates
publication-ready seaborn plots saved per-resolution to:
    visualisation/plots_64/
    visualisation/plots_256/
    visualisation/plots_512/

Usage:
    python visualisation/plot_parameter.py
"""

import os
import sys

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so model imports work
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from thop import profile, clever_format

# ---------------------------------------------------------------------------
# Import factory functions from model.model
# ---------------------------------------------------------------------------
from model.model import (
    UNet, UNetSE, UNet3PlusSE, UNetMFF, UNetSEResnet, CustomSegformer,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_CHANNELS = 7
NUM_CLASSES = 2
RESOLUTIONS = [64, 256, 512]  # spatial resolutions to profile

# Colour palette — colourblind-safe, muted tones for journal use
PALETTE = sns.color_palette("Set2", n_colors=8)


# ---------------------------------------------------------------------------
# Helper: build all models
# ---------------------------------------------------------------------------

def _build_models():
    """Instantiate each model; skip those that require missing checkpoints."""
    specs = [
        ("UNet",          lambda: UNet(INPUT_CHANNELS, NUM_CLASSES, topology="ENC_4_DEC_4")),
        ("UNet-SE",       lambda: UNetSE(INPUT_CHANNELS, NUM_CLASSES, topology="ENC_4_DEC_4")),
        ("UNet3+SE",      lambda: UNet3PlusSE(INPUT_CHANNELS, NUM_CLASSES)),
        ("UNet-MFF",      lambda: UNetMFF(INPUT_CHANNELS, NUM_CLASSES, topology="ENC_4_DEC_4")),
        ("UNet-SE\n(ResNet50)", lambda: UNetSEResnet(INPUT_CHANNELS, NUM_CLASSES)),
        ("SegFormer\n(MiT-B3)", lambda: CustomSegformer(INPUT_CHANNELS, NUM_CLASSES,
                                                         base_model="nvidia/mit-b3")),
    ]
    models = {}
    for name, builder in specs:
        try:
            models[name] = builder()
        except Exception as e:
            print(f"  [SKIPPED] {name} — {e}")
    return models


# ---------------------------------------------------------------------------
# Compute metrics at a given resolution
# ---------------------------------------------------------------------------

def _compute_metrics(models: dict, resolution: int) -> pd.DataFrame:
    """Return a DataFrame with columns: Model, Params_M, GFLOPs."""
    records = []
    dummy = torch.randn(1, INPUT_CHANNELS, resolution, resolution)

    print(f"\n  Profiling at {resolution}×{resolution}:")
    for name, model in models.items():
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())

        # GFLOPs via thop
        try:
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
            gflops = flops / 1e9
        except Exception:
            gflops = float("nan")

        records.append({
            "Model": name,
            "Params_M": total_params / 1e6,
            "GFLOPs": gflops,
        })
        params_fmt, flops_fmt = clever_format([total_params, flops if not np.isnan(gflops) else 0], "%.2f")
        print(f"    {name:25s}  Params: {params_fmt:>10s}   GFLOPs: {flops_fmt if not np.isnan(gflops) else 'N/A':>10s}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plot helpers — journal style
# ---------------------------------------------------------------------------

_current_save_dir = None  # set per resolution

def _journal_style():
    """Apply a clean serif style suitable for journal figures."""
    sns.set_theme(style="whitegrid", font="serif", font_scale=1.15)
    plt.rcParams.update({
        "font.family": "serif",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.alpha": 0.35,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


def _save(fig, name):
    """Save as both PNG and PDF."""
    for ext in ("png", "pdf"):
        path = os.path.join(_current_save_dir, f"{name}.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"    → saved {name}.png / .pdf")


# ---------------------------------------------------------------------------
# Plot 1 & 2: Horizontal bar charts (params / GFLOPs)
# ---------------------------------------------------------------------------

def plot_horizontal_bar(df: pd.DataFrame, col: str, label: str, filename: str,
                        resolution: int):
    _journal_style()
    sorted_df = df.sort_values(col, ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 4))

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(sorted_df))]
    bars = ax.barh(sorted_df["Model"], sorted_df[col], color=colors,
                   edgecolor="white", linewidth=0.6, height=0.6)

    # value annotations
    for bar, val in zip(bars, sorted_df[col]):
        ax.text(bar.get_width() + sorted_df[col].max() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=10, fontweight="medium")

    ax.set_xlabel(label, fontsize=12, labelpad=8)
    ax.set_title(f"{label}  ({resolution}×{resolution} input)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_xlim(0, sorted_df[col].max() * 1.18)
    sns.despine(left=True)
    fig.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# Plot 3 & 4: Lollipop charts (params / GFLOPs)
# ---------------------------------------------------------------------------

def plot_lollipop(df: pd.DataFrame, col: str, label: str, filename: str,
                  resolution: int):
    _journal_style()
    sorted_df = df.sort_values(col, ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 4))

    y_pos = range(len(sorted_df))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(sorted_df))]

    ax.hlines(y=y_pos, xmin=0, xmax=sorted_df[col], color=colors,
              linewidth=2.2, zorder=2)
    ax.scatter(sorted_df[col], y_pos, color=colors, s=90, zorder=3,
               edgecolors="white", linewidth=1.2)

    # value annotations
    for i, val in enumerate(sorted_df[col]):
        ax.text(val + sorted_df[col].max() * 0.03, i,
                f"{val:.2f}", va="center", fontsize=10, fontweight="medium")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sorted_df["Model"])
    ax.set_xlabel(label, fontsize=12, labelpad=8)
    ax.set_title(f"{label}  ({resolution}×{resolution} input)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, sorted_df[col].max() * 1.2)
    sns.despine(left=True)
    fig.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# Plot 5: Grouped bar chart (params + GFLOPs side by side)
# ---------------------------------------------------------------------------

def plot_combined_grouped(df: pd.DataFrame, resolution: int):
    _journal_style()
    melted = df.melt(id_vars="Model", value_vars=["Params_M", "GFLOPs"],
                     var_name="Metric", value_name="Value")
    melted["Metric"] = melted["Metric"].map({
        "Params_M": "Parameters (M)",
        "GFLOPs": "GFLOPs",
    })

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=melted, y="Model", x="Value", hue="Metric",
                palette=[PALETTE[0], PALETTE[1]], edgecolor="white",
                linewidth=0.6, ax=ax)

    ax.set_xlabel("Value", fontsize=12, labelpad=8)
    ax.set_title(f"Model Complexity: Params vs GFLOPs  ({resolution}×{resolution})",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(title="", frameon=True, fancybox=False, edgecolor="#cccccc",
              fontsize=10, loc="lower right")
    sns.despine(left=True)
    fig.tight_layout()
    _save(fig, "combined_grouped_bar")


# ---------------------------------------------------------------------------
# Plot 6: Scatter — params vs GFLOPs (efficiency view)
# ---------------------------------------------------------------------------

def plot_scatter(df: pd.DataFrame, resolution: int):
    _journal_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
    ax.scatter(df["Params_M"], df["GFLOPs"], c=colors, s=160,
               edgecolors="white", linewidth=1.4, zorder=3)

    # label each point
    for _, row in df.iterrows():
        ax.annotate(row["Model"],
                    (row["Params_M"], row["GFLOPs"]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9.5, fontweight="medium",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7))

    ax.set_xlabel("Parameters (M)", fontsize=12, labelpad=8)
    ax.set_ylabel("GFLOPs", fontsize=12, labelpad=8)
    ax.set_title(f"Model Efficiency: Params vs Compute  ({resolution}×{resolution})",
                 fontsize=14, fontweight="bold", pad=12)
    sns.despine()
    fig.tight_layout()
    _save(fig, "combined_scatter")


# ---------------------------------------------------------------------------
# Generate all plots for a single resolution
# ---------------------------------------------------------------------------

def _generate_plots(df: pd.DataFrame, resolution: int):
    """Generate all 6 plot variants for the given resolution."""
    print(f"\n  Generating plots for {resolution}×{resolution} …\n")

    plot_horizontal_bar(df, "Params_M", "Parameters (M)", "params_horizontal_bar", resolution)
    plot_lollipop(df, "Params_M", "Parameters (M)", "params_lollipop", resolution)
    plot_horizontal_bar(df, "GFLOPs", "GFLOPs", "gflops_horizontal_bar", resolution)
    plot_lollipop(df, "GFLOPs", "GFLOPs", "gflops_lollipop", resolution)
    plot_combined_grouped(df, resolution)
    plot_scatter(df, resolution)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _current_save_dir

    print("\n" + "=" * 70)
    print("  Model Parameter & GFLOPs Profiler  (multi-resolution)")
    print("=" * 70)

    # Build models once (parameters don't change with resolution)
    models = _build_models()

    for res in RESOLUTIONS:
        print("\n" + "#" * 70)
        print(f"  Resolution: {res}×{res}")
        print("#" * 70)

        # Set save directory for this resolution
        _current_save_dir = os.path.join(SCRIPT_DIR, f"plots_{res}")
        os.makedirs(_current_save_dir, exist_ok=True)

        # Compute metrics at this resolution
        df = _compute_metrics(models, res)

        print("\n" + "-" * 70)
        print(df.to_string(index=False))
        print("-" * 70)

        # Generate all plots
        _generate_plots(df, res)

        print(f"\n  Plots saved to: {_current_save_dir}")

    print("\n" + "=" * 70)
    print("  Done! All resolutions profiled.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
