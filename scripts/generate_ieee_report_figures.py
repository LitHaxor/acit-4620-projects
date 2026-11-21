#!/usr/bin/env python3
"""Generate the 5 publication-quality figures for report/ieee_report.tex.

All figures are computed directly from the cleaned PM25Vision metadata,
authoritative 5-fold CV and one-shot test results in results/REPORT_NUMBERS.md,
and tracked per-epoch learning curves in results/curves/ (with fallback to
outputs/curves/).

Outputs (saved at 300 DPI in report/figures/):
  1. ieee_fig1_dataset_splits.png  (2x1 single-column: AQI bins + spatial CDF)
  2. ieee_fig2_architecture.png    (full-width figure*: PFRML spine + Fusion)
  3. ieee_fig3_backbones_fold3.png (2x1 single-column: transfer gap + Fold 3)
  4. ieee_fig4_pfrml_collapse.png  (2x1 single-column: PFRML ladder + loss curves)
  5. ieee_fig5_fusion_test.png     (2x1 single-column: CMA-3 ep 7 + final test CIs)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import compute_test_distance_mask
from src.data.cleaning import load_and_clean_metadata
from src.data.splits import (
    build_cv_splits,
    compute_val_to_train_nearest_km,
)

FIG_DIR = ROOT / "report" / "figures"
RESULTS_CURVES_DIR = ROOT / "results" / "curves"
OUTPUTS_CURVES_DIR = ROOT / "outputs" / "curves"

# Colorblind-safe academic palette (Okabe-Ito / IEEE style)
PAL = {
    "navy": "#1B365D",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "amber": "#E69F00",
    "orange": "#D55E00",
    "red": "#C0392B",
    "purple": "#8E44AD",
    "slate": "#6C7A89",
    "dark": "#222222",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8.6,
            "axes.titlesize": 9.3,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.6,
            "axes.labelweight": "bold",
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "legend.frameon": True,
            "legend.framealpha": 0.94,
            "legend.edgecolor": "#CCCCCC",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def aqi_to_bin(y: np.ndarray) -> np.ndarray:
    bins = np.zeros(len(y), dtype=int)
    bins[(y > 50) & (y <= 100)] = 1
    bins[(y > 100) & (y <= 150)] = 2
    bins[(y > 150) & (y <= 200)] = 3
    bins[(y > 200) & (y <= 300)] = 4
    bins[y > 300] = 5
    return bins


def load_5fold_curves(exp_id: str, group_mode: str = "station") -> pd.DataFrame:
    dfs = []
    for f in range(5):
        fname = f"{exp_id}_{group_mode}_fold{f}.csv"
        p = RESULTS_CURVES_DIR / fname
        if not p.exists():
            p = OUTPUTS_CURVES_DIR / fname
        df = pd.read_csv(p)
        df["fold"] = f
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Figure 1: US EPA AQI Bin Distribution & Nearest-Station Spatial Separation
# ---------------------------------------------------------------------------
def make_fig1(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.8, 5.5))

    dist_km, mask_gt25 = compute_test_distance_mask(train_df, test_df, min_km=25.0)
    gt25_df = test_df[mask_gt25]

    bin_labels = [
        "Good\n(0–50)",
        "Mod.\n(51–100)",
        "USG\n(101–150)",
        "Unhealthy\n(151–200)",
        "Very Unh.\n(201–300)",
        "Hazardous\n(>300)",
    ]
    splits = [
        ("Cleaned Train (n=8,176 | 2,527 st.)", train_df["pm25"].values, PAL["blue"]),
        ("Full Test (n=2,920 | 729 st.)", test_df["pm25"].values, PAL["amber"]),
        ("Test >25 km (n=717 | 205 st.)", gt25_df["pm25"].values, PAL["orange"]),
    ]

    x = np.arange(6)
    w = 0.26
    for idx, (label, vals, color) in enumerate(splits):
        counts = np.bincount(aqi_to_bin(vals), minlength=6)
        pcts = 100.0 * counts / len(vals)
        bars = ax1.bar(
            x + (idx - 1) * w,
            pcts,
            width=w,
            label=label,
            color=color,
            alpha=0.90,
            edgecolor="white",
        )
        for b, pct in zip(bars, pcts):
            if pct >= 0.4:
                ax1.text(
                    b.get_x() + b.get_width() / 2,
                    pct + 0.6,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6.3,
                    rotation=90,
                )

    ax1.axvspan(3.5, 5.5, color="#FDECEC", alpha=0.58, zorder=0)
    ax1.text(
        4.5,
        41.5,
        "Severe Tail (AQI > 200)\n13.3% of Train (n=1,086)",
        ha="center",
        va="top",
        fontsize=7.3,
        color=PAL["red"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=PAL["red"], lw=0.7),
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, fontsize=7.4)
    ax1.set_ylabel("Share of Split (%)")
    ax1.set_ylim(0, 45.5)
    ax1.set_title("(a) US EPA AQI 6-Bin Distribution & Long-Tail Regime")
    ax1.legend(loc="upper left", fontsize=7.0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b): Empirical CDF of distance to nearest training station
    st_df, st_folds = build_cv_splits(
        train_df, n_splits=5, group_mode="station", spatial_cluster_km=25.0
    )
    sc_df, sc_folds = build_cv_splits(
        train_df, n_splits=5, group_mode="spatial_cluster", spatial_cluster_km=25.0
    )

    st_dists = np.concatenate(
        [compute_val_to_train_nearest_km(st_df, tr, va) for tr, va in st_folds]
    )
    sc_dists = np.concatenate(
        [compute_val_to_train_nearest_km(sc_df, tr, va) for tr, va in sc_folds]
    )

    for label, d_arr, color, ls in [
        (
            f"Station 5-Fold CV (med={np.median(st_dists):.1f} km)",
            st_dists,
            PAL["blue"],
            "-",
        ),
        (
            f"25 km Spatial-Cluster CV (med={np.median(sc_dists):.1f} km)",
            sc_dists,
            PAL["green"],
            "-",
        ),
        (
            f"Full Test vs. Train (med={np.median(dist_km):.1f} km)",
            dist_km,
            PAL["amber"],
            "--",
        ),
        (
            f"Test >25 km Subset (med={np.median(dist_km[mask_gt25]):.1f} km)",
            dist_km[mask_gt25],
            PAL["orange"],
            "-.",
        ),
    ]:
        s = np.sort(np.maximum(d_arr, 0.05))
        y = np.linspace(0, 100, len(s))
        ax2.plot(s, y, label=label, color=color, linestyle=ls, linewidth=1.9)

    ax2.axvline(25.0, color=PAL["red"], linestyle=":", linewidth=1.5)
    ax2.text(
        29.0,
        14.0,
        "25 km urban\ncutoff",
        color=PAL["red"],
        fontsize=7.5,
        fontweight="bold",
    )
    ax2.set_xscale("log")
    ax2.set_xlim(0.05, 3000)
    ax2.set_ylim(0, 103)
    ax2.set_xlabel("Haversine Distance to Nearest Training Station (km, log scale)")
    ax2.set_ylabel("Cumulative Images (%)")
    ax2.set_title("(b) Spatial Separation Across CV and Test Splits")
    ax2.legend(loc="upper left", fontsize=6.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    out_path = FIG_DIR / "ieee_fig1_dataset_splits.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: Unified Visual Spine, PFRML Ablation Rungs, and Multimodal Fusion
# ---------------------------------------------------------------------------
def _box(
    ax,
    x,
    y,
    w,
    h,
    title,
    sub="",
    fc="#F4F8FB",
    ec="#1B365D",
    title_c="#111111",
    lw=1.3,
    fs=8.4,
    sub_fs=7.1,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(patch)
    if sub:
        ax.text(
            x + w / 2,
            y + h * 0.68,
            title,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold",
            color=title_c,
            zorder=4,
        )
        ax.text(
            x + w / 2,
            y + h * 0.30,
            sub,
            ha="center",
            va="center",
            fontsize=sub_fs,
            color="#333333",
            zorder=4,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold",
            color=title_c,
            zorder=4,
        )


def _arrow(ax, x1, y1, x2, y2, color="#333333", ls="-", lw=1.4) -> None:
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>,head_length=5,head_width=3.2",
        color=color,
        linestyle=ls,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(arr)


def make_fig2() -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 4.5))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    ax.grid(False)

    ax.text(
        0.5,
        0.96,
        "Unified Multi-Stage Visual Spine, PFRML Ablation Rungs (R1 vs. R2 + T1–T4), and Multimodal Fusion Network",
        ha="center",
        va="center",
        fontsize=10.4,
        fontweight="bold",
        color=PAL["navy"],
    )

    # Vision Stream (Middle-Upper Row)
    _box(
        ax,
        0.015,
        0.53,
        0.115,
        0.22,
        "RGB Street Image",
        "224×224×3 Bilinear\nPhysics-Preserving\n(No Color Jitter)",
        fc="#EEF6FC",
        ec=PAL["blue"],
    )
    _box(
        ax,
        0.155,
        0.53,
        0.155,
        0.22,
        "4-Stage Backbone",
        "MobileNetV3-S (Pretrained)\nStages [16, 24, 48, 576]\n(or MDNet-T Scratch)",
        fc="#E8F3FA",
        ec=PAL["blue"],
    )
    _box(
        ax,
        0.335,
        0.53,
        0.150,
        0.22,
        "DCM Neck",
        "1×1 Conv → D=224\nAlign 14×14, Sum\n(T4: Concat Stalls)",
        fc="#E8F3FA",
        ec=PAL["blue"],
    )
    _box(
        ax,
        0.510,
        0.53,
        0.165,
        0.22,
        "Light Transformer",
        "2 Layers, 4 Heads\n196+CLS Tokens, 2D Pos\nGAP (T1: Attn Pool)",
        fc="#E8F3FA",
        ec=PAL["blue"],
    )

    _arrow(ax, 0.130, 0.64, 0.155, 0.64, PAL["blue"])
    _arrow(ax, 0.310, 0.64, 0.335, 0.64, PAL["blue"])
    _arrow(ax, 0.485, 0.64, 0.510, 0.64, PAL["blue"])

    # Latent chi_i node
    _box(
        ax,
        0.705,
        0.57,
        0.070,
        0.14,
        "χ_i",
        "224-D\nVisual",
        fc="#FFF7E6",
        ec=PAL["amber"],
    )
    _arrow(ax, 0.675, 0.64, 0.705, 0.64, PAL["blue"])

    # Optional PhysicsPriorGate T2 (Top Row)
    _box(
        ax,
        0.18,
        0.80,
        0.54,
        0.11,
        "Optional Toggle T2: Differentiable Physics Prior Gate (4 Optical Scalars)",
        "[Luminance Entropy, RMS Contrast, Dark-Channel Transmission t, Extinction −ln(t)] → Sigmoid Gate",
        fc="#F5F5F5",
        ec=PAL["slate"],
        lw=1.0,
        fs=7.6,
        sub_fs=6.7,
    )
    _arrow(ax, 0.075, 0.75, 0.18, 0.855, PAL["slate"], ls="--", lw=1.0)
    _arrow(ax, 0.72, 0.855, 0.740, 0.71, PAL["slate"], ls="--", lw=1.0)

    # Vision Aux Head (Upper Right)
    _box(
        ax,
        0.810,
        0.70,
        0.175,
        0.19,
        "Aux Vision Head",
        "χ_i → 256 → 1 (R1 Reg Head)\nL_aux (λ_aux = 0.5)\nDetached From Metadata",
        fc="#E8F8F2",
        ec=PAL["green"],
        lw=1.5,
    )
    _arrow(ax, 0.775, 0.66, 0.810, 0.78, PAL["green"], lw=1.5)

    # Prototype Branch (R2 vs R1 contrast)
    _box(
        ax,
        0.31,
        0.33,
        0.37,
        0.15,
        "Rung R2 Only: 6-Class EMA Prototype Head (OFF in R1)",
        "Unit-Hypersphere Prototypes p_k (T3: Class Weights)\nc_proto = Σ P(y=k|χ_i) p_k  [R2: 48.14 vs. R1: 45.50 CV MAE]",
        fc="#FDECEC",
        ec=PAL["red"],
        lw=1.1,
        fs=7.5,
        sub_fs=6.7,
    )
    _arrow(ax, 0.715, 0.57, 0.665, 0.48, PAL["red"], ls="--", lw=1.1)

    # Metadata Stream (Bottom Row)
    _box(
        ax,
        0.015,
        0.09,
        0.155,
        0.19,
        "GPS & Month Input",
        "5-D Unit-Sphere + Cyclic\n[x, y, z, sin(m), cos(m)]",
        fc="#FFF0E6",
        ec=PAL["orange"],
    )
    _box(
        ax,
        0.195,
        0.09,
        0.175,
        0.19,
        "Metadata MLP (E1)",
        "5 → 64 → 32 (2,497 params)\nRegional Prior m_i (32-D)\n(Collapses >25 km alone)",
        fc="#FFF0E6",
        ec=PAL["orange"],
    )
    _box(
        ax,
        0.395,
        0.09,
        0.240,
        0.19,
        "Modality Dropout & Eval Gate",
        "Zeroes m_i w/ prob p_drop ∈ {0.3, 0.0}\nEval: Normal (m_i active) vs.\nMetadata-Zeroed (meta_off, m_i = 0)",
        fc="#FFF0E6",
        ec=PAL["orange"],
    )

    _arrow(ax, 0.170, 0.185, 0.195, 0.185, PAL["orange"])
    _arrow(ax, 0.370, 0.185, 0.395, 0.185, PAL["orange"])

    # Fusion Concat + Main Reg Head (Bottom Right)
    _box(
        ax,
        0.675,
        0.09,
        0.120,
        0.19,
        "Fusion Concat",
        "[χ_i ; m_i]\n224 + 32 = 256-D",
        fc="#F3ECF8",
        ec=PAL["purple"],
        lw=1.5,
    )
    _box(
        ax,
        0.815,
        0.09,
        0.170,
        0.19,
        "Main Reg. Head",
        "256 → 256 → 1 (Scaled AQI)\nF_FUSION_R1 (2.02M, 7ep)\nTest MAE: 35.77 / 39.22",
        fc="#E8F8F2",
        ec=PAL["green"],
        lw=1.7,
    )

    _arrow(ax, 0.635, 0.185, 0.675, 0.185, PAL["orange"], lw=1.5)
    _arrow(ax, 0.740, 0.57, 0.740, 0.28, PAL["blue"], lw=1.6)
    _arrow(ax, 0.795, 0.185, 0.815, 0.185, PAL["purple"], lw=1.6)

    ax.text(
        0.5,
        0.02,
        "Best Overall Configuration (F_FUSION_R1): Pretrained MobileNetV3-S + DCM (Sum) + Light Transformer (GAP) + Metadata MLP, Prototypes OFF (R1), CMA-3 Early Stop at N=7 Epochs.",
        ha="center",
        va="center",
        fontsize=7.8,
        style="italic",
        color="#222222",
    )

    plt.tight_layout()
    out_path = FIG_DIR / "ieee_fig2_architecture.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 3: Image-Only Backbones Transfer Gap & Fold-3 Tail Sensitivity
# ---------------------------------------------------------------------------
def make_fig3() -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.8, 5.6))

    models = [
        ("Constant Median (E0_median)", 66.21, 4.89, 0.0000, PAL["slate"]),
        ("Haze-Stats Ridge (E0_ridge)", 64.11, 4.40, 0.0317, PAL["slate"]),
        ("MDNet-T Scratch 60ep (E3)", 53.29, 6.38, 0.1951, PAL["orange"]),
        ("ResNet-18 Scratch 30ep (E2)", 53.27, 7.11, 0.1954, PAL["orange"]),
        ("MobileNetV3-S Pretrained (E2)", 45.66, 8.41, 0.3103, PAL["blue"]),
        ("MobileNetV3-S + MD-Neck (E3b)", 44.92, 8.54, 0.3214, PAL["blue"]),
        ("ResNet-18 Pretrained 40ep (E2)", 44.05, 8.70, 0.3346, PAL["blue"]),
        ("GPS + Month MLP (E1_metadata)", 43.15, 7.47, 0.3483, PAL["amber"]),
    ]
    y_pos = np.arange(len(models))
    maes = [m[1] for m in models]
    sds = [m[2] for m in models]
    colors = [m[4] for m in models]
    labels = [m[0] for m in models]

    ax1.barh(
        y_pos,
        maes,
        xerr=sds,
        height=0.62,
        color=colors,
        alpha=0.89,
        edgecolor="white",
        capsize=2.8,
        error_kw={"ecolor": "#333333", "lw": 1.0},
    )
    for i, (_, mae, sd, skill, _) in enumerate(models):
        sk_txt = (
            f" {mae:.2f}±{sd:.2f} ({skill:+.1%})"
            if skill > 0
            else f" {mae:.2f}±{sd:.2f} (Ref)"
        )
        ax1.text(
            mae + sd + 0.8,
            i,
            sk_txt,
            va="center",
            ha="left",
            fontsize=6.8,
            fontweight="bold" if mae < 46 else "normal",
        )

    ax1.axvline(66.21, color=PAL["slate"], linestyle=":", linewidth=1.1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=7.3)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 118)
    ax1.set_xlabel("5-Fold Station CV Last-Epoch MAE (Mean ± SD)")
    ax1.set_title("(a) Transfer Learning Gap (−7.6 to −9.2 MAE) in Station CV")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b): Per-Fold MAE profiles showing Fold 3 tail-variance spike
    folds = np.arange(5)
    per_fold = {
        "Constant Median (E0)": ([67.18, 62.31, 63.71, 74.33, 63.50], PAL["slate"], ":", "o"),
        "MDNet-T Scratch (E3)": ([48.37, 50.74, 53.97, 64.06, 49.31], PAL["orange"], "--", "s"),
        "MobileNetV3-S Pre (E2)": ([36.27, 46.10, 42.96, 59.21, 43.77], PAL["blue"], "-", "^"),
        "MobileNetV3-S+MD (E3b)": ([34.51, 45.40, 43.31, 58.25, 43.14], PAL["sky"], "-", "P"),
        "ResNet-18 Pre (E2)": ([34.85, 43.35, 42.06, 58.45, 41.55], PAL["green"], "-", "D"),
        "Metadata MLP (E1)": ([39.71, 38.83, 38.11, 56.09, 42.98], PAL["amber"], "-.", "v"),
    }
    ax2.axvspan(2.68, 3.32, color="#FDECEC", alpha=0.65, zorder=0)
    for name, (vals, col, ls, mk) in per_fold.items():
        ax2.plot(
            folds,
            vals,
            label=name,
            color=col,
            linestyle=ls,
            marker=mk,
            linewidth=1.7,
            markersize=5.0,
        )

    ax2.text(
        3.0,
        30.5,
        "Fold 3 Tail Spike (Val SD=108.2)\n4 Grid Cells = 76.9% of Bins 5–6",
        ha="center",
        va="bottom",
        fontsize=6.9,
        color=PAL["red"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=PAL["red"], lw=0.7),
    )
    ax2.set_xticks(folds)
    ax2.set_xticklabels([f"Fold {f}" for f in folds], fontsize=7.6)
    ax2.set_ylabel("Validation MAE (AQI)")
    ax2.set_ylim(28, 86)
    ax2.set_title("(b) Per-Fold MAE Trajectory & Fold-3 Severe-Tail Spike")
    ax2.legend(loc="upper left", ncol=2, fontsize=6.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    out_path = FIG_DIR / "ieee_fig3_backbones_fold3.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 4: PFRML 5-Fold Ablation Ladder & Optimization Failure Diagnostics
# ---------------------------------------------------------------------------
def make_fig4() -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.8, 5.9))

    ladder = [
        ("R1: No Prototypes (w/o proto)", 45.50, 8.64, "−0.16 vs E2 (4/5)", PAL["green"]),
        ("E2: MobileNetV3-S Pretrained", 45.66, 8.41, "Reference", PAL["blue"]),
        ("R2+T3: + Class Weights", 47.06, 8.01, "−1.08 vs R2 (4/5)", PAL["sky"]),
        ("R2+all: PFRML Full (T1+T2+T3)", 47.16, 7.15, "−0.98 vs R2 (4/5)", PAL["sky"]),
        ("R2+T2: + Physics Prior Gate", 47.57, 6.73, "−0.57 vs R2 (4/5)", PAL["sky"]),
        ("R2 Eq. 5: reg_input=chi", 48.04, 8.20, "+2.54 vs R1 (0/5)", PAL["amber"]),
        ("R2: PFRML Baseline (chi_proto)", 48.14, 7.61, "+2.63 vs R1 (1/5)", PAL["amber"]),
        ("R2+T1: + Attention Pooling", 48.40, 7.74, "+0.27 vs R2 (2/5)", PAL["amber"]),
        ("R2+T4: + DCM Concat [Fail]", 60.51, 5.41, "+15.01 vs R1 (0/5)", PAL["orange"]),
        ("Extra B: MDNet-T+PFRML [Stall]", 66.56, 4.88, "+21.05 vs R1 (0/5)", PAL["red"]),
    ]
    y = np.arange(len(ladder))
    maes = [r[1] for r in ladder]
    sds = [r[2] for r in ladder]
    cols = [r[4] for r in ladder]

    ax1.barh(
        y,
        maes,
        xerr=sds,
        height=0.64,
        color=cols,
        alpha=0.89,
        edgecolor="white",
        capsize=2.5,
        error_kw={"ecolor": "#333333", "lw": 0.95},
    )
    for i, (_, m, sd, note, _) in enumerate(ladder):
        ax1.text(
            m + sd + 0.7,
            i,
            f"{m:.2f}±{sd:.2f} [{note}]",
            va="center",
            ha="left",
            fontsize=6.5,
            fontweight="bold" if i == 0 else "normal",
        )

    ax1.axvline(45.50, color=PAL["green"], linestyle="--", linewidth=1.1)
    ax1.set_yticks(y)
    ax1.set_yticklabels([r[0] for r in ladder], fontsize=7.1)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 125)
    ax1.set_xlabel("5-Fold Station CV Last-Epoch MAE (Mean ± SD)")
    ax1.set_title("(a) PFRML 5-Fold Station CV Ablation Ladder")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b): Train/Val loss trajectories
    curve_specs = [
        (
            "E4_mdnet_t_baseline_ep60",
            "train_loss",
            "MDNet-T+PFRML train_loss (Stalled ~2.17)",
            PAL["red"],
            "-",
        ),
        (
            "E4_mobilenetv3_s_dcm_concat",
            "train_loss",
            "R2+T4 DCM Concat train_loss (2.06→1.92)",
            PAL["orange"],
            "-",
        ),
        (
            "E4_mobilenetv3_s_baseline",
            "val_loss",
            "PFRML R2 val_loss (CE Overconfidence U-Curve)",
            PAL["blue"],
            "--",
        ),
        (
            "E4_mobilenetv3_s_baseline",
            "train_loss",
            "PFRML R2 train_loss (Converges 1.99→0.46)",
            PAL["blue"],
            "-",
        ),
        (
            "E3_mdnet_t_reg_lr1e-3_ep60",
            "train_loss",
            "MDNet-T Reg-Only train_loss (0.85→0.03)",
            PAL["green"],
            "-",
        ),
    ]
    for exp_id, col_name, lbl, color, ls in curve_specs:
        df = load_5fold_curves(exp_id, "station")
        g = df.groupby("epoch")[col_name].agg(["mean", "std"]).reset_index()
        ax2.plot(
            g["epoch"],
            g["mean"],
            label=lbl,
            color=color,
            linestyle=ls,
            linewidth=1.8,
        )
        ax2.fill_between(
            g["epoch"],
            g["mean"] - g["std"],
            g["mean"] + g["std"],
            color=color,
            alpha=0.11,
        )

    ax2.annotate(
        "CE overconfidence:\nval_loss rises ep 4→30\nwhile val MAE falls",
        xy=(22, 2.38),
        xytext=(34.0, 2.50),
        arrowprops=dict(arrowstyle="->", color=PAL["blue"], lw=1.1),
        fontsize=6.8,
        color=PAL["navy"],
        bbox=dict(boxstyle="round,pad=0.22", fc="#EEF6FC", ec=PAL["blue"], lw=0.7),
    )

    ax2.set_xlabel("Training Epoch")
    ax2.set_ylabel("5-Fold Mean Loss (±1 SD)")
    ax2.set_xlim(1, 60)
    ax2.set_ylim(0, 3.05)
    ax2.set_title("(b) Loss Trajectories: Val-Loss U-Curve & Optimization Stalls")
    ax2.legend(loc="center right", bbox_to_anchor=(0.99, 0.46), fontsize=6.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    out_path = FIG_DIR / "ieee_fig4_pfrml_collapse.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 5: CMA-3 Early-Stopping Epoch Selection & One-Shot Test Generalization
# ---------------------------------------------------------------------------
def make_fig5() -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.8, 5.9))

    df_p00 = load_5fold_curves("E5_fusion_r1_p00", "spatial_cluster")
    g = df_p00.groupby("epoch")["mae"].agg(["mean", "std"]).reset_index()
    cma3 = g["mean"].rolling(window=3, center=True, min_periods=1).mean()

    ax1.fill_between(
        g["epoch"],
        g["mean"] - g["std"],
        g["mean"] + g["std"],
        color=PAL["blue"],
        alpha=0.14,
        label="5-Fold ±1 SD",
    )
    ax1.plot(
        g["epoch"],
        g["mean"],
        color=PAL["blue"],
        alpha=0.65,
        linewidth=1.4,
        marker="o",
        markersize=3.2,
        label="30ep R1 Fusion Mean Val MAE",
    )
    ax1.plot(
        g["epoch"],
        cma3,
        color=PAL["navy"],
        linewidth=2.1,
        label="3-Epoch Centered MA (CMA-3)",
    )

    ax1.axhline(
        46.00,
        color=PAL["amber"],
        linestyle="--",
        linewidth=1.3,
        label="E1 Metadata MLP (46.00)",
    )
    ax1.axhline(
        50.69,
        color=PAL["slate"],
        linestyle=":",
        linewidth=1.3,
        label="E2 Image-Only Spatial (50.69)",
    )

    ax1.scatter([7], [45.74], color=PAL["green"], s=70, zorder=6, edgecolor="black")
    ax1.scatter(
        [7],
        [44.40],
        color=PAL["red"],
        marker="*",
        s=115,
        zorder=7,
        edgecolor="black",
        label="Fresh 7ep Run (44.40±7.30)",
    )
    ax1.annotate(
        "CMA-3 Min at N = 7\n30ep MAE = 45.74 (CMA-3 = 46.26)\nFresh 7ep CV = 44.40 (Wins 5/5 Folds)",
        xy=(7, 44.8),
        xytext=(11.5, 55.2),
        arrowprops=dict(arrowstyle="->", color=PAL["green"], lw=1.2),
        fontsize=6.8,
        fontweight="bold",
        color=PAL["green"],
        bbox=dict(boxstyle="round,pad=0.24", fc="#E8F8F2", ec=PAL["green"], lw=0.8),
    )

    ax1.set_xlabel("Training Epoch (25 km Spatial-Cluster 5-Fold CV)")
    ax1.set_ylabel("Validation MAE (AQI)")
    ax1.set_xlim(1, 30)
    ax1.set_ylim(37, 66)
    ax1.set_title("(a) CMA-3 Early-Stopping Epoch Selection (N = 7) in Spatial CV")
    ax1.legend(loc="upper right", fontsize=6.4, ncol=2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b): Final One-Shot Test Set vs Unseen-Region >25 km Subset
    models = [
        ("F_FUSION_R1\n(Fusion 7ep)", 35.77, 30.47, 42.72, 39.22, 30.85, 47.74, 0.61, 0.37),
        ("F_E2\n(Vision Only)", 41.12, 33.97, 50.79, 39.60, 28.06, 54.03, 0.42, 0.27),
        ("E1_metadata\n(GPS+Month)", 40.34, 34.45, 48.31, 49.84, 40.18, 58.81, 0.50, -0.04),
        ("F_PFRML_full\n(PFRML Vision)", 45.04, 37.51, 54.04, 46.79, 34.22, 61.06, 0.29, -0.07),
        ("F_FUSION_R1\n(meta_off)", 47.56, 39.57, 58.12, 44.88, 32.15, 58.48, 0.34, 0.13),
    ]

    x = np.arange(len(models))
    w = 0.36
    full_vals = [m[1] for m in models]
    full_err = [[m[1] - m[2] for m in models], [m[3] - m[1] for m in models]]
    gt25_vals = [m[4] for m in models]
    gt25_err = [[m[4] - m[5] for m in models], [m[6] - m[4] for m in models]]

    ax2.bar(
        x - w / 2,
        full_vals,
        yerr=full_err,
        width=w,
        color=PAL["blue"],
        alpha=0.90,
        capsize=2.8,
        edgecolor="white",
        error_kw={"ecolor": "#222222", "lw": 0.95},
        label="Full Test (n=2,920, 729 st.)",
    )
    ax2.bar(
        x + w / 2,
        gt25_vals,
        yerr=gt25_err,
        width=w,
        color=PAL["orange"],
        alpha=0.90,
        capsize=2.8,
        edgecolor="white",
        error_kw={"ecolor": "#222222", "lw": 0.95},
        label="Unseen >25 km (n=717, 205 st.)",
    )

    for i, m in enumerate(models):
        ax2.text(
            x[i] - w / 2,
            m[3] + 0.9,
            f"{m[1]:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.6,
            fontweight="bold" if i == 0 else "normal",
            color=PAL["navy"],
        )
        ax2.text(
            x[i] + w / 2,
            m[6] + 0.9,
            f"{m[4]:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.6,
            fontweight="bold" if i in (0, 2) else "normal",
            color="#8A3300",
        )

    ax2.annotate(
        "Metadata collapses >25 km:\nMAE 40.3→49.8 (R² = −0.04)",
        xy=(2 + w / 2, 65.5),
        xytext=(2.35, 73.5),
        arrowprops=dict(arrowstyle="->", color=PAL["red"], lw=1.1),
        fontsize=6.6,
        fontweight="bold",
        color=PAL["red"],
        bbox=dict(boxstyle="round,pad=0.22", fc="#FDECEC", ec=PAL["red"], lw=0.7),
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels([m[0] for m in models], fontsize=7.1)
    ax2.set_ylabel("Test MAE [95% Station-Bootstrap CI]")
    ax2.set_ylim(0, 84)
    ax2.set_title("(b) One-Shot Held-Out Test vs. Unseen-Region (>25 km) MAE")
    ax2.legend(loc="upper left", fontsize=6.7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    out_path = FIG_DIR / "ieee_fig5_fusion_test.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    print("Loading and cleaning PM25Vision metadata...")
    train_df, test_df = load_and_clean_metadata(
        ROOT / "datasets" / "train" / "metadata.csv",
        ROOT / "datasets" / "test" / "metadata.csv",
    )

    f1 = make_fig1(train_df, test_df)
    print(f"Saved {f1}")
    f2 = make_fig2()
    print(f"Saved {f2}")
    f3 = make_fig3()
    print(f"Saved {f3}")
    f4 = make_fig4()
    print(f"Saved {f4}")
    f5 = make_fig5()
    print(f"Saved {f5}")
    print("All 5 IEEE report figures generated successfully.")


if __name__ == "__main__":
    main()
