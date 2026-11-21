#!/usr/bin/env python3
"""Generate 5 story-driven, intuitive figures for report/ieee_story_report.tex.

These figures tell the chronological story of the project in clear, accessible
visual language while remaining 100% faithful to results/REPORT_NUMBERS.md:
  1. story_fig1_eda_can_of_worms.png (full-width figure*: real street photos
     across the 6 AQI bins + the cleaning funnel/long tail + the 25 km urban
     clustering & Fold-3 hotspot diagnostic)
  2. story_fig2_two_papers_and_pivot.png (full-width figure*: how we combined
     Paper 2 MDNet + Paper 1 PFRML + Literature ideas, where prototypes hurt,
     and how our final 7-epoch R1 + GPS/Month Fusion works)
  3. story_fig3_first_experiments_to_transfer_win.png (single-column 2x1:
     from our first failed/scratch experiments to the ImageNet pretraining win
     + the Fold-3 smog spike across all 5 folds)
  4. story_fig4_why_pfrml_prototypes_backfired.png (single-column 2x1:
     peeling the PFRML ablation ladder + the learning-curve smoking gun showing
     flatlined runs and the validation-loss U-curve)
  5. story_fig5_gps_cheat_code_and_fusion_win.png (single-column 2x1:
     catching the epoch-7 overfitting sweet spot in 25 km city-cluster CV +
     final test vs. >25 km unseen-region showdown)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image

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
TRAIN_IMG_DIR = ROOT / "datasets" / "train" / "images"

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
            "legend.fontsize": 7.3,
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
# Story Figure 1: Opening the PM25Vision "Can of Worms" (Samples + EDA)
# ---------------------------------------------------------------------------
def make_story_fig1(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(11.4, 5.7))
    gs = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.0, 1.38],
        hspace=0.44,
    )
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[0], wspace=0.22)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.22)

    # Top row: 6 real sample images (one from each US EPA AQI bin)
    bin_specs = [
        (0, "Bin 1: Good\n(0–50 AQI)", "#2ECC71", 25.0),
        (1, "Bin 2: Moderate\n(51–100 AQI)", "#F1C40F", 75.0),
        (2, "Bin 3: Sensitive\n(101–150 AQI)", "#E67E22", 125.0),
        (3, "Bin 4: Unhealthy\n(151–200 AQI)", "#E74C3C", 175.0),
        (4, "Bin 5: Very Unh.\n(201–300 AQI)", "#8E44AD", 250.0),
        (5, "Bin 6: Hazardous\n(>300 AQI)", "#641E16", 420.0),
    ]
    train_bins = aqi_to_bin(train_df["pm25"].values)

    for b_idx, b_title, b_col, target_aqi in bin_specs:
        ax_img = fig.add_subplot(gs_top[0, b_idx])
        sub = train_df[train_bins == b_idx].copy()
        sub["diff"] = (sub["pm25"] - target_aqi).abs()
        sub = sub.sort_values(["diff", "image_id"])
        chosen_img = None
        chosen_row = None
        for _, row in sub.iterrows():
            cand = TRAIN_IMG_DIR / f"{row['image_id']}.jpg"
            if not cand.exists():
                cand = ROOT / row["filename"]
            if cand.exists():
                chosen_img = cand
                chosen_row = row
                break

        if chosen_img is not None:
            im = Image.open(chosen_img).convert("RGB").resize((224, 160))
            ax_img.imshow(im)
            aqi_val = float(chosen_row["pm25"])
            lat_val = float(chosen_row["latitude"])
            lon_val = float(chosen_row["longitude"])
            ax_img.set_xlabel(
                f"Measured AQI: {aqi_val:.0f}\nGPS: ({lat_val:.1f}°, {lon_val:.1f}°)",
                fontsize=7.0,
                fontweight="bold",
                labelpad=3,
            )
        ax_img.set_title(b_title, fontsize=8.0, fontweight="bold", color=b_col, pad=4)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for sp in ax_img.spines.values():
            sp.set_edgecolor(b_col)
            sp.set_linewidth(2.0)

    # Bottom-Left: The Long-Tail Imbalance & Cleaning Audit
    ax1 = fig.add_subplot(gs_bot[0, 0])
    dist_km, mask_gt25 = compute_test_distance_mask(train_df, test_df, min_km=25.0)
    gt25_df = test_df[mask_gt25]

    bin_labels = [
        "Good\n(0–50)",
        "Moderate\n(51–100)",
        "Sensitive\n(101–150)",
        "Unhealthy\n(151–200)",
        "Very Unh.\n(201–300)",
        "Hazardous\n(>300)",
    ]
    splits = [
        ("Cleaned Train (8,176 photos, 2,527 st.)", train_df["pm25"].values, PAL["blue"]),
        ("Full Test (2,920 photos, 729 st.)", test_df["pm25"].values, PAL["amber"]),
        ("Far Test >25 km (717 photos, 205 st.)", gt25_df["pm25"].values, PAL["orange"]),
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
                    fontsize=6.4,
                    rotation=90,
                )

    ax1.axvspan(3.5, 5.5, color="#FDECEC", alpha=0.60, zorder=0)
    ax1.text(
        4.55,
        46.0,
        "Can of Worms #1 & #2:\n• Purged 18 leaked photos\n• Deduplicated stations\n• Tail (>200) is only 13.3%!",
        ha="center",
        va="top",
        fontsize=6.8,
        color=PAL["red"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=PAL["red"], lw=0.8),
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, fontsize=7.5)
    ax1.set_ylabel("Share of Photos (%)")
    ax1.set_ylim(0, 49.0)
    ax1.set_title("(a) EDA Surprise #1: Data Leakage & The 13.3% Severe Tail")
    ax1.legend(loc="upper left", fontsize=6.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Bottom-Right: The GPS Shortcut Loophole (Distance CDF)
    ax2 = fig.add_subplot(gs_bot[0, 1])
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
            f"Station 5-Fold CV (median = {np.median(st_dists):.1f} km away)",
            st_dists,
            PAL["blue"],
            "-",
        ),
        (
            f"Full Test Set (median = {np.median(dist_km):.1f} km away)",
            dist_km,
            PAL["amber"],
            "--",
        ),
        (
            f"25 km City-Cluster CV (median = {np.median(sc_dists):.1f} km away)",
            sc_dists,
            PAL["green"],
            "-",
        ),
        (
            f"Unseen Regions >25 km (median = {np.median(dist_km[mask_gt25]):.1f} km away)",
            dist_km[mask_gt25],
            PAL["orange"],
            "-.",
        ),
    ]:
        s = np.sort(np.maximum(d_arr, 0.05))
        y = np.linspace(0, 100, len(s))
        ax2.plot(s, y, label=label, color=color, linestyle=ls, linewidth=2.0)

    ax2.axvline(25.0, color=PAL["red"], linestyle=":", linewidth=1.6)
    ax2.annotate(
        "Can of Worms #3 (The GPS Cheat Code):\n~75% of Test & Station-CV photos sit\nwithin 25 km of a training station!\nA GPS-only model can just memorize cities.",
        xy=(25.0, 74.0),
        xytext=(38.0, 26.0),
        arrowprops=dict(arrowstyle="->", color=PAL["red"], lw=1.2),
        fontsize=6.9,
        fontweight="bold",
        color=PAL["red"],
        bbox=dict(boxstyle="round,pad=0.25", fc="#FDECEC", ec=PAL["red"], lw=0.8),
    )
    ax2.set_xscale("log")
    ax2.set_xlim(0.05, 3000)
    ax2.set_ylim(0, 104)
    ax2.set_xlabel("Distance to Nearest Training Station (km, log scale)")
    ax2.set_ylabel("Cumulative Share of Photos (%)")
    ax2.set_title("(b) EDA Surprise #2: Why Standard Station Splits Let GPS Cheat")
    ax2.legend(loc="upper left", fontsize=6.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    out_path = FIG_DIR / "story_fig1_eda_can_of_worms.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Story Figure 2: Our Architectural Journey (From Two Papers to Our Pivot)
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
    sub_fs=7.0,
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
            y + h * 0.69,
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


def make_story_fig2() -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 4.5))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    ax.grid(False)

    ax.text(
        0.5,
        0.96,
        "How We Built & Tested Our Models: Combining Paper 1 (PFRML), Paper 2 (MDNet), Literature Ideas & GPS/Month Fusion",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color=PAL["navy"],
    )

    # Vision Stream
    _box(
        ax,
        0.015,
        0.53,
        0.115,
        0.22,
        "1. Street Photo",
        "224×224 Resize\nKeep Sky + Ground\nNo Color Jitter!",
        fc="#EEF6FC",
        ec=PAL["blue"],
    )
    _box(
        ax,
        0.155,
        0.53,
        0.155,
        0.22,
        "2. Visual Backbone",
        "Pretrained MobileNetV3-S\n(Wins: 45.66 MAE)\nvs. Scratch MDNet-T (53.29)",
        fc="#E8F3FA",
        ec=PAL["blue"],
    )
    _box(
        ax,
        0.335,
        0.53,
        0.150,
        0.22,
        "3. Paper 1 DCM Neck",
        "Aligns 4 Stages to 14×14\nElement-wise Sum (224-D)\n(T4 Concat Stalls: 60.51)",
        fc="#E8F3FA",
        ec=PAL["blue"],
    )
    _box(
        ax,
        0.510,
        0.53,
        0.165,
        0.22,
        "4. Light Transformer",
        "2 Layers, 4 Heads\n196 Spatial Tokens\nAverage Pool (or T1 Attn)",
        fc="#E8F3FA",
        ec=PAL["blue"],
    )

    _arrow(ax, 0.130, 0.64, 0.155, 0.64, PAL["blue"])
    _arrow(ax, 0.310, 0.64, 0.335, 0.64, PAL["blue"])
    _arrow(ax, 0.485, 0.64, 0.510, 0.64, PAL["blue"])

    _box(
        ax,
        0.705,
        0.57,
        0.072,
        0.14,
        "Image Vec",
        "224 Numbers\n(Haze Cues)",
        fc="#FFF7E6",
        ec=PAL["amber"],
    )
    _arrow(ax, 0.675, 0.64, 0.705, 0.64, PAL["blue"])

    # Literature Physics Gate T2
    _box(
        ax,
        0.17,
        0.80,
        0.56,
        0.11,
        "Literature Idea T2: Differentiable Physics Gate (4 Handcrafted Haze Numbers)",
        "Measures Brightness Entropy, Contrast, Dark-Channel Haze & Extinction → Blends with Image Vector",
        fc="#F5F5F5",
        ec=PAL["slate"],
        lw=1.0,
        fs=7.5,
        sub_fs=6.7,
    )
    _arrow(ax, 0.075, 0.75, 0.17, 0.855, PAL["slate"], ls="--", lw=1.0)
    _arrow(ax, 0.73, 0.855, 0.741, 0.71, PAL["slate"], ls="--", lw=1.0)

    # Aux Vision Head
    _box(
        ax,
        0.810,
        0.70,
        0.175,
        0.19,
        "Helper Vision Head",
        "Predicts AQI From Photo\nAlone (Weight 0.5)\nKeeps Backbone Honest!",
        fc="#E8F8F2",
        ec=PAL["green"],
        lw=1.5,
    )
    _arrow(ax, 0.777, 0.66, 0.810, 0.78, PAL["green"], lw=1.5)

    # Prototype Branch (What hurt!)
    _box(
        ax,
        0.29,
        0.33,
        0.39,
        0.15,
        "Paper 1's 6-Class Prototype Head (R2 — We Found This Hurts!)",
        "Forces photo into 6 AQI buckets (~41% acc) + T3 Class Weights\nDrags regression down from 45.50 (R1: OFF) to 48.14 (R2: ON)!",
        fc="#FDECEC",
        ec=PAL["red"],
        lw=1.2,
        fs=7.5,
        sub_fs=6.7,
    )
    _arrow(ax, 0.715, 0.57, 0.665, 0.48, PAL["red"], ls="--", lw=1.1)

    # Metadata Stream
    _box(
        ax,
        0.015,
        0.09,
        0.155,
        0.19,
        "5. Where & When",
        "GPS (Lat, Lon as 3D x,y,z)\n+ Month (sin, cos)\nOnly 5 Numbers!",
        fc="#FFF0E6",
        ec=PAL["orange"],
    )
    _box(
        ax,
        0.195,
        0.09,
        0.175,
        0.19,
        "Tiny Metadata MLP",
        "5 → 64 → 32 (2,497 params)\nGreat in same city (40.34)\nCrashes >25 km (49.84)!",
        fc="#FFF0E6",
        ec=PAL["orange"],
    )
    _box(
        ax,
        0.395,
        0.09,
        0.240,
        0.19,
        "GPS Switch (Dropout / Audit)",
        "Can randomly hide GPS in training\nAt test time we test Normal vs.\nGPS Unplugged (meta_off = 0)",
        fc="#FFF0E6",
        ec=PAL["orange"],
    )

    _arrow(ax, 0.170, 0.185, 0.195, 0.185, PAL["orange"])
    _arrow(ax, 0.370, 0.185, 0.395, 0.185, PAL["orange"])

    # Fusion Concat + Main Reg Head
    _box(
        ax,
        0.675,
        0.09,
        0.120,
        0.19,
        "6. Fuse Both",
        "224 Photo + 32 GPS\n= 256 Numbers\n(Prototypes OFF)",
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
        "7. Final AQI Prediction",
        "Stop Early at Epoch 7!\nFull Test MAE: 35.77 (Best)\n>25 km MAE: 39.22 (Best)",
        fc="#E8F8F2",
        ec=PAL["green"],
        lw=1.7,
    )

    _arrow(ax, 0.635, 0.185, 0.675, 0.185, PAL["orange"], lw=1.5)
    _arrow(ax, 0.741, 0.57, 0.741, 0.28, PAL["blue"], lw=1.6)
    _arrow(ax, 0.795, 0.185, 0.815, 0.185, PAL["purple"], lw=1.6)

    ax.text(
        0.5,
        0.02,
        "Our Winning Recipe (F_FUSION_R1): Keep the pretrained multi-stage visual spine (R1), drop the 6-class prototypes, fuse with GPS/month, and stop at Epoch 7!",
        ha="center",
        va="center",
        fontsize=7.8,
        style="italic",
        color="#222222",
    )

    plt.tight_layout()
    out_path = FIG_DIR / "story_fig2_two_papers_and_pivot.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Story Figure 3: First Experiments Flop -> Transfer Learning Saves Vision
# ---------------------------------------------------------------------------
def make_story_fig3() -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.8, 5.7))

    models = [
        ("1. First Combo: MDNet-T+PFRML [Flop!]", 66.56, 4.88, "Worse than guessing!", PAL["red"]),
        ("2. Blind Guess: Constant Median (E0)", 66.21, 4.89, "Zero-skill baseline", PAL["slate"]),
        ("3. 18 Haze Numbers: Ridge Reg (E0)", 64.11, 4.40, "Barely helps (+3.2%)", PAL["slate"]),
        ("4. Paper 2 MDNet-T From Scratch (E3)", 53.29, 6.38, "Data-starved (+19.5%)", PAL["orange"]),
        ("5. ResNet-18 From Scratch (E2)", 53.27, 7.11, "Ties MDNet-T (+19.5%)", PAL["orange"]),
        ("6. Pretrained MobileNetV3-Small (E2)", 45.66, 8.41, "Big jump! (+31.0%)", PAL["blue"]),
        ("7. Our Mashup: MobileNetV3+MD-Neck", 44.92, 8.54, "Only 1.04M params (+32.1%)", PAL["green"]),
        ("8. Pretrained ResNet-18 (E2, 40ep)", 44.05, 8.70, "Best pure vision (+33.5%)", PAL["blue"]),
        ("9. No-Pixel Cheat: GPS+Month MLP (E1)", 43.15, 7.47, "Wait—no photos?! (+34.8%)", PAL["amber"]),
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
        height=0.64,
        color=colors,
        alpha=0.89,
        edgecolor="white",
        capsize=2.6,
        error_kw={"ecolor": "#333333", "lw": 0.95},
    )
    for i, (_, mae, sd, note, _) in enumerate(models):
        ax1.text(
            mae + sd + 0.7,
            i,
            f"{mae:.2f}±{sd:.2f} [{note}]",
            va="center",
            ha="left",
            fontsize=6.3,
            fontweight="bold" if i in (0, 6, 7, 8) else "normal",
        )

    ax1.axvline(66.21, color=PAL["slate"], linestyle=":", linewidth=1.1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=7.0)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 145)
    ax1.set_xlabel("5-Fold Station CV Error (Last-Epoch MAE ± SD, Lower is Better)")
    ax1.set_title("(a) From Our First Failed Combo to the Pretraining Breakthrough")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b): Why 1 Fold Wasn't Enough (The Fold-3 Monster)
    folds = np.arange(5)
    per_fold = {
        "Blind Median Guess": ([67.18, 62.31, 63.71, 74.33, 63.50], PAL["slate"], ":", "o"),
        "MDNet-T (Scratch)": ([48.37, 50.74, 53.97, 64.06, 49.31], PAL["orange"], "--", "s"),
        "MobileNetV3-S (Pre)": ([36.27, 46.10, 42.96, 59.21, 43.77], PAL["blue"], "-", "^"),
        "Our MobileNetV3+MD": ([34.51, 45.40, 43.31, 58.25, 43.14], PAL["green"], "-", "P"),
        "ResNet-18 (Pre)": ([34.85, 43.35, 42.06, 58.45, 41.55], PAL["navy"], "-", "D"),
        "GPS+Month Only": ([39.71, 38.83, 38.11, 56.09, 42.98], PAL["amber"], "-.", "v"),
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
        "The 'Monster' Fold 3:\nDelhi & Monterrey extreme smog\nstations land here (+15–20 MAE!)",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=PAL["red"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=PAL["red"], lw=0.7),
    )
    ax2.set_xticks(folds)
    ax2.set_xticklabels([f"Fold {f}" for f in folds], fontsize=7.6)
    ax2.set_ylabel("Validation Error (MAE)")
    ax2.set_ylim(28, 86)
    ax2.set_title("(b) Why We Tested All 5 Folds: Fold 3 Packs the Extreme Smog")
    ax2.legend(loc="upper left", ncol=2, fontsize=6.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    out_path = FIG_DIR / "story_fig3_first_experiments_to_transfer_win.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Story Figure 4: Detective Work on PFRML (Ablation Ladder + Loss Curves)
# ---------------------------------------------------------------------------
def make_story_fig4() -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.8, 5.9))

    ladder = [
        ("R1: Keep Neck, DROP Prototypes!", 45.50, 8.64, "Best! Beats E2 on 4/5 folds", PAL["green"]),
        ("E2: Plain Pretrained MobileNetV3-S", 45.66, 8.41, "Simple baseline to beat", PAL["blue"]),
        ("R2+T3: Prototypes + Class Weights", 47.06, 8.01, "Recovers 1.08 vs R2", PAL["sky"]),
        ("R2+all: Prototypes + T1+T2+T3", 47.16, 7.15, "Best 6-class Acc (41.5%)", PAL["sky"]),
        ("R2+T2: Prototypes + Physics Gate", 47.57, 6.73, "Slightly helps severe tail", PAL["sky"]),
        ("R2 (Eq. 5): Prototypes (No Concat)", 48.04, 8.20, "Loses 5/5 folds to R1!", PAL["amber"]),
        ("R2: Paper 1 PFRML Default", 48.14, 7.61, "Prototypes add +2.63 error", PAL["amber"]),
        ("R2+T1: Prototypes + Attn Pooling", 48.40, 7.74, "No improvement", PAL["amber"]),
        ("R2+T4: Concat 4 Stages in DCM", 60.51, 5.41, "Bottleneck breaks training!", PAL["orange"]),
        ("MDNet-T + PFRML (From Scratch)", 66.56, 4.88, "Total optimization stall!", PAL["red"]),
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
            fontsize=6.3,
            fontweight="bold" if i == 0 else "normal",
        )

    ax1.axvline(45.50, color=PAL["green"], linestyle="--", linewidth=1.1)
    ax1.set_yticks(y)
    ax1.set_yticklabels([r[0] for r in ladder], fontsize=7.0)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 165)
    ax1.set_xlabel("5-Fold Station CV Error (Last-Epoch MAE ± SD)")
    ax1.set_title("(a) Peeling the PFRML Onion: Dropping Prototypes (R1) Wins!")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b): The Smoking-Gun Learning Curves
    curve_specs = [
        (
            "E4_mdnet_t_baseline_ep60",
            "train_loss",
            "MDNet-T+PFRML train loss (Flatlined at ~2.17!)",
            PAL["red"],
            "-",
        ),
        (
            "E4_mobilenetv3_s_dcm_concat",
            "train_loss",
            "T4 DCM Concat train loss (Stuck at 2.06→1.92)",
            PAL["orange"],
            "-",
        ),
        (
            "E4_mobilenetv3_s_baseline",
            "val_loss",
            "Working R2 val loss (Rises after ep 4: U-curve!)",
            PAL["blue"],
            "--",
        ),
        (
            "E4_mobilenetv3_s_baseline",
            "train_loss",
            "Working R2 train loss (Learns smoothly 1.99→0.46)",
            PAL["blue"],
            "-",
        ),
        (
            "E3_mdnet_t_reg_lr1e-3_ep60",
            "train_loss",
            "Plain MDNet-T Reg train loss (Drops 0.85→0.03)",
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
        "Why 'Best Val Loss' is a trap:\n6-class classifier gets overconfident\nafter epoch 4, making val_loss rise\neven while AQI MAE keeps improving!",
        xy=(21, 2.36),
        xytext=(31.5, 2.38),
        arrowprops=dict(arrowstyle="->", color=PAL["blue"], lw=1.1),
        fontsize=6.5,
        color=PAL["navy"],
        bbox=dict(boxstyle="round,pad=0.22", fc="#EEF6FC", ec=PAL["blue"], lw=0.7),
    )

    ax2.set_xlabel("Training Epoch")
    ax2.set_ylabel("5-Fold Average Loss (±1 SD)")
    ax2.set_xlim(1, 60)
    ax2.set_ylim(0, 3.15)
    ax2.set_title("(b) The Smoking Gun: Flatlined Runs & The Val-Loss U-Curve Trap")
    ax2.legend(loc="center right", bbox_to_anchor=(0.99, 0.44), fontsize=6.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    out_path = FIG_DIR / "story_fig4_why_pfrml_prototypes_backfired.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Story Figure 5: Catching the GPS Cheat Code & Winning With Epoch-7 Fusion
# ---------------------------------------------------------------------------
def make_story_fig5() -> Path:
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
        label="30-Epoch Fusion Val MAE",
    )
    ax1.plot(
        g["epoch"],
        cma3,
        color=PAL["navy"],
        linewidth=2.1,
        label="3-Epoch Moving Avg (CMA-3)",
    )

    ax1.axhline(
        46.00,
        color=PAL["amber"],
        linestyle="--",
        linewidth=1.3,
        label="GPS+Month Only (46.00)",
    )
    ax1.axhline(
        50.69,
        color=PAL["slate"],
        linestyle=":",
        linewidth=1.3,
        label="Photo Only in City CV (50.69)",
    )

    ax1.scatter([7], [45.74], color=PAL["green"], s=70, zorder=6, edgecolor="black")
    ax1.scatter(
        [7],
        [44.40],
        color=PAL["red"],
        marker="*",
        s=120,
        zorder=7,
        edgecolor="black",
        label="Fresh 7-Epoch Run (44.40!)",
    )
    ax1.annotate(
        "Epoch-7 Sweet Spot (N=7):\nStops before memorizing cities!\nFresh 7ep run = 44.40 MAE\n(Wins 5/5 folds vs. Photo & GPS)",
        xy=(7, 44.8),
        xytext=(11.5, 52.8),
        arrowprops=dict(arrowstyle="->", color=PAL["green"], lw=1.2),
        fontsize=6.5,
        fontweight="bold",
        color=PAL["green"],
        bbox=dict(boxstyle="round,pad=0.22", fc="#E8F8F2", ec=PAL["green"], lw=0.8),
    )

    ax1.set_xlabel("Training Epoch (25 km City-Cluster 5-Fold CV)")
    ax1.set_ylabel("Validation Error (MAE)")
    ax1.set_xlim(1, 30)
    ax1.set_ylim(37, 68)
    ax1.set_title("(a) Stopping at Epoch 7 Before Fusion Memorizes Training Cities")
    ax1.legend(loc="upper right", fontsize=6.3, ncol=2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b): The Final Showdown (Full Test vs >25 km Unseen Regions)
    models = [
        ("7ep Fusion\n(F_FUSION_R1)", 35.77, 30.47, 42.72, 39.22, 30.85, 47.74),
        ("Photo Only\n(F_E2)", 41.12, 33.97, 50.79, 39.60, 28.06, 54.03),
        ("GPS+Month\n(Metadata MLP)", 40.34, 34.45, 48.31, 49.84, 40.18, 58.81),
        ("Full PFRML\n(F_PFRML)", 45.04, 37.51, 54.04, 46.79, 34.22, 61.06),
        ("Fusion No-GPS\n(meta_off)", 47.56, 39.57, 58.12, 44.88, 32.15, 58.48),
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
        "Gotcha! Beyond 25 km:\nGPS crashes (49.8, R² = −0.04)\nPhotos (39.6) & Fusion (39.2) win!",
        xy=(2 + w / 2, 65.5),
        xytext=(2.30, 74.0),
        arrowprops=dict(arrowstyle="->", color=PAL["red"], lw=1.1),
        fontsize=6.4,
        fontweight="bold",
        color=PAL["red"],
        bbox=dict(boxstyle="round,pad=0.22", fc="#FDECEC", ec=PAL["red"], lw=0.7),
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels([m[0] for m in models], fontsize=7.0)
    ax2.set_ylabel("Final Test MAE [95% Station CI]")
    ax2.set_ylim(0, 88)
    ax2.set_title("(b) Final Test Showdown: Catching the GPS Shortcut Beyond 25 km")
    ax2.legend(loc="upper left", fontsize=6.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    out_path = FIG_DIR / "story_fig5_gps_cheat_code_and_fusion_win.png"
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

    f1 = make_story_fig1(train_df, test_df)
    print(f"Saved {f1}")
    f2 = make_story_fig2()
    print(f"Saved {f2}")
    f3 = make_story_fig3()
    print(f"Saved {f3}")
    f4 = make_story_fig4()
    print(f"Saved {f4}")
    f5 = make_story_fig5()
    print(f"Saved {f5}")
    print("All 5 story report figures generated successfully.")


if __name__ == "__main__":
    main()
