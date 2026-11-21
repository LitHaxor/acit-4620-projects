#!/usr/bin/env python3
"""Generate publication-quality figures for the PM25Vision research report.

All numbers align strictly with `outputs/results/REPORT_NUMBERS.md` and are
computed directly from cleaned metadata, per-epoch curves (`outputs/curves/`),
result CSVs (`outputs/results/`), and saved test predictions (`outputs/predictions/`
and `outputs/checkpoints/E1_metadata_mlp/full_train/preds.npz`).

Outputs PNG (300 DPI) figures into `outputs/figures/`.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import compute_test_distance_mask  # noqa: E402
from src.data.cleaning import load_and_clean_metadata  # noqa: E402
from src.data.splits import build_cv_splits, compute_val_to_train_nearest_km  # noqa: E402

FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
CURVES_DIR = PROJECT_ROOT / "outputs" / "curves"
PREDS_DIR = PROJECT_ROOT / "outputs" / "predictions"

PAL = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "amber": "#E69F00",
    "sky": "#56B4E9",
    "navy": "#1B365D",
    "slate": "#5A6570",
    "red": "#B22222",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 8.0,
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
        p = CURVES_DIR / f"{exp_id}_{group_mode}_fold{f}.csv"
        df = pd.read_csv(p)
        df["fold"] = f
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Figure 1: Dataset Distribution & Spatial Split Leakage Isolation
# ---------------------------------------------------------------------------
def make_fig1(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))

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
            x + (idx - 1) * w, pcts, width=w, label=label, color=color, alpha=0.9, edgecolor="white"
        )
        for b, pct in zip(bars, pcts):
            if pct >= 0.4:
                ax1.text(
                    b.get_x() + b.get_width() / 2,
                    pct + 0.55,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    rotation=90,
                )

    ax1.axvspan(3.5, 5.5, color="#FDECEC", alpha=0.55, zorder=0)
    ax1.text(
        4.5,
        29.5,
        "High-Pollution Tail\n(AQI > 200: 13.3% train)",
        ha="center",
        va="top",
        fontsize=8.0,
        color=PAL["red"],
        fontweight="bold",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels)
    ax1.set_ylabel("Share of Split (%)")
    ax1.set_ylim(0, 39.5)
    ax1.set_title("(a) US EPA AQI Bin Distribution & Long-Tail Regime")
    ax1.legend(loc="upper left", fontsize=7.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b): CDF of distance to nearest training image
    st_df, st_folds = build_cv_splits(train_df, n_splits=5, group_mode="station", spatial_cluster_km=25.0)
    sc_df, sc_folds = build_cv_splits(train_df, n_splits=5, group_mode="spatial_cluster", spatial_cluster_km=25.0)

    st_dists = np.concatenate([compute_val_to_train_nearest_km(st_df, tr, va) for tr, va in st_folds])
    sc_dists = np.concatenate([compute_val_to_train_nearest_km(sc_df, tr, va) for tr, va in sc_folds])

    for label, d_arr, color, ls in [
        (f"Station 5-Fold CV (med={np.median(st_dists):.1f} km)", st_dists, PAL["blue"], "-"),
        (f"Spatial Cluster 5-Fold CV (med={np.median(sc_dists):.1f} km)", sc_dists, PAL["green"], "-"),
        (f"Full Test vs. Train (med={np.median(dist_km):.1f} km)", dist_km, PAL["amber"], "--"),
        (f"Test >25 km Subset (med={np.median(dist_km[mask_gt25]):.1f} km)", dist_km[mask_gt25], PAL["orange"], "-."),
    ]:
        s = np.sort(np.maximum(d_arr, 0.05))
        y = np.linspace(0, 100, len(s))
        ax2.plot(s, y, label=label, color=color, linestyle=ls, linewidth=2.0)

    ax2.axvline(25.0, color=PAL["red"], linestyle=":", linewidth=1.6)
    ax2.text(28.0, 15, "25 km leakage\nhorizon", color=PAL["red"], fontsize=8.2, fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_xlim(0.05, 3000)
    ax2.set_ylim(0, 103)
    ax2.set_xlabel("Haversine Distance to Nearest Training Image (km, log scale)")
    ax2.set_ylabel("Cumulative Share of Images (%)")
    ax2.set_title("(b) Spatial Separation Across CV & Test Protocols")
    ax2.legend(loc="upper left", fontsize=7.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_dataset_spatial_splits.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: PFRML + Multimodal Fusion System Architecture Schematic
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, title, sub="", fc="#F4F8FB", ec="#1B365D", title_c="#111111", lw=1.3, fs=8.2, sub_fs=7.0):
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
        ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=title_c, zorder=4)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=sub_fs, color="#333333", zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=title_c, zorder=4)


def _arrow(ax, x1, y1, x2, y2, color="#333333", ls="-", lw=1.4):
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


def make_fig2() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 5.6))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    ax.grid(False)

    ax.text(
        0.5,
        0.96,
        "Unified PFRML Visual Spine, Architectural Toggles (T1–T4), and F_FUSION_R1 Multimodal Architecture",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=PAL["navy"],
    )

    # Vision Stream (Middle-Upper Row)
    _box(ax, 0.015, 0.54, 0.115, 0.22, "RGB Image", "224×224×3\nPhysics-safe\n(no jitter)", fc="#EEF6FC", ec=PAL["blue"])
    _box(ax, 0.155, 0.54, 0.155, 0.22, "MobileNetV3-S", "ImageNet Pretrained\nStages [16,24,48,576]\n(0.93M backbone)", fc="#E8F3FA", ec=PAL["blue"])
    _box(ax, 0.335, 0.54, 0.150, 0.22, "DCM Module", "1×1 Conv → 224-D\nAlign 14×14, Sum\n(T4: Concat fails)", fc="#E8F3FA", ec=PAL["blue"])
    _box(ax, 0.510, 0.54, 0.165, 0.22, "Light Transformer", "2 Layers, 4 Heads\n196+CLS tokens\nGAP (T1: AttnPool)", fc="#E8F3FA", ec=PAL["blue"])

    _arrow(ax, 0.130, 0.65, 0.155, 0.65, PAL["blue"])
    _arrow(ax, 0.310, 0.65, 0.335, 0.65, PAL["blue"])
    _arrow(ax, 0.485, 0.65, 0.510, 0.65, PAL["blue"])

    # Latent chi node
    _box(ax, 0.705, 0.58, 0.070, 0.14, "χ_i", "224-D\nVisual", fc="#FFF7E6", ec=PAL["amber"])
    _arrow(ax, 0.675, 0.65, 0.705, 0.65, PAL["blue"])

    # Optional PhysicsPriorGate T2 (wide enough for crisp padding)
    _box(
        ax,
        0.24,
        0.81,
        0.46,
        0.105,
        "Optional PhysicsPriorGate (T2 — disabled in F_FUSION_R1)",
        "4 scalars [Entropy, Contrast, Transmission t, Extinction −ln(t)] → Sigmoid Gate",
        fc="#F5F5F5",
        ec=PAL["slate"],
        lw=1.0,
        fs=7.8,
        sub_fs=6.9,
    )
    _arrow(ax, 0.075, 0.76, 0.24, 0.862, PAL["slate"], ls="--", lw=1.0)
    _arrow(ax, 0.70, 0.862, 0.740, 0.72, PAL["slate"], ls="--", lw=1.0)

    # Vision Aux Head (Upper Right)
    _box(ax, 0.815, 0.71, 0.170, 0.18, "Vision Aux Head", "χ_i → 256 → 1\nL_aux (wt = 0.5)\nDetached from Meta", fc="#E8F8F2", ec=PAL["green"], lw=1.5)
    _arrow(ax, 0.775, 0.67, 0.815, 0.79, PAL["green"], lw=1.5)

    # Prototype Branch placed cleanly to the left of the chi_i -> Fusion Concat vertical line (x < 0.68)
    _box(
        ax,
        0.34,
        0.34,
        0.335,
        0.145,
        "Prototype Branch (R2 / Full ONLY — OFF in R1)",
        "6 L2-Norm EMA Prototypes (T3: Class Weights)\nc̃ = Σ p_k p'_k  [Degrades MAE: 48.1 vs 45.5]",
        fc="#FDECEC",
        ec=PAL["red"],
        lw=1.1,
        fs=7.6,
        sub_fs=6.8,
    )
    _arrow(ax, 0.715, 0.58, 0.660, 0.485, PAL["red"], ls="--", lw=1.1)

    # Metadata Stream (Bottom Row)
    _box(ax, 0.015, 0.10, 0.155, 0.19, "Location & Time", "5-D Tabular Vector\n[x, y, z, sin(m), cos(m)]", fc="#FFF0E6", ec=PAL["orange"])
    _box(ax, 0.200, 0.10, 0.175, 0.19, "MetadataMLP", "5 → 64 → 32 (2.5k)\nGeospatial Prior\n(Memorizes <25 km)", fc="#FFF0E6", ec=PAL["orange"])
    _box(ax, 0.405, 0.10, 0.230, 0.19, "Modality Dropout Gate", "Zeroes m w/ prob p_drop\np=0.0 (best ep=7) vs p=0.3\n[eval: normal / meta_off]", fc="#FFF0E6", ec=PAL["orange"])

    _arrow(ax, 0.170, 0.195, 0.200, 0.195, PAL["orange"])
    _arrow(ax, 0.375, 0.195, 0.405, 0.195, PAL["orange"])

    # Fusion Concat + Main Reg Head (Bottom Right)
    _box(ax, 0.680, 0.10, 0.115, 0.19, "Fusion Concat", "[χ_i ; m]\n224 + 32 = 256-D", fc="#F3ECF8", ec=PAL["purple"], lw=1.5)
    _box(ax, 0.820, 0.10, 0.165, 0.19, "Main Reg. Head", "256 → 256 → 1\nPredicted AQI / 100\nF_FUSION_R1: 2.02M", fc="#E8F8F2", ec=PAL["green"], lw=1.7)

    _arrow(ax, 0.635, 0.195, 0.680, 0.195, PAL["orange"], lw=1.5)
    # Vertical route from chi_i straight down into Fusion Concat (no box crossing!)
    _arrow(ax, 0.740, 0.58, 0.740, 0.29, PAL["blue"], lw=1.6)
    _arrow(ax, 0.795, 0.195, 0.820, 0.195, PAL["purple"], lw=1.6)

    ax.text(
        0.5,
        0.025,
        "Winning Config (F_FUSION_R1): MobileNetV3-S (pretrained) + DCM (sum) + LightTransformer (GAP) + MetadataMLP, prototypes OFF (R1), p_drop=0.0, 7-epoch CMA-3 stopping → Test MAE 35.77 (Full) / 39.22 (>25 km).",
        ha="center",
        va="center",
        fontsize=8.1,
        style="italic",
        color="#222222",
    )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_architecture_fusion.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Station 5-Fold CV Baselines, Transfer Gap & Fold 3 Heterogeneity
# ---------------------------------------------------------------------------
def make_fig3() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.7))

    models = [
        ("E0_median (Const. Median)", 66.21, 4.89, 0.0000, PAL["slate"]),
        ("E0_ridge (18 Haze Stats)", 64.11, 4.40, 0.0317, PAL["slate"]),
        ("E3_mdnet_t_reg (Scratch 60ep)", 53.29, 6.38, 0.1951, PAL["orange"]),
        ("E2_resnet18_scratch (30ep)", 53.27, 7.11, 0.1954, PAL["orange"]),
        ("E2_mobilenetv3_s_pretrained", 45.66, 8.41, 0.3103, PAL["blue"]),
        ("E3b_mobilenetv3_s_md_neck", 44.92, 8.54, 0.3214, PAL["blue"]),
        ("E2_resnet18_pretrained (40ep)", 44.05, 8.70, 0.3346, PAL["blue"]),
        ("E1_metadata_mlp (GPS+Month)", 43.15, 7.47, 0.3483, PAL["amber"]),
    ]
    y_pos = np.arange(len(models))
    maes = [m[1] for m in models]
    sds = [m[2] for m in models]
    colors = [m[4] for m in models]
    labels = [m[0] for m in models]

    ax1.barh(y_pos, maes, xerr=sds, height=0.62, color=colors, alpha=0.88, edgecolor="white", capsize=3.5, error_kw={"ecolor": "#333333", "lw": 1.1})
    for i, (_, mae, sd, skill, _) in enumerate(models):
        sk_txt = f" {mae:.2f}±{sd:.2f} ({skill:+.1%})" if skill > 0 else f" {mae:.2f}±{sd:.2f} (Ref)"
        ax1.text(mae + sd + 0.8, i, sk_txt, va="center", ha="left", fontsize=7.6, fontweight="bold" if mae < 46 else "normal")

    ax1.axvline(66.21, color=PAL["slate"], linestyle=":", linewidth=1.2)
    # Clean callout in bottom-right clear area pointing up-left to pretrained MobileNetV3-S
    ax1.annotate(
        "ImageNet Transfer Gap:\n−7.6 to −9.2 MAE",
        xy=(52.7, 4.05),
        xytext=(88.0, 5.5),
        arrowprops=dict(arrowstyle="->", color=PAL["navy"], lw=1.2),
        fontsize=7.8,
        fontweight="bold",
        color=PAL["navy"],
        bbox=dict(boxstyle="round,pad=0.3", fc="#EEF6FC", ec=PAL["blue"], lw=0.8),
    )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 125)
    ax1.set_xlabel("5-Fold station CV Last-Epoch MAE (Mean ± SD)")
    ax1.set_title("(a) Transfer Learning & Metadata Baseline (station CV)")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b): Per-Fold MAE profiles showing Fold 3 tail-variance spike
    folds = np.arange(5)
    per_fold = {
        "E0_median": ([67.18, 62.31, 63.71, 74.33, 63.50], PAL["slate"], ":", "o"),
        "E2_resnet18_scratch": ([48.41, 50.21, 53.62, 65.44, 48.67], PAL["orange"], "--", "s"),
        "E2_mobilenetv3_s_pre": ([36.27, 46.10, 42.96, 59.21, 43.77], PAL["blue"], "-", "^"),
        "E2_resnet18_pre_ep40": ([34.85, 43.35, 42.06, 58.45, 41.55], PAL["green"], "-", "D"),
        "E1_metadata_mlp": ([39.71, 38.83, 38.11, 56.09, 42.98], PAL["amber"], "-.", "v"),
    }
    ax2.axvspan(2.65, 3.35, color="#FDECEC", alpha=0.65, zorder=0)
    for name, (vals, col, ls, mk) in per_fold.items():
        ax2.plot(folds, vals, label=name, color=col, linestyle=ls, marker=mk, linewidth=1.9, markersize=6)

    ax2.text(
        3.0,
        32.0,
        "Fold 3 Tail Outlier\nVal AQI SD = 108.2\n(4 cells = 76.9% of Bins 5–6)",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color=PAL["red"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PAL["red"], lw=0.8),
    )
    ax2.set_xticks(folds)
    ax2.set_xticklabels([f"Fold {f}" for f in folds])
    ax2.set_ylabel("Validation MAE (AQI)")
    ax2.set_ylim(29, 85)
    ax2.set_title("(b) Per-Fold MAE Sensitivity & Fold-3 Tail Concentration")
    ax2.legend(loc="upper left", ncol=2, fontsize=7.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_baselines_and_backbones.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: PFRML Ablation Ladder & Optimization Collapse Diagnostics
# ---------------------------------------------------------------------------
def make_fig4() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.9))

    ladder = [
        ("R1: No Prototypes (w/o proto)", 45.50, 8.64, "−0.16 vs E2 (4/5)", PAL["green"]),
        ("E2: MobileNetV3-S Pretrained", 45.66, 8.41, "Reference", PAL["blue"]),
        ("R2+T3: + Class Weights", 47.06, 8.01, "−1.08 vs R2 (4/5)", PAL["sky"]),
        ("R2+all: PFRML Full (T1+T2+T3)", 47.16, 7.15, "−0.98 vs R2 (4/5)", PAL["sky"]),
        ("R2+T2: + Physics Prior Gate", 47.57, 6.73, "−0.57 vs R2 (4/5)", PAL["sky"]),
        ("R2 (Eq. 5): reg_input=chi", 48.04, 8.20, "−0.09 vs R2 (2/5)", PAL["amber"]),
        ("R2: PFRML Baseline (chi_proto)", 48.14, 7.61, "+2.63 vs R1 (1/5)", PAL["amber"]),
        ("R2+T1: + Attention Pooling", 48.40, 7.74, "+0.27 vs R2 (2/5)", PAL["amber"]),
        ("R2+T4: + DCM Concat [FAIL]", 60.51, 5.41, "+15.01 vs R1 (0/5)", PAL["orange"]),
        ("Extra B: MDNet-T PFRML [COLLAPSE]", 66.56, 4.88, "+21.05 vs R1 (0/5)", PAL["red"]),
    ]
    y = np.arange(len(ladder))
    maes = [r[1] for r in ladder]
    sds = [r[2] for r in ladder]
    cols = [r[4] for r in ladder]

    ax1.barh(y, maes, xerr=sds, height=0.64, color=cols, alpha=0.88, edgecolor="white", capsize=3, error_kw={"ecolor": "#333", "lw": 1.0})
    for i, (_, m, sd, note, _) in enumerate(ladder):
        ax1.text(m + sd + 0.7, i, f"{m:.2f}±{sd:.2f} [{note}]", va="center", ha="left", fontsize=7.3)

    ax1.axvline(45.50, color=PAL["green"], linestyle="--", linewidth=1.2)
    ax1.set_yticks(y)
    ax1.set_yticklabels([r[0] for r in ladder], fontsize=8.0)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 96)
    ax1.set_xlabel("5-Fold station CV Last-Epoch MAE (Mean ± SD)")
    ax1.set_title("(a) PFRML Ablation Ladder & Toggle Impacts")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b): Train/Val loss trajectories
    curve_specs = [
        ("E4_mdnet_t_baseline_ep60", "train_loss", "MDNet-T PFRML (train_loss: flat ~2.17)", PAL["red"], "-"),
        ("E4_mobilenetv3_s_dcm_concat", "train_loss", "DCM Concat T4 (train_loss: 2.06→1.92)", PAL["orange"], "-"),
        ("E4_mobilenetv3_s_baseline", "train_loss", "MobileNetV3-S PFRML R2 (train_loss: 2.00→0.46)", PAL["blue"], "-"),
        ("E4_mobilenetv3_s_baseline", "val_loss", "MobileNetV3-S PFRML R2 (val_loss: U-curve)", PAL["blue"], "--"),
        ("E3_mdnet_t_reg_lr1e-3_ep60", "train_loss", "MDNet-T Reg-Only E3 (train_loss: 0.85→0.03)", PAL["green"], "-"),
    ]
    for exp_id, col_name, lbl, color, ls in curve_specs:
        df = load_5fold_curves(exp_id, "station")
        g = df.groupby("epoch")[col_name].agg(["mean", "std"]).reset_index()
        ax2.plot(g["epoch"], g["mean"], label=lbl, color=color, linestyle=ls, linewidth=2.0)
        ax2.fill_between(g["epoch"], g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.12)

    ax2.annotate(
        "CE overconfidence:\nval_loss rises ep 4→30\nwhile val MAE improves",
        xy=(22, 2.38),
        xytext=(33.5, 2.52),
        arrowprops=dict(arrowstyle="->", color=PAL["blue"], lw=1.2),
        fontsize=7.5,
        color=PAL["navy"],
        bbox=dict(boxstyle="round,pad=0.25", fc="#EEF6FC", ec=PAL["blue"], lw=0.8),
    )

    ax2.set_xlabel("Training Epoch")
    ax2.set_ylabel("5-Fold Mean Loss (± 1 SD shaded)")
    ax2.set_xlim(1, 60)
    ax2.set_ylim(0, 3.05)
    ax2.set_title("(b) Optimization Collapse & Multi-Task Overconfidence")
    ax2.legend(loc="center right", bbox_to_anchor=(0.99, 0.48), fontsize=7.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_pfrml_ablation_and_collapse.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Multimodal Fusion Spatial CV & CMA-3 Early Stopping
# ---------------------------------------------------------------------------
def make_fig5() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))

    df_p00 = load_5fold_curves("E5_fusion_r1_p00", "spatial_cluster")
    g = df_p00.groupby("epoch")["mae"].agg(["mean", "std"]).reset_index()
    cma3 = g["mean"].rolling(window=3, center=True, min_periods=1).mean()

    ax1.fill_between(g["epoch"], g["mean"] - g["std"], g["mean"] + g["std"], color=PAL["blue"], alpha=0.14, label="5-Fold ±1 SD")
    ax1.plot(g["epoch"], g["mean"], color=PAL["blue"], alpha=0.65, linewidth=1.5, marker="o", markersize=3.5, label="5-Fold Mean Val MAE")
    ax1.plot(g["epoch"], cma3, color=PAL["navy"], linewidth=2.3, label="3-Epoch Centered MA (CMA-3)")

    ax1.axhline(46.00, color=PAL["amber"], linestyle="--", linewidth=1.4, label="E1 Metadata MLP (46.00)")
    ax1.axhline(50.69, color=PAL["slate"], linestyle=":", linewidth=1.4, label="E2 Image-Only Spatial (50.69)")

    ax1.scatter([7], [45.74], color=PAL["green"], s=90, zorder=6, edgecolor="black")
    ax1.annotate(
        "Optimal N = 7\nMean MAE = 45.74\nCMA-3 = 46.26 (min ≥ ep 6)",
        xy=(7, 45.74),
        xytext=(12.2, 54.0),
        arrowprops=dict(arrowstyle="->", color=PAL["green"], lw=1.4),
        fontsize=7.9,
        fontweight="bold",
        color=PAL["green"],
        bbox=dict(boxstyle="round,pad=0.3", fc="#E8F8F2", ec=PAL["green"], lw=1.0),
    )

    ax1.set_xlabel("Training Epoch")
    ax1.set_ylabel("Validation MAE (spatial_cluster 5-Fold CV)")
    ax1.set_xlim(1, 30)
    ax1.set_ylim(37, 67)
    ax1.set_title("(a) Fusion Overfitting Trajectory & CMA-3 Epoch Selection")
    ax1.legend(loc="upper right", fontsize=7.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b): Normal vs metadata-zeroed evaluation
    cats = [
        "E0_median\n(Baseline)",
        "Fusion Full\n(p=0.3, 30ep)",
        "E2_spatial\n(Vision Only)",
        "Fusion R1\n(p=0.3, 30ep)",
        "Fusion R1\n(p=0.0, 30ep)",
        "E1_meta_mlp\n(Meta Only)",
        "Fusion R1\n(p=0.0, ep=7)",
    ]
    normal_mae = [66.14, 51.68, 50.69, 49.18, 47.45, 46.00, 44.40]
    normal_sd = [5.60, 6.31, 5.60, 6.63, 6.71, 7.33, 7.30]
    meta_off_mae = [np.nan, 54.94, np.nan, 52.28, 53.54, np.nan, 52.58]
    meta_off_sd = [np.nan, 5.80, np.nan, 5.95, 5.49, np.nan, 5.06]

    x = np.arange(len(cats))
    w = 0.36
    for i in range(len(cats)):
        if np.isnan(meta_off_mae[i]):
            col = PAL["slate"] if i == 0 else (PAL["blue"] if i == 2 else PAL["amber"])
            ax2.bar(x[i], normal_mae[i], yerr=normal_sd[i], width=0.48, color=col, alpha=0.88, capsize=3.5, edgecolor="white")
            ax2.text(x[i], normal_mae[i] + normal_sd[i] + 0.8, f"{normal_mae[i]:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        else:
            col_n = PAL["green"] if i == len(cats) - 1 else PAL["navy"]
            ax2.bar(x[i] - w / 2, normal_mae[i], yerr=normal_sd[i], width=w, color=col_n, alpha=0.92, capsize=3, edgecolor="white", label="Normal Eval" if i == 1 else "")
            ax2.bar(x[i] + w / 2, meta_off_mae[i], yerr=meta_off_sd[i], width=w, color=PAL["purple"], alpha=0.85, capsize=3, edgecolor="white", label="Metadata Zeroed (meta_off)" if i == 1 else "")
            ax2.text(x[i] - w / 2, normal_mae[i] + normal_sd[i] + 0.7, f"{normal_mae[i]:.2f}", ha="center", va="bottom", fontsize=7.1, fontweight="bold")
            ax2.text(x[i] + w / 2, meta_off_mae[i] + meta_off_sd[i] + 0.7, f"{meta_off_mae[i]:.2f}", ha="center", va="bottom", fontsize=6.9)

    ax2.axhline(46.00, color=PAL["amber"], linestyle="--", linewidth=1.2, alpha=0.85)
    ax2.annotate(
        "Beats E1 (5/5 folds)\n& E2_spatial (5/5 folds)",
        xy=(6 - w / 2, 44.40),
        xytext=(4.1, 28.5),
        arrowprops=dict(arrowstyle="->", color=PAL["green"], lw=1.3),
        fontsize=7.7,
        fontweight="bold",
        color=PAL["green"],
        bbox=dict(boxstyle="round,pad=0.25", fc="#E8F8F2", ec=PAL["green"], lw=0.8),
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(cats, fontsize=7.4)
    ax2.set_ylabel("5-Fold spatial_cluster CV MAE (Mean ± SD)")
    ax2.set_ylim(24, 78)
    ax2.set_title("(b) spatial_cluster CV: Normal vs. meta_off Ablation")
    ax2.legend(loc="upper right", fontsize=7.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_fusion_early_stopping.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Final One-Shot Test Set & >25 km Generalization (+ Bootstrap Forest)
# ---------------------------------------------------------------------------
def make_fig6() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0))

    models = [
        ("F_FUSION_R1\n(Fusion ep=7)", 35.77, 30.47, 42.72, 39.22, 30.85, 47.74, 0.6070, 0.3743),
        ("F_E2\n(Vision Only)", 41.12, 33.97, 50.79, 39.60, 28.06, 54.03, 0.4215, 0.2747),
        ("E1_metadata_mlp\n(GPS+Month)", 40.34, 34.45, 48.31, 49.84, 40.18, 58.81, 0.4982, -0.0393),
        ("F_PFRML_full\n(PFRML Vision)", 45.04, 37.51, 54.04, 46.79, 34.22, 61.06, 0.2914, -0.0660),
        ("F_FUSION_R1\n(meta_off)", 47.56, 39.57, 58.12, 44.88, 32.15, 58.48, 0.3411, 0.1346),
    ]

    x = np.arange(len(models))
    w = 0.36
    full_vals = [m[1] for m in models]
    full_err = [[m[1] - m[2] for m in models], [m[3] - m[1] for m in models]]
    gt25_vals = [m[4] for m in models]
    gt25_err = [[m[4] - m[5] for m in models], [m[6] - m[4] for m in models]]

    ax1.bar(x - w / 2, full_vals, yerr=full_err, width=w, color=PAL["blue"], alpha=0.9, capsize=3.5, edgecolor="white", label="Full Test (n=2,920, 729 st.)")
    ax1.bar(x + w / 2, gt25_vals, yerr=gt25_err, width=w, color=PAL["orange"], alpha=0.9, capsize=3.5, edgecolor="white", label="Unseen >25 km (n=717, 205 st.)")

    for i, m in enumerate(models):
        ax1.text(x[i] - w / 2 - 0.02, m[3] + 0.9, f"{m[1]:.2f}\n(.{int(round(m[7]*100)):02d})" if m[7] >= 0 else f"{m[1]:.2f}\n({m[7]:.2f})", ha="center", va="bottom", fontsize=6.9, fontweight="bold" if i == 0 else "normal")
        r2_str = f"R²={m[8]:.2f}"
        ax1.text(x[i] + w / 2 + 0.02, m[6] + 0.9, f"{m[4]:.2f}\n({r2_str})", ha="center", va="bottom", fontsize=6.8, fontweight="bold" if i in (0, 2) else "normal")

    ax1.annotate(
        "Metadata collapses >25 km:\n+9.50 MAE, R² 0.50 → −0.04",
        xy=(2 + w / 2, 59.0),
        xytext=(2.55, 69.5),
        arrowprops=dict(arrowstyle="->", color=PAL["red"], lw=1.2),
        fontsize=7.6,
        fontweight="bold",
        color=PAL["red"],
        bbox=dict(boxstyle="round,pad=0.25", fc="#FDECEC", ec=PAL["red"], lw=0.8),
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels([m[0] for m in models], fontsize=7.6)
    ax1.set_ylabel("Test MAE (95% Station-Bootstrap CI)")
    ax1.set_ylim(0, 79)
    ax1.set_title("(a) Full Test vs. Unseen-Region (>25 km) Generalization")
    ax1.legend(loc="upper left", fontsize=7.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b): Paired Station-Bootstrap MAE Differences
    pairs = [
        ("F_FUSION_R1 − F_PFRML_full", -9.26, -12.52, -6.06, -7.57, -14.29, -1.46),
        ("F_FUSION_R1 − F_E2", -5.35, -8.62, -2.26, -0.38, -8.09, +5.99),
        ("F_FUSION_R1 − E1_metadata_mlp", -4.57, -8.31, -1.26, -10.63, -18.78, -2.59),
        ("F_FUSION_R1 − meta_off", -11.79, -16.00, -7.74, -5.66, -11.75, +0.92),
        ("F_E2 − F_PFRML_full", -3.92, -6.27, -1.58, -7.18, -11.92, -2.53),
        ("F_E2 − E1_metadata_mlp", +0.77, -3.46, +4.78, -10.24, -20.50, +0.86),
    ]
    yp = np.arange(len(pairs))
    ax2.axvline(0.0, color="#222222", linestyle="--", linewidth=1.2)
    for i, (lbl, d_f, lo_f, hi_f, d_g, lo_g, hi_g) in enumerate(pairs):
        ax2.errorbar(
            d_f,
            i - 0.14,
            xerr=[[d_f - lo_f], [hi_f - d_f]],
            fmt="o",
            color=PAL["blue"],
            capsize=3.5,
            markersize=6,
            lw=1.8,
            label="Full Test (n=2,920)" if i == 0 else "",
        )
        ax2.errorbar(
            d_g,
            i + 0.14,
            xerr=[[d_g - lo_g], [hi_g - d_g]],
            fmt="s",
            color=PAL["orange"],
            capsize=3.5,
            markersize=5.5,
            lw=1.8,
            label="Test >25 km (n=717)" if i == 0 else "",
        )
        ax2.text(hi_f + 0.45, i - 0.14, f"{d_f:+.2f}", va="center", ha="left", fontsize=7.2, color=PAL["blue"], fontweight="bold", bbox=dict(boxstyle="square,pad=0.06", fc="white", ec="none", alpha=0.85))
        ax2.text(hi_g + 0.45, i + 0.14, f"{d_g:+.2f}", va="center", ha="left", fontsize=7.2, color=PAL["orange"], fontweight="bold", bbox=dict(boxstyle="square,pad=0.06", fc="white", ec="none", alpha=0.85))

    ax2.set_yticks(yp)
    ax2.set_yticklabels([p[0] for p in pairs], fontsize=8.0)
    ax2.set_ylim(5.85, -1.15)  # extra top space for legend
    ax2.set_xlim(-22.8, 10.2)
    ax2.set_xlabel("Paired MAE Difference (Model A − Model B; Negative = Model A Better)")
    ax2.set_title("(b) Paired Station-Bootstrap Differences (95% CI, 1,000 Resamples)")
    ax2.legend(loc="upper left", ncol=2, fontsize=7.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_final_test_generalization.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Distance-Stratified Decay & EPA Long-Tail Error Analysis
# ---------------------------------------------------------------------------
def make_fig7(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))

    dist_km, _ = compute_test_distance_mask(train_df, test_df, min_km=25.0)
    test_order = test_df["image_id"].astype(str).values

    e1_npz = np.load(PROJECT_ROOT / "outputs/checkpoints/E1_metadata_mlp/full_train/preds.npz")
    f_e2 = pd.read_csv(PREDS_DIR / "F_E2.csv")
    e1_df = f_e2[["image_id", "station_id", "y_true"]].copy()
    e1_df["y_pred"] = np.asarray(e1_npz["y_pred"], dtype=np.float64)

    preds = {
        "E1_metadata_mlp (Meta)": (e1_df, PAL["amber"], "--", "v"),
        "F_PFRML_full (Vision)": (pd.read_csv(PREDS_DIR / "F_PFRML_full.csv"), PAL["purple"], ":", "s"),
        "F_E2 (Vision Only)": (f_e2, PAL["blue"], "-", "^"),
        "F_FUSION_R1 (Fusion)": (pd.read_csv(PREDS_DIR / "F_FUSION_R1.csv"), PAL["green"], "-", "o"),
    }
    for k in list(preds.keys()):
        dfp, col, ls, mk = preds[k]
        dfp = dfp.copy()
        dfp["image_id"] = dfp["image_id"].astype(str)
        dfp = dfp.set_index("image_id").loc[test_order].reset_index()
        preds[k] = (dfp, col, ls, mk)

    dist_bins = [(0, 5), (5, 25), (25, 100), (100, 10000)]
    dist_labels = [
        "0–5 km\n(n=1,276 | 299 st.)",
        "5–25 km\n(n=927 | 246 st.)",
        "25–100 km\n(n=498 | 151 st.)",
        ">100 km\n(n=219 | 55 st.)",
    ]
    xd = np.arange(len(dist_bins))
    for k, (dfp, col, ls, mk) in preds.items():
        maes = []
        for lo, hi in dist_bins:
            m = (dist_km >= lo) & (dist_km < hi)
            maes.append(np.abs(dfp["y_true"].values[m] - dfp["y_pred"].values[m]).mean())
        ax1.plot(xd, maes, label=k, color=col, linestyle=ls, marker=mk, linewidth=2.1, markersize=6.5)
        for xi, val in zip(xd, maes):
            if k.startswith("E1") and xi == 3:
                ax1.text(xi, val + 1.3, f"{val:.1f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=col)
            elif k.startswith("F_E2") and xi == 3:
                ax1.text(xi, val - 2.3, f"{val:.1f}", ha="center", va="top", fontsize=7.8, fontweight="bold", color=col)
            elif k.startswith("F_FUSION") and xi < 3:
                ax1.text(xi, val - 2.1, f"{val:.1f}", ha="center", va="top", fontsize=7.6, fontweight="bold", color=col)

    ax1.axvline(1.5, color=PAL["red"], linestyle=":", linewidth=1.3)
    ax1.text(1.56, 58.0, "25 km boundary →\n(unseen regions)", color=PAL["red"], fontsize=7.8, fontweight="bold")
    ax1.set_xticks(xd)
    ax1.set_xticklabels(dist_labels, fontsize=8.0)
    ax1.set_ylabel("Mean Absolute Error (MAE)")
    ax1.set_ylim(25, 64)
    ax1.set_title("(a) Test MAE by Distance to Nearest Training Station")
    ax1.legend(loc="upper left", fontsize=7.7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b): Per-EPA-bin MAE on Full Test Set
    test_bins = aqi_to_bin(test_df["pm25"].values)
    bin_names = [
        "Good\n(n=629)",
        "Mod.\n(n=630)",
        "USG\n(n=630)",
        "Unh.\n(n=630)",
        "V. Unh.\n(n=275)",
        "Haz.\n(n=126)",
    ]
    xb = np.arange(6)
    w = 0.20
    order = ["F_FUSION_R1 (Fusion)", "F_E2 (Vision Only)", "E1_metadata_mlp (Meta)", "F_PFRML_full (Vision)"]
    for idx, k in enumerate(order):
        dfp, col, _, _ = preds[k]
        vals = [np.abs(dfp["y_true"].values[test_bins == b] - dfp["y_pred"].values[test_bins == b]).mean() for b in range(6)]
        bars = ax2.bar(xb + (idx - 1.5) * w, vals, width=w, color=col, alpha=0.9, edgecolor="white", label=k)
        if idx == 0:
            for b_rect, v in zip(bars, vals):
                ax2.text(
                    b_rect.get_x() + b_rect.get_width() / 2,
                    v + 2.5,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    fontweight="bold",
                    color=col,
                    rotation=90 if v > 100 else 0,
                )

    ax2.axvspan(3.5, 5.5, color="#FDECEC", alpha=0.55, zorder=0)
    ax2.text(
        4.5,
        191,
        "Tail Regimes (AQI > 200)\nF_FUSION_R1 Haz. MAE: 126.6\n(vs. F_E2 155.4, E1 146.8)",
        ha="center",
        va="top",
        fontsize=7.6,
        color=PAL["red"],
        fontweight="bold",
    )
    ax2.set_xticks(xb)
    ax2.set_xticklabels(bin_names, fontsize=8.0)
    ax2.set_ylabel("Mean Absolute Error (MAE)")
    ax2.set_ylim(0, 198)
    ax2.set_title("(b) Full Test MAE Across US EPA Severity Categories")
    ax2.legend(loc="upper left", fontsize=7.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig7_distance_and_tail_errors.png")
    plt.close(fig)


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_and_clean_metadata(
        PROJECT_ROOT / "datasets/train/metadata.csv",
        PROJECT_ROOT / "datasets/test/metadata.csv",
    )

    make_fig1(train_df, test_df)
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    make_fig7(train_df, test_df)
    print(f"Generated 7 figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
