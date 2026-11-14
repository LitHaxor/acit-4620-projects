"""Evaluation metrics, fold aggregation, and statistical significance helpers.

Includes:
- Regression metrics: MAE, RMSE, PLCC (Pearson r), SRCC (Spearman rho), R2.
- Classification metrics: 6-class accuracy, +-1 adjacent-class accuracy
  (Paper 4: Nathvani et al., Sci. Total Environ. 2023, Sec. 3.6), Macro-F1.
- Long-tail metrics for US EPA bins 5-6 (0-indexed bins {4, 5}, AQI > 200):
  tail MAE and tail Macro-F1.
- Cross-validation fold aggregation (mean +- std) and paired Wilcoxon
  signed-rank test across folds (Paper 17: Han et al. 2025; Paper 11: Kumaar & Deepak 2025).
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, root_mean_squared_error

from src.data.cleaning import aqi_to_bin_index

TAIL_BINS: Tuple[int, int] = (4, 5)  # 1-indexed EPA bins 5-6 (201-300, 301-600)
ALL_BINS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)

METRIC_ORDER: Tuple[str, ...] = (
    "mae",
    "rmse",
    "plcc",
    "srcc",
    "r2",
    "acc",
    "pm1_acc",
    "macro_f1",
    "tail_mae",
    "tail_f1",
)


def _safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    res = stats.pearsonr(y_true, y_pred)
    val = float(res.statistic)
    return 0.0 if np.isnan(val) else val


def _safe_spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    res = stats.spearmanr(y_true, y_pred)
    val = float(res.statistic)
    return 0.0 if np.isnan(val) else val


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute continuous AQI regression metrics (MAE, RMSE, PLCC, SRCC, R2)."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(root_mean_squared_error(yt, yp))
    plcc = _safe_pearsonr(yt, yp)
    srcc = _safe_spearmanr(yt, yp)
    r2 = float(r2_score(yt, yp)) if len(yt) >= 2 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "plcc": plcc,
        "srcc": srcc,
        "r2": r2,
    }


def compute_classification_metrics(
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
) -> Dict[str, float]:
    """Compute 6-class accuracy, +-1 adjacent-class accuracy, and Macro-F1."""
    yt = np.asarray(y_true_bin, dtype=np.int64).ravel()
    yp = np.asarray(y_pred_bin, dtype=np.int64).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true_bin {yt.shape} vs y_pred_bin {yp.shape}")

    acc = float(accuracy_score(yt, yp))
    pm1_acc = float(np.mean(np.abs(yt - yp) <= 1))
    macro_f1 = float(
        f1_score(yt, yp, labels=list(ALL_BINS), average="macro", zero_division=0.0)
    )
    return {
        "acc": acc,
        "pm1_acc": pm1_acc,
        "macro_f1": macro_f1,
    }


def compute_tail_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
    tail_bins: Tuple[int, ...] = TAIL_BINS,
) -> Dict[str, float]:
    """Compute MAE and Macro-F1 for tail bins 5-6 (0-indexed {4, 5}, AQI > 200)."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    yt_b = np.asarray(y_true_bin, dtype=np.int64).ravel()
    yp_b = np.asarray(y_pred_bin, dtype=np.int64).ravel()

    mask = np.isin(yt_b, tail_bins)
    if np.any(mask):
        tail_mae = float(mean_absolute_error(yt[mask], yp[mask]))
    else:
        tail_mae = float("nan")

    tail_f1 = float(
        f1_score(yt_b, yp_b, labels=list(tail_bins), average="macro", zero_division=0.0)
    )
    return {
        "tail_mae": tail_mae,
        "tail_f1": tail_f1,
    }


def compute_all_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    y_true_bin: Optional[Sequence[int]] = None,
    y_pred_bin: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Compute the full Step 2+ evaluation metric suite on AQI (unscaled) targets."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()

    yt_b = (
        np.asarray(y_true_bin, dtype=np.int64).ravel()
        if y_true_bin is not None
        else aqi_to_bin_index(yt)
    )
    yp_b = (
        np.asarray(y_pred_bin, dtype=np.int64).ravel()
        if y_pred_bin is not None
        else aqi_to_bin_index(yp)
    )

    out: Dict[str, float] = {}
    out.update(compute_regression_metrics(yt, yp))
    out.update(compute_classification_metrics(yt_b, yp_b))
    out.update(compute_tail_metrics(yt, yp, yt_b, yp_b))
    return out


def summarize_fold_metrics(
    fold_metrics: List[Dict[str, float]],
    ddof: int = 1,
) -> Dict[str, Tuple[float, float]]:
    """Aggregate a list of per-fold metric dicts into {metric: (mean, std)}."""
    if not fold_metrics:
        raise ValueError("fold_metrics must be non-empty.")

    keys = list(fold_metrics[0].keys())
    summary: Dict[str, Tuple[float, float]] = {}
    for k in keys:
        vals = np.asarray([float(m[k]) for m in fold_metrics], dtype=np.float64)
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            summary[k] = (float("nan"), float("nan"))
        elif len(valid) == 1:
            summary[k] = (float(valid[0]), 0.0)
        else:
            effective_ddof = ddof if len(valid) > ddof else 0
            summary[k] = (
                float(np.mean(valid)),
                float(np.std(valid, ddof=effective_ddof)),
            )
    return summary


def format_mean_std(
    summary: Dict[str, Tuple[float, float]],
    keys: Optional[Iterable[str]] = None,
    precision_map: Optional[Dict[str, int]] = None,
) -> Dict[str, str]:
    """Format {metric: (mean, std)} into 'mean +- std' strings."""
    default_prec = {
        "mae": 2,
        "rmse": 2,
        "plcc": 4,
        "srcc": 4,
        "r2": 4,
        "acc": 4,
        "pm1_acc": 4,
        "macro_f1": 4,
        "tail_mae": 2,
        "tail_f1": 4,
    }
    if precision_map:
        default_prec.update(precision_map)

    selected = list(keys) if keys is not None else list(summary.keys())
    formatted: Dict[str, str] = {}
    for k in selected:
        mean_v, std_v = summary[k]
        p = default_prec.get(k, 4)
        if np.isnan(mean_v):
            formatted[k] = "nan"
        else:
            formatted[k] = f"{mean_v:.{p}f} +- {std_v:.{p}f}"
    return formatted


def paired_wilcoxon_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Run paired Wilcoxon signed-rank test across folds (Paper 17 / Paper 11)."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape or len(a) == 0:
        raise ValueError("scores_a and scores_b must have the same non-zero length.")
    diff = a - b
    if np.allclose(diff, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "mean_diff": 0.0}
    res = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "mean_diff": float(np.mean(diff)),
    }
