import os
import time
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from statsmodels.stats.contingency_tables import mcnemar

from Service.ObjectDetector.yolo import YOLOModel
from app import IGNITE
from IO_Module.logger import Logger

# =====================================================================
# SHARED UTILITIES & METRIC COMPUTATION
# =====================================================================

def build_dataset_manifest(safe_dir: str, fire_dir: str) -> pd.DataFrame:
    """Builds a structured dataset manifest from image directories."""
    Logger.info("Building dataset manifest...")
    records = []
    valid_exts = (".jpg", ".jpeg", ".png")
    targets = [(safe_dir, 0), (fire_dir, 1)]

    for dir_path, y_oracle in targets:
        if not os.path.exists(dir_path):
            Logger.info(f"Warning: Directory not found — {dir_path}")
            continue

        for fname in os.listdir(dir_path):
            if fname.lower().endswith(valid_exts):
                img_path = os.path.join(dir_path, fname)
                group_id = fname.split("_")[0] if "_" in fname else fname
                records.append({
                    "path": img_path,
                    "y_oracle": y_oracle,
                    "group_id": group_id
                })

    df = pd.DataFrame(records)
    Logger.info(f"Dataset manifest initialized: {len(df)} total image instances loaded.")
    return df


def predict_yolo(img: np.ndarray, yolo_instance: YOLOModel) -> tuple[int, float]:
    """Runs YOLO inference and returns (binary_prediction, latency_ms)."""
    t0 = time.perf_counter()
    preds = yolo_instance.predict(img)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    pred_class = 0 if preds is None or len(preds) == 0 else 1
    return pred_class, latency_ms


def predict_ignite(img: np.ndarray, ignite_instance: IGNITE) -> tuple[int, float]:
    """Runs IGNITE neuro-symbolic inference and returns (binary_prediction, latency_ms)."""
    t0 = time.perf_counter()
    value = ignite_instance._parse(img)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    pred_class = value[1][1][0] if value else 0
    return pred_class, latency_ms


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, latencies: list[float]) -> dict:
    """Computes safety metrics under the HSO protocol."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    avg_latency = float(np.mean(latencies)) if len(latencies) > 0 else 0.0

    return {
        "Accuracy": accuracy,
        "FAR": far,
        "Precision": precision,
        "Recall": recall,
        "Avg_Latency_ms": avg_latency
    }


# =====================================================================
# RESEARCH QUESTION EVALUATION FUNCTIONS
# =====================================================================

def RQ1(manifest: pd.DataFrame, yolo_instance: YOLOModel, ignite_instance: IGNITE) -> pd.DataFrame:
    """
    RQ1: Baseline System Efficacy & Latency
    Evaluates end-to-end performance across the full benchmark dataset.
    """
    Logger.info("Conducting RQ1: Baseline System Efficacy Evaluation...")

    y_true = manifest["y_oracle"].values
    paths = manifest["path"].values

    preds_yolo, preds_ignite = [], []
    lat_yolo, lat_ignite = [], []

    for img_path in paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        p_y, l_y = predict_yolo(img, yolo_instance)
        p_i, l_i = predict_ignite(img, ignite_instance)

        preds_yolo.append(p_y)
        preds_ignite.append(p_i)
        lat_yolo.append(l_y)
        lat_ignite.append(l_i)

    m_yolo = compute_metrics(y_true, np.array(preds_yolo), lat_yolo)
    m_ignite = compute_metrics(y_true, np.array(preds_ignite), lat_ignite)

    df_rq1 = pd.DataFrame([
        {"System": "Standalone YOLO", **m_yolo},
        {"System": "IGNITE Framework", **m_ignite}
    ])

    report_str = (
        "\n" + "=" * 65 + "\n"
        " RQ1: Baseline System Efficacy Evaluation Report\n"
        + "=" * 65 + "\n"
        + df_rq1.to_string(index=False) + "\n"
        + "=" * 65
    )
    Logger.report(report_str)
    return df_rq1


def RQ2(manifest: pd.DataFrame, yolo_instance: YOLOModel, ignite_instance: IGNITE, n_splits: int = 5) -> dict:
    """
    RQ2: Operational Stability via Cross-Validation
    Runs Stratified Group K-Fold CV and returns pooled out-of-fold data for RQ3.
    """
    Logger.info(f"Conducting RQ2: {n_splits}-Fold Cross-Validation & Stability Evaluation...")

    X = manifest["path"].values
    y = manifest["y_oracle"].values
    groups = manifest["group_id"].values

    unique_groups = len(np.unique(groups))
    if unique_groups >= n_splits:
        Logger.info(f"Splitting strategy: StratifiedGroupKFold ({n_splits} folds, {unique_groups} groups)")
        splitter = StratifiedGroupKFold(n_splits=n_splits)
        splits = list(splitter.split(X, y, groups))
    else:
        Logger.info(f"Splitting strategy: StratifiedKFold ({n_splits} folds)")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(splitter.split(X, y))

    fold_metrics_yolo, fold_metrics_ignite = [], []
    oof_y_true, oof_y_yolo, oof_y_ignite = [], [], []

    for fold, (_, test_idx) in enumerate(splits, 1):
        Logger.info(f"Evaluating Fold {fold}/{n_splits}...")
        test_paths = X[test_idx]
        test_y = y[test_idx]

        preds_yolo, preds_ignite = [], []
        lat_yolo, lat_ignite = [], []

        for img_path in test_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue

            p_y, l_y = predict_yolo(img, yolo_instance)
            p_i, l_i = predict_ignite(img, ignite_instance)

            preds_yolo.append(p_y)
            preds_ignite.append(p_i)
            lat_yolo.append(l_y)
            lat_ignite.append(l_i)

        m_yolo = compute_metrics(test_y, np.array(preds_yolo), lat_yolo)
        m_ignite = compute_metrics(test_y, np.array(preds_ignite), lat_ignite)

        fold_metrics_yolo.append(m_yolo)
        fold_metrics_ignite.append(m_ignite)

        oof_y_true.extend(test_y)
        oof_y_yolo.extend(preds_yolo)
        oof_y_ignite.extend(preds_ignite)

    df_yolo = pd.DataFrame(fold_metrics_yolo)
    df_ignite = pd.DataFrame(fold_metrics_ignite)

    # Report Mean ± Std Dev across folds
    metrics_to_report = ["Accuracy", "FAR", "Precision", "Recall", "Avg_Latency_ms"]
    summary_rows = []
    for m in metrics_to_report:
        summary_rows.append({
            "Metric": m,
            "Standalone YOLO (µ ± σ)": f"{df_yolo[m].mean():.4f} ± {df_yolo[m].std():.4f}",
            "IGNITE Framework (µ ± σ)": f"{df_ignite[m].mean():.4f} ± {df_ignite[m].std():.4f}"
        })

    summary_df = pd.DataFrame(summary_rows)
    report_str = (
        "\n" + "=" * 65 + "\n"
        f" RQ2: Operational Stability ({n_splits}-Fold Mean ± Std Dev) Report\n"
        + "=" * 65 + "\n"
        + summary_df.to_string(index=False) + "\n"
        + "=" * 65
    )
    Logger.report(report_str)

    return {
        "y_true": np.array(oof_y_true),
        "y_yolo": np.array(oof_y_yolo),
        "y_ignite": np.array(oof_y_ignite)
    }


def RQ3(oof_data: dict, alpha: float = 0.05):
    """
    RQ3: Statistical Significance via McNemar's Test
    Evaluates off-diagonal discordant pairs across pooled out-of-fold predictions.
    """
    Logger.info("Conducting RQ3: McNemar Statistical Significance Analysis...")

    y_true = oof_data["y_true"]
    y_yolo = oof_data["y_yolo"]
    y_ignite = oof_data["y_ignite"]

    yolo_correct = (y_yolo == y_true)
    ignite_correct = (y_ignite == y_true)

    a = int(np.sum(yolo_correct & ignite_correct))
    b = int(np.sum(yolo_correct & ~ignite_correct))
    c = int(np.sum(~yolo_correct & ignite_correct))
    d = int(np.sum(~yolo_correct & ~ignite_correct))

    table = [[a, b], [c, d]]
    discordant_count = b + c

    if discordant_count < 25:
        Logger.info("Notice: Discordant count < 25 -> Running Exact Binomial Test.")
        result = mcnemar(table, exact=True)
    else:
        Logger.info("Notice: Discordant count >= 25 -> Running Asymptotic Test with Yates's Correction.")
        result = mcnemar(table, exact=False, correction=True)

    is_significant = result.pvalue < alpha
    p_val_fmt = f"{result.pvalue:.5e}" if result.pvalue < 1e-3 else f"{result.pvalue:.5f}"

    report_str = (
        "\n" + "=" * 65 + "\n"
        " RQ3: McNemar Statistical Significance Report\n"
        + "=" * 65 + "\n"
        f" Contingency Table Matrix:\n"
        f"   a (Both Correct)               : {a}\n"
        f"   b (YOLO Correct, IGNITE Wrong) : {b}\n"
        f"   c (YOLO Wrong, IGNITE Correct) : {c}\n"
        f"   d (Both Wrong)                 : {d}\n"
        f" Total Discordant Pairs (b + c)   : {discordant_count}\n"
        f" ---------------------------------------------------\n"
        f" Test Statistic : {result.statistic:.4f}\n"
        f" p-value        : {p_val_fmt}\n"
        f" Decision       : {'STATISTICALLY SIGNIFICANT' if is_significant else 'NOT STATISTICALLY SIGNIFICANT'} (alpha = {alpha})\n"
        + "=" * 65
    )
    Logger.report(report_str)


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    SAFE_DIR = "Dataset/evaluation_slices/safe"
    FIRE_DIR = "Dataset/evaluation_slices/fire_present_set"
    YOLO_WEIGHTS = "Service//ObjectDetector//fire.pt"

    Logger.info("Initializing perception backbones and IGNITE framework...")
    yolo_instance = YOLOModel(YOLO_WEIGHTS)
    ignite_instance = IGNITE()

    manifest = build_dataset_manifest(SAFE_DIR, FIRE_DIR)

    if len(manifest) > 0:
        # Run baseline metrics (RQ1)
        RQ1(manifest, yolo_instance, ignite_instance)

        # Run stability cross-validation (RQ2) and capture out-of-fold predictions
        oof_data = RQ2(manifest, yolo_instance, ignite_instance, n_splits=5)

        # Run statistical significance test (RQ3) on out-of-fold predictions
        RQ3(oof_data)
        Logger.flush()#A Insurance Method to Flush All Reports
    else:
        Logger.error("Evaluation aborted: Dataset manifest is empty.")
        
        
    