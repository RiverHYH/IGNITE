import os
import time
import cv2
import numpy as np
import pandas as pd

import torch

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from statsmodels.stats.contingency_tables import mcnemar

from PIL import Image
from typing import Callable, Any
import argparse

from Service.ObjectDetector.yolo import YOLOModel
from Service.Predicate.affordanceEmbedder import GeometricPredicateExtractor
from app import IGNITE
from IO_Module.videoCapture import CameraStream
from IO_Module.logger import Logger
from IO_Module.throughputMonitor import PerformanceMonitor
from IO_Module.cameraList import list_cameras_windows
from IO_Module.boundingBoxDrawer import draw_predictions

import torch.nn.functional as F

def build_dataset_manifest(safe_dir: str, fire_dir: str) -> pd.DataFrame:
    """
    Builds a structured dataset manifest from image directories.
    
    Args:
        safe_dir: Path to directory containing safe images (class 0).
        fire_dir: Path to directory containing fire images (class 1).
        
    Returns:
        DataFrame with columns: path, y_oracle, group_id.
    """
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
    """
    Runs YOLO inference and returns (binary_prediction, latency_ms).
    
    Args:
        img: Input image array.
        yolo_instance: YOLO model instance detecting fire.
        
    Returns:
        Tuple of binary prediction and latency in milliseconds.
    """
    t0 = time.perf_counter()
    preds = yolo_instance.predict(img)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    pred_class = 0 if preds is None or len(preds) == 0 else 1 # No Bounding Boxes for Flame and Smoke indicates No Fire
    return pred_class, latency_ms


def predict_ignite(img: np.ndarray, ignite_instance: IGNITE) -> tuple[int, float]:
    """
    Runs IGNITE neuro-symbolic inference and returns (binary_prediction, latency_ms).
    
    Args:
        img: Input image array.
        ignite_instance: IGNITE model instance.
        
    Returns:
        Tuple of binary prediction and latency in milliseconds.
    """
    t0 = time.perf_counter()
    value = ignite_instance._parse(img)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    pred_class = value[1][1][0] if value else 0
    return pred_class, latency_ms


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, latencies: list[float]) -> dict:
    """
    Computes General Benchmark metrics including accuracy, precision, recall, and latency.
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        latencies: Inference latencies in milliseconds.
    Returns:
        Dictionary of metrics.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # False Alarm Rate = FP / (FP + TN)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall (Sensitivity) = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    avg_latency = float(np.mean(latencies)) if len(latencies) > 0 else 0.0

    return {
        "Accuracy": accuracy,
        "FAR": far,
        "Precision": precision,
        "Recall": recall,
        "Avg_Latency_ms": avg_latency,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }


# =====================================================================
# RESEARCH QUESTION EVALUATION FUNCTIONS
# =====================================================================

def Test1(manifest: pd.DataFrame, yolo_instance: YOLOModel, ignite_instance: IGNITE) -> tuple[pd.DataFrame, dict]:
    """
    T1: Baseline System Efficacy & Latency
    Evaluates single-pass inference across the fully isolated benchmark dataset (D_HSO).
    Args:
        manifest: Dataset manifest.
        yolo_instance: YOLO model instance.
        ignite_instance: IGNITE model instance.
    Returns:
        Tuple containing DataFrame of evaluation results and a dictionary with raw data.
    """
    Logger.info("Conducting T1: Baseline System Efficacy Evaluation on Isolated Benchmark...")

    paths = manifest["path"].values
    y_oracle = manifest["y_oracle"].values

    valid_y_true = []
    preds_yolo, preds_ignite = [], []
    lat_yolo, lat_ignite = [], []

    for img_path, y_true in zip(paths, y_oracle):
        img = cv2.imread(img_path)
        if img is None:
            Logger.info(f"Warning: Could not read image at {img_path}")
            continue

        p_y, l_y = predict_yolo(img, yolo_instance)
        p_i, l_i = predict_ignite(img, ignite_instance)

        valid_y_true.append(y_true)
        preds_yolo.append(p_y)
        preds_ignite.append(p_i)
        lat_yolo.append(l_y)
        lat_ignite.append(l_i)

    y_true_arr = np.array(valid_y_true)
    yolo_preds_arr = np.array(preds_yolo)
    ignite_preds_arr = np.array(preds_ignite)

    m_yolo = compute_metrics(y_true_arr, yolo_preds_arr, lat_yolo)
    m_ignite = compute_metrics(y_true_arr, ignite_preds_arr, lat_ignite)

    df_rq1 = pd.DataFrame([
        {"System": "Standalone YOLO", **m_yolo},
        {"System": "IGNITE Framework", **m_ignite}
    ])

    report_str = (
        "\n" + "=" * 65 + "\n"
        " T1: Baseline System Efficacy Evaluation Report\n"
        + "=" * 65 + "\n"
        + df_rq1.to_string(index=False) + "\n"
        + "=" * 65
    )
    Logger.report(report_str)

    eval_data = {
        "y_true": y_true_arr,
        "y_yolo": yolo_preds_arr,
        "y_ignite": ignite_preds_arr
    }

    return df_rq1, eval_data


def Test2(manifest: pd.DataFrame, yolo_instance: YOLOModel, ignite_instance: IGNITE, n_splits: int = 5) -> None:
    """
    T2: Operational Stability via Cross-Validation
    Runs Stratified (Group) K-Fold CV to measure stability and variance across subsets of D_HSO.
    Args:
        manifest: Dataset manifest.
        yolo_instance: YOLO model instance.
        ignite_instance: IGNITE model instance.
        n_splits: Number of folds for cross-validation.
    Returns:
        None
    """
    Logger.info(f"Conducting T2: {n_splits}-Fold Cross-Validation & Stability Evaluation...")

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

    for fold, (_, test_idx) in enumerate(splits, 1):
        Logger.info(f"Evaluating Fold {fold}/{n_splits}...")
        test_paths = X[test_idx]
        test_y = y[test_idx]

        valid_test_y = []
        preds_yolo, preds_ignite = [], []
        lat_yolo, lat_ignite = [], []

        for img_path, y_true in zip(test_paths, test_y):
            img = cv2.imread(img_path)
            if img is None:
                continue

            p_y, l_y = predict_yolo(img, yolo_instance)
            p_i, l_i = predict_ignite(img, ignite_instance)

            valid_test_y.append(y_true)
            preds_yolo.append(p_y)
            preds_ignite.append(p_i)
            lat_yolo.append(l_y)
            lat_ignite.append(l_i)

        m_yolo = compute_metrics(np.array(valid_test_y), np.array(preds_yolo), lat_yolo)
        m_ignite = compute_metrics(np.array(valid_test_y), np.array(preds_ignite), lat_ignite)

        fold_metrics_yolo.append(m_yolo)
        fold_metrics_ignite.append(m_ignite)

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
        f" T2: Operational Stability ({n_splits}-Fold Mean ± Std Dev) Report\n"
        + "=" * 65 + "\n"
        + summary_df.to_string(index=False) + "\n"
        + "=" * 65
    )
    Logger.report(report_str)


def Test3(eval_data: dict, alpha: float = 0.05) -> None:
    """
    T3: Statistical Significance via McNemar's Test
    Evaluates off-diagonal discordant pairs from isolated single-pass test predictions.
    Args:
        eval_data: Dictionary containing evaluation data. Passing down from Test 1
        alpha: Significance level.
    """
    Logger.info("Conducting T3: McNemar Statistical Significance Analysis on Isolated Holdout Set...")

    y_true = eval_data["y_true"]
    y_yolo = eval_data["y_yolo"]
    y_ignite = eval_data["y_ignite"]

    yolo_correct = (y_yolo == y_true)
    ignite_correct = (y_ignite == y_true)
    #Purpose: XOR Operations
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
        " T3: McNemar Statistical Significance Report (Isolated D_HSO)\n"
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

def Test4(manifest_df: pd.DataFrame, ignite_pipeline=IGNITE()) -> dict:
    """
    Evaluates Outer Branch Collapse Rate (OBCR) using counterfactual occlusion.
    Filters outcome[0] for 'flame' and 'smoke' classes to build full hazard ROI masks.
    
    Args:
        manifest_df (pd.DataFrame): Dataset manifest from `build_dataset_manifest`.
        ignite_pipeline: Pipeline instance implementing `._parse(image)`.
                        
    Returns:
        dict: Metric dictionary containing `summary_str` and `details_df`.
    """
    Logger.info("Initiating Counterfactual Occlusion Probing (OBCR)...")
    
    # Filter strictly for ground-truth hazard instances (y_oracle == 1)
    pos_df = manifest_df[manifest_df["y_oracle"] == 1].copy()
    if pos_df.empty:
        Logger.info("Warning: No positive hazard instances found in manifest.")
        return {"obcr": 0.0, "total_tested": 0, "collapsed_count": 0, "details_df": pd.DataFrame()}

    total_valid_probes = 0
    collapsed_count = 0
    records = []

    for _, row in pos_df.iterrows():
        img_path = row["path"]
        img = cv2.imread(img_path)
        if img is None:
            Logger.info(f"Warning: Failed to read image — {img_path}")
            continue

        h, w, _ = img.shape

        # 1. Baseline Inference Pass via _parse()
        try:
            outcome = ignite_pipeline._parse(img, (0.25, 0.5))
            
            # Safely extract detections list from outcome[0]
            detections = outcome[0] if isinstance(outcome, (tuple, list)) and len(outcome) > 0 else []
            
            # Safely navigate nested decision y_orig from outcome[1][1][0]
            y_orig = None
            if len(outcome) > 1 and isinstance(outcome[1], (tuple, list)) and len(outcome[1]) > 1:
                if isinstance(outcome[1][1], (tuple, list)) and len(outcome[1][1]) > 0:
                    y_orig = outcome[1][1][0]

        except (IndexError, TypeError, AttributeError) as e:
            Logger.info(f"Warning: Baseline output parsing error on frame {img_path} — {e}")
            continue

        # Extract all hazard bounding boxes matching 'flame' or 'smoke'
        b_fire_list = [
            item for item in detections 
            if isinstance(item, dict) and item.get("class_name") in ("flame", "smoke")
        ]

        # Skip false-negative baselines (must have active baseline alert + detected hazard items)
        if y_orig != 1 or not b_fire_list:
            continue

        # 2. Counterfactual Masking Pass (I_i^occ)
        img_occ = img.copy()
        
        for item in b_fire_list:
            bbox_data = item.get("bbox")
            if bbox_data is None:
                continue

            # Transfer CUDA tensors to CPU list before casting
            if isinstance(bbox_data, torch.Tensor):
                bbox_coords = bbox_data.detach().cpu().squeeze().tolist()
            else:
                bbox_coords = bbox_data

            # Guard against unexpected index/tuple lengths during unpacking
            if not isinstance(bbox_coords, (tuple, list)) or len(bbox_coords) < 4:
                continue

            x1, y1, x2, y2 = map(int, bbox_coords[:4])

            # Clamp bounding box coordinates to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            img_occ[y1:y2, x1:x2] = 0  # Zero out detected hazard ROI

        # 3. Probe Inference Pass on Occluded Frame via _parse()
        try:
            # Maintained threshold parameters (0.25, 0.5) consistently with baseline pass
            outcome_occ = ignite_pipeline._parse(img_occ, (0.25, 0.5))
            
            # Safely navigate nested decision y_occ from outcome_occ[1][1][0]
            y_occ = 0  # Default to 0 (collapsed/safe) if missing
            if isinstance(outcome_occ, (tuple, list)) and len(outcome_occ) > 1:
                if isinstance(outcome_occ[1], (tuple, list)) and len(outcome_occ[1]) > 1:
                    if isinstance(outcome_occ[1][1], (tuple, list)) and len(outcome_occ[1][1]) > 0:
                        y_occ = outcome_occ[1][1][0]

        except (IndexError, TypeError, AttributeError) as e:
            Logger.info(f"Warning: Occluded pass output parsing error on frame {img_path} — {e}")
            continue

        # 4. Check for Outer Branch Collapse (\hat{Y}_IGNITE(I^occ) == 0)
        is_collapsed = (y_occ == 0)
        
        total_valid_probes += 1
        if is_collapsed:
            collapsed_count += 1

        records.append({
            "path": img_path,
            "group_id": row.get("group_id", None),
            "y_orig": y_orig,
            "y_occ": y_occ,
            "hazards_masked": len(b_fire_list),
            "collapsed": is_collapsed
        })

    # Calculate final dataset metric
    obcr_score = (collapsed_count / total_valid_probes) if total_valid_probes > 0 else 0.0
    
    # Formatted print-out string report
    final_result = (
        "\n==================================================\n"
        "    IGNITE Diagnostic Evaluation: OBCR Probing    \n"
        "==================================================\n"
        f" Ground-Truth Hazard Instances Loaded : {len(pos_df)}\n"
        f" Active Baseline Probes Executed      : {total_valid_probes}\n"
        f" Successfully Collapsed Alerts        : {collapsed_count}\n"
        f" Outer Branch Collapse Rate (OBCR)    : {obcr_score * 100:.2f}%\n"
        "=================================================="
    )

    # Log structured string summary
    Logger.report(final_result)

    return {
        "obcr": obcr_score,
        "total_tested": total_valid_probes,
        "collapsed_count": collapsed_count,
        "summary_str": final_result,
        "details_df": pd.DataFrame(records)
    }
    
def Test5(
    manifest_df: pd.DataFrame,
    fire_detector: Any,
    obj_detector: Any,
    tau_margin: float = 0.15,
    max_logged_violations: int = 15,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Evaluates Predicate Confidence Margin (Delta_margin) using deterministic
    temperature-scaled outputs from GeometricPredicateExtractor.
    Args:
        manifest_df (pd.DataFrame): Dataset manifest from `build_dataset_manifest`.
        fire_detector: Fire detector instance implementing `.predict(image)`.
        obj_detector: Object detector instance implementing `.predict(image)`.
        tau_margin (float, optional): Margin threshold for violation reporting. Defaults to 0.15.
        max_logged_violations (int, optional): Maximum number of violations to log. Defaults to 15.
        device (str, optional): Device to use for inference. Defaults to "cpu".
    """
    Logger.info("Initialising zero-shot GeometricPredicateExtractor diagnostic audit...")
    extractor = GeometricPredicateExtractor().to(device)
    extractor.eval()

    all_margins = []
    violation_records = []
    ued_violations = 0
    total_pairs_evaluated = 0
    active_frames = 0

    for _, row in manifest_df.iterrows():
        img_path = row["path"]

        if not os.path.exists(img_path):
            continue

        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            Logger.info(f"Failed to open image {img_path}: {e}")
            continue

        # Execute dual detector predictions
        fire_detections = fire_detector.predict(img_path)
        obj_detections = obj_detector.predict(img_path)

        if not fire_detections or not obj_detections:
            continue

        active_frames += 1

        # Scale raw pixel coordinates to normalized [0, 1] XYXY float32 Tensors
        norm_scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=device)

        boxes_f = [
            (det["bbox"].to(device).float() if isinstance(det["bbox"], torch.Tensor)
             else torch.tensor(det["bbox"], dtype=torch.float32, device=device)) / norm_scale
            for det in fire_detections
        ]

        boxes_o = [
            (det["bbox"].to(device).float() if isinstance(det["bbox"], torch.Tensor)
             else torch.tensor(det["bbox"], dtype=torch.float32, device=device)) / norm_scale
            for det in obj_detections
        ]

        gt_pred = row.get("gt_predicate", None)
        if pd.isna(gt_pred) or gt_pred not in extractor.predicates:
            gt_pred = None

        with torch.no_grad():
            for f_idx, box_f in enumerate(boxes_f):
                for o_idx, box_o in enumerate(boxes_o):
                    # Direct signature call: returns (pred_class, margin, sim_dict)
                    pred_class, margin, sim_dict = extractor.get_predicate(
                        box_f=box_f,
                        box_o=box_o,
                        gt_predicate=gt_pred,
                    )

                    margin_val = float(margin)
                    all_margins.append(margin_val)
                    total_pairs_evaluated += 1

                    # Extract target vs competitor probabilities directly from sim_dict
                    if gt_pred is not None:
                        s_target = sim_dict[gt_pred]
                        s_comp = max(v for k, v in sim_dict.items() if k != gt_pred)
                    else:
                        sorted_probs = sorted(sim_dict.values(), reverse=True)
                        s_target = sorted_probs[0]
                        s_comp = sorted_probs[1] if len(sorted_probs) > 1 else 0.0

                    if margin_val < tau_margin:
                        ued_violations += 1
                        violation_records.append({
                            "image": os.path.basename(img_path),
                            "pair": f"f{f_idx}-o{o_idx}",
                            "gt_pred": gt_pred if gt_pred is not None else "N/A",
                            "pred_class": pred_class,
                            "s_target": s_target,
                            "s_competitor": s_comp,
                            "delta_margin": margin_val,
                        })

    if total_pairs_evaluated == 0:
        Logger.report("IGNITE DIAGNOSTIC REPORT: No valid bounding box pairs detected across dataset manifest.")
        return {}

    margins_arr = np.array(all_margins)
    ued_rate = (ued_violations / total_pairs_evaluated) * 100.0

    metrics = {
        "total_manifest_images": len(manifest_df),
        "frames_with_dual_detections": active_frames,
        "total_candidate_pairs": total_pairs_evaluated,
        "mean_delta_margin": float(margins_arr.mean()),
        "std_delta_margin": float(margins_arr.std()),
        "min_delta_margin": float(margins_arr.min()),
        "max_delta_margin": float(margins_arr.max()),
        "ued_rate_pct": ued_rate,
        "ued_violations": ued_violations,
    }

    report_lines = [
        "IGNITE DUAL-DETECTOR DIAGNOSTIC REPORT: ZERO-SHOT MARGIN & UED AUDIT",
        "=" * 70,
        f"Extractor Mode          : Zero-Shot Sinusoidal Template Matching",
        f"Total Manifest Frames   : {metrics['total_manifest_images']}",
        f"Frames Active (Both)    : {active_frames}",
        f"Total BBox Pairs Tested : {total_pairs_evaluated}",
        f"Tau Margin Threshold    : {tau_margin:.2f}",
        "-" * 70,
        f"Mean Delta Margin (\\Delta) : {metrics['mean_delta_margin']:.4f} ± {metrics['std_delta_margin']:.4f}",
        f"Min / Max Margin        : {metrics['min_delta_margin']:.4f} / {metrics['max_delta_margin']:.4f}",
        f"UED Violation Rate      : {metrics['ued_rate_pct']:.2f}% ({ued_violations}/{total_pairs_evaluated})",
        "-" * 70,
        "FAILING PAIR PROBABILITY BREAKDOWN (Delta < Tau):",
        f"{'Image':<20} | {'Pair':<7} | {'GT':<6} | {'Pred':<7} | {'s_target':<9} | {'s_comp':<9} | {'Delta':<7}",
        "-" * 70,
    ]

    for v in violation_records[:max_logged_violations]:
        report_lines.append(
            f"{v['image'][:19]:<20} | {v['pair']:<7} | {str(v['gt_pred']):<6} | "
            f"{str(v['pred_class']):<7} | {v['s_target']:<9.4f} | {v['s_competitor']:<9.4f} | {v['delta_margin']:<7.4f}"
        )

    if len(violation_records) > max_logged_violations:
        report_lines.append(f"... and {len(violation_records) - max_logged_violations} additional failing pairs omitted from log summary.")

    report_lines.append("=" * 70)

    Logger.report("\n" + "\n".join(report_lines))

    return metrics
# MAIN ENTRY POINT
# =====================================================================
def runExperiment():
    """Automated experiment runner"""
    SAFE_DIR = "Dataset//evaluation_slices//safe"
    FIRE_DIR = "Dataset//evaluation_slices//fire_present_set//images"
    YOLO_WEIGHTS = "Service//ObjectDetector//fire.pt"
    
    Logger.info("Initializing perception backbones and IGNITE framework...")
    yolo_instance = YOLOModel(YOLO_WEIGHTS)
    ignite_instance = IGNITE()
    
    manifest = build_dataset_manifest(SAFE_DIR, FIRE_DIR)
    
    if len(manifest) > 0:
            # Run baseline metrics on isolated holdout set (RQ1) and extract single-pass predictions
            df_rq1, eval_data = Test1(manifest, yolo_instance, ignite_instance)
    
            # Run stability cross-validation (RQ2)
            Test2(manifest, yolo_instance, ignite_instance, n_splits=10)
    
            # Run statistical significance test (RQ3) on direct single-pass holdout predictions
            Test3(eval_data)
            Test4(manifest, ignite_instance)
            Test5(manifest,YOLOModel("Service//ObjectDetector//fire.pt"),YOLOModel("Service//ObjectDetector//obj1.pt"))
            
            Logger.flush()  # Insurance Method to Flush All Reports
    else:
            Logger.error("Evaluation aborted: Dataset manifest is empty.")
            
def runThruputPerformance(tframe=1000):
    """Automated Throughput Performance Test Runner"""
    SKIP_FRAMES = 15
    monitor = PerformanceMonitor()

    # Camera selection logic
    cams = list_cameras_windows()
    if not cams:
        Logger.error("Application Terminated Due to No video devices found.")
        exit(1)

    print("Available Devices Are:")
    for i, name in enumerate(cams):
        print(f"[{i}]: {name}")

    device_no = int(input("Select device index: "))
    device_name = cams[device_no]

    TARGET_TEST_FRAMES = tframe
    frame_counter = 0

    print(
        f"[INFO] Running IGNITE performance audit for {TARGET_TEST_FRAMES} frames..."
    )

    application = IGNITE()
    with CameraStream(device_name=device_name) as stream:
        for frame in stream.frames():
            t_loop_start = time.perf_counter()

            # 1. IO Stage
            t_io_start = time.perf_counter()
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            t_io_end = time.perf_counter()

            # 2. Inference Stage
            t_inf_start = time.perf_counter()
            if frame_counter % SKIP_FRAMES == 0:
                outcome = application._parse(bgr)
            else:
                outcome = None
            t_inf_end = time.perf_counter()

            # 3. Drawing Stage
            t_draw_start = time.perf_counter()
            if outcome:
                result, decision = outcome
                draw_predictions(bgr, result, in_place=True)
            t_draw_end = time.perf_counter()

            # Display
            cv2.imshow("IGNITE", bgr)
            key = cv2.waitKey(1) & 0xFF

            # Loop calculation
            t_loop_end = time.perf_counter()

            # Calculate Stage Latencies (ms)
            io_ms = (t_io_end - t_io_start) * 1000
            inf_ms = (t_inf_end - t_inf_start) * 1000
            draw_ms = (t_draw_end - t_draw_start) * 1000
            loop_ms = (t_loop_end - t_loop_start) * 1000

            # Record Latencies & FPS
            monitor.record_latency(io_ms, inf_ms, draw_ms, loop_ms)
            fps_now = monitor.record_fps(loop_ms)

            # Record System & Process Memory Resources
            mem_now = monitor.record_memory()
            gpu_mem_now, gpu_util_now = monitor.record_gpu()

            # Frame drop detection
            if monitor.detect_frame_drop(loop_ms):
                print(
                    f"[DROP] Frame {frame_counter}: {loop_ms:.2f} ms (FPS={fps_now:.2f})"
                )

            frame_counter += 1
            if frame_counter >= TARGET_TEST_FRAMES or key == 27:
                break

    cv2.destroyAllWindows()

    # Generate Performance Summary
    gpu_summary = ""
    if monitor.gpu_available and len(monitor.gpu_mem_usage) > 0:
        gpu_summary = (
            f"Peak GPU Memory: {max(monitor.gpu_mem_usage):.2f} MB\n"
            f"Peak GPU Utilization: {max(monitor.gpu_util):.2f}%\n"
            f"GPU Memory Leak Detected: {monitor.detect_gpu_leak()}\n"
        )

    summary = (
        "\n=== IGNITE PERFORMANCE SUMMARY ===\n"
        f"Frames Analyzed: {frame_counter}\n"
        f"Mean FPS: {np.mean(monitor.fps):.2f}\n"
        f"Peak CPU Memory: {max(monitor.mem_usage):.2f} MB\n"
        f"CPU Memory Leak Detected: {monitor.detect_memory_leak()}\n"
        f"{gpu_summary}"
        "\nLatency Stats:\n"
        "IO: " + str(monitor.stats(monitor.io_lat)) + "\n"
        "Inference: " + str(monitor.stats(monitor.inf_lat)) + "\n"
        "Draw: " + str(monitor.stats(monitor.draw_lat)) + "\n"
        "Loop: " + str(monitor.stats(monitor.loop_lat))
    )
    Logger.report(summary)
    Logger.flush()

    # Generate plot (Outputs auto-scaled subplots with integrated RAM & VRAM AUC shading)
    monitor.plot(
        save_path="reports/run_thru.png",
        kwargs={"TargetFPS": 30},
    )
    return None

def runAblation(
    manifest_df: pd.DataFrame,
    fire_detector: Any,
    obj_detector: Any,
    extractor_cls: Any,
    tau_margin: float = 0.15,
    temperatures: list[float] | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Standalone temperature ablation study runner (Unsupervised Confidence Calibration).
    Args:
        manifest_df: Dataset manifest dataframe.
        fire_detector: Fire detector instance.
        obj_detector: Object detector instance.
        extractor_cls: Similarity extractor class.
        tau_margin: Tau margin for thresholding.
        temperatures: Temperature values to test.
        device: Device to run on.
    """
    if temperatures is None:
        temperatures = [0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10, 0.20, 0.50]

    Logger.info("Initialising Standalone Temperature Ablation Study (Unlabeled Mode)...")
    extractor = extractor_cls().to(device)
    extractor.eval()

    raw_sims_list = []
    total_pairs_evaluated = 0
    active_frames = 0

    # -------------------------------------------------------------------------
    # STEP 1: Single-Pass Detector Inference & Raw Similarity Extraction
    # -------------------------------------------------------------------------
    for _, row in manifest_df.iterrows():
        img_path = row["path"]
        if not os.path.exists(img_path):
            continue

        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            Logger.info(f"Failed to open image {img_path}: {e}")
            continue

        fire_detections = fire_detector.predict(img_path)
        obj_detections = obj_detector.predict(img_path)

        if not fire_detections or not obj_detections:
            continue

        active_frames += 1
        norm_scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=device)

        boxes_f = [
            (det["bbox"].to(device).float() if isinstance(det["bbox"], torch.Tensor)
             else torch.tensor(det["bbox"], dtype=torch.float32, device=device)) / norm_scale
            for det in fire_detections
        ]
        boxes_o = [
            (det["bbox"].to(device).float() if isinstance(det["bbox"], torch.Tensor)
             else torch.tensor(det["bbox"], dtype=torch.float32, device=device)) / norm_scale
            for det in obj_detections
        ]

        with torch.no_grad():
            for box_f in boxes_f:
                for box_o in boxes_o:
                    # Exact CXCYWH conversion matching get_predicate
                    wf = torch.clamp(box_f[2] - box_f[0], min=1e-6)
                    hf = torch.clamp(box_f[3] - box_f[1], min=1e-6)
                    cxf, cyf = box_f[0] + wf / 2.0, box_f[1] + hf / 2.0

                    wo = torch.clamp(box_o[2] - box_o[0], min=1e-6)
                    ho = torch.clamp(box_o[3] - box_o[1], min=1e-6)
                    cxo, cyo = box_o[0] + wo / 2.0, box_o[1] + ho / 2.0

                    # Construct spatial layout tensor t_live
                    t_live = torch.tensor([
                        (cxf - cxo) / wo,
                        (cyf - cyo) / ho,
                        torch.log(wf / wo),
                        torch.log(hf / ho)
                    ], dtype=torch.float32, device=device)

                    # Multi-frequency wave projection & unit normalization
                    v_live = extractor._compute_wave_embedding(t_live)
                    v_live_norm = v_live / torch.norm(v_live, p=2)

                    # Raw unscaled inner product against template anchors
                    raw_sims = torch.mv(extractor.anchors_norm, v_live_norm)  # Shape: (|C^p|,)

                    raw_sims_list.append(raw_sims)
                    total_pairs_evaluated += 1

    if total_pairs_evaluated == 0:
        Logger.report("IGNITE ABLATION REPORT: No valid candidate bounding box pairs detected across dataset manifest.")
        return pd.DataFrame()

    # Stack cached similarity vectors: shape (N, |C^p|)
    raw_sims_tensor = torch.stack(raw_sims_list).to(device)

    # -------------------------------------------------------------------------
    # STEP 2: Vectorised Temperature Grid Search
    # -------------------------------------------------------------------------
    ablation_records = []

    for T in temperatures:
        scaled_logits = raw_sims_tensor / T
        probs = F.softmax(scaled_logits, dim=-1)

        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        margins = sorted_probs[:, 0] - sorted_probs[:, 1]

        ued_count = (margins < tau_margin).sum().item()
        ued_rate_pct = (ued_count / total_pairs_evaluated) * 100.0

        mean_margin = margins.mean().item()
        std_margin = margins.std().item()

        log_probs = torch.log(probs + 1e-12)
        entropy = -(probs * log_probs).sum(dim=-1).mean().item()

        ablation_records.append({
            "Temp (T)": T,
            "UED Rate (%)": round(ued_rate_pct, 2),
            "Mean Margin (Δ)": round(mean_margin, 4),
            "Std Margin": round(std_margin, 4),
            "Mean Entropy": round(entropy, 4)
        })

    df_results = pd.DataFrame(ablation_records)

    # -------------------------------------------------------------------------
    # STEP 3: Report String Payload Construction
    # -------------------------------------------------------------------------
    report_lines = [
        "IGNITE PREDICATE EMBEDDER: UNSUPERVISED TEMPERATURE ABLATION REPORT",
        "=" * 70,
        f"Total Manifest Frames   : {len(manifest_df)}",
        f"Frames Active (Both)    : {active_frames}",
        f"Total BBox Pairs Tested : {total_pairs_evaluated}",
        f"Tau Margin Threshold    : {tau_margin:.2f}",
        "-" * 70,
        df_results.to_string(index=False),
        "=" * 70,
    ]

    Logger.report("\n" + "\n".join(report_lines))

    return df_results

if __name__ == "__main__":
    SAFE_DIR = "Dataset//evaluation_slices//safe"
    FIRE_DIR = "Dataset//evaluation_slices//fire_present_set//images"
    manifest = build_dataset_manifest(SAFE_DIR, FIRE_DIR)
        
    if len(manifest) > 0:
        runAblation(manifest,YOLOModel("Service//ObjectDetector//fire.pt"),YOLOModel("Service//ObjectDetector//obj1.pt"),extractor_cls=GeometricPredicateExtractor)
        Logger.flush()