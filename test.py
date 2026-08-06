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
    """Computes safety metrics under the HSO protocol with division safeguards."""
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
    Returns summary DataFrame and raw prediction dictionaries for McNemar's test.
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
    Evaluates off-diagonal discordant pairs from isolated single-pass test predictions on D_HSO.
    """
    Logger.info("Conducting T3: McNemar Statistical Significance Analysis on Isolated Holdout Set...")

    y_true = eval_data["y_true"]
    y_yolo = eval_data["y_yolo"]
    y_ignite = eval_data["y_ignite"]

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
    Evaluates Predicate Confidence Margin (\Delta_margin) using deterministic
    temperature-scaled outputs from GeometricPredicateExtractor.
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IGNITE Diagnostic Margin & UED Evaluation Runner")
    parser.add_argument("--weights", type=str, required=False, help="Path to GeometricPredicateExtractor checkpoint")
    parser.add_argument("--tau", type=float, default=0.15, help="Safety threshold margin tau (default: 0.15)")
    parser.add_argument("--device", type=str, default="cpu", help="Device target ('cpu' or 'cuda')")
    args = parser.parse_args()

    # Place execution wrapper instantiation here when executed directly as CLI
# MAIN ENTRY POINT
# =====================================================================
def runExperiment():
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
    SKIP_FRAMES = 15
    monitor = PerformanceMonitor()

    # PURPOSE: For the purpose of monitoring each stage, the script isn't using activate() for full-pipeline execution.
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

            # IO
            t_io_start = time.perf_counter()
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            t_io_end = time.perf_counter()

            # Inference
            t_inf_start = time.perf_counter()
            if frame_counter % SKIP_FRAMES == 0:
                outcome = application._parse(bgr)
            else:
                outcome = None
            t_inf_end = time.perf_counter()

            # Drawing
            t_draw_start = time.perf_counter()
            if outcome:
                result, decision = outcome
                draw_predictions(bgr, result, in_place=True)
            t_draw_end = time.perf_counter()

            # Display
            cv2.imshow("IGNITE", bgr)
            key = cv2.waitKey(1) & 0xFF

            # Loop end
            t_loop_end = time.perf_counter()

            # Latencies
            io_ms = (t_io_end - t_io_start) * 1000
            inf_ms = (t_inf_end - t_inf_start) * 1000
            draw_ms = (t_draw_end - t_draw_start) * 1000
            loop_ms = (t_loop_end - t_loop_start) * 1000

            monitor.record_latency(io_ms, inf_ms, draw_ms, loop_ms)

            # Memory Recording
            mem_now = monitor.record_memory()

            # GPU Recording (ADDED THIS)
            gpu_mem_now, gpu_util_now = monitor.record_gpu()

            # FPS
            fps_now = monitor.record_fps(loop_ms)

            # Drop detection
            if monitor.detect_frame_drop(loop_ms):
                print(
                    f"[DROP] Frame {frame_counter}: {loop_ms:.2f} ms (FPS={fps_now:.2f})"
                )

            frame_counter += 1
            if frame_counter >= TARGET_TEST_FRAMES or key == 27:
                break

    cv2.destroyAllWindows()

    # Summary Generation (Include GPU stats if available)
    gpu_summary = ""
    if monitor.gpu_available and len(monitor.gpu_mem_usage) > 0:
        gpu_summary = (
            f"Peak GPU Memory: {max(monitor.gpu_mem_usage):.2f} MB\n"
            f"GPU Memory Leak Detected: {monitor.detect_gpu_leak()}\n"
        )

    summary = (
        "\n=== IGNITE PERFORMANCE SUMMARY ===\n"
        f"Frames: {frame_counter}\n"
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

    # Plotting (Will now populate GPU graphs)
    monitor.plot(
        save_path="reports/run_thru.png",
        kwargs={"RAM": 16, "VRAM": 4, "TargetFPS": 30},
    )
    return None


if __name__ == "__main__":
    SAFE_DIR = "Dataset//evaluation_slices//safe"
    FIRE_DIR = "Dataset//evaluation_slices//fire_present_set//images"
    manifest = build_dataset_manifest(SAFE_DIR, FIRE_DIR)
        
    if len(manifest) > 0:
        runExperiment()
        Logger.flush()