import os
import time
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from statsmodels.stats.contingency_tables import mcnemar

from Service.ObjectDetector.yolo import YOLOModel
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


# =====================================================================
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
    runThruputPerformance()