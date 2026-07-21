import os
import cv2
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from Service.ObjectDetector.yolo import YOLOModel
from app import IGNITE

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SAFE_DIR = "Dataset/evaluation_slices/safe"
FIRE_DIR = "Dataset/evaluation_slices/fire_present_set"

# Your model instances (passed externally)
YOLO_instance = YOLOModel("Service//ObjectDetector//fire.pt")
IGNITE_instance = IGNITE()

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_yolo_pred(img, YOLO_instance):
    """Return 1 if YOLO detects fire/smoke, else 0."""
    preds = YOLO_instance.predict(img)
    return 0 if preds is None or len(preds) == 0 else 1

def get_ignite_pred(img, IGNITE_instance):
    """Return IGNITE prediction (0 or 1)."""
    value=IGNITE_instance._parse(img)
    if value:
        return value[1][1][0]
    return 0

def load_image(path):
    """Load image in BGR format."""
    return cv2.imread(path)

# ---------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------

# McNemar contingency table:
#                IGNITE
#              Correct   Wrong
# YOLO Correct    a        b
# YOLO Wrong      c        d
#
# We only need b and c for McNemar.

a = b = c = d = 0

def evaluate_directory(dir_path, y_oracle, YOLO_instance, IGNITE_instance):
    global a, b, c, d

    for fname in os.listdir(dir_path):
        if not fname.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(dir_path, fname)
        img = load_image(img_path)

        pred_yolo = get_yolo_pred(img, YOLO_instance)
        pred_ignite = get_ignite_pred(img, IGNITE_instance)

        yolo_correct = int(pred_yolo == y_oracle)
        ignite_correct = int(pred_ignite == y_oracle)

        # Fill contingency table
        if yolo_correct == 1 and ignite_correct == 1:
            a += 1
        elif yolo_correct == 1 and ignite_correct == 0:
            b += 1
        elif yolo_correct == 0 and ignite_correct == 1:
            c += 1
        else:
            d += 1

# Evaluate both sets
evaluate_directory(SAFE_DIR, 0, YOLO_instance, IGNITE_instance)
evaluate_directory(FIRE_DIR, 1, YOLO_instance, IGNITE_instance)

# ---------------------------------------------------------
# McNemar Test
# ---------------------------------------------------------

table = [[a, b],
         [c, d]]

# Calculate total discordant pairs (disagreements)
discordant_count = b + c

print("--------------------------------------------------")
print("McNemar Contingency Table:")
print(f"a (both correct):               {a}")
print(f"b (YOLO correct, IGNITE wrong): {b}")
print(f"c (YOLO wrong, IGNITE correct): {c}")
print(f"d (both wrong):                 {d}")
print(f"Total Discordant Pairs (b + c): {discordant_count}")
print("--------------------------------------------------")

# Dynamically choose the correct test based on sample size
if discordant_count < 25:
    print("Notice: Low discordant count (< 25). Running Exact Binomial Test.")
    result = mcnemar(table, exact=True)
else:
    print("Notice: Sufficient discordant count (>= 25). Running Asymptotic Test with Yates's Correction.")
    result = mcnemar(table, exact=False, correction=True)

print("McNemar Test Result:")
print(f"Statistic: {result.statistic}")
print(f"P-value:   {result.pvalue}")
print("--------------------------------------------------")

if result.pvalue < 0.05:
    print("Significant difference detected between YOLO and IGNITE.")
else:
    print("No significant difference detected between YOLO and IGNITE.")
