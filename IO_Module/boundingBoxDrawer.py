import cv2
import torch
import time
from typing import List, Dict, Any, Union

def draw_predictions(
    img_path_or_array: Union[str, Any], 
    predictions: List[Dict[str, Any]], 
    output_path=None,
    in_place=False
) -> Any:
    """
    Draws bounding boxes, class labels, and confidence scores onto an image frame.
    Optimized for high-FPS live streams via optional in-place array manipulation.
    """
    # 1. Handle image ingestion securely
    if isinstance(img_path_or_array, str):
        image = cv2.imread(img_path_or_array)
        if image is None:
            raise FileNotFoundError(f"Could not read image frame from path: {img_path_or_array}")
    else:
        # For live streaming, use in_place=True to modify the frame matrix directly and maximize FPS
        image = img_path_or_array if in_place else img_path_or_array.copy()

    # 2. Iterate through parsed target objects
    print(predictions)
    for pred in predictions:
        bbox = pred["bbox"]
        class_name = pred["class_name"]
        confidence = pred["confidence"]
        
        # Safe unboxing conversion for PyTorch Tensors vs primitive Python sequences
        if isinstance(bbox, torch.Tensor):
            xmin, ymin, xmax, ymax = bbox.cpu().numpy().astype(int)
        else:
            xmin, ymin, xmax, ymax = [int(coord) for coord in bbox]
            
        conf_score = float(confidence.item()) if isinstance(confidence, torch.Tensor) else float(confidence)

        # Assign high-contrast alert coloring (BGR system)
        box_color = (0, 0, 255) if class_name.lower() in ["fire", "flame", "smoke"] else (0, 255, 0)

        # Draw the primary spatial bounding box rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), box_color, thickness=2)

        # Build clean string tag allocation payload
        label_text = f"{class_name} {conf_score:.2f}"

        # Construct a background banner behind text to preserve scannability
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness=1)
        
        # Prevent drawing banner outside the upper bounds of the image screen matrix
        y_banner_min = max(0, ymin - text_height - 6)

        # Draw solid color banner anchor
        cv2.rectangle(
            image, 
            (xmin, y_banner_min), 
            (xmin + text_width, ymin), 
            box_color, 
            thickness=cv2.FILLED
        )

        # Render white text over the solid banner block
        cv2.putText(
            image, 
            label_text, 
            (xmin, ymin - 4 if ymin - text_height - 6 > 0 else ymin + text_height), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            thickness=1, 
            lineType=cv2.LINE_AA
        )

    # 3. Handle data flushing output configurations
    if output_path:
        cv2.imwrite(output_path, image)

    return image