import cv2
import torch
from typing import List, Dict, Any, Union

def draw_predictions(img_path_or_array: Union[str, Any], predictions: List[Dict[str, Any]], output_path=None) -> Any:
    """
    Draws bounding boxes, class labels, and confidence scores onto an image frame.
    
    Args:
        img_path_or_array: String path to the target image file, or an already loaded cv2 numpy matrix.
        predictions: The flat list of dictionaries returned by YOLOModel.predict().
        output_path: Optional file path string to save the annotated result image to disk.
    """
    # 1. Handle image ingestion securely
    if isinstance(img_path_or_array, str):
        image = cv2.imread(img_path_or_array)
        if image is None:
            raise FileNotFoundError(f"Could not read image frame from path: {img_path_or_array}")
    else:
        image = img_path_or_array.copy()  # Create a copy to prevent polluting your input frame memory

    # 2. Iterate through parsed target objects
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
        # Uses explicit red for dangerous anomalies, clean green for environmental structural items
        box_color = (0, 0, 255) if class_name.lower() in ["flame", "smoke"] else (0, 255, 0)

        # Draw the primary spatial bounding box rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), box_color, thickness=2)

        # Build clean string tag allocation payload
        label_text = f"{class_name} {conf_score:.2f}"

        # Construct a background banner behind text to preserve scannability across diverse contrast backgrounds
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness=1)
        
        # Draw solid color banner anchor
        cv2.rectangle(
            image, 
            (xmin, ymin - text_height - 6), 
            (xmin + text_width, ymin), 
            box_color, 
            thickness=cv2.FILLED
        )

        # Render white text over the solid banner block
        cv2.putText(
            image, 
            label_text, 
            (xmin, ymin - 4), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            thickness=1, 
            lineType=cv2.LINE_AA
        )

    # 3. Handle data flushing output configurations
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"[+] Annotated diagnostic image flushed to storage: {output_path}")

    return image