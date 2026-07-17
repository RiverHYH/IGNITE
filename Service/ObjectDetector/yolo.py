from ultralytics import YOLO
from IO_Module.logger import Logger
from typing import List, Union, Dict, Any

class YOLOModel:
    def __init__(self, version: str = "yolo26n.pt",fine_tuning:bool=False,is_cuda:bool=True,**kwargs):
        self.__ver = version
        self.__fine_tuning=fine_tuning
        self.__yolo = YOLO(self.__ver,**kwargs)
        if is_cuda:
            self.__yolo.to('cuda')
            
        
        
    @property
    def version(self):
        return self.__ver
        
    @version.setter
    def version(self, version: str):
        """Re-instantiates the core instance cleanly when weights change."""
        self.__ver = version
        self.__yolo = YOLO(self.__ver)
        
        
    @property
    def device(self):
        return self.__yolo.device
    
    @property
    def fine_tuned(self):
        return self.__fine_tuning
    
        
    def train(self, data, epochs: int = 10, **kwargs):
        Logger.hook_stdout()
        Logger.info(f"Model {self.version} is Training...")
        self.__yolo.train(data=data, epochs=epochs, **kwargs)
        self.__fine_tuning = True
        return Logger.info('Base Model Training Complete.')
    
    def predict(self, img, classes: List[Union[int, str]] = list(), use_tensors: bool = True, topk:int=4,**kwargs) -> List[Dict[str, Any]]:
        """
        Generates standardized, filtered bounding boxes for any customizable list of classes.
        
        Args:
            img: Input image matrix, canvas frame, or file path string.
            classes: Optional list of class strings or integer IDs to filter predictions.
            use_tensors: If True, leaves bboxes as GPU/CPU tensors for fast downstream geometric math.
            topk: The number of topk predictions to return.
        """
        resolved_ids = None
        
        # Resolve mixed string/integer target lists using this instance's specific metadata taxonomy
        if len(classes)!=0:
            name_to_id = {name: idx for idx, name in self.__yolo.names.items()}
            resolved_ids = []
            for c in classes:
                if isinstance(c, str):
                    if c in name_to_id:
                        resolved_ids.append(name_to_id[c])
                    else:
                        Logger.info(f"Warning: Class '{c}' not found in model registry ({self.__ver}).")
                elif isinstance(c, int):
                    resolved_ids.append(c)

        # Execute single instance forward pass
        results = self.__yolo.predict(img, classes=resolved_ids, verbose=False, **kwargs)
        
        parsed_boxes = []
        if not results or len(results) == 0 or results[0].boxes is None:
            return parsed_boxes
            
        boxes = results[0].boxes
        cls_tensor = boxes.cls.int()
        xyxy_tensor = boxes.xyxy if use_tensors else boxes.xyxy.cpu().tolist()
        conf_tensor = boxes.conf if use_tensors else boxes.conf.cpu().tolist()

        # Build structural, task-agnostic payload output
        for i in range(len(cls_tensor)):
            class_id = int(cls_tensor[i])
            parsed_boxes.append({
                "class_id": class_id,
                "class_name": self.__yolo.names[class_id],
                "bbox": xyxy_tensor[i],
                "confidence": conf_tensor[i]
            })
        topk=min(topk,len(parsed_boxes))    
        return parsed_boxes[:topk]
    
    
    
    
    
    
if __name__=='__main__':
    import torch
    test_msg="Operator is conducting a Unit Test Run."
    test1=YOLOModel()
    test_msg+=f' The Unit Test Run is ran on {test1.version}'
    if torch.cuda.is_available():
        test_msg+=' The GPU is available, therefore the test is ran on GPU'
    else:
        test_msg+=' The GPU is not available, therefore the test run is ran on CPU'
    Logger.info(test_msg)
    
    test1.train(data='Dataset//HFD//data.yaml',
                epochs=200,
                imgsz=640,
                batch=6,     
                device=0,
                workers=6)