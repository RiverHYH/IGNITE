# IGINIE Application Starts Here
from typing import List
from IO_Module.logger import Logger
import cv2
from IO_Module.cameraList import list_cameras_windows
from IO_Module.videoCapture import CameraStream
from IO_Module.boundingBoxDrawer import draw_predictions
from Service.ObjectDetector.yolo import YOLOModel
from Service.Semantic.continuousLatentInferencer import ContinuousLatentInferencer
from Service.Predicate.affordanceEmbedder import GeometricPredicateExtractor,generate_semantic_prompt
from pathlib import Path
import numpy as np
frame_counter = 0
SKIP_FRAMES = 15 # Run AI inference every 10th frame


class IGNITE:
    def __init__(self):
        """
        Interface of 'Integrated Geospatial Navigation and Inference for Thermal Events'.
        """
        Logger.info("IGNITE Preparing...")
        self.__obj_model = YOLOModel("Service//ObjectDetector//obj1.pt")
        Logger.info(f"Object Detector Loaded onto {self.__obj_model.device}")
        self.__fire_model = YOLOModel("Service//ObjectDetector//fire.pt")
        Logger.info(f"Fire Detector Loaded onto {self.__fire_model.device}")
        self.__latent_inferencer = ContinuousLatentInferencer()
        self.__affordance_extractor = GeometricPredicateExtractor()
    
    @property
    def object_detector(self):
        """
        Get the running object detector version.
        """
        return self.__obj_model.version
    @object_detector.setter
    def object_detector(self, version: str):
        """
        Set the object detector version.
        """
        self.__obj_model.version = version
        
    @property
    def fire_detector(self):
        """
        Get the running fire detector version.
        """
        return self.__fire_model.version
    @fire_detector.setter
    def fire_detector(self, version: str):
        """
        Set the fire detector version.
        """
        self.__fire_model.version = version
        
    def _camera_select(self):
        """
        Private Method. Select camera device from available options.
        """
        
        cams = list_cameras_windows()
        if not cams:
            Logger.error("Application Terminated Due to No video devices found.")
            exit(1)
        print("Available Device Are:")
        for i, name in enumerate(cams):
            print(f"[{i}]: {name}")

        device_no = int(input("Select device index: "))
        device_name = cams[device_no]
        return device_name
    
    def activate(self,streaming=False,img_path:str="",result_path:str="",**kwargs):
        """
        Initiate IGNITE Application
        Args:
            streaming (bool,optional): Whether to activate in streaming mode. Default to False
            img_path (str,optional): Path to image file if not streaming. Default to Null
            result_path (str,optional): Path to save results. Default to the same directory of input image
            **kwargs: Additional arguments to pass to the YOLO detectors.
        """
        if streaming:
            device_name=self._camera_select()
            Logger.info(f"IGNITE Activated,receiving from {device_name}")
            Logger.debug("CRITICAL: Click the VIDEO WINDOW before pressing 'Esc' to quit.")
            window_name = "IGNITE"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 540)
            with CameraStream(device_name=device_name) as stream:
                for frame in stream.frames():
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # Only run the heavy AI pipeline periodically
                    if frame_counter % SKIP_FRAMES == 0:
                        outcome=self._parse(bgr,conf=(0.25,0.65),**kwargs)
                        if outcome:
                            result=outcome[0]
                            decision=outcome[1]
                            draw_predictions(bgr,result,in_place=True)
                            print(decision)
                    cv2.imshow(window_name, bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key== 27:
                        Logger.info("Sytem Shutting as Intended")
                        break
                    
            cv2.destroyAllWindows()
        else:
            if img_path=="":
                Logger.error("IGNITE Activation Failed,No Image Path Provided")
                return None
            path=Path(img_path)
            Logger.info(f"IGNITE Activated,receiving from {path.name}")
            outcome=self._parse(img_path)
            if outcome:
                result=outcome[0]
                decision=outcome[1]
                result_path=str(path.parent)+f"{path.stem}_result.jpg" if result_path=="" else result_path
                draw_predictions(cv2.imread(img_path),result,output_path=result_path)
                Logger.report(f"IGNITE Inference Report\nMatched Record: {decision[0]}\n Final_Decision: {decision[1]}\n Confidence: {decision[2]}")
            else:
                Logger.report(f"IGNITE Inference Report\n[ERROR]Failure to Generate Full Report,Please Trace Log")

                        
    def _parse(self,frame,conf=(0.25,0.65),**kwargs):
        """
        Private Method. Parse the frame and generate the final report.
        Args:
            frame (ndarray | str): Frame to be passed into the YOLO models. Input image matrix, canvas frame, or file path string.
            conf (tuple,optional): Confidence threshold for fire and object detection. Default to (0.25,0.65)
            **kwargs: Additional arguments to pass to the YOLO detectors.
        """
        
       # 1. Run ONLY the fire model first (Fast pass)
        fresult = self.__fire_model.predict(frame,conf=conf[0],**kwargs)
        
        # 2. Extract the actual detected boxes from the first results object
        result=fresult
        fire_detected = fresult[0]["class_id"]>-1 if fresult else False
        # PURPOSE:SHORT-CIRCUIT: If no fire is in the scene, drop the frame immediately
        if not fire_detected:
            Logger.debug("No fire localized in Perceptual Layer. Suppressing downstream computation.")
            return tuple()
        
        # PURPOSE:Short-CIRCUIT: Only look for environmental assets if a fire hazard is present
        oresult = self.__obj_model.predict(frame, conf=conf[1],**kwargs)
        obj_detected = oresult[0]['class_id'] > -1 if oresult else False
        result+=oresult
        if not obj_detected:
            Logger.debug("Fire present, but no context objects localized. Proceeding with caution.")
            # Handle or return accordingly based on your Stage 2 expectations
            # Pythonic check for empty lists
            if not fresult or not oresult:
            # If no fire or object is detected, there is no need to proceed
                Logger.debug("Empty Frame elements. Or Potential Failure of Target Capturing. Proceeding")
                decision=(["No Objects Detected"],[1],[0.0])#purpose: cautious alarm
                Logger.info(f"Matched Record: {decision[0]}\n Final_Decision: {decision[1]}\n Confidence: {decision[2]}")
                return result,decision

                # Cross-examine every detected fire element against every environment object
        for f_det in fresult:
            # Safe explicit dictionary key lookups
            fid = f_det["class_id"]
            fbox = f_det["bbox"]
            subject = "flame" if fid == 0 else "smoke"
                        
            for o_det in oresult:
                oname = o_det["class_name"]
                obox = o_det["bbox"]
                obj = oname
                            
                # Purpose: Stage 2 Geometric Inference
                predicate = self.__affordance_extractor.get_predicate(fbox, obox)
                        
                if predicate not in self.__affordance_extractor.predicates:
                    Logger.debug(f"Predicate {predicate} not in affordance extractor")
                    return tuple()
                            
                # Purpose: Stage 3 Symbolic Assembly
                triplet = generate_semantic_prompt(subject, predicate, obj)
                Logger.info(f"Triplet Identified: {triplet},parsing...")
                # Purpose: Stage 4 Continuous Latent Inference
                decision=self.__latent_inferencer.get_top_k(triplet,k=1)
                # Purpose: Stage 5 Final Decision
                Logger.info(f"Matched Record: {decision[0]}\n Final_Decision: {decision[1]}\n Confidence: {decision[2]}")
                return result,decision
                        
                    
                
if __name__=='__main__':
    test=IGNITE()
    #kwargs are passed as a single dictionary to ensure readability 
    test.activate(streaming=True,kwargs={"half":True,"imgsz":640,"vid_stride":True})
           
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
