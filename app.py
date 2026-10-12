# IGINIE Application Starts Here
from typing import List
from utils.logger import Logger
import cv2
from utils.cameraList import list_cameras_windows
from utils.videoCapture import CameraStream
from utils.boundingBoxDrawer import draw_predictions
from Service.ObjectDetector.yolo import YOLOModel
from Service.Semantic.continuousLatentInferencer import ContinuousLatentInferencer
from Service.Predicate.affordanceEmbedder import GeometricPredicateExtractor, generate_semantic_prompt
from pathlib import Path
import numpy as np
import time

# ---------------------------------------------------------------------------
# E1 additions: render decoupling + CUDA warm-up + frame-level metrics
# (productisation improvements, P1/P2).
# All of these are "new" dependencies; no existing module's external
# semantics are modified (C-02).
# ---------------------------------------------------------------------------
from config import E1Config
from utils.renderPipeline import FrameSnapshot, make_default_pipeline
from utils.warmup import ReadinessState, warmup_if_enabled
from utils.metrics import FrameRecord, MetricsCollector, vram_snapshot

# Note: in the baseline, frame_counter / SKIP_FRAMES are module-level globals;
# frame_counter was never incremented (always 0), and activate(skip_frames=)
# assigned to a local variable, so the parameter had no effect.
# E1 keeps the same names for compatibility, but the downsampling logic now
# uses an explicit per-instance counter.
frame_counter = 0
SKIP_FRAMES = 15


class IGNITE:
    def __init__(self, config=None, **config_overrides):
        """
        Interface of 'Integrated Geospatial Navigation and Inference for Thermal Events'.

        Args:
            config: An E1Config instance / a configuration dictionary / a YAML
                    path / None (when None, environment variables are read).
                    Both E1 switches default to off, so behaviour matches the
                    baseline when no config is supplied.
            **config_overrides: Of the form render_enabled=True, warmup_enabled=True.
        """
        Logger.info("IGNITE Preparing...")

        # ---- E1: resolve and validate the configuration ----
        self.__cfg = E1Config.resolve(config, **config_overrides).validate()
        self.readiness = ReadinessState()

        self.__obj_model = YOLOModel("Service//ObjectDetector//obj1.pt")
        Logger.info(f"Object Detector Loaded onto {self.__obj_model.device}")
        self.__fire_model = YOLOModel("Service//ObjectDetector//fire.pt")
        Logger.info(f"Fire Detector Loaded onto {self.__fire_model.device}")
        self.__latent_inferencer = ContinuousLatentInferencer()
        self.__affordance_extractor:GeometricPredicateExtractor = GeometricPredicateExtractor()

        # ---- E1: render pipeline + metrics collector (enabled lazily by switch) ----
        self.__pipeline = make_default_pipeline(self.__cfg.render, draw_fn=draw_predictions)
        self.__metrics = MetricsCollector(
            enabled=self.__cfg.metrics.enabled,
            warmup_frames=self.__cfg.metrics.warmup_frames,
            budget_ms=self.__cfg.metrics.latency_budget_ms,
            target_p95_ms=self.__cfg.metrics.latency_target_p95_ms,
            first_frame_factor=self.__cfg.metrics.first_frame_factor,
        )
        self.__warmup_report = None

        # ---- E1: CUDA warm-up (FR-04/05), absorbing the one-off cold-start cost up front ----
        self._warmup()

    # ======================================================================
    # E1 addition: warm-up orchestration
    # ======================================================================
    def _warmup(self):
        """
        Run the CUDA warm-up and set readiness.

        The dummy forward pass covers both the detector and the embedder (R3);
        the service state is only entered once warm-up completes.
        When warm-up is disabled, an empty report with ready=True is returned,
        so the calling logic needs no branch.
        """
        report = warmup_if_enabled(
            self.__cfg.warmup,
            detector_probe=self.__probe_detector,
            embedder_probe=self.__probe_embedder,
            logger=Logger,
        )
        self.readiness.update(report)
        self.__warmup_report = report
        if report.ready:
            Logger.info(f"IGNITE readiness=READY ({report.summary()})")
        else:
            Logger.error(f"IGNITE readiness=NOT-READY :: {report.error}")

    def __probe_detector(self, frame):
        """
        Detection dummy forward pass, covering both the fire and obj models
        plus the downstream geometric layer.

        Note: forward pass only; no decision side effects are produced
        (G4 zero precision loss).
        """
        kwargs = {"half": self.__cfg.warmup.half, "imgsz": self.__cfg.warmup.imgsz}
        for _ in range(max(1, self.__cfg.warmup.warm_iters)):
            self.__fire_model.predict(frame, conf=0.25, **kwargs)
            self.__obj_model.predict(frame, conf=0.50, **kwargs)
        return True

    def __probe_embedder(self, prompt: str):
        """
        Embedder dummy forward pass (R3: the detector alone must not be warmed up).

        This walks the full BGE encoding + matrix retrieval path, warming up
        both the kernels and the VRAM along the vector retrieval path.
        """
        self.__latent_inferencer.get_top_k(prompt, k=1)
        return True

    # ======================================================================
    # Existing property accessors (unchanged, preserving C-02)
    # ======================================================================
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

    # ======================================================================
    # E1 addition: read-only observation interface (matches the D-6 readiness interface doc)
    # ======================================================================
    @property
    def config(self) -> E1Config:
        """The currently active E1 configuration (read-only use)."""
        return self.__cfg

    @property
    def warmup_report(self):
        """Warm-up report; None when warm-up has not run."""
        return self.__warmup_report

    def metrics_summary(self):
        """Return the current metrics summary object (including the AC-01/02/03 criteria)."""
        return self.__metrics.summary()

    def probe(self):
        """
        readiness probe: read directly by HTTP / logs / load-test scripts.
        Typical return: {"ready": bool, "serving": bool, "warmup": {...}}
        """
        return self.readiness.probe()

    def render_stats(self):
        """Render-side statistics: per-call draw time, dropped frames, peak queue depth."""
        return self.__pipeline.stats()

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
    
    def activate(self,streaming=False,img_path:str="",result_path:str="",skip_frames=15,**kwargs):
        """
        Initiate IGNITE Application
        Args:
            streaming (bool,optional): Whether to activate in streaming mode. Default to False
            img_path (str,optional): Path to image file if not streaming. Default to Null
            result_path (str,optional): Path to save results. Default to the same directory of input image
            skip_frames (int,optional): Skip frames for faster inference. Default to 15
            **kwargs: Additional arguments to pass to the YOLO detectors.
        """
        if streaming:
            return self._activate_streaming(skip_frames=skip_frames, **kwargs)
        else:
            return self._activate_static(img_path=img_path, result_path=result_path)

    # ----------------------------------------------------------------------
    # E1 addition: streaming main loop (render decoupling + warm-up + metrics)
    # ----------------------------------------------------------------------
    def _activate_streaming(self, skip_frames=15, **kwargs):
        """
        Streaming-mode main loop.

        Three key differences from the baseline:
        1. **Render decoupling**: draw and cv2 presentation move to a separate
           render thread; the inference thread submits non-blockingly via
           `RenderPipeline.submit()` and **no longer waits for draw to return** (FR-01).
        2. **Downsampling fix**: the baseline used the module-level
           `frame_counter % SKIP_FRAMES`, but `frame_counter` was always 0 and
           `skip_frames` was assigned to a local variable, so the parameter had
           no effect. Here an explicit per-instance counter is used, so
           `skip_frames` takes effect properly.
        3. **Metrics isolation**: `parse_ms` only measures the inference critical
           path; draw time is recorded separately by the render thread, and no
           such entry is produced at all when the overlay is disabled (FR-08).
        """
        skip = max(1, int(skip_frames))
        device_name = self._camera_select()
        Logger.info(f"IGNITE Activated,receiving from {device_name}")
        Logger.debug("CRITICAL: Click the VIDEO WINDOW before pressing 'Esc' to quit.")

        window_name = self.__cfg.render.name
        uses_own_window = self.__cfg.render.enabled

        # When decoupling is off, still open the window and present inside the main
        # loop as in the baseline (so the control group behaves identically)
        if not uses_own_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 540)

        self.__pipeline.start()
        self.__metrics.start()

        local_idx = 0          # Explicit per-instance frame counter (replaces the broken global frame_counter)
        first_steady_seen = False

        try:
            with CameraStream(device_name=device_name) as stream:
                for frame in stream.frames():
                    local_idx += 1
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Only time frames that take part in inference: frames skipped by downsampling are not inference frames
                    if local_idx % skip != 0:
                        # FR-02: frames without inference are not submitted for rendering, so the draw rate is not inflated by the stream frame rate
                        continue

                    t0 = time.perf_counter()
                    outcome = self._parse(bgr, conf=(0.25, 0.65), **kwargs)
                    parse_ms = (time.perf_counter() - t0) * 1000.0

                    result = outcome[0] if outcome else []
                    decision = outcome[1] if outcome else None

                    # FR-07: the first N inference frames are marked as warm-up frames and excluded from steady-state statistics
                    seq = self.__metrics_summary_len() + 1
                    is_warmup = seq <= self.__cfg.metrics.warmup_frames

                    # ---- Critical-path fork: hand draw off to a separate thread; the inference thread returns immediately ----
                    if uses_own_window:
                        # Render frame-rate limiting is decided internally by RenderPipeline via rate_limit (FR-02)
                        self.__pipeline.submit(
                            FrameSnapshot(
                                seq=seq,
                                timestamp=t0,
                                frame=bgr,
                                predictions=result,
                                decision=decision,
                                is_warmup=is_warmup,
                            )
                        )
                    else:
                        # Baseline path (decoupling off): still draw and present synchronously inside the main loop
                        if result and self.__cfg.render.draw_enabled:
                            t_draw = time.perf_counter()
                            draw_predictions(bgr, result, in_place=True)
                            self.__draw_ms = (time.perf_counter() - t_draw) * 1000.0
                        cv2.imshow(window_name, bgr)

                    if decision:
                        print(decision)

                    # ---- E1: metrics persistence ----
                    alloc, reserved = vram_snapshot()
                    self.__metrics.record(
                        FrameRecord(
                            seq=seq,
                            timestamp=t0,
                            parse_ms=parse_ms,
                            is_warmup=is_warmup,
                            # FR-08: a real draw time can only occur when decoupling is enabled
                            draw_ms=(
                                self.__pipeline.stats().get("draw_ms_last")
                                if uses_own_window and self.__cfg.render.draw_enabled
                                else None
                            ),
                            queue_depth=int(self.__pipeline.stats().get("queue_depth_now", 0))
                            if uses_own_window
                            else None,
                            vram_alloc_mb=alloc,
                            vram_reserved_mb=reserved,
                        )
                    )

                    if not first_steady_seen and not is_warmup:
                        first_steady_seen = True
                        self.readiness.mark_serving()

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        Logger.info("Sytem Shutting as Intended")
                        break
        finally:
            self.__pipeline.stop()
            # When decoupling is off the window is owned by the main thread; when it is on
            # the render thread destroys it itself
            if not uses_own_window:
                cv2.destroyAllWindows()

            # AC-07: on exit, backfill the render-side statistics and emit the conclusion
            stats = self.__pipeline.stats()
            self.__metrics.note_render(
                dropped=int(stats.get("dropped", 0)),
                max_queue_depth=float(stats.get("max_queue_depth", 0)),
            )
            summary = self.__metrics.summary()
            Logger.report(f"IGNITE E1 run summary\n{summary.report()}")

    def __metrics_summary_len(self) -> int:
        """Number of frames recorded so far (used to generate seq, without triggering statistics computation)."""
        return len(self.__metrics.records)

    # ----------------------------------------------------------------------
    # Static mode (single-image inference): no behavioural change, only warm-up and readiness marking added
    # ----------------------------------------------------------------------
    def _activate_static(self, img_path: str = "", result_path: str = ""):
        """
        Static (single-image) mode. The logic matches the baseline; only a
        readiness mark is added. NFR-04 requires this mode's per-frame
        (per-image) output to match the baseline, so no render decoupling
        change is made here.
        """
        if img_path == "":
            Logger.error("IGNITE Activation Failed,No Image Path Provided")
            return None
        path = Path(img_path)
        Logger.info(f"IGNITE Activated,receiving from {path.name}")
        outcome = self._parse(img_path)
        if outcome:
            result = outcome[0]
            decision = outcome[1]
            result_path = str(path.parent)+f"{path.stem}_result.jpg" if result_path == "" else result_path
            draw_predictions(cv2.imread(img_path), result, output_path=result_path)
            Logger.report(f"IGNITE inference report\nMatched Record: {decision[0]}\n Final_Decision: {decision[1]}\n Confidence: {decision[2]}")
        else:
            Logger.report(f"IGNITE inference report\n[ERROR]Failure to Generate Full Report,Please Trace Log")
        self.readiness.mark_serving()

    # ======================================================================
    # Inference core: kept line-by-line identical to the baseline logic (the precondition for NFR-04 zero accuracy regression)
    # ======================================================================
    def _parse(self,frame,conf=(0.25,0.50),**kwargs):
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
                predicate,*_ = self.__affordance_extractor.get_predicate(fbox,obox)
                        
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
    # E1: read configuration from environment variables (IGNITE_RENDER_ENABLED / IGNITE_WARMUP_ENABLED etc.);
    # when unset both switches are off — i.e. default behaviour matches the baseline (C-02 / NFR-04 zero regression).
    cfg = E1Config.from_env().validate()
    test = IGNITE(cfg)
    # kwargs are passed as a single dictionary to ensure readability.
    # Note: the baseline here was `activate(streaming=True, kwargs={...})`, which
    # double-wrapped kwargs; E1 spreads them instead, so the argument semantics match
    # the activate() docstring (without changing YOLO's default behaviour).
    test.activate(streaming=True, half=True, imgsz=640, vid_stride=True)
