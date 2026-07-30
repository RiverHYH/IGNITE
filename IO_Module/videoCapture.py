import queue
import threading
import av
from IO_Module.logger import Logger

# Mute FFmpeg internal warnings globally
av.logging.set_level(av.logging.ERROR)


class CameraStream:
    def __init__(
        self, 
        device_name: str, 
        width=None, 
        height=None, 
        fps=None, 
        buffer_size: str = "100M",
        max_queue_size: int = 1
    ):
        """
        Threaded Camera Streaming Module
        Args:
            device_name: Name of the camera device. Defined by Windows Device Registry
            width: Optional video width
            height: Optional video height
            fps: Optional frames per second
            buffer_size: Video buffer size
            max_queue_size: Maximum size of the frame queue
        """
        self.device_name = device_name
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        # Bounded Queue: maxsize=1 ensures the consumer ALWAYS processes the freshest frame
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.worker_thread = None
        self.container = None

    def _worker(self):
        """Producer Thread: Reads DirectShow stream continuously to prevent buffer accumulation."""
        options = {
            "rtbufsize": self.buffer_size,
            "vcodec": "mjpeg"
        }
        if self.width and self.height:
            options["video_size"] = f"{self.width}x{self.height}"
        if self.fps:
            options["framerate"] = str(self.fps)

        try:
            self.container = av.open(
                f"video={self.device_name}",
                format="dshow",
                options=options
            )
            stream = next(s for s in self.container.streams if s.type == "video")

            for packet in self.container.demux(stream):
                if not self.running:
                    break

                for frame in packet.decode():
                    img_array = frame.to_ndarray(format="rgb24")

                    # If queue is full (inference thread is busy), drop the stale frame
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.frame_queue.put(img_array)

        except Exception as e:
            Logger.error(f"Error in camera reader thread: {e}")
        finally:
            if self.container:
                self.container.close()

    def start(self):
        """Spawns the background ingestion worker thread."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        Logger.info(f"Threaded ingestion started for {self.device_name}.")

    def frames(self):
        """Consumer Endpoint: Yields latest decoded frame to the processing loop."""
        while self.running:
            try:
                # Wait up to 1 second for next frame from background thread
                frame = self.frame_queue.get(timeout=1.0)
                yield frame
            except queue.Empty:
                if not self.running:
                    break

    def stop(self):
        """Safely terminates background thread and cleans up hardware resource."""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        Logger.info(f"Threaded ingestion stopped for {self.device_name}.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()