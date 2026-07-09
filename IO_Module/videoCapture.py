import av
from IO_Module.logger import Logger

class CameraStream:
    def __init__(self, device_name: str, width=None, height=None, fps=None):
        self.device_name = device_name
        self.width = width
        self.height = height
        self.fps = fps
        self.container = None
        self.stream = None

    def start(self):
        options = {}

        # Only set options if explicitly requested
        if self.width and self.height:
            options["video_size"] = f"{self.width}x{self.height}"
        if self.fps:
            options["framerate"] = str(self.fps)

        try:
            self.container = av.open(
                f"video={self.device_name}",
                format="dshow",
                options=options if options else None
            )
            # pick first video stream
            self.stream = next(s for s in self.container.streams if s.type == "video")
            Logger.info(f"Connected to {self.device_name}")
        except Exception as e:
            Logger.error(f"Failed to open device {self.device_name}: {e}")
            raise

    def frames(self):
        for packet in self.container.demux(self.stream):
            for frame in packet.decode():
                yield frame.to_ndarray(format="rgb24")

    def close(self):
        if self.container:
            self.container.close()
            self.container = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()



if __name__ == "__main__": 
    import cv2
    from cameraList import list_cameras_windows
    
    cams = list_cameras_windows()
    if not cams:
        print("No video devices found.")
        exit(1)
    print("Available Device Are:")
    for i, name in enumerate(cams):
        print(f"[{i}]: {name}")

    device_no = int(input("Select device index: "))
    device_name = cams[device_no]

    Logger.debug("CRITICAL: Click the VIDEO WINDOW before pressing 'Esc' to quit.")

    window_name = "Test Frame"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    # You can pass width/height/fps if you’ve probed them and know they’re valid
    with CameraStream(device_name=device_name) as stream:
        for frame in stream.frames():
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, bgr)
            key = cv2.waitKey(5) & 0xFF
            if key== 27:break

    cv2.destroyAllWindows()
