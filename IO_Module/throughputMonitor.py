import os
import time
import ctypes
import psutil
import numpy as np
import matplotlib.pyplot as plt

try:
    import pynvml
except ImportError:
    pynvml = None


class PerformanceMonitor:
    def __init__(self, gpu_index=0):
        self.process = psutil.Process(os.getpid())

        # CPU metrics
        self.io_lat = []
        self.inf_lat = []
        self.draw_lat = []
        self.loop_lat = []
        self.mem_usage = []
        self.fps = []

        # GPU metrics
        self.gpu_mem_usage = []
        self.gpu_util = []
        self.gpu_available = False
        self.gpu_handle = None

        # Initialization
        self._init_nvml(gpu_index)

        # Baseline memory for AUC & total system RAM
        self.baseline_mem = self.process.memory_info().rss / 1024 / 1024
        self.total_system_mem = psutil.virtual_memory().total / 1024 / 1024

    # ---------------------------------------------------------
    # NVML Initialization
    # ---------------------------------------------------------
    def _init_nvml(self, gpu_index):
        if pynvml is None:
            print("[WARN] pynvml not installed. GPU monitoring disabled.")
            return

        # 1. Standard init
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.gpu_available = True
            print(f"[INFO] NVML initialized standardly for GPU {gpu_index}")
            return
        except Exception:
            pass

        # 2. Manual DLL injection fallback
        possible_paths = [
            r"C:\Windows\System32\nvml.dll",
            r"C:\Windows\SysWOW64\nvml.dll",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll",
            r"C:\Program Files\NVIDIA Corporation\nvml.dll",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path)
                    pynvml.nvmlLib = ctypes.CDLL(path)
                    pynvml.nvmlInit()
                    self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                    self.gpu_available = True
                    print(f"[INFO] NVML initialized via manual injection: {path}")
                    return
                except Exception as e:
                    print(f"[WARN] Failed NVML injection via {path}: {e}")

        print("[WARN] NVML initialization failed. GPU monitoring disabled.")
        self.gpu_available = False

    # ---------------------------------------------------------
    # Recording functions
    # ---------------------------------------------------------
    def record_latency(self, io, inf, draw, loop):
        self.io_lat.append(io)
        self.inf_lat.append(inf)
        self.draw_lat.append(draw)
        self.loop_lat.append(loop)

    def record_memory(self):
        mem = self.process.memory_info().rss / 1024 / 1024
        self.mem_usage.append(mem)
        return mem

    def record_gpu(self):
        if not self.gpu_available:
            self.gpu_mem_usage.append(0)
            self.gpu_util.append(0)
            return 0, 0

        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)

            mem_mb = info.used / 1024 / 1024
            util_pct = util.gpu

            self.gpu_mem_usage.append(mem_mb)
            self.gpu_util.append(util_pct)
            return mem_mb, util_pct
        except Exception as e:
            print(f"[WARN] Error reading GPU metrics: {e}")
            self.gpu_mem_usage.append(0)
            self.gpu_util.append(0)
            return 0, 0

    def record_fps(self, loop_latency_ms):
        fps_val = 1000.0 / loop_latency_ms if loop_latency_ms > 0 else 0
        self.fps.append(fps_val)
        return fps_val

    # ---------------------------------------------------------
    # Leak & Performance Detection
    # ---------------------------------------------------------
    def detect_memory_leak(self, threshold_mb=50):
        if len(self.mem_usage) < 50:
            return False
        return (self.mem_usage[-1] - self.mem_usage[0]) > threshold_mb

    def detect_gpu_leak(self, threshold_mb=200):
        if len(self.gpu_mem_usage) < 50:
            return False
        return (self.gpu_mem_usage[-1] - self.gpu_mem_usage[0]) > threshold_mb

    def detect_frame_drop(self, loop_latency_ms, threshold_ms=33.33):
        return loop_latency_ms > threshold_ms

    # ---------------------------------------------------------
    # Statistics Helper
    # ---------------------------------------------------------
    def stats(self, arr):
        if len(arr) == 0:
            return {"mean": 0, "min": 0, "max": 0, "std": 0}
        return {
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
        }

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    def plot(self, save_path=None, kwargs=None):
        RAM_GB = kwargs.get("RAM", None) if kwargs else None
        VRAM_GB = kwargs.get("VRAM", None) if kwargs else None
        TargetFPS = kwargs.get("TargetFPS", None) if kwargs else None

        RAM_MB = RAM_GB * 1024 if RAM_GB else self.total_system_mem
        VRAM_MB = VRAM_GB * 1024 if VRAM_GB else None
        
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 8,
            'axes.labelsize': 8,
            'axes.titlesize': 9,
            'legend.fontsize': 7,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'lines.linewidth': 1.0,
        })
        total_plots = 7 if self.gpu_available else 5
        # 2. Set dimensions: 6.5 in width (full ACL text width), 7.5 in height (leaves room for caption)
        fig, axes = plt.subplots(total_plots, 1, figsize=(6.5, 7.5), sharex=True)
        idx = 0

        # 1. Loop latency + drop markers
        axes[idx].plot(self.loop_lat, color="blue")
        drops = [i for i, v in enumerate(self.loop_lat) if v > 33.33]
        if drops:
            axes[idx].scatter(
                drops,
                [self.loop_lat[i] for i in drops],
                color="red",
                label="Dropped Frame (>33.3ms)",
            )
        axes[idx].set_title("Total Loop Latency (ms)")
        axes[idx].set_ylabel("ms")
        axes[idx].legend(loc="upper right")
        idx += 1

        # 2. Component latencies
        axes[idx].plot(self.io_lat, label="IO", alpha=0.7)
        axes[idx].plot(self.inf_lat, label="Inference", alpha=0.7)
        axes[idx].plot(self.draw_lat, label="Draw", alpha=0.7)
        axes[idx].set_title("Component Latencies (ms)")
        axes[idx].set_ylabel("ms")
        axes[idx].legend(loc="upper right")
        idx += 1

        # 3. CPU Memory
        axes[idx].plot(self.mem_usage, color="orange", label="CPU Memory")
        if len(self.mem_usage) > 1:
            x = np.arange(len(self.mem_usage))
            m, b = np.polyfit(x, self.mem_usage, 1)
            axes[idx].plot(x, m * x + b, color="black", linestyle="--", label="Leak Trend")
        axes[idx].set_title("CPU Memory Usage (MB)")
        axes[idx].set_ylabel("MB")
        axes[idx].set_ylim(0, RAM_MB)
        axes[idx].legend(loc="upper right")
        idx += 1

        # 4 & 5. GPU Metrics (If GPU is available)
        if self.gpu_available:
            # GPU VRAM
            axes[idx].plot(self.gpu_mem_usage, color="red", label="GPU VRAM Usage")
            axes[idx].set_title("GPU Memory Usage (MB)")
            axes[idx].set_ylabel("MB")
            if VRAM_MB:
                axes[idx].set_ylim(0, VRAM_MB)
            axes[idx].legend(loc="upper right")
            idx += 1

            # GPU Utilization %
            axes[idx].plot(self.gpu_util, color="purple", label="GPU Utilization (%)")
            axes[idx].set_title("GPU Utilization (%)")
            axes[idx].set_ylabel("%")
            axes[idx].set_ylim(0, 100)
            axes[idx].legend(loc="upper right")
            idx += 1

        # 6. FPS
        axes[idx].plot(self.fps, color="green", label="FPS")
        if TargetFPS:
            axes[idx].axhline(
                TargetFPS, color="red", linestyle="--", label=f"Target FPS ({TargetFPS})"
            )
        axes[idx].set_title("FPS")
        axes[idx].set_ylabel("Frames/s")
        axes[idx].legend(loc="upper right")
        idx += 1

        # 7. Memory AUC
        total_mem = np.array(self.mem_usage) if len(self.mem_usage) > 0 else np.array([0])
        non_func_mem = np.full_like(total_mem, self.baseline_mem)

        axes[idx].plot(total_mem, color="blue", label="Total Memory")
        axes[idx].plot(non_func_mem, color="gray", label="Non-function Baseline")
        axes[idx].fill_between(
            np.arange(len(total_mem)),
            non_func_mem,
            total_mem,
            color="cyan",
            alpha=0.4,
            label="Function Memory (AUC)",
        )
        axes[idx].set_title("Memory AUC (Function vs Non-function)")
        axes[idx].set_ylabel("MB")
        axes[idx].set_ylim(0, RAM_MB)
        axes[idx].legend(loc="upper right")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            plt.tight_layout()

            # 3. Save as vector PDF (recommended for LaTeX) or high-DPI PNG
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[INFO] Plot saved to {save_path}")

        plt.show()