# camera_probe.py
import subprocess

def probe_dshow_options(device_name: str):
    cmd = [
        "ffmpeg",
        "-f", "dshow",
        "-list_options", "true",
        "-i", f"video={device_name}"
    ]
    result = subprocess.run(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    modes = []
    for line in result.stderr.splitlines():
        line = line.strip()
        if "pixel_format" in line and "fps" in line:
            modes.append(line)
    return modes

if __name__ == "__main__":
    from cameraList import list_cameras_windows
    cams = list_cameras_windows()
    for i, name in enumerate(cams):
        print(f"[{i}]: {name}")
    idx = int(input("Select device index to probe: "))
    for m in probe_dshow_options(cams[idx]):
        print(m)
