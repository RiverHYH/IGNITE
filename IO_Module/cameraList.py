# camera_list.py
import subprocess
import re

def list_cameras_windows():
    cmd = ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy']
    result = subprocess.run(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )

    lines = result.stderr.splitlines()
    cameras = []

    pattern = re.compile(r'\[dshow.*\]\s+"(.+)"\s+\((video)\)', re.IGNORECASE)

    for line in lines:
        m = pattern.search(line)
        if m:
            cameras.append(m.group(1))

    return cameras

if __name__ == "__main__":
    for i, name in enumerate(list_cameras_windows()):
        print(f"[{i}]: {name}")
