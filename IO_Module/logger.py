import os
import sys
import threading
import queue
from datetime import datetime,timezone
import traceback


class Logger:
    
    LOG_DIR = "logs"
    _log_queue = queue.Queue()
    _worker_started = False

    @classmethod
    def _start_worker(cls):
        
        if cls._worker_started:
            return

        cls._worker_started = True

        def worker():
            while True:
                log_path, entry = cls._log_queue.get()
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(entry)
                except Exception:
                    pass
                cls._log_queue.task_done()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    @classmethod
    def _get_log_path(cls):
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        utc_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(cls.LOG_DIR, f"{utc_date}.log")


    @classmethod
    def _write(cls, level: str, message: str):
        cls._start_worker()

        log_path = cls._get_log_path()
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}\n"

        cls._log_queue.put((log_path, entry))
        return log_path

    @classmethod
    def info(cls, message: str):
        return cls._write("INFO", message)

    @classmethod
    def debug(cls, message: str):
        return cls._write("DEBUG", message)

    @classmethod
    def error(cls, message: str):
        return cls._write("ERROR", message)
    @classmethod
    def hook_interruption(cls):
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                cls._write("ERROR", "KeyboardInterrupt detected — process aborted by user")
            else:
                # Format the full stack trace into a clean string
                fmt_traceback = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                cls._write("CRITICAL", f"Unhandled Exception:\n{fmt_traceback}")

            # Ensure the background queue writes the crash data to disk before exiting!
            cls._log_queue.join() 
            
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        
        sys.excepthook = handle_exception
    @classmethod
    def hook_stdout(cls):
        cls._start_worker()

        class StreamHook:
            def write(self, text):
                # 1. Check for carriage returns BEFORE stripping whitespace
                if "\r" in text:
                    return
                
                cleaned_text = text.strip()
                if not cleaned_text:
                    return

                # 2. Catch progress bars that might bypass the \r check
                # (tqdm bars often contain '|' and '%')
                if "%" in cleaned_text and ("|" in cleaned_text or "it/s" in cleaned_text or "fps" in cleaned_text):
                    return

                cls._write("YOLO", cleaned_text)

            def flush(self):
                pass
        sys.stdout = StreamHook()
        sys.stderr = StreamHook()

