import os
import sys
import threading
import time
from datetime import datetime, timezone
import traceback


class Logger:
    
    LOG_DIR = "logs"
    
    # 5-second aggregation buffer states
    _buffer = {}  # Schema: {(level, message): count}
    _lock = threading.Lock()
    _flush_interval = 5.0
    _worker_started = False

    @classmethod
    def _start_worker(cls):
        if cls._worker_started:
            return

        cls._worker_started = True

        def worker():
            while True:
                # Sleep for the designated window block before flushing cached lines
                time.sleep(cls._flush_interval)
                cls.flush()

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

        # Deduplicate and count occurrences across the current 5-second window frame
        with cls._lock:
            key = (level, message)
            if key not in cls._buffer:
                cls._buffer[key] = 0
            cls._buffer[key] += 1
            
        return cls._get_log_path()

    @classmethod
    def flush(cls):
        """
        Gathers all compiled logs from the memory buffer cache, compresses 
        repetitive lines, and commits them cleanly to physical storage.
        """
        with cls._lock:
            if not cls._buffer:
                return
            # Deep-copy and release the memory buffer immediately to prevent loop blockades
            active_snapshots = cls._buffer
            cls._buffer = {}

        log_path = cls._get_log_path()
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        
        entries = []
        for (level, message), count in active_snapshots.items():
            if count > 1:
                # Compresses repetitive entries using the requested format structure
                entry = f"[{timestamp}] [{level}] {message}*{count}\n"
            else:
                entry = f"[{timestamp}] [{level}] {message}\n"
            entries.append(entry)

        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.writelines(entries)
        except Exception:
            pass

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
    def critical(cls, message:str):
        return cls._write("CRITICAL", message)
    
    @classmethod
    def report(cls, message:str):
        return cls._write("REPORT", message)

    @classmethod
    def hook_interruption(cls):
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                cls._write("ERROR", "KeyboardInterrupt detected — process aborted by user")
            else:
                fmt_traceback = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                cls.critical(f"Unhandled Exception:\n{fmt_traceback}")

            # CRITICAL: Force an immediate flush of the buffer so crash stack traces 
            # are recorded instantly before the main Python application finishes exiting!
            cls.flush() 
            
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        
        sys.excepthook = handle_exception

    @classmethod
    def hook_stdout(cls):
        cls._start_worker()

        class StreamHook:
            def write(self, text):
                if "\r" in text:
                    return
                
                cleaned_text = text.strip()
                if not cleaned_text:
                    return

                if "%" in cleaned_text and ("|" in cleaned_text or "it/s" in cleaned_text or "fps" in cleaned_text):
                    return

                cls._write("YOLO", cleaned_text)

            def flush(self):
                pass
                
        sys.stdout = StreamHook()
        sys.stderr = StreamHook()