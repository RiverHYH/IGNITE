import os
import sys
import threading
import time
from datetime import datetime, timezone
import traceback


class Logger:
    
    LOG_DIR = "logs"# Set Default Directory
    REPORT_DIR = "reports" #Set Default Directory
    
    # 5-second aggregation buffer states
    _buffer = {}          # Operational logs schema: {(level, message): count}
    _report_buffer = []   # Reports schema: [(timestamp, report_dir, message)]
    
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
        """
        Returns the log file path for writing log messages.
        """
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        utc_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(cls.LOG_DIR, f"{utc_date}.log")

    @classmethod
    def _get_report_path(cls, report_dir=None):
        target_dir = report_dir if report_dir else cls.REPORT_DIR
        os.makedirs(target_dir, exist_ok=True)
        utc_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(target_dir, f"{utc_date}_report.log")

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
    def report(cls, message: str, report_dir = None):
        """
        Asynchronously buffers and writes evaluation reports to designated directory.
        Args:
            message (str): Report content.
            report_dir (str, optional): Target directory for the report. Defaults to None (uses default report directory).
        """
        cls._start_worker()
        target_dir = report_dir if report_dir else cls.REPORT_DIR
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")#Messages are reorded in Exact Time under UTC

        with cls._lock:
            cls._report_buffer.append((timestamp, target_dir, message))

        return cls._get_report_path(target_dir)

    @classmethod
    def flush(cls):
        """
        Gathers all compiled logs and reports from memory buffers and 
        commits them cleanly to physical storage files.
        """
        with cls._lock:
            if not cls._buffer and not cls._report_buffer:
                return
            
            # Deep-copy and release memory buffers immediately
            active_snapshots = cls._buffer
            active_reports = cls._report_buffer
            cls._buffer = {}
            cls._report_buffer = []

        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")

        # ---------------------------------------------------------
        # 1. Flush Operational Logs
        # ---------------------------------------------------------
        if active_snapshots:
            log_path = cls._get_log_path()
            entries = []
            for (level, message), count in active_snapshots.items():
                if count > 1:
                    entry = f"[{timestamp}] [{level}] {message}*{count}\n"
                else:
                    entry = f"[{timestamp}] [{level}] {message}\n"
                entries.append(entry)

            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.writelines(entries)
            except Exception:
                pass

        # ---------------------------------------------------------
        # 2. Flush Evaluation Reports (Grouped by target directory)
        # ---------------------------------------------------------
        if active_reports:
            reports_by_dir = {}
            for ts, r_dir, msg in active_reports:
                if r_dir not in reports_by_dir:
                    reports_by_dir[r_dir] = []
                
                report_entry = f"[{ts}] [REPORT]\n{msg}\n\n"
                reports_by_dir[r_dir].append(report_entry)

            for r_dir, entries in reports_by_dir.items():
                report_path = cls._get_report_path(r_dir)
                try:
                    with open(report_path, "a", encoding="utf-8") as f:
                        f.writelines(entries)
                except Exception:
                    pass

    @classmethod
    def info(cls, message: str):
        """
        Tag Log Message as INFO
        """
        return cls._write("INFO", message)

    @classmethod
    def debug(cls, message: str):
        """
        Tag Log Message as DEBUG
        """
        return cls._write("DEBUG", message)

    @classmethod
    def error(cls, message: str):
        """
        Tag Log Message as ERROR
        """
        return cls._write("ERROR", message)
    
    @classmethod
    def critical(cls, message: str):
        """
        Tag Log Message as CRITICAL
        """
        return cls._write("CRITICAL", message)

    @classmethod
    def hook_interruption(cls):
        """
        Hooks the KeyboardInterrupt exception to flush logs and reports on process exit.
        """
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                cls._write("ERROR", "KeyboardInterrupt detected — process aborted by user")
            else:
                fmt_traceback = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                cls.critical(f"Unhandled Exception:\n{fmt_traceback}")

            # Force an immediate flush of operational logs AND pending reports on process exit
            cls.flush() 
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        
        sys.excepthook = handle_exception

    @classmethod
    def hook_stdout(cls):
        """
        Hooks the standard output and error streams to capture and log YOLO messages.
        """
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