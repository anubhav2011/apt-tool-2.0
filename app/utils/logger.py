import logging
from datetime import datetime, timedelta
import os
from types import TracebackType
from typing import Mapping
from pathlib import Path
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv
import glob

load_dotenv()
ENV = os.getenv("APP_ENV")


def is_debug_env() -> bool:
    """True when DEBUG=1|true|yes|on (case-insensitive)."""
    v = (os.getenv("DEBUG", "") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_enabled(key: str, default: bool = True) -> bool:
    """False for 0|false|no|off|n (case-insensitive); empty uses default."""
    v = (os.getenv(key) or "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off", "n")


def _parse_log_level() -> int:
    if is_debug_env():
        raw = "DEBUG"
    else:
        raw = (os.getenv("LOG_LEVEL", "INFO") or "INFO").strip().upper()
    mapping = {
        "DEBUG":    logging.DEBUG,
        "INFO":     logging.INFO,
        "WARNING":  logging.WARNING,
        "WARN":     logging.WARNING,
        "ERROR":    logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return mapping.get(raw, logging.INFO)


def _resolve_log_paths(root_dir: Path) -> tuple[Path, Path]:
    """
    Returns (log file path, directory used for old-log cleanup glob).
    LOG_FILE: relative paths are resolved under project root.
    If unset: debug_logs/debug_logs_{local date}.log.
    """
    lf = (os.getenv("LOG_FILE") or "").strip()
    if lf:
        p = Path(lf)
        if not p.is_absolute():
            p = root_dir / p
        p = p.resolve()
        return p, p.parent
    current_date = datetime.now().astimezone().strftime("%Y-%m-%d")
    logs_dir = (root_dir / "debug_logs").resolve()
    return logs_dir / f"debug_logs_{current_date}.log", logs_dir


def is_proctoring_diagnostics_enabled() -> bool:
    """
    When true, emit verbose proctoring pipeline logs (per-frame samples,
    event breakdowns, full scoring metrics). Controlled by env:
    ENABLE_PROCTORING_DIAGNOSTICS=1|true|yes|on
    """
    v = (os.getenv("ENABLE_PROCTORING_DIAGNOSTICS", "") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


class DebugLogger:
    """
    App logger. Env:
      DEBUG — if true, log level is DEBUG (overrides LOG_LEVEL for app handlers).
      LOG_LEVEL — DEBUG|INFO|WARNING|ERROR when DEBUG is off (default INFO).
      LOG_FILE — path to the rotating log file; relative → project root. If unset,
                 uses debug_logs/debug_logs_{local date}.log.
      LOG_ENABLE_FILE — if false, no file handler (stdout/stderr only; Docker/K8s).
      LOG_ENABLE_CONSOLE — if false, file only (default true).
      LOG_FILE_MAX_BYTES — rotating file size (default 10485760 = 10 MiB).
      LOG_FILE_BACKUP_COUNT — rotated files to keep (default 5).
      ENABLE_PROCTORING_DIAGNOSTICS — verbose proctoring logs (default off).
    """

    def __init__(self, logger_name="debug_logger"):
        self._logger_name = logger_name
        self._root_dir = Path(__file__).resolve().parents[2]
        self._logs_dir = ""
        self._log_file_path: Path | None = None
        self._level = _parse_log_level()
        self._debug_logger: logging.Logger = self._create_logger()
        self._clean_old_logs()

    def _create_logger(self):
        """Creates handlers from LOG_ENABLE_FILE / LOG_ENABLE_CONSOLE and LOG_FILE."""
        logger = logging.getLogger(self._logger_name)
        logger.setLevel(self._level)

        use_file = _env_enabled("LOG_ENABLE_FILE", True)
        use_console = _env_enabled("LOG_ENABLE_CONSOLE", True)
        if not use_file and not use_console:
            use_console = True

        if use_file:
            self._log_file_path, logs_dir_path = _resolve_log_paths(self._root_dir)
            self._logs_dir = str(logs_dir_path)
            log_parent = self._log_file_path.parent
            if not log_parent.exists():
                log_parent.mkdir(parents=True, exist_ok=True)
        else:
            self._log_file_path = None
            self._logs_dir = ""

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        if use_file and use_console:
            console_format = logging.Formatter("%(message)s")
        else:
            console_format = file_format

        if use_file:
            assert self._log_file_path is not None
            log_filename = str(self._log_file_path)
            max_bytes = int(os.getenv("LOG_FILE_MAX_BYTES", "10485760") or "10485760")
            backup_count = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5") or "5")
            file_handler = RotatingFileHandler(
                log_filename,
                maxBytes=max(1_048_576, max_bytes),
                backupCount=max(0, backup_count),
                encoding="utf-8",
            )
            file_handler.setLevel(self._level)
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)

        if use_console:
            console_handler = logging.StreamHandler()
            if hasattr(console_handler.stream, "reconfigure"):
                try:
                    console_handler.stream.reconfigure(encoding="utf-8")
                except Exception:
                    pass
            console_handler.setLevel(self._level)
            console_handler.setFormatter(console_format)
            logger.addHandler(console_handler)

        logger.propagate = False

        return logger

    def _clean_old_logs(self, days_to_keep=30):
        if not self._logs_dir:
            return []
        try:
            # Retention uses local wall time (matches default log filename date).
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            log_files = glob.glob(os.path.join(self._logs_dir, "*.log"))

            deleted_files = []
            for log_file in log_files:
                file_mod_time = datetime.fromtimestamp(os.path.getmtime(log_file))
                if file_mod_time < cutoff_date:
                    os.remove(log_file)
                    deleted_files.append(log_file)
                    self._debug_logger.debug("Deleted old log file: %s", log_file)

            if deleted_files:
                self._debug_logger.debug(
                    "Log cleanup: removed %s old file(s)", len(deleted_files)
                )
            else:
                self._debug_logger.debug("Log cleanup: no old log files to delete")

            return deleted_files

        except Exception as e:
            self._debug_logger.warning("Error cleaning up old logs: %s", e)
            return []

    def info(self,
             msg: object,
             *args: object,
             exc_info: None | bool | tuple[type[BaseException], BaseException, TracebackType | None] | tuple[
                 None, None, None] | BaseException = None,
             stack_info: bool = False,
             stacklevel: int = 1,
             extra: Mapping[str, object] | None = None) -> None:
        self._debug_logger.info(msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra,
                                stacklevel=stacklevel)

    def debug(self,
              msg: object,
              *args: object,
              exc_info: None | bool | tuple[type[BaseException], BaseException, TracebackType | None] | tuple[
                  None, None, None] | BaseException = None,
              stack_info: bool = False,
              stacklevel: int = 1,
              extra: Mapping[str, object] | None = None) -> None:
        self._debug_logger.debug(msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra,
                                 stacklevel=stacklevel)

    def warning(self,
                msg: object,
                *args: object,
                exc_info: None | bool | tuple[type[BaseException], BaseException, TracebackType | None] | tuple[
                    None, None, None] | BaseException = None,
                stack_info: bool = False,
                stacklevel: int = 1,
                extra: Mapping[str, object] | None = None) -> None:
        self._debug_logger.warning(msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra,
                                   stacklevel=stacklevel)

    def error(self,
              msg: object,
              *args: object,
              exc_info: None | bool | tuple[type[BaseException], BaseException, TracebackType | None] | tuple[
                  None, None, None] | BaseException = None,
              stack_info: bool = False,
              stacklevel: int = 1,
              extra: Mapping[str, object] | None = None) -> None:
        self._debug_logger.error(msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra,
                                 stacklevel=stacklevel)

    def critical(self,
                 msg: object,
                 *args: object,
                 exc_info: None | bool | tuple[type[BaseException], BaseException, TracebackType | None] | tuple[
                     None, None, None] | BaseException = None,
                 stack_info: bool = False,
                 stacklevel: int = 1,
                 extra: Mapping[str, object] | None = None) -> None:
        self._debug_logger.critical(msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra,
                                    stacklevel=stacklevel)

    def exception(self,
                  msg: object,
                  *args: object,
                  exc_info: None | bool | tuple[type[BaseException], BaseException, TracebackType | None] | tuple[
                      None, None, None] | BaseException = True,
                  stack_info: bool = False,
                  stacklevel: int = 1,
                  extra: Mapping[str, object] | None = None) -> None:
        self._debug_logger.exception(msg, *args, exc_info=exc_info, stack_info=stack_info, extra=extra,
                                     stacklevel=stacklevel)


debug_logger = DebugLogger()
