import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

_log_dir: Optional[Path] = None
DEFAULT_DATE_FORMAT = "%d/%m/%Y %H:%M:%S"


def get_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(log_dir: Union[str, Path] = "_logs", run_name: Optional[str] = None, level: int = logging.INFO, log_filename: str = "run.log", create_run_dir: bool = True) -> Path:
    global _log_dir
    log_dir = Path(log_dir)
    if create_run_dir:
        stamp = get_run_timestamp()
        log_dir = log_dir / (f"{run_name}_{stamp}" if run_name else f"run_{stamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    _log_dir = log_dir
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers = []
    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(level)
    stream.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt=DEFAULT_DATE_FORMAT))
    root.addHandler(stream)
    file_handler = logging.FileHandler(log_dir / log_filename, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt=DEFAULT_DATE_FORMAT))
    root.addHandler(file_handler)
    root.info("Logging initialized")
    return log_dir


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt=DEFAULT_DATE_FORMAT, stream=sys.stdout)
    return logger


def get_log_dir() -> Optional[Path]:
    return _log_dir


class LogContext:
    def __init__(self, log_dir: Union[str, Path], run_name: Optional[str] = None, level: int = logging.INFO):
        self.log_dir = log_dir
        self.run_name = run_name
        self.level = level
        self._previous_handlers = []
        self._previous_level = None

    def __enter__(self) -> Path:
        root = logging.getLogger()
        self._previous_handlers = root.handlers.copy()
        self._previous_level = root.level
        return setup_logging(self.log_dir, self.run_name, self.level)

    def __exit__(self, exc_type, exc_val, exc_tb):
        root = logging.getLogger()
        root.handlers = self._previous_handlers
        if self._previous_level is not None:
            root.setLevel(self._previous_level)
        return False


def log_exception(logger: logging.Logger, exc: Exception, context: str = "") -> None:
    logger.exception(f"{context}: {exc}" if context else str(exc))


def create_workflow_logger(workflow_name: str, mode: str = "production", level: int = logging.INFO) -> logging.Logger:
    if mode not in ("production", "test"):
        raise ValueError("mode must be 'production' or 'test'")
    setup_logging(Path.cwd() / "_logs" / mode, workflow_name, level)
    return get_logger(workflow_name)


def datedPrint(msg: str, logger: Optional[logging.Logger] = None) -> None:
    if logger is not None:
        logger.info(msg)
    else:
        print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {msg}")
        sys.stdout.flush()
