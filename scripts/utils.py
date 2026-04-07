import logging
from pathlib import Path
from typing import Optional


def ensure_parent(path: Path) -> None:
    """Create parent directories for a path if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)


def setup_logger(log_path: Path, name: str = "feature_pipeline") -> logging.Logger:
    """Return a configured logger that writes to log_path."""
    ensure_parent(log_path)
    logger = logging.getLogger(name + str(log_path))
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def resolve_path(path_str: str, dataset_root: Path) -> Path:
    """Resolve a file path that may be relative to the dataset root."""
    p = Path(path_str)
    return dataset_root / p if not p.is_absolute() else p


def optional_relative(path: Path, root: Path) -> str:
    """Return a path relative to root when possible, otherwise absolute."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def file_size(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size
    except OSError:
        return None
