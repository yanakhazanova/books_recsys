"""
Install a Tee on sys.stdout / sys.stderr that mirrors every write to a log
file. The /api/v1/logs/stream SSE endpoint tails this file so the frontend
can show live progress during long-running operations (training, SHAP, ...).

Install is idempotent — calling install_log_capture() more than once is a
no-op.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional


_lock = threading.Lock()
_installed = False
_log_path: Optional[Path] = None


class _Tee:
    def __init__(self, original, path: Path):
        self._original = original
        self._path = path
        self._fh = None

    def _ensure_open(self):
        if self._fh is None or getattr(self._fh, 'closed', True):
            self._fh = open(self._path, 'a', buffering=1, encoding='utf-8',
                            errors='replace')

    def write(self, data):
        try:
            self._original.write(data)
        except Exception:
            pass
        try:
            self._ensure_open()
            self._fh.write(data)
        except Exception:
            pass
        return len(data) if isinstance(data, str) else 0

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass
        try:
            if self._fh:
                self._fh.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._original.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._original.fileno()

    def __getattr__(self, name):
        return getattr(self._original, name)


def install_log_capture(path: str | Path = "data/logs/server.log") -> Path:
    """Install the Tee. Returns the log file path."""
    global _installed, _log_path
    with _lock:
        if _installed and _log_path is not None:
            return _log_path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(p, 'a', encoding='utf-8') as f:
                f.write("\n==== session start ====\n")
        except Exception:
            pass
        sys.stdout = _Tee(sys.stdout, p)
        sys.stderr = _Tee(sys.stderr, p)
        _installed = True
        _log_path = p
        return p


def get_log_path() -> Optional[Path]:
    return _log_path
