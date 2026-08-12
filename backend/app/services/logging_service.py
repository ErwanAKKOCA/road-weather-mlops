from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock


class JSONLInferenceLogger:
    """Structured inference events to stdout and, optionally, a JSONL file."""

    def __init__(self, path: Path | None = None, logger_name: str = "inference") -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._logger = logging.getLogger(logger_name)

    def log(self, record: dict) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            **record,
        }
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self._logger.info(line)

        if self.path is None:
            return
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
