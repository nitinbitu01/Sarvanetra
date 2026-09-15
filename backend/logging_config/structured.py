import logging
import logging.handlers
import json
import os
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for field in [
            "camera_id", "crime_type", "danger_score", "severity",
            "reid_id", "plate_text", "officer_id", "worker_id", "is_simulated"
        ]:
            if hasattr(record, field):
                entry[field] = getattr(record, field)
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(log_level: str = "INFO", log_file: str = "logs/sentinel.jsonl"):
    os.makedirs("logs", exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Console — human readable
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s %(message)s"
    ))

    # File — JSON structured
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=100 * 1024 * 1024, backupCount=10
    )
    fh.setFormatter(JSONFormatter())

    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(fh)
