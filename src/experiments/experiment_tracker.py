"""
Append-only experiment tracker.
Every training/eval run appends one record to reports/experiments.json.
Never overwrites history.
"""

import json
import os
import datetime
from typing import Any, Dict, List

EXPERIMENTS_PATH = os.path.join("reports", "experiments.json")


def log_experiment(record: Dict[str, Any]) -> None:
    """
    Append record to experiments.json. Fields logged per run:
    id, model, feature_set_hash, split info, threshold, all metrics, created_at, notes.
    """
    os.makedirs("reports", exist_ok=True)

    existing: List[Dict[str, Any]] = []
    if os.path.exists(EXPERIMENTS_PATH):
        try:
            with open(EXPERIMENTS_PATH, "r") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = []

    record.setdefault("id", len(existing) + 1)
    record.setdefault("created_at", datetime.datetime.utcnow().isoformat())

    existing.append(record)

    with open(EXPERIMENTS_PATH, "w") as f:
        json.dump(existing, f, indent=2, default=str)


def load_experiments() -> List[Dict[str, Any]]:
    """Return all experiment records."""
    if not os.path.exists(EXPERIMENTS_PATH):
        return []
    with open(EXPERIMENTS_PATH, "r") as f:
        return json.load(f)
