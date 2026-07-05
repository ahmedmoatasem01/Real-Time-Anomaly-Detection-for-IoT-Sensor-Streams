"""
Model registry — single source of truth for all trained model artifacts and test metrics.
Backed by models/model_registry.json. Thread-safe for sequential use.
"""

import json
import os
import datetime
from typing import Any, Dict, List, Optional

REGISTRY_PATH = os.path.join("models", "model_registry.json")


def _load() -> List[Dict[str, Any]]:
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def _save(registry: List[Dict[str, Any]]) -> None:
    os.makedirs("models", exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def register(model_meta: Dict[str, Any]) -> None:
    """
    Upsert a model entry. If a record with the same 'name' exists it is replaced.
    model_meta must include at minimum: name, type, artifact_path.
    """
    registry = _load()
    # Remove stale entry for same model name
    registry = [r for r in registry if r.get("name") != model_meta["name"]]
    model_meta.setdefault("registered_at", datetime.datetime.utcnow().isoformat())
    registry.append(model_meta)
    _save(registry)


def list_models() -> List[Dict[str, Any]]:
    """Return all registered models."""
    return _load()


def get(name: str) -> Optional[Dict[str, Any]]:
    """Return a single model entry by name, or None."""
    for entry in _load():
        if entry.get("name") == name:
            return entry
    return None


def set_production(name: str) -> bool:
    """
    Mark model `name` as production (is_production=True); demote all others.
    Returns True if the model was found.
    """
    registry = _load()
    found = False
    for entry in registry:
        if entry.get("name") == name:
            entry["is_production"] = True
            found = True
        else:
            entry["is_production"] = False
    if found:
        _save(registry)
    return found
