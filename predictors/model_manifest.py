"""
Model manifest — keeps a machine-readable record of trained models.

The manifest is stored next to the model artifacts in data/models/ and contains
enough metadata to reproduce or audit a prediction model.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sanitize(value: Any) -> Any:
    """Replace non-JSON floats (NaN, Infinity) with None recursively."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value

MODELS_DIR = Path(__file__).parent.parent / "data" / "models"
MANIFEST_PATH = MODELS_DIR / "model_manifest.json"


def _hash_file(path: Path) -> str:
    """Return SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_manifest(
    name: str,
    feature_cols: list[str],
    hyperparameters: dict[str, Any],
    dataset_hash: str | None = None,
    calibration: bool = True,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest dictionary for a trained model."""
    paths = {
        "outcome": str(MODELS_DIR / f"{name}_outcome.pkl"),
        "home_goals": str(MODELS_DIR / f"{name}_home_goals.pkl"),
        "away_goals": str(MODELS_DIR / f"{name}_away_goals.pkl"),
        "meta": str(MODELS_DIR / f"{name}_meta.json"),
    }
    manifest = {
        "model_name": name,
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_cols": feature_cols,
        "hyperparameters": _sanitize(hyperparameters),
        "calibration": calibration,
        "dataset_hash": dataset_hash,
        "artifact_paths": paths,
        "artifact_hashes": {k: _hash_file(Path(v)) for k, v in paths.items() if Path(v).exists()},
        "metrics": _sanitize(metrics) or {},
        "extra": _sanitize(extra) or {},
    }
    return manifest


def load_manifest() -> dict[str, Any] | None:
    """Load the current manifest if it exists."""
    if not MANIFEST_PATH.exists():
        return None
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: dict[str, Any], path: Path | str | None = None) -> Path:
    """Persist manifest to disk."""
    target = Path(path) if path is not None else MANIFEST_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, default=str, allow_nan=False), encoding="utf-8")
    return target


def hash_dataset(csv_path: Path | str | None = None) -> str:
    """Return a short SHA-256 hash of the canonical dataset."""
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "Oloraculo.Web" / "Data" / "historical_results.csv"
    return _hash_file(Path(csv_path))
