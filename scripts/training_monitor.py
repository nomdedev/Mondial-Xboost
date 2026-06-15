#!/usr/bin/env python3
"""
Training monitor — emite estado en vivo del entrenamiento.

Escribe `data/models/training_status.json` para que el dashboard HTML
pueda hacer polling del progreso de Optuna en tiempo real.

Usage:
    from scripts.training_monitor import monitor
    monitor.start(total_trials=100)
    monitor.update(trial_completed=1, best_test_accuracy=58.5)
    monitor.complete()
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "data" / "models"
STATUS_PATH = MODELS_DIR / "training_status.json"


class TrainingMonitor:
    """Thread-safe-ish file-based monitor for training progress."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or STATUS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = self._default()
        self._start_time: float = 0.0

    def _default(self) -> dict[str, Any]:
        return {
            "status": "idle",
            "phase": "idle",
            "batch": 0,
            "total_trials": 0,
            "completed_trials": 0,
            "best_test_accuracy": 0.0,
            "best_log_loss": 99.0,
            "best_overfit_gap": 0.0,
            "best_brier": 1.0,
            "elapsed_seconds": 0,
            "eta_seconds": 0,
            "recent_events": [],
            "last_update": datetime.now(UTC).isoformat(),
        }

    def _write(self) -> None:
        self._data["last_update"] = datetime.now(UTC).isoformat()
        try:
            self.path.write_text(json.dumps(self._data, indent=2, default=str), encoding="utf-8")
        except OSError:
            pass  # Best-effort: do not crash training if write fails.

    def _add_event(self, message: str) -> None:
        events = self._data.get("recent_events", [])
        events.append({"time": datetime.now(UTC).isoformat(), "message": message})
        self._data["recent_events"] = events[-50:]  # Keep last 50.

    def _update_eta(self) -> None:
        completed = self._data.get("completed_trials", 0)
        total = self._data.get("total_trials", 0)
        elapsed = time.time() - self._start_time
        self._data["elapsed_seconds"] = int(elapsed)
        if completed > 0 and total > completed:
            avg_per_trial = elapsed / completed
            remaining = total - completed
            self._data["eta_seconds"] = int(avg_per_trial * remaining)
        else:
            self._data["eta_seconds"] = 0

    def start(self, total_trials: int, batch: int = 1, phase: str = "tuning") -> None:
        """Mark training as started."""
        self._data = self._default()
        self._data["status"] = "running"
        self._data["phase"] = phase
        self._data["batch"] = batch
        self._data["total_trials"] = total_trials
        self._start_time = time.time()
        self._add_event(f"Batch {batch} iniciado: {total_trials} trials")
        self._write()

    def set_phase(self, phase: str) -> None:
        """Update current phase (tuning/analysis/retraining/completed)."""
        self._data["phase"] = phase
        self._add_event(f"Fase: {phase}")
        self._update_eta()
        self._write()

    def update(
        self,
        trial_completed: int,
        test_accuracy: float | None = None,
        val_accuracy: float | None = None,
        log_loss: float | None = None,
        brier: float | None = None,
        overfit_gap: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Report a completed trial."""
        self._data["completed_trials"] = trial_completed

        if test_accuracy is not None and test_accuracy > self._data.get("best_test_accuracy", 0.0):
            self._data["best_test_accuracy"] = round(test_accuracy, 4)
            msg = f"Nuevo mejor trial {trial_completed}: test accuracy {test_accuracy:.2f}%"
            if log_loss is not None:
                msg += f", log loss {log_loss:.4f}"
            self._add_event(msg)

        if log_loss is not None and log_loss < self._data.get("best_log_loss", 99.0):
            self._data["best_log_loss"] = round(log_loss, 4)

        if brier is not None and brier < self._data.get("best_brier", 1.0):
            self._data["best_brier"] = round(brier, 4)

        if overfit_gap is not None:
            current_best_gap = self._data.get("best_overfit_gap", 0.0)
            # Prefer lower gap when accuracy is similar, but keep best acc primary.
            if overfit_gap < current_best_gap:
                self._data["best_overfit_gap"] = round(overfit_gap, 4)

        self._update_eta()
        self._write()

    def complete(self, message: str = "Entrenamiento completado") -> None:
        """Mark training as completed."""
        self._data["status"] = "completed"
        self._data["phase"] = "completed"
        self._data["eta_seconds"] = 0
        self._update_eta()
        self._add_event(message)
        self._write()

    def error(self, message: str) -> None:
        """Mark training as failed."""
        self._data["status"] = "error"
        self._add_event(f"ERROR: {message}")
        self._update_eta()
        self._write()

    def reset(self) -> None:
        """Reset to idle state."""
        self._data = self._default()
        self._write()


# Global singleton for convenience.
monitor = TrainingMonitor()


def load_status() -> dict[str, Any]:
    """Load current training status from disk."""
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return TrainingMonitor()._default()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        monitor.reset()
        print(f"Monitor reset: {STATUS_PATH}")
    else:
        print(json.dumps(load_status(), indent=2))
