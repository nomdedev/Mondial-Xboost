"""Vercel entrypoint wrapper for the Mondial-Xboost FastAPI bridge."""

from __future__ import annotations

from predictors.api import app

__all__ = ["app"]
