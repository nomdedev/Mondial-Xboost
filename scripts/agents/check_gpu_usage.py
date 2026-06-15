#!/usr/bin/env python3
"""
GPU / CUDA Auditor — Mondial-Xboost
====================================

Detecta si XGBoost puede usar CUDA y, si pynvml está disponible,
verifica el uso de VRAM durante el entrenamiento.

Salida JSON con status PASS / WARNING / BLOCK.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def detect_device() -> str:
    """Replica la detección de loop_engineering.py para consistencia."""
    try:
        info = xgb.build_info()
        if not info.get("USE_CUDA", False):
            return "cpu"
        X = np.random.RandomState(42).rand(100, 4)
        y = np.random.RandomState(43).randint(0, 3, size=100)
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "device": "cuda",
            "tree_method": "hist",
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 3,
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            xgb.train(params, dtrain, num_boost_round=5)
            for warning in w:
                msg = str(warning.message).lower()
                if "no visible gpu" in msg or "changed from gpu to cpu" in msg:
                    return "cpu"
        return "cuda"
    except Exception:
        return "cpu"


def vram_info() -> dict[str, Any]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_gb = mem.total / (1024 ** 3)
        used_gb = mem.used / (1024 ** 3)
        free_gb = mem.free / (1024 ** 3)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "available": True,
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "used_percent": round(used_gb / total_gb * 100, 1) if total_gb else 0.0,
            "gpu_utilization": util.gpu,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def audit(vram_warning: float = 85.0, vram_block: float = 95.0) -> dict[str, Any]:
    device = detect_device()
    info: dict[str, Any] = {"device": device, "cuda_supported": xgb.build_info().get("USE_CUDA", False)}

    if device == "cpu":
        info["status"] = "WARNING"
        info["message"] = "XGBoost correrá en CPU. Verificar driver NVIDIA/CUDA si se espera GPU."
        return info

    vram = vram_info()
    info["vram"] = vram

    if not vram.get("available"):
        info["status"] = "WARNING"
        info["message"] = "GPU detectada pero no se pudo leer VRAM (instalar pynvml para monitoreo)."
        return info

    used = vram.get("used_percent", 0)
    if used >= vram_block:
        info["status"] = "BLOCK"
        info["message"] = f"VRAM crítica: {used:.1f}% usada. Riesgo de OOM."
    elif used >= vram_warning:
        info["status"] = "WARNING"
        info["message"] = f"VRAM alta: {used:.1f}% usada. Considerar reducir max_depth/n_estimators."
    else:
        info["status"] = "PASS"
        info["message"] = f"GPU lista: {used:.1f}% VRAM usada."

    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU / CUDA Auditor")
    parser.add_argument("--vram-warning", type=float, default=85.0, help="%% VRAM para WARNING")
    parser.add_argument("--vram-block", type=float, default=95.0, help="%% VRAM para BLOCK")
    parser.add_argument("--json", action="store_true", help="Imprimir solo JSON")
    args = parser.parse_args()

    result = audit(vram_warning=args.vram_warning, vram_block=args.vram_block)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"[{result['status']}] {result['message']}")

    return 0 if result["status"] in ("PASS", "WARNING") else 1


if __name__ == "__main__":
    raise SystemExit(main())
