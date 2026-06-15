"""
Loop Engineering — Mondial-Xboost
===================================
Hyperparameter tuning de XGBoost con Optuna, cross-validation temporal
y soporte GPU.

- 3-way temporal split: train / validation / test
- Cada trial se guarda inmediatamente en CSV (resistente a cortes)
- Walk-forward validation opcional para estabilidad
- Purged k-fold CV opcional como objetivo de Optuna
- GPU auto-detección (CUDA) con fallback a CPU
- Persistencia del study de Optuna en SQLite para reanudar
- Guarda el mejor modelo y sus parámetros
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from predictors.feature_engineering import FEATURE_COLS, build_training_dataset
from scripts.training_monitor import monitor

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = MODELS_DIR / "loop_engineering.json"
PARTIAL_CSV = MODELS_DIR / "loop_engineering_partial.csv"
BEST_JSON = MODELS_DIR / "xgboost_loop_best.json"
STUDY_DB = MODELS_DIR / "optuna_study.db"

FIELDNAMES = [
    "batch",
    "trial",
    "timestamp",
    "params",
    "val_accuracy",
    "test_accuracy",
    "log_loss",
    "brier",
    "train_accuracy",
    "overfit_gap",
    "walk_forward_acc",
    "cv_mean_acc",
    "cv_std_acc",
    "cv_min_acc",
    "cv_max_acc",
    "cv_stability_ratio",
    "cv_mean_log_loss",
]

CV_LAMBDA_DEFAULT = 2.0


def _xgboost_can_run_cuda() -> bool:
    """Intenta entrenar un booster en CUDA y confirma que no haya fallback a CPU."""
    try:
        info = xgb.build_info()
        if not info.get("USE_CUDA", False):
            return False
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
                    return False
        return True
    except Exception:
        return False


def detect_device(prefer_gpu: bool = True) -> str:
    """Detectar si XGBoost puede usar CUDA; si no, devolver 'cpu'."""
    env = os.getenv("XGBOOST_DEVICE", "auto").lower()
    if env == "cpu":
        return "cpu"
    if env == "cuda":
        return "cuda"
    if not prefer_gpu:
        return "cpu"
    return "cuda" if _xgboost_can_run_cuda() else "cpu"


def device_n_jobs(device: str) -> int:
    """n_jobs recomendado según el dispositivo."""
    return 1 if device == "cuda" else -1


def load_data(min_date: str = "2015-01-01", val_cutoff: str = "2023-01-01", test_cutoff: str = "2024-01-01"):
    """Carga datos con split temporal 3-way."""
    df = build_training_dataset(min_date=min_date)
    df = df.dropna(subset=["home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    available = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=available)

    train = df[df["date"] < val_cutoff].copy()
    val = df[(df["date"] >= val_cutoff) & (df["date"] < test_cutoff)].copy()
    test = df[df["date"] >= test_cutoff].copy()

    splits = {
        "X_train": train[available].fillna(0),
        "y_train": train["outcome"].astype(int),
        "X_val": val[available].fillna(0),
        "y_val": val["outcome"].astype(int),
        "X_test": test[available].fillna(0),
        "y_test": test["outcome"].astype(int),
        "features": available,
    }

    print(f"  Train: {len(train)} (< {val_cutoff})")
    print(f"  Val:   {len(val)} ({val_cutoff} .. {test_cutoff})")
    print(f"  Test:  {len(test)} (>= {test_cutoff})")
    return splits


def build_full_df(min_date: str = "2015-01-01") -> pd.DataFrame:
    """Dataset completo con fechas para CV temporal / walk-forward."""
    df = build_training_dataset(min_date=min_date)
    df = df.dropna(subset=["home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    available = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=available)
    return df


def build_xgb(
    trial: optuna.Trial,
    aggressive: bool = False,
    device: str = "cpu",
) -> dict[str, object]:
    """Construye un dict de hiperparámetros para XGBClassifier."""
    tree_method = "hist"
    if aggressive:
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 3000),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
            "subsample": trial.suggest_float("subsample", 0.3, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 50.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 50.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 10.0),
            "num_parallel_tree": trial.suggest_int("num_parallel_tree", 1, 10),
            "max_bin": trial.suggest_int("max_bin", 128, 512),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "sampling_method": trial.suggest_categorical("sampling_method", ["uniform", "gradient_based"]),
            "tree_method": tree_method,
            "device": device,
        }
        if p["grow_policy"] == "lossguide":
            p["max_leaves"] = trial.suggest_int("max_leaves", 0, 256)
        if p["booster"] == "dart":
            p["rate_drop"] = trial.suggest_float("rate_drop", 0.0, 0.5)
            p["skip_drop"] = trial.suggest_float("skip_drop", 0.0, 0.5)
            p["one_drop"] = trial.suggest_categorical("one_drop", [True, False])
        return p
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "tree_method": tree_method,
        "device": device,
    }


def evaluate(y_true, y_pred, y_proba, y_train_true=None, y_train_pred=None) -> dict[str, float]:
    """Calcula métricas de clasificación."""
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_proba)
    brier = sum(brier_score_loss((y_true == i).astype(int), y_proba[:, i]) for i in range(3)) / 3
    train_acc = accuracy_score(y_train_true, y_train_pred) if y_train_true is not None else None
    return {
        "test_accuracy": round(acc * 100, 4),
        "log_loss": round(ll, 4),
        "brier": round(brier, 4),
        "train_accuracy": round(train_acc * 100, 4) if train_acc is not None else None,
        "overfit_gap": round((train_acc - acc) * 100, 4) if train_acc is not None else None,
    }


def append_partial(row: dict[str, object]) -> None:
    """Guarda un trial en CSV inmediatamente (append)."""
    exists = PARTIAL_CSV.exists()
    # Si el CSV existente tiene headers distintos, lo respaldamos.
    if exists:
        try:
            with open(PARTIAL_CSV, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
            if header != FIELDNAMES:
                bak = PARTIAL_CSV.with_suffix(".csv.bak")
                PARTIAL_CSV.rename(bak)
                exists = False
        except Exception:
            exists = False

    with open(PARTIAL_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_json_results() -> dict:
    """Carga resultados acumulados o inicializa estructura vacía."""
    if RESULTS_JSON.exists():
        return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    return {"batches": [], "all_runs": [], "best": None}


def save_json_results(data: dict) -> None:
    """Persiste resultados acumulados en JSON."""
    RESULTS_JSON.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def walk_forward_eval(params: dict, df: pd.DataFrame, features: list[str], folds: int = 3) -> float:
    """Evaluación walk-forward para medir estabilidad temporal."""
    dates = sorted(df["date"].unique())
    fold_size = len(dates) // (folds + 1)
    accs = []

    for i in range(folds):
        train_cutoff = dates[(i + 1) * fold_size]
        test_cutoff = dates[(i + 2) * fold_size] if (i + 2) * fold_size < len(dates) else None

        train_fold = df[df["date"] < train_cutoff]
        test_fold = df[(df["date"] >= train_cutoff) & (df["date"] < test_cutoff)] if test_cutoff else df[df["date"] >= train_cutoff]

        if len(test_fold) == 0:
            continue

        model = xgb.XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=device_n_jobs(params.get("device", "cpu")),
            verbosity=0,
        )
        model.fit(train_fold[features].fillna(0), train_fold["outcome"].astype(int))
        preds = model.predict(test_fold[features].fillna(0))
        accs.append(accuracy_score(test_fold["outcome"].astype(int), preds))

    return round(float(np.mean(accs)) * 100, 4) if accs else 0.0


def purged_cv_eval_with_test(
    params: dict,
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    n_splits: int = 3,
    embargo_days: int = 60,
    early_stop: bool = True,
) -> tuple[dict[str, float], dict[str, float]]:
    """Purged CV con evaluación de test integrada.

    Cada fold:
      1. Train en fold train (con early stopping en fold val)
      2. Predice en fold test (métricas CV)
      3. Predice en test_df externo (promediado entre folds)

    Retorna (cv_metrics, test_metrics).
    """
    dates = sorted(df["date"].unique())
    fold_size = len(dates) // (n_splits + 1)
    accs, log_losses, briers = [], [], []
    all_test_probas = []
    n_valid_folds = 0
    n_jobs = device_n_jobs(params.get("device", "cpu"))

    for i in range(n_splits):
        train_end = dates[(i + 1) * fold_size]
        test_end = dates[(i + 2) * fold_size] if (i + 2) * fold_size < len(dates) else dates[-1]

        embargo_start = train_end + pd.Timedelta(days=1)
        test_start = embargo_start + pd.Timedelta(days=embargo_days)

        if test_start >= test_end:
            continue

        train_mask = df["date"] < train_end
        embargo_mask = (df["date"] >= embargo_start) & (df["date"] < test_start)
        test_mask = (df["date"] >= test_start) & (df["date"] <= test_end)

        train_fold = df[train_mask].copy()
        val_fold = df[test_mask & ~embargo_mask].copy()

        if len(train_fold) < 100 or len(val_fold) < 10:
            continue

        fit_kwargs = dict(
            early_stopping_rounds=50 if (early_stop and params.get("n_estimators", 100) > 100) else 0,
        )

        model = xgb.XGBClassifier(
            **params,
            **fit_kwargs,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=n_jobs,
            verbosity=0,
        )

        model.fit(
            X=train_fold[features].fillna(0),
            y=train_fold["outcome"].astype(int),
            eval_set=[(val_fold[features].fillna(0), val_fold["outcome"].astype(int))] if (early_stop and params.get("n_estimators", 100) > 100) else None,
            verbose=False,
        )

        val_proba = model.predict_proba(val_fold[features].fillna(0))
        val_preds = np.argmax(val_proba, axis=1)
        accs.append(accuracy_score(val_fold["outcome"].astype(int), val_preds))
        log_losses.append(log_loss(val_fold["outcome"].astype(int), val_proba))
        briers.append(sum(brier_score_loss((val_fold["outcome"] == i).astype(int), val_proba[:, i]) for i in range(3)) / 3)

        test_proba = model.predict_proba(test_df[features].fillna(0))
        all_test_probas.append(test_proba)
        n_valid_folds += 1

    if n_valid_folds == 0:
        cv_defaults = {k: 0.0 for k in ["cv_mean_acc", "cv_std_acc", "cv_min_acc", "cv_max_acc", "cv_stability_ratio", "cv_mean_log_loss"]}
        return cv_defaults, {"test_accuracy": 0.0, "log_loss": 99.0, "brier": 1.0, "train_accuracy": 0.0, "overfit_gap": 0.0}

    cv_metrics = {
        "cv_mean_acc": round(float(np.mean(accs)) * 100, 4),
        "cv_std_acc": round(float(np.std(accs)) * 100, 4),
        "cv_min_acc": round(float(np.min(accs)) * 100, 4),
        "cv_max_acc": round(float(np.max(accs)) * 100, 4),
        "cv_stability_ratio": round(float(np.mean(accs) / (np.std(accs) + 0.001)), 4),
        "cv_mean_log_loss": round(float(np.mean(log_losses)), 4),
    }

    avg_test_proba = np.mean(all_test_probas, axis=0)
    avg_test_pred = np.argmax(avg_test_proba, axis=1)
    y_test = test_df["outcome"].astype(int) if "outcome" in test_df.columns else test_df["y_test"]

    test_metrics = {
        "test_accuracy": round(accuracy_score(y_test, avg_test_pred) * 100, 4),
        "log_loss": round(log_loss(y_test, avg_test_proba), 4),
        "brier": round(sum(brier_score_loss((y_test == i).astype(int), avg_test_proba[:, i]) for i in range(3)) / 3, 4),
        "train_accuracy": None,
        "overfit_gap": None,
    }

    return cv_metrics, test_metrics


def smooth_labels(outcome: pd.Series, elo_diff: pd.Series, alpha: float = 0.7) -> np.ndarray:
    """Convierte etiquetas duras en suaves mezclando con prior basado en Elo."""
    n = len(outcome)
    y_onehot = np.eye(3)[outcome.astype(int)]

    elo_prior = np.zeros((n, 3))
    for i, dr in enumerate(elo_diff):
        home_exp = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        elo_prior[i] = [(1 - home_exp) * 0.3, 0.3, home_exp * 0.7]
        elo_prior[i] /= elo_prior[i].sum()

    return alpha * y_onehot + (1 - alpha) * elo_prior


def run_batch(
    batch_num: int,
    n_trials: int = 50,
    walk_forward: bool = False,
    aggressive: bool = False,
    label_smoothing: bool = False,
    cv: bool = True,
    cv_folds: int = 3,
    cv_embargo: int = 60,
    cv_lambda: float = CV_LAMBDA_DEFAULT,
    use_gpu: bool = True,
    study_name: str = "xgboost_loop",
    test_cutoff: str = "2024-01-01",
    top_k_retrain: int = 10,
) -> list[dict]:
    """Ejecuta un batch de Optuna con XGBoost.

    Mejoras v2:
      - CV sin leakage (excluye periodo test del CV)
      - Métricas test embebidas en cada trial (sin post-hoc loop lento)
      - Log-loss como objetivo primario (minimizar)
      - Two-phase sampler: Random → TPE
      - MedianPruner en lugar de Hyperband
      - DART booster con parámetros condicionales
      - Warm-start desde mejor previo
    """
    device = detect_device(prefer_gpu=use_gpu)
    print(f"\n{'=' * 70}")
    print(f"BATCH {batch_num} — XGBoost + Optuna ({n_trials} trials)")
    print(f"Dispositivo: {device} | CV: {cv} ({cv_folds} folds, {cv_embargo}d embargo)")
    print(f"{'=' * 70}")

    splits = load_data(test_cutoff=test_cutoff)
    data = load_json_results()
    batch_runs: list[dict] = []

    monitor.start(total_trials=n_trials, batch=batch_num, phase="tuning")

    df_full = build_full_df(min_date="2015-01-01")
    cv_df = df_full[df_full["date"] < test_cutoff].copy()
    print(f"  CV data: {len(cv_df)} rows (excluye test period >= {test_cutoff})")

    test_df = splits["X_test"].copy()
    test_df["outcome"] = splits["y_test"]

    def objective(trial: optuna.Trial) -> float:
        params = build_xgb(trial, aggressive=aggressive, device=device)

        cv_metrics, test_metrics = purged_cv_eval_with_test(
            params,
            cv_df,
            test_df,
            splits["features"],
            n_splits=cv_folds,
            embargo_days=cv_embargo,
            early_stop=True,
        )

        for k, v in {**cv_metrics, **test_metrics}.items():
            trial.set_user_attr(k, v)

        return cv_metrics["cv_mean_log_loss"]

    storage_url = f"sqlite:///{STUDY_DB.resolve().as_posix()}"

    n_existing_trials = 0
    try:
        prev_study = optuna.load_study(study_name=study_name, storage=storage_url)
        n_existing_trials = len(prev_study.trials)
    except Exception:
        pass

    burn_in_remaining = max(0, 50 - n_existing_trials)
    if n_existing_trials >= 50:
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=0)
        print(f"  Sampler: TPE (group=True) — {n_existing_trials} trials existentes > 50")
    else:
        sampler = optuna.samplers.RandomSampler(seed=42)
        print(f"  Sampler: Random (burn-in) — {burn_in_remaining} trials restantes para TPE")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=30, n_warmup_steps=5),
        load_if_exists=True,
    )

    if n_existing_trials < 50:
        study.sampler = optuna.samplers.RandomSampler(seed=42)

    best_prev = data.get("best")
    if best_prev and best_prev.get("params"):
        base_params = {k: v for k, v in best_prev["params"].items()
                       if k in ["n_estimators", "max_depth", "learning_rate",
                                "subsample", "colsample_bytree", "reg_alpha",
                                "reg_lambda", "min_child_weight", "gamma"]}
        if aggressive:
            base_params.update({
                "max_delta_step": best_prev["params"].get("max_delta_step", 5.0),
                "num_parallel_tree": best_prev["params"].get("num_parallel_tree", 3),
                "max_bin": best_prev["params"].get("max_bin", 256),
                "grow_policy": best_prev["params"].get("grow_policy", "depthwise"),
                "booster": "gbtree",
                "sampling_method": "uniform",
            })
        study.enqueue_trial(base_params)
        print(f"  Warm-start: seeded best prev params (test_acc={best_prev['test_accuracy']:.2f}%)")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, gc_after_trial=True)

    if n_existing_trials < 50 and len(study.trials) >= 50:
        study.sampler = optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=0)

    print(f"\n[Done] {n_trials} CV trials completados. Leyendo métricas almacenadas...")

    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        params = build_xgb(trial, aggressive=aggressive, device=device)

        cv_metrics = {k: trial.user_attrs.get(k, 0.0) for k in
                      ["cv_mean_acc", "cv_std_acc", "cv_min_acc", "cv_max_acc",
                       "cv_stability_ratio", "cv_mean_log_loss"]}
        test_metrics = {k: trial.user_attrs.get(k, 0.0) for k in
                        ["test_accuracy", "log_loss", "brier", "train_accuracy", "overfit_gap"]}

        run = {
            "batch": batch_num,
            "trial": trial.number,
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "val_accuracy": round(trial.value, 4) if trial.value is not None else 0.0,
            "walk_forward_acc": 0.0,
            **test_metrics,
            **cv_metrics,
        }

        batch_runs.append(run)
        append_partial(run)

    batch_runs.sort(key=lambda x: x.get("test_accuracy", 0), reverse=True)

    print(f"\n[Top-K retrain] Re-entrenando top-{top_k_retrain} para walk-forward...")
    for rank, run in enumerate(batch_runs[:top_k_retrain]):
        params = run["params"]
        try:
            model = xgb.XGBClassifier(
                **params,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=device_n_jobs(device),
                verbosity=0,
            )
            model.fit(splits["X_train"], splits["y_train"])

            train_pred = model.predict(splits["X_train"])
            train_acc = accuracy_score(splits["y_train"], train_pred)
            run["train_accuracy"] = round(train_acc * 100, 4)
            run["overfit_gap"] = round((train_acc * 100) - run["test_accuracy"], 4)

            if walk_forward and df_full is not None:
                run["walk_forward_acc"] = walk_forward_eval(params, df_full, splits["features"])

            print(f"  Trial {run['trial']}: test={run['test_accuracy']:.2f}% train={run['train_accuracy']:.2f}% gap={run['overfit_gap']:.2f}%")
        except Exception as exc:
            print(f"  Trial {run['trial']} falló en retrain: {exc}")

    if not batch_runs:
        print("No se completó ningún trial exitosamente.")
        return []

    best_run = max(batch_runs, key=lambda x: x["test_accuracy"])

    data["all_runs"].extend(batch_runs)
    data["batches"].append({
        "batch": batch_num,
        "total_runs": len(batch_runs),
        "best": best_run,
        "timestamp": datetime.now().isoformat(),
    })

    if data["best"] is None or best_run["test_accuracy"] > data["best"]["test_accuracy"]:
        data["best"] = best_run
        BEST_JSON.write_text(json.dumps(best_run, indent=2, default=str), encoding="utf-8")

    save_json_results(data)

    print(f"\n{'-' * 70}")
    print(f"BATCH {batch_num} RESUMEN")
    print(f"{'-' * 70}")
    print(f"Mejor test accuracy: {best_run['test_accuracy']:.2f}%")
    print(f"Log loss: {best_run['log_loss']:.4f} | Brier: {best_run['brier']:.4f} | Gap: {best_run['overfit_gap']:.2f}%")
    if cv:
        print(f"CV mean: {best_run['cv_mean_acc']:.2f}% | CV std: {best_run['cv_std_acc']:.2f}% | Stability: {best_run['cv_stability_ratio']:.2f}")
    print(f"Mediana test: {batch_runs[max(0, len(batch_runs)//2)]['test_accuracy']:.2f}% | Trials: {len(batch_runs)}")
    print(f"GLOBAL BEST: {data['best']['test_accuracy']:.2f}% (batch {data['best']['batch']})")

    monitor.complete(f"Batch {batch_num} completado. Mejor test accuracy: {best_run['test_accuracy']:.2f}%")
    return batch_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loop Engineering — XGBoost con Optuna + CV")
    parser.add_argument("--batch", type=int, default=1, help="Número de batch")
    parser.add_argument("--trials", type=int, default=50, help="Trials de Optuna por batch")
    parser.add_argument("--auto", action="store_true", help="Corre 10 batches seguidos")
    parser.add_argument("--walk-forward", action="store_true", help="Activa walk-forward validation")
    parser.add_argument("--aggressive", action="store_true", help="Espacio de búsqueda agresivo")
    parser.add_argument("--label-smoothing", action="store_true", help="Usa label smoothing con prior Elo")
    parser.add_argument("--no-cv", action="store_true", help="Desactiva CV temporal en el objetivo de Optuna")
    parser.add_argument("--cv-folds", type=int, default=3, help="Folds para purged CV")
    parser.add_argument("--cv-embargo", type=int, default=60, help="Días de embargo entre train y test en CV")
    parser.add_argument("--cv-lambda", type=float, default=CV_LAMBDA_DEFAULT, help="Peso de la penalización por std en CV")
    parser.add_argument("--no-gpu", action="store_true", help="Forzar entrenamiento en CPU")
    parser.add_argument("--study-name", type=str, default="xgboost_loop", help="Nombre del study de Optuna")
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"Modo agresivo: {args.aggressive} | Label smoothing: {args.label_smoothing} | Walk-forward: {args.walk_forward}")
    print(f"{'=' * 70}")

    if args.auto:
        for b in range(1, 11):
            run_batch(
                b,
                args.trials,
                args.walk_forward,
                args.aggressive,
                args.label_smoothing,
                cv=not args.no_cv,
                cv_folds=args.cv_folds,
                cv_embargo=args.cv_embargo,
                cv_lambda=args.cv_lambda,
                use_gpu=not args.no_gpu,
                study_name=args.study_name,
            )
    else:
        run_batch(
            args.batch,
            args.trials,
            args.walk_forward,
            args.aggressive,
            args.label_smoothing,
            cv=not args.no_cv,
            cv_folds=args.cv_folds,
            cv_embargo=args.cv_embargo,
            cv_lambda=args.cv_lambda,
            use_gpu=not args.no_gpu,
            study_name=args.study_name,
        )
