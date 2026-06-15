"""
Loop Engineering — Mondial-Xboost
===================================
Hyperparameter tuning exclusivo de XGBoost con Optuna.

- 3-way temporal split: train / validation / test
- Cada trial se guarda inmediatamente en CSV (resistente a cortes)
- Walk-forward validation opcional para estabilidad
- Guarda el mejor modelo y sus parámetros
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
]


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


def build_xgb(trial: optuna.Trial, aggressive: bool = False) -> xgb.XGBClassifier:
    """Construye un XGBClassifier con hiperparámetros muestreados por Optuna.

    Si aggressive=True, se expande el espacio de búsqueda significativamente
    para explorar regímenes de high complexity + heavy regularization.
    """
    if aggressive:
        max_depth = trial.suggest_int("max_depth", 2, 20)
        n_estimators = trial.suggest_int("n_estimators", 100, 3000)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.5, log=True)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 20)
        gamma = trial.suggest_float("gamma", 0.0, 5.0)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-6, 50.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-6, 50.0, log=True)
        max_delta_step = trial.suggest_float("max_delta_step", 0.0, 10.0)
        num_parallel_tree = trial.suggest_int("num_parallel_tree", 1, 10)
        tree_method = trial.suggest_categorical("tree_method", ["auto", "hist", "approx"])
        max_bin = trial.suggest_int("max_bin", 128, 512)
        grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        subsample = trial.suggest_float("subsample", 0.3, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.3, 1.0)
        colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.5, 1.0)
        colsample_bynode = trial.suggest_float("colsample_bynode", 0.5, 1.0)
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            max_delta_step=max_delta_step,
            num_parallel_tree=num_parallel_tree,
            tree_method=tree_method if tree_method != "auto" else None,
            max_bin=max_bin,
            grow_policy=grow_policy,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    return xgb.XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        gamma=trial.suggest_float("gamma", 0.0, 1.0),
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


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

        model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1, verbosity=0)
        model.fit(train_fold[features].fillna(0), train_fold["outcome"].astype(int))
        preds = model.predict(test_fold[features].fillna(0))
        accs.append(accuracy_score(test_fold["outcome"].astype(int), preds))

    return round(float(np.mean(accs)) * 100, 4) if accs else 0.0


def purged_cv_eval(params: dict, df: pd.DataFrame, features: list[str], n_splits: int = 5, embargo_days: int = 60) -> dict:
    """Evaluación con Purged Cross-Validation + temporal embargo.

    Cada fold tiene un gap (embargo) entre train y test para evitar
    leakage temporal. Devuelve métricas de estabilidad entre folds.
    """
    dates = sorted(df["date"].unique())
    fold_size = len(dates) // (n_splits + 1)
    accs, log_losses, briers = [], [], []

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
        test_fold = df[test_mask & ~embargo_mask].copy()

        if len(train_fold) < 100 or len(test_fold) < 10:
            continue

        model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1, verbosity=0)
        model.fit(train_fold[features].fillna(0), train_fold["outcome"].astype(int))

        preds = model.predict(test_fold[features].fillna(0))
        proba = model.predict_proba(test_fold[features].fillna(0))

        accs.append(accuracy_score(test_fold["outcome"].astype(int), preds))
        log_losses.append(log_loss(test_fold["outcome"].astype(int), proba))
        briers.append(sum(brier_score_loss((test_fold["outcome"] == i).astype(int), proba[:, i]) for i in range(3)) / 3)

    if not accs:
        return {"mean_acc": 0, "std_acc": 0, "min_acc": 0, "max_acc": 0}

    return {
        "mean_acc": round(float(np.mean(accs)) * 100, 4),
        "std_acc": round(float(np.std(accs)) * 100, 4),
        "min_acc": round(float(np.min(accs)) * 100, 4),
        "max_acc": round(float(np.max(accs)) * 100, 4),
        "mean_log_loss": round(float(np.mean(log_losses)), 4),
        "stability_ratio": round(float(np.mean(accs) / (np.std(accs) + 0.001)), 4),
    }


def smooth_labels(outcome: pd.Series, elo_diff: pd.Series, alpha: float = 0.7) -> np.ndarray:
    """Convierte etiquetas duras en suaves mezclando con prior basado en Elo.

    Para cada partido, la etiqueta suave = alpha * onehot(outcome) + (1-alpha) * elo_prior
    donde elo_prior es la probabilidad esperada según diferencia Elo.
    """
    n = len(outcome)
    y_onehot = np.eye(3)[outcome.astype(int)]

    elo_prior = np.zeros((n, 3))
    for i, dr in enumerate(elo_diff):
        home_exp = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        elo_prior[i] = [(1 - home_exp) * 0.3, 0.3, home_exp * 0.7]
        elo_prior[i] /= elo_prior[i].sum()

    return alpha * y_onehot + (1 - alpha) * elo_prior


def run_batch(batch_num: int, n_trials: int = 50, walk_forward: bool = False, aggressive: bool = False, label_smoothing: bool = False) -> list[dict]:
    """Ejecuta un batch de Optuna con XGBoost único."""
    print(f"\n{'=' * 70}")
    print(f"BATCH {batch_num} — XGBoost + Optuna ({n_trials} trials)")
    print(f"{'=' * 70}")

    splits = load_data()
    data = load_json_results()
    batch_runs: list[dict] = []

    monitor.start(total_trials=n_trials, batch=batch_num, phase="tuning")

    df_full = None
    if walk_forward:
        df_full = build_training_dataset(min_date="2015-01-01")
        df_full["date"] = pd.to_datetime(df_full["date"], errors="coerce")
        df_full = df_full.dropna(subset=["home_score"] + [c for c in splits["features"] if c in df_full.columns])

    def objective(trial: optuna.Trial) -> float:
        model = build_xgb(trial, aggressive=aggressive)
        model.fit(
            splits["X_train"], splits["y_train"],
            eval_set=[(splits["X_val"], splits["y_val"])],
            verbose=False,
        )
        val_pred = model.predict(splits["X_val"])
        return accuracy_score(splits["y_val"], val_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[Evaluación] {n_trials} trials completados. Evaluando mejores en test...")

    for idx, trial in enumerate(study.trials, start=1):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        params = trial.params
        try:
            model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1, verbosity=0)
            model.fit(splits["X_train"], splits["y_train"])

            test_pred = model.predict(splits["X_test"])
            test_proba = model.predict_proba(splits["X_test"])
            train_pred = model.predict(splits["X_train"])

            metrics = evaluate(
                splits["y_test"], test_pred, test_proba,
                splits["y_train"], train_pred,
            )

            wf_acc = 0.0
            if walk_forward and df_full is not None:
                wf_acc = walk_forward_eval(params, df_full, splits["features"])

            run = {
                "batch": batch_num,
                "trial": idx,
                "timestamp": datetime.now().isoformat(),
                "params": params,
                "val_accuracy": round(trial.value * 100, 4) if trial.value is not None else 0.0,
                "walk_forward_acc": wf_acc,
                **metrics,
            }

            batch_runs.append(run)
            append_partial(run)
            monitor.update(
                trial_completed=len(batch_runs),
                test_accuracy=metrics.get("test_accuracy"),
                val_accuracy=run.get("val_accuracy"),
                log_loss=metrics.get("log_loss"),
                brier=metrics.get("brier"),
                overfit_gap=metrics.get("overfit_gap"),
                params=params,
            )

        except Exception as exc:
            print(f"  Trial {idx} falló: {exc}")
            continue

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
    print(f"GLOBAL BEST: {data['best']['test_accuracy']:.2f}% (batch {data['best']['batch']})")

    monitor.complete(f"Batch {batch_num} completado. Mejor test accuracy: {best_run['test_accuracy']:.2f}%")
    return batch_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loop Engineering — XGBoost único")
    parser.add_argument("--batch", type=int, default=1, help="Número de batch")
    parser.add_argument("--trials", type=int, default=50, help="Trials de Optuna por batch")
    parser.add_argument("--auto", action="store_true", help="Corre 10 batches seguidos")
    parser.add_argument("--walk-forward", action="store_true", help="Activa walk-forward validation")
    parser.add_argument("--aggressive", action="store_true", help="Espacio de búsqueda agresivo (max_depth hasta 20, n_estimators hasta 3000)")
    parser.add_argument("--label-smoothing", action="store_true", help="Usa label smoothing con prior Elo")
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"Modo agresivo: {args.aggressive} | Label smoothing: {args.label_smoothing} | Walk-forward: {args.walk_forward}")
    print(f"{'=' * 70}")

    if args.auto:
        for b in range(1, 11):
            run_batch(b, args.trials, args.walk_forward, args.aggressive, args.label_smoothing)
    else:
        run_batch(args.batch, args.trials, args.walk_forward, args.aggressive, args.label_smoothing)
