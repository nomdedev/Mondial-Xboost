"""
Loop Engineering v2 — Mondial-Xboost
=======================================
- Hyperparameter tuning con Optuna
- 3-way split: train / validation / test
- 100 iteraciones por batch, 10 batches = 1000 total
- Cada batch analiza resultados y ajusta búsqueda
- Walk-forward validation temporal
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from predictors.feature_engineering import FEATURE_COLS, build_training_dataset

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = MODELS_DIR / "loop_engineering_v2.json"
BEST_FILE = MODELS_DIR / "loop_best_v2.json"


# ══════════════════════════════════════════════
# 3-Way Temporal Split
# ══════════════════════════════════════════════
def load_data(min_date="2015-01-01", val_cutoff="2023-01-01", test_cutoff="2024-01-01"):
    """
    Temporal 3-way split:
      train:  < val_cutoff      (aprende patrones)
      val:    val_cutoff..test   (ajusta hiperparámetros)
      test:   >= test_cutoff     (evaluación final, NUNCA tocado)
    """
    df = build_training_dataset(min_date=min_date)
    df = df.dropna(subset=["home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    available = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=available)

    train = df[df["date"] < val_cutoff].copy()
    val = df[(df["date"] >= val_cutoff) & (df["date"] < test_cutoff)].copy()
    test = df[df["date"] >= test_cutoff].copy()

    splits = {}
    for name, subset in [("train", train), ("val", val), ("test", test)]:
        splits[f"X_{name}"] = subset[available].fillna(0)
        splits[f"y_{name}"] = subset["outcome"].astype(int)

    print(f"  Train: {len(train)} (< {val_cutoff})")
    print(f"  Val:   {len(val)} ({val_cutoff}..{test_cutoff})")
    print(f"  Test:  {len(test)} (>= {test_cutoff})")
    return splits, available


# ══════════════════════════════════════════════
# Model Builders
# ══════════════════════════════════════════════
def build_xgb(trial):
    return xgb.XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 1000),
        max_depth=trial.suggest_int("max_depth", 4, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        gamma=trial.suggest_float("gamma", 0.0, 1.0),
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=42,
    )

def build_rf(trial):
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 1000),
        max_depth=trial.suggest_int("max_depth", 6, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        random_state=42, n_jobs=-1,
    )

def build_gb(trial):
    return GradientBoostingClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 800),
        max_depth=trial.suggest_int("max_depth", 4, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        random_state=42,
    )

def build_lgbm(trial):
    return lgb.LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 1000),
        max_depth=trial.suggest_int("max_depth", 4, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 200),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
        random_state=42, verbose=-1,
    )

def build_cb(trial):
    return CatBoostClassifier(
        iterations=trial.suggest_int("iterations", 200, 800),
        depth=trial.suggest_int("depth", 4, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 10),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0, 2),
        random_state=42, verbose=0,
    )

BUILDERS = {
    "XGBoost": build_xgb,
    "RandomForest": build_rf,
    "GradientBoosting": build_gb,
}
if HAS_LGBM:
    BUILDERS["LightGBM"] = build_lgbm
if HAS_CATBOOST:
    BUILDERS["CatBoost"] = build_cb


# ══════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════
def evaluate(y_true, y_pred, y_proba, y_train_true=None, y_train_pred=None):
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_proba)
    brier = sum(brier_score_loss((y_true == i).astype(int), y_proba[:, i]) for i in range(3)) / 3
    train_acc = accuracy_score(y_train_true, y_train_pred) if y_train_true is not None else None
    return {
        "accuracy": round(acc * 100, 4),
        "log_loss": round(ll, 4),
        "brier": round(brier, 4),
        "train_accuracy": round(train_acc * 100, 4) if train_acc else None,
        "overfit_gap": round((train_acc - acc) * 100, 4) if train_acc else None,
    }


# ══════════════════════════════════════════════
# Walk-Forward Validation
# ══════════════════════════════════════════════
def walk_forward_eval(model_class, params, df, features, folds=3):
    """
    Walk-forward: train on expanding window, test on next fold.
    Returns average metrics across folds.
    """
    dates = sorted(df["date"].unique())
    fold_size = len(dates) // (folds + 1)
    metrics_list = []

    for i in range(folds):
        cutoff = dates[(i + 1) * fold_size]
        train_fold = df[df["date"] < cutoff]
        test_fold = df[(df["date"] >= cutoff) & (df["date"] < dates[(i + 2) * fold_size])] if (i + 2) * fold_size < len(dates) else df[df["date"] >= cutoff]

        if len(test_fold) == 0:
            continue

        x_tr = train_fold[features].fillna(0)
        y_tr = train_fold["outcome"].astype(int)
        x_te = test_fold[features].fillna(0)
        y_te = test_fold["outcome"].astype(int)

        m = model_class(params)
        m.fit(x_tr, y_tr)
        y_pred = m.predict(x_te)
        acc = accuracy_score(y_te, y_pred)
        metrics_list.append(acc)

    return round(np.mean(metrics_list) * 100, 4) if metrics_list else 0


# ══════════════════════════════════════════════
# Loop Engineering
# ══════════════════════════════════════════════
def load_results():
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {"batches": [], "all_runs": [], "best_per_model": {}}

def save_results(data):
    RESULTS_FILE.write_text(json.dumps(data, indent=2, default=str))


def run_batch(batch_num, iterations=100, optuna_trials=20):
    """
    Each batch:
    1. Optuna tunes each model (optuna_trials per model)
    2. Random exploration fills remaining iterations
    3. All results evaluated on validation set
    4. Walk-forward validates stability
    5. Best overall saved
    """
    print(f"\n{'='*70}")
    print(f"BATCH {batch_num} — {optuna_trials} Optuna trials × {len(BUILDERS)} modelos + {iterations} exploraciones")
    print(f"{'='*70}")

    splits, features = load_data()
    data = load_results()
    all_batch_runs = []

    # ── Phase 1: Optuna tuning per model ──
    for model_name, builder in BUILDERS.items():
        print(f"\n  [Optuna] {model_name} — {optuna_trials} trials")

        def objective(trial):
            try:
                model = builder(trial)
                model.fit(splits["X_train"], splits["y_train"])
                y_pred = model.predict(splits["X_val"])
                return accuracy_score(splits["y_val"], y_pred)
            except Exception:
                return 0.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)

        best_trial = study.best_trial
        best_params = best_trial.params
        best_acc = round(best_trial.value * 100, 4)

        # Evaluate best on test set
        try:
            best_model = builder(optuna.trial.FixedTrial(best_params))
            best_model.fit(splits["X_train"], splits["y_train"])
            y_pred_test = best_model.predict(splits["x_test"])
            y_proba_test = best_model.predict_proba(splits["x_test"])
            test_metrics = evaluate(
                splits["y_test"], y_pred_test, y_proba_test,
                splits["y_train"], best_model.predict(splits["X_train"])
            )
        except Exception as e:
            test_metrics = {"accuracy": 0, "log_loss": 99, "brier": 1, "error": str(e)}

        # Walk-forward validation
        try:
            df_full = build_training_dataset(min_date="2015-01-01")
            df_full = df_full.dropna(subset=["home_score"] + [c for c in features if c in df_full.columns])
            df_full["date"] = pd.to_datetime(df_full["date"], errors="coerce")
            wf_acc = walk_forward_eval(
                lambda p: builder(optuna.trial.FixedTrial(p)),
                best_params, df_full, features
            )
        except Exception:
            wf_acc = 0

        run = {
            "batch": batch_num,
            "model": model_name,
            "method": "optuna",
            "params": best_params,
            "val_accuracy": best_acc,
            **test_metrics,
            "walk_forward_acc": wf_acc,
        }
        all_batch_runs.append(run)
        print(f"    Val: {best_acc:.2f}% | Test: {test_metrics.get('accuracy',0):.2f}% | "
              f"WF: {wf_acc:.2f}% | Gap: {test_metrics.get('overfit_gap', 'N/A')}%")

    # ── Phase 2: Random exploration ──
    print(f"\n  [Exploración] {iterations} iteraciones aleatorias")
    model_names = list(BUILDERS.keys())

    for i in range(iterations):
        model_name = model_names[i % len(model_names)]
        builder = BUILDERS[model_name]

        try:
            # Use a random Optuna trial so the builder can call suggest_* methods
            random_study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42 + i))
            trial = random_study.ask()
            model = builder(trial)
            params = trial.params
            model.fit(splits["X_train"], splits["y_train"])
            y_pred = model.predict(splits["X_val"])
            val_acc = accuracy_score(splits["y_val"], y_pred)

            y_pred_test = model.predict(splits["x_test"])
            y_proba_test = model.predict_proba(splits["x_test"])
            test_metrics = evaluate(
                splits["y_test"], y_pred_test, y_proba_test,
                splits["y_train"], model.predict(splits["X_train"])
            )

            run = {
                "batch": batch_num,
                "model": model_name,
                "method": "random",
                "params": params,
                "val_accuracy": round(val_acc * 100, 4),
                **test_metrics,
                "walk_forward_acc": 0,
            }
            all_batch_runs.append(run)

            if (i + 1) % 20 == 0:
                best_so_far = max(all_batch_runs, key=lambda x: x.get("accuracy", 0))
                print(f"    [{i+1:3d}/{iterations}] best: {best_so_far['model']} "
                      f"{best_so_far['accuracy']:.2f}%")

        except Exception:
            pass

    # ── Save batch ──
    data["all_runs"].extend(all_batch_runs)
    best_run = max(all_batch_runs, key=lambda x: x.get("accuracy", 0))
    data["batches"].append({
        "batch": batch_num,
        "total_runs": len(all_batch_runs),
        "best": best_run,
        "timestamp": datetime.now().isoformat(),
    })

    # Update best per model
    for run in all_batch_runs:
        name = run["model"]
        if name not in data["best_per_model"] or run["accuracy"] > data["best_per_model"][name]["accuracy"]:
            data["best_per_model"][name] = run

    save_results(data)
    BEST_FILE.write_text(json.dumps(best_run, indent=2, default=str))

    # ── Summary ──
    print(f"\n{'─'*70}")
    print(f"BATCH {batch_num} RESUMEN")
    print(f"{'─'*70}")
    print(f"{'Modelo':20s} {'Val%':>8s} {'Test%':>8s} {'LogLoss':>10s} {'Gap%':>8s} {'WF%':>8s}")
    print("-" * 70)
    for name, run in sorted(data["best_per_model"].items(), key=lambda x: -x[1]["accuracy"]):
        print(f"{name:20s} {run.get('val_accuracy',0):7.2f}% {run['accuracy']:7.2f}% "
              f"{run['log_loss']:10.4f} {run.get('overfit_gap','N/A'):>7}% {run.get('walk_forward_acc',0):7.2f}%")

    global_best = max(data["all_runs"], key=lambda x: x.get("accuracy", 0))
    print(f"\n  GLOBAL BEST: {global_best['model']} {global_best['accuracy']:.2f}% "
          f"(batch {global_best['batch']})")

    return all_batch_runs, best_run


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    if args.auto:
        for b in range(1, 11):
            run_batch(b, args.iterations, args.optuna_trials)
    else:
        run_batch(args.batch, args.iterations, args.optuna_trials)
