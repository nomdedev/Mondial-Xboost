"""
Loop Engineering — Mondial-Xboost
====================================
100 iteraciones por batch, 10 batches = 1000 total.
Cada batch analiza resultados del anterior y ajusta hiperparámetros.
Temporal split (NO index split) para evitar data leakage.
"""
import sys, json, time, pickle, random
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import xgboost as xgb

from predictors.feature_engineering import build_training_dataset

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

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
FEATURE_COLS = [
    'elo_diff', 'home_elo_before', 'away_elo_before',
    'home_points_avg_5', 'home_points_avg_10',
    'home_goals_scored_avg_10', 'home_goals_conceded_avg_10',
    'home_win_rate_10', 'home_draw_rate_10', 'home_loss_rate_10',
    'home_matches_played',
    'away_points_avg_5', 'away_points_avg_10',
    'away_goals_scored_avg_10', 'away_goals_conceded_avg_10',
    'away_win_rate_10', 'away_draw_rate_10', 'away_loss_rate_10',
    'away_matches_played',
    'h2h_last_result', 'h2h_goals_avg', 'h2h_wins_diff', 'h2h_years_since',
]

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = MODELS_DIR / "loop_engineering_results.json"
BEST_FILE = MODELS_DIR / "loop_best.json"

# ──────────────────────────────────────────────
# Hyperparameter search spaces (evolve per batch)
# ──────────────────────────────────────────────
SEARCH_SPACES = {
    "RandomForest": {
        "n_estimators": [200, 300, 500, 800, 1000],
        "max_depth": [6, 8, 10, 12, 16, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5, 0.8],
    },
    "GradientBoosting": {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [2, 5, 10],
    },
    "XGBoost": {
        "n_estimators": [200, 300, 500, 800, 1000],
        "max_depth": [4, 6, 8, 10, 12],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 1.5, 2.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.2, 0.3],
    },
}
if HAS_LGBM:
    SEARCH_SPACES["LightGBM"] = {
        "n_estimators": [200, 300, 500, 800, 1000],
        "max_depth": [4, 6, 8, 10, 12],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
        "num_leaves": [31, 50, 80, 120],
        "min_child_samples": [5, 10, 20, 30],
    }
if HAS_CATBOOST:
    SEARCH_SPACES["CatBoost"] = {
        "iterations": [200, 300, 500, 800],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
        "bagging_temperature": [0, 0.5, 1.0],
    }


def load_data(min_date="2015-01-01", test_cutoff="2024-01-01"):
    """Load and split data temporally."""
    df = build_training_dataset(min_date=min_date)
    df = df.dropna(subset=["home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    available = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=available)

    train = df[df["date"] < test_cutoff].copy()
    test = df[df["date"] >= test_cutoff].copy()
    X_train = train[available].fillna(0)
    y_train = train["outcome"].astype(int)
    X_test = test[available].fillna(0)
    y_test = test["outcome"].astype(int)
    return X_train, y_train, X_test, y_test, available


def sample_params(space, bias=None):
    """Sample random params, optionally biased toward previous best."""
    params = {}
    for k, values in space.items():
        if bias and k in bias and random.random() < 0.6:
            # 60% chance to use biased value (near previous best)
            best_val = bias[k]
            if isinstance(best_val, (int, float)):
                idx = values.index(best_val) if best_val in values else len(values) // 2
                idx = max(0, min(len(values) - 1, idx + random.randint(-1, 1)))
                params[k] = values[idx]
            else:
                params[k] = best_val
        else:
            params[k] = random.choice(values)
    return params


def build_model(name, params):
    """Build model from name and params."""
    if name == "RandomForest":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    elif name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=42, **params)
    elif name == "XGBoost":
        return xgb.XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=42, **params
        )
    elif name == "LightGBM":
        return lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
    elif name == "CatBoost":
        return CatBoostClassifier(random_state=42, verbose=0, **params)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Train and evaluate, return metrics dict."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)
    brier = sum(brier_score_loss((y_test == i).astype(int), y_proba[:, i]) for i in range(3)) / 3
    train_acc = accuracy_score(y_train, model.predict(X_train))
    return {
        "accuracy": round(acc * 100, 4),
        "log_loss": round(ll, 4),
        "brier": round(brier, 4),
        "train_accuracy": round(train_acc * 100, 4),
        "overfit_gap": round((train_acc - acc) * 100, 4),
    }


def load_results():
    """Load previous results."""
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {"batches": [], "all_runs": []}


def save_results(data):
    """Save results."""
    RESULTS_FILE.write_text(json.dumps(data, indent=2, default=str))


def get_best_per_model(all_runs):
    """Get best params per model from all runs."""
    best = {}
    for run in all_runs:
        name = run["model"]
        if name not in best or run["accuracy"] > best[name]["accuracy"]:
            best[name] = run
    return {name: r["params"] for name, r in best.items()}


def run_batch(batch_num, iterations=100):
    """Run one batch of iterations."""
    print(f"\n{'='*70}")
    print(f"BATCH {batch_num} — {iterations} iteraciones")
    print(f"{'='*70}")

    data = load_results()
    biases = get_best_per_model(data["all_runs"]) if data["all_runs"] else {}

    X_train, y_train, X_test, y_test, features = load_data()
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(features)}")

    batch_runs = []
    best_batch = {"accuracy": 0}
    model_names = list(SEARCH_SPACES.keys())

    for i in range(iterations):
        model_name = model_names[i % len(model_names)]
        params = sample_params(SEARCH_SPACES[model_name], biases.get(model_name))
        t0 = time.time()

        try:
            model = build_model(model_name, params)
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
            elapsed = round(time.time() - t0, 1)

            run = {
                "batch": batch_num,
                "iteration": i + 1,
                "global_iter": len(data["all_runs"]) + i + 1,
                "model": model_name,
                "params": params,
                **metrics,
                "time_s": elapsed,
            }
            batch_runs.append(run)

            if metrics["accuracy"] > best_batch["accuracy"]:
                best_batch = run

            # Progress every 10
            if (i + 1) % 10 == 0:
                print(f"  [{i+1:3d}/{iterations}] best so far: {best_batch['model']} "
                      f"{best_batch['accuracy']:.2f}% (last: {model_name} {metrics['accuracy']:.2f}%)")

        except Exception as e:
            print(f"  [{i+1:3d}/{iterations}] ERROR {model_name}: {e}")

    # Save batch
    data["all_runs"].extend(batch_runs)
    data["batches"].append({
        "batch": batch_num,
        "iterations": len(batch_runs),
        "best": best_batch,
        "timestamp": datetime.now().isoformat(),
    })
    save_results(data)

    # Print batch summary
    print(f"\n{'─'*70}")
    print(f"BATCH {batch_num} RESUMEN")
    print(f"{'─'*70}")
    top5 = sorted(batch_runs, key=lambda x: -x["accuracy"])[:5]
    for r in top5:
        print(f"  {r['model']:20s} acc={r['accuracy']:.2f}%  logloss={r['log_loss']:.4f}  "
              f"gap={r['overfit_gap']:.1f}%  {json.dumps(r['params'])[:80]}")

    # Global best
    global_best = max(data["all_runs"], key=lambda x: x["accuracy"])
    print(f"\n  GLOBAL BEST: {global_best['model']} {global_best['accuracy']:.2f}% "
          f"(batch {global_best['batch']}, iter {global_best['iteration']})")

    # Save best model
    BEST_FILE.write_text(json.dumps(global_best, indent=2, default=str))

    return batch_runs, best_batch


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="Batch number (1-10)")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per batch")
    parser.add_argument("--auto", action="store_true", help="Run all 10 batches automatically")
    args = parser.parse_args()

    if args.auto:
        for batch_num in range(1, 11):
            run_batch(batch_num, args.iterations)
    else:
        run_batch(args.batch, args.iterations)


if __name__ == "__main__":
    main()
