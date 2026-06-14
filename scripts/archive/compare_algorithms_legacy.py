"""
ML Model Comparison + Loop Engineering — Mondial-Xboost
=========================================================
Compara todos los algoritmos: RF, GBM, XGBoost, LightGBM, CatBoost.
Usa temporal split (NO index split) para evitar leakage.
Loop engineering: cada iteración ajusta hiperparámetros y re-entrena.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from predictors.feature_engineering import FEATURE_COLS, build_training_dataset

# Try importing optional libraries
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


def load_data(min_date="2015-01-01", test_cutoff="2024-01-01"):
    """Load and split data temporally (NO index split)."""
    print(f"Loading data (min_date={min_date}, test_cutoff={test_cutoff})...")
    df = build_training_dataset(min_date=min_date)
    df = df.dropna(subset=["home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    available = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=available)

    train = df[df["date"] < test_cutoff].copy()
    test = df[df["date"] >= test_cutoff].copy()

    x_train = train[available].fillna(0)
    y_train = train["outcome"].astype(int)
    x_test = test[available].fillna(0)
    y_test = test["outcome"].astype(int)

    print(f"  Train: {len(train)} partidos (< {test_cutoff})")
    print(f"  Test:  {len(test)} partidos (>= {test_cutoff})")
    print(f"  Features: {len(available)}")
    return x_train, y_train, x_test, y_test, available


def evaluate(name, model, x_test, y_test, x_train=None, y_train=None):
    """Evaluate a model and return metrics."""
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)
    brier = sum(brier_score_loss((y_test == i).astype(int), y_proba[:, i]) for i in range(3)) / 3

    train_acc = None
    if x_train is not None:
        train_pred = model.predict(x_train)
        train_acc = accuracy_score(y_train, train_pred)

    result = {
        "name": name,
        "accuracy": round(acc * 100, 2),
        "log_loss": round(ll, 4),
        "brier": round(brier, 4),
        "train_accuracy": round(train_acc * 100, 2) if train_acc else None,
        "overfit_gap": round((train_acc - acc) * 100, 2) if train_acc else None,
    }
    return result


def run_comparison():
    """Run all algorithms and compare."""
    x_train, y_train, x_test, y_test, features = load_data()
    results = []

    # ── 1. Random Forest ──
    print("\n[1/5] Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1
    )
    rf.fit(x_train, y_train)
    results.append(evaluate("RandomForest", rf, x_test, y_test, x_train, y_train))
    print(f"  → {results[-1]['accuracy']}% acc, {results[-1]['log_loss']} logloss")

    # ── 2. Gradient Boosting (sklearn) ──
    print("\n[2/5] Gradient Boosting (sklearn)...")
    gb = GradientBoostingClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, min_samples_split=5, random_state=42
    )
    gb.fit(x_train, y_train)
    results.append(evaluate("GradientBoosting", gb, x_test, y_test, x_train, y_train))
    print(f"  → {results[-1]['accuracy']}% acc, {results[-1]['log_loss']} logloss")

    # ── 3. XGBoost ──
    print("\n[3/5] XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", eval_metric="mlogloss", random_state=42
    )
    xgb_model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
    results.append(evaluate("XGBoost", xgb_model, x_test, y_test, x_train, y_train))
    print(f"  → {results[-1]['accuracy']}% acc, {results[-1]['log_loss']} logloss")

    # ── 4. LightGBM ──
    if HAS_LGBM:
        print("\n[4/5] LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1
        )
        lgb_model.fit(x_train, y_train)
        results.append(evaluate("LightGBM", lgb_model, x_test, y_test, x_train, y_train))
        print(f"  → {results[-1]['accuracy']}% acc, {results[-1]['log_loss']} logloss")
    else:
        print("\n[4/5] LightGBM... SKIP (not installed)")

    # ── 5. CatBoost ──
    if HAS_CATBOOST:
        print("\n[5/5] CatBoost...")
        cb_model = CatBoostClassifier(
            iterations=500, depth=8, learning_rate=0.05,
            random_state=42, verbose=0
        )
        cb_model.fit(x_train, y_train)
        results.append(evaluate("CatBoost", cb_model, x_test, y_test, x_train, y_train))
        print(f"  → {results[-1]['accuracy']}% acc, {results[-1]['log_loss']} logloss")
    else:
        print("\n[5/5] CatBoost... SKIP (not installed)")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("COMPARACIÓN DE ALGORITMOS (temporal split, sin data leakage)")
    print("=" * 70)
    print(f"{'Modelo':20s} {'Acc%':>8s} {'LogLoss':>10s} {'Brier':>8s} {'Train%':>8s} {'Gap':>6s} {'Status':>8s}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: -x["accuracy"]):
        status = "PASS" if r["accuracy"] >= 85 else "NEEDS WORK"
        gap = f"{r['overfit_gap']:.1f}" if r["overfit_gap"] else "N/A"
        print(f"{r['name']:20s} {r['accuracy']:7.2f}% {r['log_loss']:10.4f} {r['brier']:8.4f} "
              f"{r.get('train_accuracy', 'N/A'):>7}% {gap:>5}% {status:>8s}")

    # Save results
    out_dir = Path(__file__).parent.parent / "models"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResultados guardados en models/comparison_results.json")

    return results


if __name__ == "__main__":
    run_comparison()
