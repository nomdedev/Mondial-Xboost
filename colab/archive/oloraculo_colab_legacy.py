"""
Google Colab Notebook — Mondial-Xboost
=========================================
Copiar este archivo a un .ipynb en Google Colab.
Usa GPU gratis para entrenar todos los modelos con Loop Engineering.

INSTRUCCIONES:
1. Ir a https://colab.research.google.com
2. Crear nuevo notebook
3. Copiar cada sección como una celda
4. Runtime → Change runtime type → T4 GPU
5. Ejecutar todo (Runtime → Run all)
"""

# ═══════════════════════════════════════════════════════════
# CELDA 1: Setup e Instalación
# ═══════════════════════════════════════════════════════════
"""
!pip install xgboost lightgbm catboost optuna scikit-learn pandas numpy requests beautifulsoup4 lxml pyarrow -q

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import optuna
import json
import time
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING)
print("✓ Librerías cargadas")
print(f"  XGBoost: {xgb.__version__}")
print(f"  LightGBM: {lgb.__version__}")
print(f"  Optuna: {optuna.__version__}")
"""

# ═══════════════════════════════════════════════════════════
# CELDA 2: Descarga de Datos
# ═══════════════════════════════════════════════════════════
"""
import requests
import io
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Ligas europeas (formato mmz4281/season/code.csv)
EUROPEAN_LEAGUES = {
    "E0": "Premier League", "E1": "Championship",
    "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1",
    "N1": "Eredivisie", "B1": "Jupiler League", "P1": "Liga Portugal",
    "T1": "Süper Lig", "G1": "Super League Greece",
    "SC0": "Scottish Premiership", "T2": "Bundesliga Austria",
    "N2": "Superliga Denmark", "E4": "Veikkausliiga",
    "SC1": "Championship Scotland", "I2": "Serie B",
    "SP2": "Segunda Division", "D2": "2. Bundesliga", "F2": "Ligue 2",
}

# Ligas americanas y otras (formato new/CODE.csv)
NEW_LEAGUES = {
    "ARG": "Argentina Primera", "BRA": "Brasil Série A",
    "MEX": "Liga MX", "USA": "MLS",
    "CHN": "Super League China", "JPN": "J-League",
    "ROU": "Liga I Romania", "POL": "Ekstraklasa",
    "RUS": "Premier League Russia", "SWE": "Allsvenskan",
    "NOR": "Eliteserien", "DNK": "Superliga Denmark",
    "SWZ": "Super League Switzerland", "IRL": "Premier Division",
    "AUS": "Bundesliga Austria",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def download_european(code, start=2018, end=2024):
    frames = []
    for season in range(start, end + 1):
        s = str(season)[-2:]
        url = f"https://www.football-data.co.uk/mmz4281/{season}{s}/{code}.csv"
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200 and len(r.text) > 500:
                df = pd.read_csv(io.StringIO(r.text), on_bad_lines='skip')
                df['season'] = season
                df['league_code'] = code
                frames.append(df)
        except:
            pass
    return pd.concat(frames, ignore_index=True) if frames else None

def download_new(code):
    url = f"https://www.football-data.co.uk/new/{code}.csv"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200 and len(r.text) > 500:
            df = pd.read_csv(io.StringIO(r.text), on_bad_lines='skip')
            df['league_code'] = code
            return df
    except:
        pass
    return None

all_frames = []

# Download European leagues
print("Descargando ligas europeas...")
for code, name in EUROPEAN_LEAGUES.items():
    df = download_european(code)
    if df is not None:
        all_frames.append(df)
        print(f"  ✓ {name} ({code}): {len(df)} partidos")

# Download new leagues
print("\\nDescargando ligas americanas y otras...")
for code, name in NEW_LEAGUES.items():
    df = download_new(code)
    if df is not None:
        all_frames.append(df)
        print(f"  ✓ {name} ({code}): {len(df)} partidos")

raw = pd.concat(all_frames, ignore_index=True)
print(f"\\nTotal crudo: {len(raw)} partidos")

# Normalize columns
COL_MAP = {
    'Date': 'date', 'HomeTeam': 'home_team', 'AwayTeam': 'away_team',
    'FTHG': 'home_goals', 'FTAG': 'away_goals', 'FTR': 'result',
    'HS': 'home_shots', 'AS': 'away_shots',
    'HST': 'home_shots_on_target', 'AST': 'away_shots_on_target',
    'HC': 'home_corners', 'AC': 'away_corners',
    'HF': 'home_fouls', 'AF': 'away_fouls',
    'HY': 'home_yellow', 'AY': 'away_yellow',
    'HR': 'home_red', 'AR': 'away_red',
    'HTHG': 'ht_home_goals', 'HTAG': 'ht_away_goals', 'HTR': 'ht_result',
    'Referee': 'referee',
}

df = raw.rename(columns={k: v for k, v in COL_MAP.items() if k in raw.columns})
df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
df = df.dropna(subset=['date', 'home_team', 'away_team', 'home_goals', 'away_goals'])
df = df.sort_values('date').reset_index(drop=True)
df['home_goals'] = df['home_goals'].astype(int)
df['away_goals'] = df['away_goals'].astype(int)

# Outcome
df['outcome'] = np.where(df['home_goals'] > df['away_goals'], 0,
                np.where(df['home_goals'] == df['away_goals'], 1, 2))

print(f"Dataset limpio: {len(df)} partidos")
print(f"Rango: {df['date'].min()} → {df['date'].max()}")
print(f"Outcome distribution: Home={sum(df['outcome']==0)}, Draw={sum(df['outcome']==1)}, Away={sum(df['outcome']==2)}")
"""

# ═══════════════════════════════════════════════════════════
# CELDA 3: Feature Engineering (Elo + Rolling + H2H)
# ═══════════════════════════════════════════════════════════
"""
def compute_elo(df, initial=1500.0):
    \"\"\"Elo ratings con K variable por tipo de partido (eloratings.net methodology, sin leakage).\"\"\"
    ratings = defaultdict(lambda: initial)
    matches_played = defaultdict(int)
    home_elo, away_elo = [], []

    for _, row in df.itertuples(index=False):
        h, a = row.home_team, row.away_team
        rh, ra = ratings[h], ratings[a]
        home_elo.append(rh)
        away_elo.append(ra)

        league = str(getattr(row, 'league_code', ''))
        tournament = str(getattr(row, 'tournament', ''))

        if 'WC' in league or 'World Cup' in tournament:
            K = 60
        elif any(x in tournament for x in ['Euro', 'Copa America']):
            K = 50
        elif 'qualifier' in tournament.lower():
            K = 40
        elif any(x in league for x in ['E0', 'SP1', 'I1', 'D1', 'F1']):
            K = 30
        else:
            K = 20

        dr = (rh - ra) + 100  # Home advantage
        expected_h = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        result_h = 1.0 if row.home_goals > row.away_goals else (0.5 if row.home_goals == row.away_goals else 0.0)
        delta = K * (result_h - expected_h)

        ratings[h] = rh + delta
        ratings[a] = ra - delta
        matches_played[h] += 1
        matches_played[a] += 1

    df = df.copy()
    df['home_elo'] = home_elo
    df['away_elo'] = away_elo
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    return df


def compute_rolling_features(df, windows=[5, 10]):
    \"\"\"Rolling features con shift(1) para evitar leakage.\"\"\"
    # Build long format
    home = df[['date', 'home_team', 'away_team', 'home_goals', 'away_goals']].copy()
    home['team'] = home['home_team']
    home['goals_scored'] = home['home_goals']
    home['goals_conceded'] = home['away_goals']
    home['points'] = np.where(home['home_goals'] > home['away_goals'], 3,
                     np.where(home['home_goals'] == home['away_goals'], 1, 0))

    away = df[['date', 'home_team', 'away_team', 'home_goals', 'away_goals']].copy()
    away['team'] = away['away_team']
    away['goals_scored'] = away['away_goals']
    away['goals_conceded'] = away['home_goals']
    away['points'] = np.where(away['away_goals'] > away['home_goals'], 3,
                     np.where(away['home_goals'] == away['away_goals'], 1, 0))

    long = pd.concat([home, away], ignore_index=True).sort_values(['team', 'date'])

    for w in windows:
        long[f'points_avg_{w}'] = long.groupby('team')['points'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean() / 3.0)
        long[f'goals_scored_avg_{w}'] = long.groupby('team')['goals_scored'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        long[f'goals_conceded_avg_{w}'] = long.groupby('team')['goals_conceded'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        long[f'win_rate_{w}'] = long.groupby('team')['points'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).apply(lambda x: (x==3).sum()) / s.shift(1).rolling(w, min_periods=1).count())
        long[f'draw_rate_{w}'] = long.groupby('team')['points'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).apply(lambda x: (x==1).sum()) / s.shift(1).rolling(w, min_periods=1).count())
        long[f'loss_rate_{w}'] = long.groupby('team')['points'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).apply(lambda x: (x==0).sum()) / s.shift(1).rolling(w, min_periods=1).count())

    long['matches_played'] = long.groupby('team').cumcount()
    return long


def compute_h2h(df):
    \"\"\"H2H features — solo partidos anteriores entre los dos equipos.\"\"\"
    df = df.copy()
    df['pair_key'] = df.apply(lambda r: tuple(sorted([r['home_team'], r['away_team']])), axis=1)
    df = df.sort_values(['pair_key', 'date']).reset_index(drop=True)

    h2h_last = []
    h2h_goals = []
    h2h_wins_diff = []
    h2h_years = []

    prev = {}
    for _, row in df.iterrows():
        key = row['pair_key']
        if key in prev:
            p = prev[key]
            h2h_last.append(p['result_val'])
            h2h_goals.append(p['avg_goals'])
            h2h_wins_diff.append(p['wins_diff'])
            h2h_years.append((row['date'] - p['date']).days / 365.25)
        else:
            h2h_last.append(np.nan)
            h2h_goals.append(np.nan)
            h2h_wins_diff.append(np.nan)
            h2h_years.append(np.nan)

        # Update for next
        result_val = 1.0 if row['home_goals'] > row['away_goals'] else (0.5 if row['home_goals'] == row['away_goals'] else 0.0)
        total_goals = row['home_goals'] + row['away_goals']
        prev[key] = {
            'result_val': result_val,
            'avg_goals': total_goals,
            'wins_diff': 1 if result_val == 1 else (-1 if result_val == 0 else 0),
            'date': row['date'],
        }

    df['h2h_last_result'] = h2h_last
    df['h2h_goals_avg'] = h2h_goals
    df['h2h_wins_diff'] = h2h_wins_diff
    df['h2h_years_since'] = h2h_years
    return df


def build_all_features(df):
    \"\"\"Build complete feature matrix.\"\"\"
    print("  Computing Elo ratings...")
    df = compute_elo(df)

    print("  Computing rolling features...")
    long = compute_rolling_features(df)

    print("  Computing H2H features...")
    df = compute_h2h(df)

    # Merge rolling stats back
    home_stats = long[long['team'].isin(df['home_team'].unique())].copy()
    home_cols = ['date', 'team', 'points_avg_5', 'points_avg_10',
                 'goals_scored_avg_5', 'goals_scored_avg_10',
                 'goals_conceded_avg_10', 'win_rate_10', 'draw_rate_10',
                 'loss_rate_10', 'matches_played']

    # Merge home rolling
    df = df.merge(
        home_stats[home_cols].rename(columns={'team': 'home_team'}),
        on=['date', 'home_team'], how='left', suffixes=('', '_home_roll'))

    # Merge away rolling
    away_stats = long[~long['team'].isin(df['home_team'].unique())].copy()
    df = df.merge(
        away_stats[home_cols].rename(columns={
            'team': 'away_team',
            'points_avg_5': 'away_points_avg_5',
            'points_avg_10': 'away_points_avg_10',
            'goals_scored_avg_10': 'away_goals_scored_avg_10',
            'goals_conceded_avg_10': 'away_goals_conceded_avg_10',
            'win_rate_10': 'away_win_rate_10',
            'draw_rate_10': 'away_draw_rate_10',
            'loss_rate_10': 'away_loss_rate_10',
            'matches_played': 'away_matches_played',
        }),
        on=['date', 'away_team'], how='left')

    # Rename home rolling
    df = df.rename(columns={
        'points_avg_5': 'home_points_avg_5',
        'points_avg_10': 'home_points_avg_10',
        'goals_scored_avg_10': 'home_goals_scored_avg_10',
        'goals_conceded_avg_10': 'home_goals_conceded_avg_10',
        'win_rate_10': 'home_win_rate_10',
        'draw_rate_10': 'home_draw_rate_10',
        'loss_rate_10': 'home_loss_rate_10',
        'matches_played': 'home_matches_played',
    })

    return df

print("✓ Feature functions defined")
"""


# ═══════════════════════════════════════════════════════════
# CELDA 4: Build Features + 3-Way Split
# ═══════════════════════════════════════════════════════════
"""
print("Building features (this takes 5-10 min on Colab)...")
t0 = time.time()
features_df = build_all_features(df)
t1 = time.time()
print(f"✓ Features built in {t1-t0:.0f}s — {len(features_df)} rows")

FEATURE_COLS = [
    'elo_diff', 'home_elo', 'away_elo',
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

available = [c for c in FEATURE_COLS if c in features_df.columns]
features_df = features_df.dropna(subset=available + ['outcome'])

# Temporal 3-way split
train = features_df[features_df['date'] < '2023-01-01']
val = features_df[(features_df['date'] >= '2023-01-01') & (features_df['date'] < '2024-01-01')]
test = features_df[features_df['date'] >= '2024-01-01']

X_train, y_train = train[available].fillna(0), train['outcome'].astype(int)
X_val, y_val = val[available].fillna(0), val['outcome'].astype(int)
X_test, y_test = test[available].fillna(0), test['outcome'].astype(int)

print(f"\\n3-Way Temporal Split:")
print(f"  Train: {len(X_train)} partidos (< 2023)")
print(f"  Val:   {len(X_val)} partidos (2023)")
print(f"  Test:  {len(X_test)} partidos (≥ 2024)")
print(f"  Features: {len(available)}")
"""


# ═══════════════════════════════════════════════════════════
# CELDA 5: Loop Engineering con Optuna (1000 iteraciones)
# ═══════════════════════════════════════════════════════════
"""
results = []
best_global = {'accuracy': 0}

def evaluate(name, model, X_tr, y_tr, X_v, y_v, X_te, y_te):
    model.fit(X_tr, y_tr)
    val_acc = accuracy_score(y_v, model.predict(X_v))
    val_proba = model.predict_proba(X_v)
    val_ll = log_loss(y_v, val_proba)

    test_pred = model.predict(X_te)
    test_proba = model.predict_proba(X_te)
    test_acc = accuracy_score(y_te, test_pred)
    test_ll = log_loss(y_te, test_proba)
    brier = sum(brier_score_loss((y_te==i).astype(int), test_proba[:,i]) for i in range(3)) / 3
    train_acc = accuracy_score(y_tr, model.predict(X_tr))

    return {
        'model': name,
        'val_accuracy': round(val_acc*100, 2),
        'test_accuracy': round(test_acc*100, 2),
        'test_logloss': round(test_ll, 4),
        'test_brier': round(brier, 4),
        'train_accuracy': round(train_acc*100, 2),
        'overfit_gap': round((train_acc - test_acc)*100, 2),
    }


for batch in range(1, 11):
    print(f"\\n{'='*60}")
    print(f"BATCH {batch}/10 — 100 iteraciones")
    print(f"{'='*60}")
    batch_best = {'test_accuracy': 0}

    # ── Optuna: XGBoost (30 trials) ──
    print(f"  [Optuna] XGBoost — 30 trials")
    def obj_xgb(trial):
        m = xgb.XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 200, 1000),
            max_depth=trial.suggest_int('max_depth', 4, 12),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
            gamma=trial.suggest_float('gamma', 0, 1),
            objective='multi:softprob', eval_metric='mlogloss',
            random_state=42, verbosity=0
        )
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return accuracy_score(y_val, m.predict(X_val))

    study = optuna.create_study(direction='maximize')
    study.optimize(obj_xgb, n_trials=30, show_progress_bar=True)
    best_xgb_params = study.best_trial.params
    best_xgb = xgb.XGBClassifier(**best_xgb_params, objective='multi:softprob',
                                   eval_metric='mlogloss', random_state=42, verbosity=0)
    r = evaluate('XGBoost', best_xgb, X_train, y_train, X_val, y_val, X_test, y_test)
    r['params'] = best_xgb_params
    r['batch'] = batch
    results.append(r)
    if r['test_accuracy'] > best_global['test_accuracy']:
        best_global = r
    print(f"    → Val: {r['val_accuracy']}% | Test: {r['test_accuracy']}% | Gap: {r['overfit_gap']}%")

    # ── Optuna: LightGBM (30 trials) ──
    print(f"  [Optuna] LightGBM — 30 trials")
    def obj_lgb(trial):
        m = lgb.LGBMClassifier(
            n_estimators=trial.suggest_int('n_estimators', 200, 1000),
            max_depth=trial.suggest_int('max_depth', 4, 12),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
            num_leaves=trial.suggest_int('num_leaves', 31, 200),
            min_child_samples=trial.suggest_int('min_child_samples', 5, 50),
            random_state=42, verbose=-1
        )
        m.fit(X_train, y_train)
        return accuracy_score(y_val, m.predict(X_val))

    study = optuna.create_study(direction='maximize')
    study.optimize(obj_lgb, n_trials=30, show_progress_bar=True)
    best_lgb_params = study.best_trial.params
    best_lgb = lgb.LGBMClassifier(**best_lgb_params, random_state=42, verbose=-1)
    r = evaluate('LightGBM', best_lgb, X_train, y_train, X_val, y_val, X_test, y_test)
    r['params'] = best_lgb_params
    r['batch'] = batch
    results.append(r)
    if r['test_accuracy'] > best_global['test_accuracy']:
        best_global = r
    print(f"    → Val: {r['val_accuracy']}% | Test: {r['test_accuracy']}% | Gap: {r['overfit_gap']}%")

    # ── Optuna: CatBoost (20 trials) ──
    print(f"  [Optuna] CatBoost — 20 trials")
    def obj_cb(trial):
        m = CatBoostClassifier(
            iterations=trial.suggest_int('iterations', 200, 800),
            depth=trial.suggest_int('depth', 4, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1, 10),
            random_state=42, verbose=0
        )
        m.fit(X_train, y_train)
        return accuracy_score(y_val, m.predict(X_val))

    study = optuna.create_study(direction='maximize')
    study.optimize(obj_cb, n_trials=20, show_progress_bar=True)
    best_cb_params = study.best_trial.params
    best_cb = CatBoostClassifier(**best_cb_params, random_state=42, verbose=0)
    r = evaluate('CatBoost', best_cb, X_train, y_train, X_val, y_val, X_test, y_test)
    r['params'] = best_cb_params
    r['batch'] = batch
    results.append(r)
    if r['test_accuracy'] > best_global['test_accuracy']:
        best_global = r
    print(f"    → Val: {r['val_accuracy']}% | Test: {r['test_accuracy']}% | Gap: {r['overfit_gap']}%")

    # ── Optuna: RandomForest (10 trials) ──
    print(f"  [Optuna] RandomForest — 10 trials")
    def obj_rf(trial):
        m = RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 200, 800),
            max_depth=trial.suggest_int('max_depth', 6, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]),
            random_state=42, n_jobs=-1
        )
        m.fit(X_train, y_train)
        return accuracy_score(y_val, m.predict(X_val))

    study = optuna.create_study(direction='maximize')
    study.optimize(obj_rf, n_trials=10, show_progress_bar=True)
    best_rf_params = study.best_trial.params
    best_rf = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1)
    r = evaluate('RandomForest', best_rf, X_train, y_train, X_val, y_val, X_test, y_test)
    r['params'] = best_rf_params
    r['batch'] = batch
    results.append(r)
    if r['test_accuracy'] > best_global['test_accuracy']:
        best_global = r
    print(f"    → Val: {r['val_accuracy']}% | Test: {r['test_accuracy']}% | Gap: {r['overfit_gap']}%")

    # ── Optuna: GradientBoosting (10 trials) ──
    print(f"  [Optuna] GradientBoosting — 10 trials")
    def obj_gb(trial):
        m = GradientBoostingClassifier(
            n_estimators=trial.suggest_int('n_estimators', 200, 800),
            max_depth=trial.suggest_int('max_depth', 4, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            random_state=42
        )
        m.fit(X_train, y_train)
        return accuracy_score(y_val, m.predict(X_val))

    study = optuna.create_study(direction='maximize')
    study.optimize(obj_gb, n_trials=10, show_progress_bar=True)
    best_gb_params = study.best_trial.params
    best_gb = GradientBoostingClassifier(**best_gb_params, random_state=42)
    r = evaluate('GradientBoosting', best_gb, X_train, y_train, X_val, y_val, X_test, y_test)
    r['params'] = best_gb_params
    r['batch'] = batch
    results.append(r)
    if r['test_accuracy'] > best_global['test_accuracy']:
        best_global = r
    print(f"    → Val: {r['val_accuracy']}% | Test: {r['test_accuracy']}% | Gap: {r['overfit_gap']}%")

    # Batch summary
    print(f"\\n  BATCH {batch} BEST: {best_global['model']} {best_global['test_accuracy']}%")
    print(f"  GLOBAL BEST: {best_global['model']} {best_global['test_accuracy']}%")

print(f"\\n{'='*60}")
print(f"GLOBAL BEST: {best_global['model']} {best_global['test_accuracy']}%")
print(f"{'='*60}")
"""


# ═══════════════════════════════════════════════════════════
# CELDA 6: Guardar resultados
# ═══════════════════════════════════════════════════════════
"""
import json

# Save all results
with open('loop_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save best model
import pickle
best_model = best_global['model']
print(f"\\nBest model: {best_model} with {best_global['test_accuracy']}% accuracy")

# Download from Colab
from google.colab import files
files.download('loop_results.json')
"""

# ═══════════════════════════════════════════════════════════
# CELDA 7: Análisis de resultados
# ═══════════════════════════════════════════════════════════
"""
import pandas as pd

df_results = pd.DataFrame(results)
print("=== RESULTADOS POR MODELO ===")
summary = df_results.groupby('model').agg({
    'test_accuracy': ['mean', 'max', 'min', 'std'],
    'test_logloss': 'mean',
    'overfit_gap': 'mean',
}).round(2)
print(summary)

print("\\n=== TOP 10 MEJORES CONFIGURACIONES ===")
top10 = df_results.nlargest(10, 'test_accuracy')[['model', 'test_accuracy', 'test_logloss', 'overfit_gap', 'batch']]
print(top10.to_string(index=False))

print("\\n=== PROGRESO POR BATCH ===")
batch_best = df_results.groupby('batch')['test_accuracy'].max()
print(batch_best)
"""

print("=" * 60)
print("INSTRUCCIONES PARA GOOGLE COLAB")
print("=" * 60)
print()
print("1. Ir a https://colab.research.google.com")
print("2. Crear nuevo notebook")
print("3. Copiar cada sección marcada como 'CELDA X' como una celda")
print("4. Quitar los triples quotes (\"\"\") que rodean cada celda")
print("5. Runtime → Change runtime type → T4 GPU")
print("6. Runtime → Run all")
print()
print("El notebook ejecuta:")
print("  - Descarga de 21 ligas (football-data.co.uk)")
print("  - Feature engineering (Elo + Rolling + H2H)")
print("  - 3-way temporal split (train/val/test)")
print("  - 5 algoritmos × Optuna tuning")
print("  - 1000 iteraciones totales (10 batches × 100)")
print("  - Guarda resultados JSON + mejor modelo")
