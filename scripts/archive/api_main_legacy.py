from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
import json
from pathlib import Path
import joblib

from predictors.feature_engineering import build_training_dataset, load_historical_results
from predictors.xgboost_predictor import XGBoostPredictor

app = FastAPI(title="Mondial-Xboost API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class TrainRequest(BaseModel):
    model_name: str = "outcome"
    min_date: str = "2018-01-01"
    test_size: float = 0.2
    hyperparams: Optional[Dict[str, Any]] = None

class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    date: Optional[str] = None
    league: Optional[str] = None

class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    log_loss: float
    brier_score: float
    trained_at: str
    dataset_size: int
    feature_importance: List[Dict[str, float]]

class TrainingJob(BaseModel):
    job_id: str
    status: str
    progress: float
    metrics: Optional[ModelMetrics] = None
    error: Optional[str] = None

# State
training_jobs: Dict[str, TrainingJob] = {}
models_cache: Dict[str, Any] = {}

@app.get("/")
def root():
    return {"status": "ok", "service": "Mondial-Xboost API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": len(models_cache)}

@app.post("/train")
def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_jobs[job_id] = TrainingJob(
        job_id=job_id,
        status="queued",
        progress=0.0
    )
    
    background_tasks.add_task(_train_async, job_id, request)
    
    return {"job_id": job_id, "status": "queued"}

def _train_async(job_id: str, request: TrainRequest):
    try:
        training_jobs[job_id].status = "running"
        training_jobs[job_id].progress = 0.1
        
        # Load data
        train = build_training_dataset(min_date=request.min_date)
        training_jobs[job_id].progress = 0.3
        
        # Train
        predictor = XGBoostPredictor()
        feature_cols = [
            'elo_diff', 'home_points_avg_5', 'home_points_avg_10',
            'home_goals_scored_avg_10', 'home_goals_conceded_avg_10',
            'home_win_rate_10', 'home_draw_rate_10', 'home_loss_rate_10',
            'away_points_avg_5', 'away_points_avg_10',
            'away_goals_scored_avg_10', 'away_goals_conceded_avg_10',
            'away_win_rate_10', 'away_draw_rate_10', 'away_loss_rate_10',
            'h2h_last_result', 'h2h_goals_avg', 'h2h_wins_diff', 'h2h_years_since',
            'home_matches_played', 'away_matches_played'
        ]
        
        train_clean = train.dropna(subset=feature_cols + ['outcome'])
        split_idx = int(len(train_clean) * (1 - request.test_size))
        train_df = train_clean.iloc[:split_idx]
        test_df = train_clean.iloc[split_idx:]
        
        X_train = train_df[feature_cols]
        y_train = train_df['outcome']
        X_test = test_df[feature_cols]
        y_test = test_df['outcome']
        
        training_jobs[job_id].progress = 0.5
        
        hyperparams = request.hyperparams or {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        import xgboost as xgb
        model = xgb.XGBClassifier(**hyperparams, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        
        training_jobs[job_id].progress = 0.8
        
        # Metrics
        from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob)
        
        # Brier score (multi-class)
        y_test_bin = pd.get_dummies(y_test).values
        brier = np.mean((y_prob - y_test_bin) ** 2)
        
        # Feature importance
        importance = []
        for feat, imp in zip(feature_cols, model.feature_importances_):
            importance.append({"feature": feat, "importance": float(imp)})
        importance.sort(key=lambda x: x['importance'], reverse=True)
        
        # Save model
        model_path = f"models/xgboost_{request.model_name}_{job_id}.pkl"
        joblib.dump(model, model_path)
        models_cache[request.model_name] = model
        
        metrics = ModelMetrics(
            model_name=request.model_name,
            accuracy=acc,
            log_loss=ll,
            brier_score=brier,
            trained_at=datetime.now().isoformat(),
            dataset_size=len(train_clean),
            feature_importance=importance
        )
        
        training_jobs[job_id].status = "completed"
        training_jobs[job_id].progress = 1.0
        training_jobs[job_id].metrics = metrics
        
        # Save metrics to file
        metrics_file = Path(f"models/metrics_{job_id}.json")
        metrics_file.write_text(json.dumps(metrics.dict(), indent=2, default=str))
        
    except Exception as e:
        training_jobs[job_id].status = "failed"
        training_jobs[job_id].error = str(e)
        raise

@app.get("/train/{job_id}")
def get_training_status(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return training_jobs[job_id]

@app.get("/models")
def list_models():
    models_dir = Path("models")
    if not models_dir.exists():
        return {"models": []}
    
    models = []
    for f in models_dir.glob("*.pkl"):
        models.append({
            "name": f.stem,
            "path": str(f),
            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat()
        })
    return {"models": models}

@app.post("/predict")
def predict(request: PredictRequest):
    if "outcome" not in models_cache:
        # Try to load latest model
        models_dir = Path("models")
        if models_dir.exists():
            model_files = sorted(models_dir.glob("xgboost_outcome_*.pkl"), key=lambda x: x.stat().st_ctime, reverse=True)
            if model_files:
                models_cache["outcome"] = joblib.load(model_files[0])
    
    if "outcome" not in models_cache:
        raise HTTPException(status_code=400, detail="No model loaded. Train a model first.")
    
    # Build features for prediction
    historical = load_historical_results()
    
    # Create a dummy fixture
    fixture = pd.DataFrame([{
        "date": pd.to_datetime(request.date) if request.date else pd.Timestamp.now(),
        "home_team": request.home_team,
        "away_team": request.away_team,
        "home_score": np.nan,
        "away_score": np.nan,
        "neutral": False
    }])
    
    from predictors.feature_engineering import build_features
    features = build_features(historical, fixture)
    
    feature_cols = [
        'elo_diff', 'home_points_avg_5', 'home_points_avg_10',
        'home_goals_scored_avg_10', 'home_goals_conceded_avg_10',
        'home_win_rate_10', 'home_draw_rate_10', 'home_loss_rate_10',
        'away_points_avg_5', 'away_points_avg_10',
        'away_goals_scored_avg_10', 'away_goals_conceded_avg_10',
        'away_win_rate_10', 'away_draw_rate_10', 'away_loss_rate_10',
        'h2h_last_result', 'h2h_goals_avg', 'h2h_wins_diff', 'h2h_years_since',
        'home_matches_played', 'away_matches_played'
    ]
    
    X = features[feature_cols].fillna(0)
    
    model = models_cache["outcome"]
    probs = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    
    outcomes = ["Away Win", "Draw", "Home Win"]
    
    return {
        "prediction": outcomes[int(pred)],
        "probabilities": {
            "away_win": round(float(probs[0]), 4),
            "draw": round(float(probs[1]), 4),
            "home_win": round(float(probs[2]), 4)
        },
        "confidence": round(float(max(probs)), 4),
        "features": X.iloc[0].to_dict()
    }

@app.get("/metrics")
def get_metrics():
    metrics_dir = Path("models")
    if not metrics_dir.exists():
        return {"metrics": []}
    
    all_metrics = []
    for f in metrics_dir.glob("metrics_*.json"):
        data = json.loads(f.read_text())
        all_metrics.append(data)
    
    return {"metrics": sorted(all_metrics, key=lambda x: x.get('trained_at', ''), reverse=True)}

@app.get("/features/importance")
def get_feature_importance():
    metrics = get_metrics()
    if not metrics["metrics"]:
        return {"features": []}
    
    latest = metrics["metrics"][0]
    return {"features": latest.get("feature_importance", [])}

@app.get("/dataset/stats")
def get_dataset_stats():
    try:
        df = load_historical_results()
        return {
            "total_matches": len(df),
            "date_range": {
                "from": df["date"].min().isoformat() if hasattr(df["date"].min(), 'isoformat') else str(df["date"].min()),
                "to": df["date"].max().isoformat() if hasattr(df["date"].max(), 'isoformat') else str(df["date"].max())
            },
            "teams": int(df["home_team"].nunique()),
            "leagues": int(df["league_code"].nunique()) if "league_code" in df.columns else 1,
            "outcome_distribution": df.groupby("result").size().to_dict() if "result" in df.columns else {}
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
