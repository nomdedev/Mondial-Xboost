# Mondial-Xboost API Architecture Audit

> Generated: 2026-06-15 | Scope: Backend/API gap analysis, Vercel deployment readiness

---

## 1. Current Architecture Overview

### Two Separate FastAPI Servers

| Server | File | Port | Purpose |
|--------|------|------|---------|
| **ML Bridge** | `predictors/api.py` | 8000 | Prediction serving + training + dashboard static |
| **Training Monitor** | `scripts/training_server.py` | 8765 | Training progress monitoring (reads JSON from disk) |

### Endpoints That Exist

#### `predictors/api.py` (the Vercel-deployed one)
| Method | Path | Purpose | Notes |
|--------|------|---------|-------|
| GET | `/` | Serve `dashboard/index.html` | Static file |
| GET | `/health` | Health check + model loaded status | ✅ |
| POST | `/train` | Train model **synchronously** | ⚠️ Blocks until done |
| POST | `/predict` | Predict match outcomes | ✅ |
| GET | `/docs` | Swagger UI | Auto-generated |

#### `scripts/training_server.py` (local only)
| Method | Path | Purpose | Notes |
|--------|------|---------|-------|
| GET | `/` | Serve dashboard HTML | |
| GET | `/status` | Read `training_status.json` | File polling |
| GET | `/results` | Read `loop_engineering.json` | File polling |
| GET | `/runs` | List all Optuna runs | From JSON |
| GET | `/canonical` | Canonical model metrics | From manifest JSON |
| GET | `/health` | Health check | |

### CORS Configuration
- **api.py**: ❌ **NO CORS middleware at all** — dashboard fetches will fail cross-origin
- **training_server.py**: ✅ Has `CORSMiddleware(allow_origins=["*"])`

---

## 2. Critical Gap Analysis

### 2A. Dashboard Calls Endpoints That DON'T EXIST

The dashboard (`dashboard/index.html`) makes calls to these endpoints that are **not implemented** in `predictors/api.py`:

| Dashboard Call | Expected Response | Status |
|----------------|-------------------|--------|
| `GET /dataset/stats` | `{total_matches, teams, outcome_distribution}` | ❌ MISSING |
| `GET /metrics` | `{metrics: [{accuracy, log_loss, trained_at}]}` | ❌ MISSING |
| `POST /train` (JSON body) | `{job_id}` with async polling | ❌ Current is sync, different signature |
| `GET /train/{job_id}` | `{status, progress, metrics}` | ❌ MISSING |
| `GET /models` | `{models: [{name, size_mb, created}]}` | ❌ MISSING |
| `GET /features/importance` | `{features: [{feature, importance}]}` | ❌ MISSING |
| `POST /predict` (home_team/away_team) | `{probabilities, prediction, confidence}` | ❌ Different request/response format |

**Impact**: The dashboard is currently a **non-functional mock** — it will show errors on every tab except the sidebar health check.

### 2B. CLI Commands With NO API Equivalent (17 of 25)

| CLI Command | API? | Long-running? | Dashboard Needs? | Priority |
|-------------|------|---------------|------------------|----------|
| `loop` (Optuna tuning) | ❌ | Yes (minutes-hours) | **HIGH** — Agent tab | P1 |
| `auto-loop` | ❌ | Yes (hours) | **HIGH** — Agent tab | P1 |
| `gates` (verify_gates) | ❌ | Medium (seconds) | Medium | P2 |
| `backtest` | ❌ | Medium | Medium | P2 |
| `auditar` (leakage audit) | ❌ | Fast | Medium | P2 |
| `elo` (comparison) | ❌ | Fast | Low | P3 |
| `data-council` | ❌ | Fast | Low | P3 |
| `manifest` | ❌ | Instant | **HIGH** — Models tab | P1 |
| `train-cold-start` | ❌ | Medium | Low | P3 |
| `train-gpu` | ❌ | Long | Low (Vercel = CPU) | P4 |
| `info` | ❌ | Instant | Low | P3 |
| `doctor` | ❌ | Instant | Low | P3 |
| `clean` | ❌ | Instant | N/A (dangerous via API) | P4 |
| `install` | ❌ | N/A | N/A | N/A |
| `test` | ❌ | Fast | N/A | N/A |
| `lint` | ❌ | Fast | N/A | N/A |
| `bridge` | ❌ | Fast | N/A | N/A |

---

## 3. Vercel Deployment — Critical Blockers

### 3A. `.vercelignore` Excludes Everything Needed

```
data/           ← No models, no CSV data, no manifests
models/
*.pkl           ← No trained model files
*.csv           ← No historical_results.csv
*.parquet
scripts/        ← No training scripts
```

**Result**: On Vercel, `/predict` will ALWAYS return 503 ("No trained model available"). The app is deployed as a **dead shell**.

### 3B. Synchronous `/train` Will Timeout

- Vercel Python functions have a **10-second timeout** (or 60s on Pro)
- Training takes 30-120 seconds minimum
- Current `/train` is `async def` but calls blocking code — will timeout

### 3C. No Authentication

- Open API with no auth keys, no rate limiting
- `/train` endpoint allows anyone to trigger expensive compute
- On Vercel, this means anyone can rack up your bill

### 3D. Static Files May Not Mount Correctly

- `app.mount("/static", StaticFiles(...))` may conflict with Vercel's routing
- The `vercel.json` routes `/static/(.*)` but Vercel Python functions don't serve static files the same way

---

## 4. Real-Time Training Updates: WebSocket vs SSE vs Polling

### Current Approach: File-based polling
- Training writes `training_status.json` to disk
- `training_server.py` reads and serves it
- Dashboard polls every 2 seconds via `setInterval`

### Analysis for `mondial start` command

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Polling** (current) | Simple, works everywhere | Latency, wasted requests | ✅ Keep for Vercel |
| **SSE** (Server-Sent Events) | One-way real-time, simple server | Vercel doesn't support long-lived connections | ❌ Not for Vercel |
| **WebSocket** | True real-time bidirectional | Vercel serverless = no persistent connections | ❌ Not for Vercel |

**Recommendation**: For Vercel deployment, **polling is the only viable option**. For local `mondial start`, consider adding optional SSE since the server runs persistently.

---

## 5. Long-Running Operations Architecture

### The Core Problem
Training, loop, auto-loop take seconds to hours. HTTP request/response can't hold that long.

### Recommended Architecture: Background Tasks + Job Queue

```
┌─────────┐     POST /train       ┌──────────────┐
│ Dashboard │ ──────────────────→ │  FastAPI App  │
│  (browser) │                    │               │
│           │ ← ── {job_id} ──── │  Creates job  │
│           │                     └──────┬───────┘
│           │                            │
│           │     GET /jobs/{id}         │ spawns
│           │ ──────────────────→  ┌─────▼──────┐
│           │ ← ── {status, ...} ─ │ Background  │
│           │     (polling)        │ Task Thread │
└─────────┘                       │             │
                                  │ Updates     │
                                  │ job store   │
                                  └─────────────┘
```

### Implementation Options

**Option A: FastAPI BackgroundTasks + In-Memory Store** (Recommended for Vercel)
```python
from fastapi import BackgroundTasks
import uuid

_jobs: dict[str, dict] = {}

@app.post("/train")
async def start_train(background_tasks: BackgroundTasks, ...):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "progress": 0}
    background_tasks.add_task(_run_training, job_id, ...)
    return {"job_id": job_id}

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    return _jobs.get(job_id, {"status": "not_found"})
```

**Option B: Celery/Redis** (Overkill for this project)

**Option C: Thread-based** (Simple, good for single-server)
```python
import threading

_jobs: dict[str, dict] = {}

def _run_training(job_id, params):
    try:
        # ... training code ...
        _jobs[job_id] = {"status": "completed", "metrics": {...}}
    except Exception as e:
        _jobs[job_id] = {"status": "failed", "error": str(e)}
```

**Verdict**: Option A (BackgroundTasks) for Vercel-compatible operations, Option C (threads) for local `mondial start` where you need full training.

---

## 6. Predict Endpoint — Production Readiness Assessment

### Current State
```python
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, engine: str = "xgboost") -> PredictResponse:
```

### Issues

| Issue | Severity | Fix |
|-------|----------|-----|
| Model loaded on first request (cold start ~2-5s) | ⚠️ Medium | Pre-load at startup or use Vercel keep-warm |
| No input validation beyond Pydantic | ⚠️ Medium | Validate team names exist in historical data |
| No caching of predictions | ⚠️ Low | Add Redis/in-memory cache for repeated queries |
| Dashboard sends different format (home_team/away_team) | 🔴 High | Align dashboard and API request schemas |
| Single fixture batch, no limits | ⚠️ Medium | Add max batch size (e.g., 100 fixtures) |
| Generic 500 for all errors | ⚠️ Medium | Specific error codes (model not found, invalid team, etc.) |
| No request logging/metrics | ⚠️ Medium | Add structured logging |

### What Works
- Pydantic models for request/response ✅
- Multi-engine support (xgboost, random_forest) ✅
- In-memory model caching ✅
- Feature engineering pipeline integration ✅

---

## 7. Endpoints Needed for Full Dashboard Functionality

### Phase 1 — Critical (Dashboard won't work without these)

```python
# Dataset statistics
GET /api/dataset/stats
→ {total_matches: int, teams: int, outcome_distribution: {H, D, A}, date_range: {min, max}}

# Model metrics history
GET /api/metrics
→ {metrics: [{name, accuracy, log_loss, trained_at, engine}]}

# List trained models
GET /api/models
→ {models: [{name, size_mb, created, engine, accuracy}]}

# Model manifest
GET /api/manifest
→ {model_name, accuracy, log_loss, top_feature, trained_at}

# Feature importance
GET /api/features/importance
→ {features: [{feature, importance}]}
```

### Phase 2 — Training & Jobs (Agent system needs these)

```python
# Start async training job
POST /api/train
Body: {engine, min_date, name, elo_decay, elo_recent}
→ {job_id: str, status: "queued"}

# Poll job status
GET /api/jobs/{job_id}
→ {status, progress, metrics, error, started_at, completed_at}

# List all jobs
GET /api/jobs
→ {jobs: [{job_id, status, progress, ...}]}
```

### Phase 3 — Analysis & Quality (Full CLI parity)

```python
# Run quality gates
POST /api/gates
→ {job_id} (async)

# Run backtest
POST /api/backtest
→ {job_id} (async)

# Temporal leakage audit
POST /api/audit
→ {results: {...}}

# Elo comparison
GET /api/elo/comparison
→ {comparison: [...]}

# Data council review
POST /api/data-council
→ {job_id} (async)
```

### Phase 4 — Convenience

```python
# Predict with simpler interface (what dashboard actually sends)
POST /api/predict
Body: {home_team: "Argentina", away_team: "Brazil", date?: "2026-06-15"}
→ {home_team, away_team, probabilities: {home_win, draw, away_win}, prediction, confidence}

# Environment info
GET /api/info
→ {python_version, dependencies: {...}, models: [...]}

# Health check (enhanced)
GET /api/health
→ {status, model_loaded, engine, uptime, version}
```

---

## 8. Security Considerations for Vercel

### Critical

| Risk | Mitigation |
|------|------------|
| **Open training endpoint** — anyone can trigger expensive compute | Add API key auth to `/train`, `/gates`, `/backtest` etc. |
| **No rate limiting** — DDoS / cost amplification | Use Vercel's built-in rate limiting or add slowapi |
| **CORS = allow all** (on training server) | Restrict to your domain(s) |
| **Model file access** — if models are served, competitors could steal them | Don't serve model files via API; only serve predictions |
| **Input injection** — team names passed to feature engineering | Validate against known team list |

### Recommended Auth Pattern

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_SECRET_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")

# Public endpoints (no auth)
@app.get("/health")
@app.get("/api/dataset/stats")
@app.get("/api/models")

# Protected endpoints (require auth)
@app.post("/train", dependencies=[Depends(verify_api_key)])
@app.post("/api/gates", dependencies=[Depends(verify_api_key)])
```

### Vercel-Specific

1. **Environment Variables**: Store `API_SECRET_KEY` in Vercel dashboard
2. **Function Size**: Keep `requirements-vercel.txt` minimal — current is good (8 deps)
3. **Cold Starts**: XGBoost model loading adds 2-5s. Consider lighter model for Vercel or pre-warming
4. **File System**: Vercel serverless has a **read-only filesystem** except `/tmp` (512MB). Can't write models to disk. Training is impossible on Vercel.
5. **Timeout**: 10s default, 60s on Pro. `/train` will always timeout. Must be async job pattern.

---

## 9. Architecture Recommendations

### Recommendation 1: Merge the Two Servers

Having `api.py` (port 8000) and `training_server.py` (port 8765) is confusing. Merge all endpoints into one FastAPI app.

```
predictors/api.py          ← Single unified API
  ├── /health
  ├── /predict
  ├── /train (async job)
  ├── /jobs/{id}
  ├── /api/dataset/stats
  ├── /api/metrics
  ├── /api/models
  ├── /api/features/importance
  ├── /api/manifest
  ├── /api/gates
  ├── /api/backtest
  └── / (dashboard)
```

### Recommendation 2: Separate Vercel vs Local Concerns

```
Vercel (read-only, no training):        Local (full capability):
  /health                                 All Vercel endpoints PLUS:
  /predict                                /train (real training)
  /api/dataset/stats                      /api/jobs/{id} (polling)
  /api/metrics                            /api/gates
  /api/models                             /api/backtest
  /api/features/importance                /api/auto-loop
  / (dashboard - read-only mode)          WebSocket for live updates
```

### Recommendation 3: Use `requirements-vercel.txt` Split

Current split is good. Add to Vercel:
```
# Vercel: serving only, no training
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
xgboost>=2.0.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
```

Training deps (optuna, selenium, etc.) stay in main `requirements.txt` only.

### Recommendation 4: Ship Pre-Trained Models for Vercel

Since Vercel can't train, you need to:
1. Train locally → produce `*.pkl` files
2. Upload to a storage service (S3, R2, Vercel Blob)
3. On Vercel cold start, download model from storage
4. Cache in memory for subsequent requests

```python
# api.py — model loading strategy
MODEL_STORAGE_URL = os.getenv("MODEL_URL", "")  # e.g., Vercel Blob URL

def _get_predictor(engine="xgboost"):
    if engine in _predictors:
        return _predictors[engine]
    
    if MODEL_STORAGE_URL:
        # Download from cloud storage
        _download_model(MODEL_STORAGE_URL, engine)
    
    # Load from local file
    predictor = XGBoostFootballPredictor.load(f"{engine}_football")
    _predictors[engine] = predictor
    return predictor
```

---

## 10. Priority Ordering

### P0 — Must Fix Before Dashboard Works
1. **Add missing endpoints** that dashboard calls (`/dataset/stats`, `/metrics`, `/models`, `/features/importance`)
2. **Add CORS** to `api.py`
3. **Align predict request format** between dashboard and API
4. **Make `/train` async** (job-based pattern)

### P1 — Must Fix Before Vercel Deploy
5. **Pre-trained model strategy** (cloud storage + download on cold start)
6. **API key authentication** for write endpoints
7. **Fix `.vercelignore`** or implement model download strategy
8. **Merge training_server.py endpoints into api.py**

### P2 — Full CLI Parity
9. **`/api/manifest`** endpoint
10. **`/api/gates`** endpoint (async)
11. **`/api/backtest`** endpoint (async)
12. **`/api/audit`** endpoint

### P3 — Nice to Have
13. **`/api/elo/comparison`** endpoint
14. **`/api/data-council`** endpoint
15. **SSE/WebSocket** for local `mondial start` real-time updates
16. **Rate limiting**
17. **Request logging & metrics**

### P4 — Low Priority / CLI Only
18. Info, doctor, clean, install, lint, test — keep as CLI-only

---

## 11. Summary of Files to Create/Modify

| File | Action | What |
|------|--------|------|
| `predictors/api.py` | **REWRITE** | Add all missing endpoints, CORS, async jobs, auth |
| `scripts/training_server.py` | **DEPRECATE** | Merge into api.py |
| `dashboard/index.html` | **FIX** | Align API calls with actual endpoints |
| `dashboard/training.js` | **FIX** | Point to single API server |
| `vercel.json` | **UPDATE** | Add model storage env vars, fix routing |
| `requirements-vercel.txt` | **KEEP** | Current is good |
| `.vercelignore` | **REVIEW** | Decide on model strategy |

---

*End of audit report.*
