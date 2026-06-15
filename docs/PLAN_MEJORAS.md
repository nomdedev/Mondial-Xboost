# Mondial-Xboost — Plan de Mejoras Exhaustivo
## Auditoría completa: CLI + Dashboard + API

**Fecha:** 2026-06-15
**Auditado por:** 3 agentes expertos (UX/CLI, Frontend, Backend/Arquitectura)

---

## HALLAZGOS CRÍTICOS (bloquean funcionalidad)

### 1. Dashboard llama endpoints que NO EXISTEN en la API
El dashboard (index.html) hace fetch a 7 endpoints que no están en `predictors/api.py`:
- `GET /dataset/stats` → NO EXISTE
- `GET /metrics` → NO EXISTE
- `POST /train` (async con job_id) → API tiene sync, sin job_id
- `GET /train/{job_id}` → NO EXISTE
- `GET /models` → NO EXISTE
- `GET /features/importance` → NO EXISTE
- `POST /predict` formato mismatch: dashboard envía `{home_team, away_team}`, API espera `{fixtures: [...]}`

**Resultado:** El dashboard muestra solo guiones "-" en todas las secciones.

### 2. Deploy en Vercel no sirve predicciones
`.vercelignore` excluye `*.pkl`, `*.csv`, `data/` → no llegan modelos entrenados.
`/predict` siempre devuelve 503 (no trained model available).

### 3. `cmd_health` usa `curl` — falla en Windows sin curl
Línea 316: shellea a curl. Debería usar `urllib` o `requests`.

### 4. `cmd_start` no verifica si el puerto ya está ocupado
Si el puerto 8000 está en uso, uvicorn crashea con error confuso.

### 5. Error en menú: `args.func` sin subcomando
Si ejecutás `mondial --algo`, crashea con `AttributeError` en vez de mostrar ayuda.

---

## FASE 0 — FIXES CRÍTICOS (esta sesión)

### 0.1 Registrar subcomando `start` en el parser
Ya creado `cmd_start`, falta agregarlo al parser y al menú.

### 0.2 Verificar puerto antes de iniciar
```python
import socket
def _port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0
```

### 0.3 Fix `args.func` AttributeError
```python
if not hasattr(args, 'func'):
    parser.print_help()
    return 1
```

### 0.4 Reemplazar `curl` en `cmd_health` con `urllib`

### 0.5 Agregar `--version` flag
```python
parser.add_argument('--version', action='version', version='%(prog)s 2.0.0')
```

---

## FASE 1 — API: Endpoints faltantes para el dashboard (prioridad alta)

### 1.1 Merge de los dos servidores en uno solo
`predictors/api.py` (puerto 8000) + `scripts/training_server.py` (puerto 8765) → UN solo servidor.
El training server tiene endpoints útiles que el dashboard necesita:
- `GET /status` → training_status.json
- `GET /results` → loop_engineering.json
- `GET /canonical` → métricas del modelo canónico

**Acción:** Mover estos endpoints a `predictors/api.py`.

### 1.2 Nuevos endpoints requeridos por el dashboard

| Endpoint | Método | Descripción | Prioridad |
|---|---|---|---|
| `/dataset/stats` | GET | Total matches, teams, outcome distribution | P0 |
| `/metrics` | GET | Lista de modelos entrenados con sus métricas | P0 |
| `/models` | GET | Lista de modelos en disco con metadata | P0 |
| `/features/importance` | GET | Feature importance del modelo actual | P0 |
| `/predict` (fix) | POST | Aceptar formato `{home_team, away_team}` del dashboard | P0 |
| `/train` (async) | POST | Iniciar entrenamiento como job background | P1 |
| `/train/{job_id}` | GET | Estado de un job de entrenamiento | P1 |
| `/gates` | POST | Ejecutar verify_gates | P1 |
| `/backtest` | POST | Ejecutar backtest | P1 |
| `/audit` | POST | Ejecutar auditoría de leakage | P1 |
| `/loop` | POST | Iniciar tuning de Optuna | P2 |
| `/auto-loop` | POST | Iniciar auto-loop engineering | P2 |
| `/manifest` | GET | Mostrar model_manifest.json | P1 |
| `/elo` | GET | Comparar Elo ratings | P2 |
| `/data-council` | POST | Ejecutar data council | P2 |
| `/clean` | POST | Limpiar artefactos | P2 |
| `/doctor` | GET | Info del entorno (doctor) | P1 |

### 1.3 Pattern para operaciones largas (train, loop, auto-loop)
Usar `BackgroundTasks` de FastAPI + job queue en memoria:
```python
from fastapi import BackgroundTasks
import uuid

_jobs: dict[str, dict] = {}

@app.post("/train")
async def train(background_tasks: BackgroundTasks, ...):
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "running", "progress": 0}
    background_tasks.add_task(_run_training, job_id, ...)
    return {"job_id": job_id, "status": "started"}

@app.get("/train/{job_id}")
async def train_status(job_id: str):
    return _jobs.get(job_id, {"status": "not_found"})
```

### 1.4 CORS middleware
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
```

---

## FASE 2 — Dashboard: Completar funcionalidad (prioridad alta)

### 2.1 Fix `showTab` — usa `event` implícito (falla en Firefox)
```javascript
function showTab(tabId, el) {
    // ... usar el parámetro 'el' en vez de 'event.target'
    el.closest('.nav-item').classList.add('active');
    window.location.hash = tabId;
}
```

### 2.2 Agregar tabs faltantes al sidebar
- ✅ Dashboard, Training, Predictions, Models, Features, Agent System
- ❌ Gates, Backtest, Manifest, Doctor, Audit

### 2.3 Sistema de notificaciones (reemplazar `alert()`)
```javascript
function showToast(message, type = 'error') { ... }
```

### 2.4 Loading states en todas las pestañas
Spinner/skeleton mientras carga data.

### 2.5 Error handling en todos los fetch
Mostrar mensaje claro cuando la API no responde.

### 2.6 Responsive design (mobile)
```css
@media (max-width: 768px) {
    .sidebar { display: none; } /* hamburger toggle */
    .grid-cols-4 { grid-template-columns: 1fr 1fr; }
}
```

### 2.7 URL hash routing
`#training`, `#predictions`, etc. para que sobreviva refresh.

### 2.8 Parar polling cuando el tab no está activo
```javascript
document.addEventListener('visibilitychange', () => { ... });
```

---

## FASE 3 — CLI: Mejoras de UX (prioridad media)

### 3.1 Menú interactivo categorizado
```
╔══════════════════════════════════════════════════════════╗
║            Mondial-Xboost — Menú principal               ║
╚══════════════════════════════════════════════════════════╝

📦 Setup
   1) Instalar dependencias          mondial instalar
   2) Doctor de portabilidad         mondial doctor

🧠 Entrenamiento
   3) Entrenar modelo                mondial entrenar
   4) Entrenar con GPU               mondial entrenar-gpu
   5) Loop engineering (Optuna)      mondial loop
   6) Auto Loop Engineering          mondial auto-loop

⚽ Predicción
   7) Predecir un partido            mondial predecir --home X --away Y

🔍 Validación
   8) Ejecutar tests                 mondial test
   9) Verificar gates                mondial gates
  10) Backtest de World Cup          mondial backtest
  11) Auditar leakage                mondial auditar

🖥️ Servidor
  12) Levantar plataforma            mondial start
  13) Solo API (bridge)              mondial servidor
  14) Health check                   mondial health

ℹ️ Info
  15) Ver manifest                   mondial manifest
  16) Info del entorno               mondial info
  17) Guía de uso                    mondial guia

   0) Salir
```

### 3.2 Aceptar nombre del comando además de número
```python
choice = input("Seleccioná: ").strip().lower()
if choice in command_map:
    # ejecutar directamente
```

### 3.3 Elapsed time en `_run()`
```python
import time
def _run(cmd, env=None):
    start = time.time()
    code = subprocess.call(cmd, ...)
    elapsed = time.time() - start
    print(f"\n  Completado en {elapsed:.1f}s")
    return code
```

### 3.4 Confirmación antes de operaciones destructivas
```python
def cmd_clean(_args):
    # ... contar archivos ...
    confirm = input(f"¿Eliminar {count} archivos? [y/N]: ")
    if confirm.lower() != 'y':
        return 0
```

### 3.5 `cmd_info` mejorado — tabla formateada con versiones

### 3.6 `cmd_guia` — paginar output o usar `less`

---

## FASE 4 — Verificaciones de calidad (prioridad media)

### 4.1 ARIA roles en el dashboard
```html
<nav role="navigation" aria-label="Navegación principal">
<button class="nav-item" aria-current="page">
<div role="tabpanel" id="dashboard">
```

### 4.2 Keyboard navigation en sidebar
Los `<div>` nav-items deben ser `<button>` con tabindex.

### 4.3 Dark scrollbar
```css
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
```

### 4.4 Dynamic histogram bins (training.js)
Bins hardcodeados [55-61] → calcular rango automáticamente.

---

## FASE 5 — Vercel deploy funcional (prioridad alta)

### 5.1 Modelo como variable de entorno o remote storage
Opciones:
- a) Subir .pkl como artifact (muy pesado)
- b) Usar un storage remoto (S3, R2, GitHub Releases)
- c) Servir solo la API/predict desde un modelo embebido pequeño
- d) Para Vercel: solo mostrar dashboard + datos estáticos, predict via Colab/otro backend

### 5.2 requirements-vercel.txt — verificar que no se pase de 500 MB
xgboost (~200MB) + sklearn (~50MB) + pandas (~50MB) + numpy (~50MB) = ~350MB
Con uvicorn + fastapi → ~360MB. Debería caber.

### 5.3 Verificar que vercel.json funcione correctamente
El `installCommand` podría no ser soportado en todas las configuraciones.

---

## FASE 6 — Nice-to-have (prioridad baja)

- [ ] Shell completions (`mondial completions`)
- [ ] `--json` output para `doctor`, `info`
- [ ] `--verbose` / `--quiet` global flags
- [ ] `mondial status` — resumen rápido de estado
- [ ] Config file (`mondial.toml`) para defaults
- [ ] `mondial actualizar` — self-update
- [ ] Chart export (PNG/SVG) en dashboard
- [ ] WebSocket para updates en tiempo real (solo local)
- [ ] Confusion matrix y calibration curve en dashboard
- [ ] Batch prediction via CSV upload

---

## ORDEN DE EJECUCIÓN RECOMENDADO

1. **FASE 0** — Fix críticos CLI (30 min)
2. **FASE 1** — API endpoints (merge servers + nuevos endpoints) (2-3h)
3. **FASE 2** — Dashboard fixes + nuevos tabs (2-3h)
4. **FASE 3** — CLI UX improvements (1-2h)
5. **FASE 5** — Vercel deploy funcional (1h)
6. **FASE 4** — Quality/accessibility (1h)
7. **FASE 6** — Nice-to-have (bajo demanda)

**Tiempo total estimado: 8-12 horas de trabajo**

---

## ARCHIVOS GENERADOS POR LA AUDITORÍA

- `docs/API_ARCHITECTURE_AUDIT.md` — Reporte detallado del agente Backend
- Este archivo: `docs/PLAN_MEJORAS.md`

---

## NOTA SOBRE EL COMANDO `mondial start`

Ya creado en `scripts/mondial_cli.py` (función `cmd_start`). Falta:
1. Registrar en el parser (`subparsers.add_parser("start", ...)`)
2. Agregar al menú interactivo (`MENU_ITEMS`)
3. Verificar puerto disponible antes de iniciar
4. Abrir navegador automáticamente
