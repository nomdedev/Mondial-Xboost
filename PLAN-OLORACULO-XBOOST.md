# Mondial-Xboost — Plan de Transformación Predictiva

## 1. Diagnóstico del Estado Actual (Mondial-Xboost v1)

### 1.1 Arquitectura Actual

```
Oloraculo.Web (C# / ASP.NET Core)
├── Predictors (Modelos Determinísticos)
│   ├── NullModel          → Prioridad 0 (fallback uniforme)
│   ├── FifaRankingModel   → Prioridad 1 (ranking FIFA)
│   ├── EloModel           → Prioridad 2 (Elo ratings)
│   ├── RecentFormModel    → Prioridad 3 (forma reciente)
│   ├── GoalModel          → Prioridad 4 (Poisson + Dixon-Coles)
│   └── GoalPlusRecentContextModel → Prioridad 5 (goles + contexto)
├── FinalPredictionSelector
│   ├── Selecciona el escalón NO degradado más alto
│   ├── Aplica calibración Elo/FIFA (15% weight) si hay consenso
│   └── Genera explicación textual
├── SimulationService
│   ├── 10,000 simulaciones Monte Carlo
│   ├── Simula grupos → clasificación → octavos → final
│   └── Probabilidades de: clasificar, QF, SF, Final, Campeón
├── Data Sources
│   ├── API-Football (lesiones, alineaciones, cuotas)
│   ├── OpenRouter/GPT-4o-mini (clasificación de disponibilidad)
│   ├── ESPN/TalkSport trackers (lesiones, parseo determinístico fallback)
│   ├── Wikipedia (FIFA rankings)
│   └── international-football.net (Elo ratings)
└── FeaturesEnum
    ├── FifaRanking, Elo, PlayerAvailability, Lineups, Odds
    └── OpponentAdjustedAttackStrength, OpponentAdjustedDefenseVulnerability, DixonColesScorelineGrid
```

### 1.2 Fortalezas
- Sistema de "escalones" (ladder) con degradación graceful
- Simulación Monte Carlo completa del torneo
- Integración con API-Football para datos en vivo
- Clasificación de disponibilidad con LLM (OpenRouter)
- Evaluación post-partido para mejora continua

### 1.3 Debilidades Críticas Identificadas

| # | Debilidad | Impacto | Prioridad |
|---|-----------|---------|-----------|
| 1 | **Sin ML/XGBoost**: Todos los predictores son modelos estadísticos clásicos. No hay ensemble learning, no hay feature engineering avanzado, no hay optimización de hiperparámetros. | Predicciones subóptimas vs mercado | CRÍTICA |
| 2 | **Sin datos de jugadores individuales**: No se usan stats de jugadores (goles, asistencias, xG, pases, tackles, etc.) para ponderar fuerza del equipo. | Ignora el 80% de la varianza real | CRÍTICA |
| 3 | **Sin head-to-head histórico**: No se analizan enfrentamientos previos entre jugadores clave. | Pierde contexto de rivalidad/táctica | ALTA |
| 4 | **Sin noticias diarias automatizadas**: Las URLs de availability son estáticas (ESPN/TalkSport). No hay scraping diario de múltiples fuentes. | Información desactualizada | ALTA |
| 5 | **Sin análisis de cuotas del mercado**: Las odds se leen pero no se usan como feature de predicción (solo se reportan). | Pierde señal de mercado | MEDIA |
| 6 | **Sin backtesting de predictores**: No hay walk-forward validation de cada modelo individual. | No se sabe qué modelo es mejor | MEDIA |
| 7 | **Sin LLM para análisis narrativo**: La explicación es determinística. No hay análisis cualitativo de forma, táctica, o momentum. | Menor valor para usuarios | MEDIA |
| 8 | **Sin sistema de agentes**: Todo es código monolítico C#. No hay pipeline de mejora autónoma. | Mejora lenta, manual | ALTA |

---

## 2. Visión: Mondial-Xboost v2

### 2.1 Objetivo
Transformar Mondial-Xboost de un oráculo estadístico clásico a un **sistema de predicción híbrido ML + LLM + Agentes** que:
- Use XGBoost como predictor principal con feature engineering avanzado
- Integre datos de jugadores individuales (API-Football + Transfermarkt)
- Scrapee noticias diarias de múltiples fuentes (lesiones, tácticas, clima)
- Use LLM para análisis narrativo y calibración de probabilidades
- Ejecute un pipeline de 7 fases con agentes autónomos
- Backtestee cada predictor con walk-forward validation
- Exporte todo el conocimiento a Obsidian vault

### 2.2 Arquitectura Target

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CAPA DE PRESENTACIÓN                          │
│  Oloraculo.Web (ASP.NET Core) + Dashboard React/Vite (opcional)     │
│  ── Predicciones por partido, tabla de probabilidades, simulación    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                      CAPA DE ORQUESTACIÓN (Agentes)                   │
│  Orchestrator (7 fases) → Domain-Expert → Architect → Backend        │
│  → Security → Developer → QA → DevOps → Production                   │
│  ── Pipeline diario: 20:00 ART (cron)                                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                    CAPA DE PREDICCIÓN (ML + LLM)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  │
│  │ XGBoost     │  │ LLM Analyst │  │ Elo Model   │  │ Poisson  │  │
│  │ Ensemble    │  │ (calibración│  │ (fallback)  │  │ + DC     │  │
│  │ (principal) │  │  narrativa) │  │             │  │          │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘  │
│  FinalPredictionSelector v2: Weighted ensemble con ML meta-learner │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                      CAPA DE FEATURE ENGINEERING                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  │
│  │ Player Stats│  │ H2H History │  │ News Scraper│  │ Odds    │  │
│  │ (API-Football│  │ (API-Football│  │ (Selenium + │  │ Feature │  │
│  │  Transfermarkt)│  │  + manual)  │  │  LLM parse) │  │         │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │ Team Form   │  │ Availability│  │ Weather/    │                │
│  │ (Elo + FIFA)│  │ (Injuries)   │  │ Travel      │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                      CAPA DE DATOS / INFRAESTRUCTURA                  │
│  SQLite/PostgreSQL → API-Football → OpenRouter → News Sources        │
│  Alpaca (opcional) → Obsidian Vault → Vercel Dashboard               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Equipo de Agentes (7 Fases / 6 Equipos)

Adaptado del agent-manager-template + tradingview-mcp experience.

### 3.1 Fase 1: Domain & Requirements — `domain-expert`

**Rol**: Experto en fútbol, estadísticas deportivas, y mercado de apuestas.
**Misión**: Validar que los requerimientos de predicción son realistas y completos.
**Tareas**:
- Definir el feature set óptimo para predicción de partidos de fútbol
- Validar que las fuentes de datos (API-Football, Transfermarkt, etc.) cubren las necesidades
- Definir métricas de éxito: log-loss, Brier score, ROI simulado, calibration
- Documentar en Obsidian: `domain/requirements.md`

**Skills**: `obsidian-vault`, `grill-with-docs`

### 3.2 Fase 2: Architecture — `architect`

**Rol**: Arquitecto de software. Diseña el sistema de ML + LLM + C#.
**Misión**: Diseñar la arquitectura de integración sin romper el código existente.
**Tareas**:
- Diseñar el bridge C# ↔ Python para XGBoost (gRPC / HTTP / CLI)
- Diseñar el schema de features para el modelo ML
- Diseñar el pipeline de datos: ETL desde API-Football → features
- Definir interfaces: `IPredictor` v2 con soporte para ML
- Documentar en Obsidian: `architecture/system-design.md`

**Skills**: `improve-codebase-architecture`, `prototype`

### 3.3 Fase 3: Backend — `api-expert` + `data-engineer`

**Rol**: Expertos en APIs y datos. Implementan los conectores y ETL.
**Misión**: Tener todas las APIs funcionales y schemas consistentes.
**Tareas**:
- Implementar conector a API-Football v3 (jugadores, estadísticas, eventos)
- Implementar scraper de noticias (Selenium + Brave) para lesiones/tácticas
- Implementar ETL de datos históricos a formato ML (CSV/Parquet)
- Implementar caché de features (Redis/SQLite)
- Documentar en Obsidian: `backend/api-contracts.md`

**Skills**: `mcp-builder`, `browser-session-cookies`

### 3.4 Fase 4: Security — `security-auditor`

**Rol**: Auditor de seguridad. Revisa APIs, keys, y datos.
**Misión**: Sin hallazgos críticos antes de desarrollar.
**Tareas**:
- Auditar manejo de API keys (API-Football, OpenRouter)
- Auditar SQL injection en queries dinámicas
- Auditar XSS en el frontend
- Validar rate limiting
- Documentar en Obsidian: `security/audit-report.md`

**Skills**: `security-auditor` (del template)

### 3.5 Fase 5: Develop — `developer` + `reviewer`

**Rol**: Implementadores y revisores de código.
**Misión**: Build exitoso, código optimizado, tests pasando.
**Tareas**:
- Implementar `XGBoostPredictor` (Python) con feature engineering
- Implementar bridge C# ↔ Python
- Implementar `LLMAnalysisService` (OpenRouter) para calibración narrativa
- Implementar `NewsScraperService` (Selenium)
- Implementar `PlayerStatsService` (API-Football)
- Implementar `BacktestService` (walk-forward validation)
- Code review obligatorio
- Documentar en Obsidian: `dev/implementation-log.md`

**Skills**: `tdd`, `request-refactor-plan`, `code-review`

### 3.6 Fase 6: QA — `tester`

**Rol**: Tester. Misión: ROMPER TODO.
**Misión**: Tests pasando, coverage documentado, todo roto y arreglado.
**Tareas**:
- Unit tests para cada predictor
- Integration tests para el pipeline ETL
- Backtest validation: ¿el modelo predice mejor que el baseline?
- A/B test: XGBoost vs Elo vs Poisson
- Edge cases: equipos sin datos, jugadores desconocidos, partidos cancelados
- Documentar en Obsidian: `qa/test-report.md`

**Skills**: `tdd`, `diagnose`

### 3.7 Fase 6.5: DevOps — `devops-infra`

**Rol**: Infraestructura y deploy.
**Misión**: Deploy listo, CI/CD verde, infra auditada.
**Tareas**:
- Configurar cron job diario (20:00 ART) para pipeline de datos
- Configurar CI/CD para tests automáticos
- Configurar monitoreo de APIs (rate limits, errores)
- Deploy dashboard a Vercel
- Documentar en Obsidian: `devops/deploy-guide.md`

**Skills**: `ci-pipeline`, `hermes-gateway-troubleshooting`

### 3.8 Fase 7: Production — `orchestrator` (principal)

**Rol**: Director del pipeline. No implementa código.
**Misión**: Verificación en producción con datos reales.
**Tareas**:
- Ejecutar pipeline completo en producción
- Verificar que las predicciones del día son coherentes
- Comparar vs mercado de apuestas (Pinnacle, Bet365)
- Ajustar pesos del ensemble si hay drift
- Documentar en Obsidian: `production/run-log.md`

**Skills**: `orchestrator` (del template)

---

## 4. Capas de Computo y LLM

### 4.1 Capa de Computo Intensivo (Python / Local)

| Componente | Tecnología | Dónde corre | Por qué |
|------------|-----------|-------------|---------|
| XGBoost Training | Python + xgboost + scikit-learn | Local (Windows) | Dataset pequeño (<100K rows), no necesita GPU |
| Feature Engineering | Python + pandas + numpy | Local | Transformaciones vectorizadas |
| Backtest Engine | Python + pandas (adaptado de tradingview-mcp v3) | Local | Walk-forward, stop-loss, position sizing |
| Data ETL | Python + requests + beautifulsoup | Local | Scraping + API calls |
| Simulation Monte Carlo | C# (existente) | Local | 10,000 sims rápidas en C# |

### 4.2 Capa de LLM (OpenRouter / API)

| Uso | Modelo | Frecuencia | Costo Est. |
|-----|--------|-----------|-------------|
| Clasificación de disponibilidad (lesiones) | GPT-4o-mini | Diario (cron) | ~$0.10/día |
| Análisis narrativo de partidos | GPT-4o / Claude 3.5 Sonnet | Por partido | ~$0.05/partido |
| Calibración de probabilidades | GPT-4o-mini | Por partido | ~$0.02/partido |
| Resumen de noticias diarias | GPT-4o-mini | Diario | ~$0.05/día |
| Meta-análisis post-partido | GPT-4o | Post-partido | ~$0.10/partido |

### 4.3 Capa de C# / ASP.NET (Core del Sistema)

| Componente | Tecnología | Rol |
|------------|-----------|-----|
| Oloraculo.Web | ASP.NET Core 8 | API + Frontend |
| Predictores Clásicos | C# | Elo, FIFA, Poisson (fallback) |
| SimulationService | C# | Monte Carlo del torneo |
| SnapshotService | C# | Export a README/JSON |
| Bridge a Python | gRPC / HTTP / Process | Llamar a XGBoost |

---

## 5. Pipeline de Datos Diario (20:00 ART)

```
Cron: 0 20 * * * (Argentina Time)

┌─────────────────┐
│ 1. NEWS SCRAPER │ ← Selenium + Brave → ESPN, TalkSport, BBC, Marca
│   (5 min)       │    → Lesiones, tácticas, rumores, clima
└────────┬────────┘
         │
┌────────▼────────┐
│ 2. API-Football │ ← Jugadores, stats, alineaciones, cuotas
│   (10 min)      │    → SQLite cache
└────────┬────────┘
         │
┌────────▼────────┐
│ 3. LLM CLASSIFY │ ← OpenRouter: clasifica disponibilidad
│   (5 min)       │    → AvailabilityClaims DB
└────────┬────────┘
         │
┌────────▼────────┐
│ 4. FEATURE ENG  │ ← Python: genera features para ML
│   (3 min)       │    → Parquet file
└────────┬────────┘
         │
┌────────▼────────┐
│ 5. XGBoost PRED │ ← Python: predice cada partido del día
│   (2 min)       │    → Probabilidades + explicación
└────────┬────────┘
         │
┌────────▼────────┐
│ 6. LLM ANALYSIS │ ← OpenRouter: análisis narrativo
│   (5 min)       │    → Contexto táctico, momentum
└────────┬────────┘
         │
┌────────▼────────┐
│ 7. ENSEMBLE     │ ← C#: combina XGBoost + Elo + Poisson + LLM
│   (1 min)       │    → Predicción final con confianza
└────────┬────────┘
         │
┌────────▼────────┐
│ 8. SIMULATION   │ ← C#: 10,000 sims del torneo
│   (2 min)       │    → Probabilidades de clasificación/campeón
└────────┬────────┘
         │
┌────────▼────────┐
│ 9. EXPORT       │ ← C#: README + JSON + Obsidian
│   (1 min)       │    → Snapshot del día
└────────┬────────┘
         │
┌────────▼────────┐
│ 10. EVALUATION  │ ← C#: compara con resultados reales
│   (background)  │    → Métricas de calidad del modelo
└─────────────────┘
```

---

## 6. Feature Engineering para XGBoost

### 6.1 Features de Equipo (Team-Level)

| Feature | Fuente | Descripción |
|---------|--------|-------------|
| `elo_diff` | Elo ratings | Diferencia Elo (home - away) |
| `fifa_rank_diff` | FIFA | Diferencia de ranking FIFA |
| `fifa_points_diff` | FIFA | Diferencia de puntos FIFA |
| `form_last_5` | API-Football | Puntos obtenidos en últimos 5 partidos |
| `goals_scored_avg` | API-Football | Promedio goles anotados últimos 10 |
| `goals_conceded_avg` | API-Football | Promedio goles recibidos últimos 10 |
| `win_rate_last_10` | API-Football | % victorias últimos 10 partidos |
| `home_advantage` | Calculado | 1 si juega de local, 0.5 si neutral |
| `tournament_experience` | Manual | Años desde última participación en mundial |

### 6.2 Features de Jugador (Player-Level Aggregated)

| Feature | Fuente | Descripción |
|---------|--------|-------------|
| `top_scorer_strength` | API-Football | Goles del máximo goleador del equipo |
| `avg_player_age` | API-Football | Edad promedio del once titular |
| `avg_player_caps` | API-Football | Partidos internacionales promedio |
| `key_players_available` | Availability | % de jugadores clave disponibles |
| `squad_depth` | API-Football | Cantidad de jugadores con >10 partidos |
| `xG_team` | API-Football | Expected goals acumulado del equipo |
| `xGA_team` | API-Football | Expected goals against acumulado |

### 6.3 Features Head-to-Head (H2H)

| Feature | Fuente | Descripción |
|---------|--------|-------------|
| `h2h_wins` | API-Football | Victorias en enfrentamientos directos |
| `h2h_goals_avg` | API-Football | Promedio goles en H2H |
| `h2h_last_result` | API-Football | Resultado del último enfrentamiento |
| `h2h_years_since` | API-Football | Años desde último enfrentamiento |

### 6.4 Features de Contexto (Context)

| Feature | Fuente | Descripción |
|---------|--------|-------------|
| `odds_implied_prob_home` | API-Football | Probabilidad implícita del mercado (local) |
| `odds_implied_prob_draw` | API-Football | Probabilidad implícita del mercado (empate) |
| `odds_implied_prob_away` | API-Football | Probabilidad implícita del mercado (visitante) |
| `news_sentiment` | LLM | Sentimiento de noticias recientes (-1 a 1) |
| `injury_severity_score` | Availability | Ponderación de lesiones (0=ninguna, 10=crítica) |
| `travel_distance_km` | Calculado | Distancia entre sedes del torneo |
| `rest_days` | Calculado | Días desde último partido |
| `knockout_pressure` | Calculado | 1 si es eliminación directa, 0 si grupo |

### 6.5 Target Variables

| Target | Descripción | Tipo |
|--------|-------------|------|
| `outcome` | 0=away win, 1=draw, 2=home win | Multiclass |
| `home_goals` | Goles del equipo local | Regression |
| `away_goals` | Goles del equipo visitante | Regression |
| `btts` | Both teams to score (0/1) | Binary |
| `over_2_5` | Over 2.5 goals (0/1) | Binary |

---

## 7. Integración con TradingView MCP (Lo que Reutilizamos)

### 7.1 Sistemas Reutilizables

| Sistema | En TradingView | En Mondial-Xboost | Adaptación |
|---------|---------------|---------------------|------------|
| **XGBoost Engine** | `scripts/backtest/engine_v3.py` | `predictors/xgboost_engine.py` | Cambiar features de precios a features de fútbol |
| **Expert Panel** | `scripts/backtest/expert_panel_backtest.py` | `predictors/expert_panel.py` | 4 agentes evaluando estrategias → 4 agentes evaluando predicciones |
| **Walk-Forward** | `engine_v3.py:walk_forward_backtest()` | `backtest/walk_forward.py` | Mismo código, diferentes datos |
| **Autoresearch Loop** | `skills/autoresearch-integration-loop` | `skills/autoresearch-football` | Loop de mejora continua aplicado a predictores |
| **Loop Engineering** | `skills/loop-engineering` | `skills/loop-engineering` | Reutilizar directamente |
| **EvoMemory** | `.agents/agents/memory-engineer.md` | `.agents/agents/football-memory-engineer.md` | Adaptar a memoria de partidos |
| **Orchestrator** | `.agents/orchestrator.md` | `.agents/orchestrator.md` | Reutilizar directamente |
| **Dashboard** | `dashboard/` (React+Vite) | `dashboard/` (nuevo o adaptado) | Mostrar predicciones en vez de señales |

### 7.2 No Reutilizamos

| Sistema | Razón |
|---------|-------|
| Alpaca Trading | No es trading financiero, es predicción deportiva |
| TradingView API | No aplica a fútbol |
| Indicadores técnicos (RSI, MACD) | No aplican a fútbol |

---

## 8. Plan de Implementación (Fases)

### Fase 0: Setup (Semana 0 — 2 días)
- [ ] Copiar skills de tradingview-mcp y agent-manager-template
- [ ] Crear estructura `.agents/` en Mondial-Xboost
- [ ] Configurar Python environment (xgboost, pandas, scikit-learn)
- [ ] Setup Obsidian vault para documentación
- [ ] Crear cron job diario (20:00 ART)

### Fase 1: Data Pipeline (Semana 1 — 5 días)
- [ ] Implementar `PlayerStatsService` (API-Football v3)
- [ ] Implementar `NewsScraperService` (Selenium + Brave)
- [ ] Implementar ETL a Parquet/CSV
- [ ] Implementar feature engineering en Python
- [ ] Tests de integración para APIs

### Fase 2: ML Predictor (Semana 2 — 5 días)
- [ ] Implementar `XGBoostPredictor` en Python
- [ ] Entrenar con datos históricos (mínimo 5 años)
- [ ] Implementar bridge C# ↔ Python
- [ ] Integrar en `FinalPredictionSelector` v2
- [ ] Backtest walk-forward vs baseline

### Fase 3: LLM Integration (Semana 3 — 4 días)
- [ ] Implementar `LLMAnalysisService` (OpenRouter)
- [ ] Prompt engineering para análisis táctico
- [ ] Calibración de probabilidades con LLM
- [ ] Integrar explicaciones narrativas en output
- [ ] A/B test: con vs sin LLM

### Fase 4: Agent Pipeline (Semana 4 — 4 días)
- [ ] Crear agentes especializados (7 fases)
- [ ] Implementar pipeline de orquestación
- [ ] Integrar con Hermes cron jobs
- [ ] Tests de pipeline end-to-end
- [ ] Documentación en Obsidian

### Fase 5: Evaluation & Tuning (Semana 5 — 3 días)
- [ ] Backtest completo del sistema
- [ ] Comparar vs mercado de apuestas
- [ ] Ajustar pesos del ensemble
- [ ] Implementar drift detection
- [ ] Reporte de performance

### Fase 6: Production (Semana 6 — 2 días)
- [ ] Deploy a producción
- [ ] Monitoreo de APIs y errores
- [ ] Dashboard en Vercel (opcional)
- [ ] Run diario automatizado
- [ ] Post-mortem y ajustes

---

## 9. Métricas de Éxito

| Métrica | Target | Cómo Medir |
|---------|--------|-----------|
| Log-loss | < 0.65 | Comparar probabilidades vs resultados reales |
| Brier Score | < 0.20 | MSE de probabilidades |
| ROI Simulado | > 5% | Apostar 1 unidad por predicción, calcular P&L |
| Calibration | ECE < 0.05 | Expected Calibration Error |
| Coverage | > 95% | % de partidos con predicción (no fallback) |
| Latency | < 5 min | Tiempo total del pipeline diario |
| Costo LLM | < $5/día | Tracking de tokens OpenRouter |

---

## 10. Estructura de Archivos Target

```
Mondial-Xboost/
├── Oloraculo.Web/                    ← C# (existente, extendido)
│   ├── Predictors/
│   │   ├── XGBoostBridge.cs          ← NUEVO: bridge a Python
│   │   ├── LLMAnalysisBridge.cs      ← NUEVO: bridge a LLM
│   │   └── ... (existentes)
│   ├── Services/
│   │   ├── PlayerStatsService.cs     ← NUEVO
│   │   ├── NewsScraperService.cs     ← NUEVO
│   │   └── ... (existentes)
│   └── ...
├── predictors/                       ← NUEVO: Python ML
│   ├── xgboost_engine.py
│   ├── feature_engineering.py
│   ├── expert_panel.py
│   ├── train.py
│   └── requirements.txt
├── backtest/                         ← NUEVO: Python backtest
│   ├── walk_forward.py
│   ├── evaluate.py
│   └── requirements.txt
├── scrapers/                         ← NUEVO: Python scrapers
│   ├── news_scraper.py
│   ├── player_stats_scraper.py
│   └── requirements.txt
├── .agents/                          ← NUEVO: agentes
│   ├── orchestrator.md
│   ├── agents/
│   │   ├── domain-expert.md
│   │   ├── architect.md
│   │   ├── api-expert.md
│   │   ├── data-engineer.md
│   │   ├── security-auditor.md
│   │   ├── developer.md
│   │   ├── reviewer.md
│   │   ├── tester.md
│   │   ├── devops-infra.md
│   │   ├── football-memory-engineer.md
│   │   └── performance-analyst.md
│   └── skills/
│       ├── autoresearch-football/
│       ├── loop-engineering/
│       └── mcp-builder/
├── data/                             ← NUEVO: datasets
│   ├── raw/
│   ├── processed/
│   └── features/
├── docs/                             ← NUEVO: documentación
│   └── (o symlink a Obsidian vault)
├── dashboard/                        ← NUEVO: React+Vite (opcional)
│   └── (adaptado de tradingview-mcp)
└── README.md                         ← Actualizado
```

---

## 11. Próximos Pasos Inmediatos

1. **Aprobación del plan** — Revisar con usuario y ajustar scope/prioridades
2. **Setup de agentes** — Crear `.agents/` y copiar skills reutilizables
3. **Fase 0** — Python env, cron, Obsidian vault
4. **Fase 1** — Empezar con `PlayerStatsService` y `NewsScraperService`
5. **Daily standup** — Cada run del cron, revisar métricas y ajustar

---

*Documento generado: 2026-06-13*
*Versión: 1.0*
*Próxima revisión: post-Fase 0*
