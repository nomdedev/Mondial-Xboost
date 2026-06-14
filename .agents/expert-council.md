# Agent System — Mondial-Xboost

## Consejo de Agentes Expertos

El sistema de agentes analiza, audita y mejora continuamente el pipeline de predicción.
Cada agente tiene un rol específico y reporta al orquestador principal.

---

## Agentes

### 1. Data Leakage Auditor (CRÍTICO)
**Prioridad:** Máxima  
**Misión:** Garantizar que no exista data leakage en ninguna parte del pipeline.

**Checks obligatorios:**
- [ ] Elo ratings: ¿usa scores del partido actual para calcular K? → LEAKAGE
- [ ] Rolling features: ¿usa shift(1) correctamente? → Verificar
- [ ] H2H: ¿incluye el partido actual en el historial? → LEAKAGE
- [ ] Train/test split: ¿features computadas antes o después del split?
- [ ] Temporal split: ¿respeta orden cronológico?
- [ ] Min_date filter: ¿fixtures filtran pero historical no? → Verificar merge

**Método de detección:**
```python
# Test 1: Elo leakage
# Si K depende del score del partido, hay leakage
# Porque el score es el LABEL

# Test 2: Feature contamination
# Entrenar con datos hasta 2022, testear 2023
# Si accuracy > 60% en Mundial real → posible leakage
# Si accuracy < 45% → underfitting, no leakage

# Test 3: Temporal consistency
# Mismo modelo, diferentes splits temporales
# Si accuracy varía mucho → posible leakage o overfitting
```

### 2. ML Model Auditor
**Prioridad:** Alta  
**Misión:** Validar que los modelos están correctamente entrenados y calibrados.

**Checks:**
- [ ] Cross-validation temporal (no random)
- [ ] Calibration curves (reliability diagrams)
- [ ] Feature importance stability across folds
- [ ] Overfitting detection (train vs test gap)
- [ ] Baseline comparison (random, Elo-only, bookmaker odds)

### 3. Data Quality Agent
**Prioridad:** Alta  
**Misión:** Verificar calidad, completitud y veracidad de los datos.

**Checks:**
- [ ] NaN detection en features críticas
- [ ] Outlier detection (goles > 10, Elo > 2500)
- [ ] Team name normalization (mismos equipos en diferentes fuentes)
- [ ] Date parsing consistency
- [ ] Duplicate detection

### 4. Feature Engineering Agent
**Prioridad:** Media-Alta  
**Misión:** Diseñar, implementar y validar nuevas features.

**Checks:**
- [ ] Feature correlation analysis (remove highly correlated)
- [ ] Feature-target relationship (mutual information)
- [ ] Temporal stability (feature distribution shift over time)
- [ ] Missing value strategy (fill vs drop)

### 5. Prediction Validation Agent
**Prioridad:** Alta  
**Misión:** Validar predicciones contra resultados reales.

**Checks:**
- [ ] Backtesting against known WC results
- [ ] Calibration: ¿las probabilidades predichas son reales?
- [ ] Brier score por bins de confianza
- [ ] ROI simulado vs bookmaker odds

---

## Flujo de Ejecución

```
┌─────────────────┐
│  Orquestador    │
│  (Main Agent)   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Auditor │ │ML Model│
│Leakage │ │Auditor │
└────┬───┘ └────┬───┘
     │          │
     ▼          ▼
┌────────┐ ┌────────┐
│Data    │ │Feature │
│Quality │ │Eng.    │
└────┬───┘ └────┬───┘
     │          │
     └────┬─────┘
          ▼
    ┌──────────┐
    │Prediction│
    │Validation│
    └──────────┘
```

---

## Reporte de Auditoría

Cada agente genera un reporte con:
1. **Status**: PASS / FAIL / WARNING
2. **Evidence**: Código que reproduce el issue
3. **Impact**: Alto / Medio / Bajo
4. **Fix**: Código para corregir
5. **Verification**: Test que verifica la corrección
