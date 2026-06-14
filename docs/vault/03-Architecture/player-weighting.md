# Peso de Jugadores en Oloráculo xBoost

## Objetivo
Convertir estadísticas individuales de jugadores de una selección en modificadores de equipo que ajusten los goles esperados del modelo base.

## Archivo
`predictors/player_weights.py`

## Flujo

```
Raw player stats
       │
       ▼
Per-90 normalization + caps + age peak + market value + injury
       │
       ▼
compute_player_score()  (0.0 - ~2.0)
       │
       ▼
compute_team_strength()
       │
       ▼
attack_modifier, defense_modifier, squad_depth_score
```

## Datos de entrada (`PlayerStats`)
- Identidad: `name`, `team`, `position`
- Rendimiento: `minutes`, `goals`, `assists`, `xg`, `xa`, `shots`, `key_passes`, `tackles`, `interceptions`, `passes`, `pass_accuracy`
- Contexto: `caps`, `age`, `market_value`, `recent_form_score`, `injury_status`

## Pesos por defecto
```python
DEFAULT_WEIGHTS = {
    "goals_per_90": 0.20,
    "assists_per_90": 0.10,
    "xg_per_90": 0.15,
    "xa_per_90": 0.08,
    "shots_per_90": 0.05,
    "key_passes_per_90": 0.05,
    "defensive_actions_per_90": 0.08,
    "pass_accuracy": 0.05,
    "minutes_share": 0.10,
    "caps": 0.08,
    "age_peak": 0.04,
    "recent_form": 0.10,
    "market_value": 0.05,
}
```
Suma = 1.0. El consejo de agentes puede proponer ajustes; el autoresearch los evalúa.

## Normalizaciones
- **Per-90**: todas las estadísticas de producción se dividen por minutos/90.
- **minutes_share**: fracción de minutos sobre 10 partidos (`min(minutes / 900, 1)`).
- **caps**: `min(caps / 50, 1)`.
- **age_peak**: factor que maximiza a los 27 años.
- **market_value**: `log1p(value) / log1p(100M)` saturado a 1.
- **injury**: multiplicador sobre el score final.

```python
INJURY_MULTIPLIER = {
    "available": 1.0,
    "doubt": 0.70,
    "injured": 0.0,
    "suspended": 0.0,
}
```

## Agregación por equipo
- `attack_modifier`: `1 + mean(score)` de los mejores FW/MF/AM.
- `defense_modifier`: `1 - 0.5 * mean(score)` de los mejores DF/DM/GK.
- `squad_depth_score`: media de los 18 mejores jugadores.
- `top_player_score`: mejor jugador.

## Uso en predicción
Los modificadores se pueden aplicar al expected goals del modelo de equipos:

```
expected_home_goals *= home_attack_modifier * away_defense_modifier
expected_away_goals *= away_attack_modifier * home_defense_modifier
```

## Calidad de datos
El `player-data-auditor` verifica:
- [ ] `minutes` > 0 para jugadores titulares.
- [ ] `injury_status` está en el enum permitido.
- [ ] `market_value` no es NaN para jugadores top.
- [ ] La suma de pesos es 1.0.
- [ ] No hay jugadores duplicados en el once.

## Próximas mejoras
1. Incorporar xG de contexto (liga vs selección).
2. Ajustar pesos por posición (portero, defensa, mediocampo, delantera).
3. Penalizar cambios bruscos de forma (lesión reciente, baja de última hora).
