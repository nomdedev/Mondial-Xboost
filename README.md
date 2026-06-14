# Mondial-Xboost

Sistema de predicción de partidos de fútbol basado en machine learning. Combina datos históricos, feature engineering anti-leakage, y un motor canónico XGBoost optimizado con Optuna para generar probabilidades de outcomes.

## [Video con lore y explicación](https://youtu.be/cvPeS0qAikw?si=yHv5wKkk5lqgYXhn)

<!-- mondial-xboost:snapshots:start -->
## Predicciones más recientes
_A medida que se recibe nueva información y se juegan partidos reales, el Mondial-Xboost ajusta sus predicciones y las publica acá. A continuación vas a encontrar las más recientes._

### Torneo

_Generado 2026-06-14 19:04 UTC a través de 10,000 simulaciones._

| Team | Group | Qualify | QF | SF | Final | Champion |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/co.svg" width="18" alt=""> Colombia | K | 80 % | 32 % | 18 % | 10 % | **5.8 %** |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/ar.svg" width="18" alt=""> Argentina | J | 85 % | 31 % | 18 % | 10 % | **5.8 %** |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/es.svg" width="18" alt=""> Spain | H | 92 % | 31 % | 18 % | 10 % | **5.7 %** |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/gb-eng.svg" width="18" alt=""> England | L | 92 % | 32 % | 18 % | 10 % | **5.6 %** |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/br.svg" width="18" alt=""> Brazil | C | 90 % | 32 % | 19 % | 10 % | **5.5 %** |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/pt.svg" width="18" alt=""> Portugal | K | 81 % | 31 % | 18 % | 10 % | **5.5 %** |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/fr.svg" width="18" alt=""> France | I | 82 % | 31 % | 17 % | 10 % | **5.3 %** |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/jp.svg" width="18" alt=""> Japan | F | 85 % | 30 % | 17 % | 9 % | **5.1 %** |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/be.svg" width="18" alt=""> Belgium | G | 81 % | 29 % | 16 % | 9 % | **4.9 %** |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/dz.svg" width="18" alt=""> Algeria | J | 81 % | 26 % | 14 % | 8 % | **4.2 %** |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/nl.svg" width="18" alt=""> Netherlands | F | 83 % | 28 % | 15 % | 8 % | **4.2 %** |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/ir.svg" width="18" alt=""> Iran | G | 79 % | 27 % | 15 % | 8 % | **4.1 %** |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/ma.svg" width="18" alt=""> Morocco | C | 88 % | 27 % | 15 % | 7 % | **3.6 %** |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/de.svg" width="18" alt=""> Germany | E | 99 % | 31 % | 16 % | 8 % | **3.5 %** |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/au.svg" width="18" alt=""> Australia | D | 97 % | 29 % | 15 % | 7 % | **3.5 %** |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/us.svg" width="18" alt=""> United States | D | 97 % | 29 % | 14 % | 7 % | **3.2 %** |

### Grupos

<details open>
<summary><strong>Group A</strong></summary>

| Match | Status | Result / Pick | H | D | A |
| --- | --- | --- | ---: | ---: | ---: |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/mx.svg" width="18" alt=""> Mexico vs <img src="MondialXboost.Web/wwwroot/flags/4x3/za.svg" width="18" alt=""> South Africa | FT | **2-0** <br><sub>Prediction: 1-0</sub> | 52 % | 28 % | 20 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/kr.svg" width="18" alt=""> South Korea vs <img src="MondialXboost.Web/wwwroot/flags/4x3/cz.svg" width="18" alt=""> Czechia | FT | **2-1** <br><sub>Prediction: 1-1</sub> | 54 % | 23 % | 23 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/za.svg" width="18" alt=""> South Africa vs <img src="MondialXboost.Web/wwwroot/flags/4x3/cz.svg" width="18" alt=""> Czechia | Jun 18 16:00 UTC | 1-1 | 32 % | 28 % | 40 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/mx.svg" width="18" alt=""> Mexico vs <img src="MondialXboost.Web/wwwroot/flags/4x3/kr.svg" width="18" alt=""> South Korea | Jun 19 01:00 UTC | 1-1 | 36 % | 29 % | 35 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/mx.svg" width="18" alt=""> Mexico vs <img src="MondialXboost.Web/wwwroot/flags/4x3/cz.svg" width="18" alt=""> Czechia | Jun 25 01:00 UTC | 1-1 | 52 % | 25 % | 23 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/za.svg" width="18" alt=""> South Africa vs <img src="MondialXboost.Web/wwwroot/flags/4x3/kr.svg" width="18" alt=""> South Korea | Jun 25 01:00 UTC | 0-1 | 20 % | 26 % | 55 % |

</details>

<details open>
<summary><strong>Group B</strong></summary>

| Match | Status | Result / Pick | H | D | A |
| --- | --- | --- | ---: | ---: | ---: |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/qa.svg" width="18" alt=""> Qatar vs <img src="MondialXboost.Web/wwwroot/flags/4x3/ch.svg" width="18" alt=""> Switzerland | FT | **1-1** <br><sub>Prediction: 1-2</sub> | 18 % | 21 % | 61 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/ca.svg" width="18" alt=""> Canada vs <img src="MondialXboost.Web/wwwroot/flags/4x3/qa.svg" width="18" alt=""> Qatar | Jun 18 22:00 UTC | 1-0 | 60 % | 23 % | 18 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/ca.svg" width="18" alt=""> Canada vs <img src="MondialXboost.Web/wwwroot/flags/4x3/ch.svg" width="18" alt=""> Switzerland | Jun 24 19:00 UTC | 1-1 | 35 % | 28 % | 36 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/ba.svg" width="18" alt=""> Bosnia and Herzegovina vs <img src="MondialXboost.Web/wwwroot/flags/4x3/qa.svg" width="18" alt=""> Qatar | Scheduled | 1-1 | 28 % | 26 % | 47 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/ba.svg" width="18" alt=""> Bosnia and Herzegovina vs <img src="MondialXboost.Web/wwwroot/flags/4x3/ch.svg" width="18" alt=""> Switzerland | Scheduled | 0-2 | 12 % | 20 % | 68 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/ca.svg" width="18" alt=""> Canada vs <img src="MondialXboost.Web/wwwroot/flags/4x3/ba.svg" width="18" alt=""> Bosnia and Herzegovina | Final | **1-1** <br><sub>Prediction: 1-0</sub> | 66 % | 22 % | 12 % |

</details>

<details open>
<summary><strong>Group C</strong></summary>

| Match | Status | Result / Pick | H | D | A |
| --- | --- | --- | ---: | ---: | ---: |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/br.svg" width="18" alt=""> Brazil vs <img src="MondialXboost.Web/wwwroot/flags/4x3/ma.svg" width="18" alt=""> Morocco | FT | **1-1** <br><sub>Prediction: 1-1</sub> | 39 % | 27 % | 34 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/ht.svg" width="18" alt=""> Haiti vs <img src="MondialXboost.Web/wwwroot/flags/4x3/gb-sct.svg" width="18" alt=""> Scotland | FT | **0-1** <br><sub>Prediction: 1-1</sub> | 33 % | 26 % | 41 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/ma.svg" width="18" alt=""> Morocco vs <img src="MondialXboost.Web/wwwroot/flags/4x3/gb-sct.svg" width="18" alt=""> Scotland | Jun 19 22:00 UTC | 1-0 | 55 % | 25 % | 20 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/br.svg" width="18" alt=""> Brazil vs <img src="MondialXboost.Web/wwwroot/flags/4x3/ht.svg" width="18" alt=""> Haiti | Jun 20 00:30 UTC | 2-0 | 64 % | 20 % | 15 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/br.svg" width="18" alt=""> Brazil vs <img src="MondialXboost.Web/wwwroot/flags/4x3/gb-sct.svg" width="18" alt=""> Scotland | Jun 24 22:00 UTC | 1-0 | 59 % | 23 % | 18 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/ma.svg" width="18" alt=""> Morocco vs <img src="MondialXboost.Web/wwwroot/flags/4x3/ht.svg" width="18" alt=""> Haiti | Jun 24 22:00 UTC | 1-1 | 61 % | 22 % | 17 % |

</details>

<details open>
<summary><strong>Group D</strong></summary>

| Match | Status | Result / Pick | H | D | A |
| --- | --- | --- | ---: | ---: | ---: |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/us.svg" width="18" alt=""> United States vs <img src="MondialXboost.Web/wwwroot/flags/4x3/py.svg" width="18" alt=""> Paraguay | FT | **4-1** <br><sub>Prediction: 1-1</sub> | 51 % | 26 % | 23 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/au.svg" width="18" alt=""> Australia vs <img src="MondialXboost.Web/wwwroot/flags/4x3/tr.svg" width="18" alt=""> Turkey | FT | **2-0** <br><sub>Prediction: 1-1</sub> | 48 % | 26 % | 27 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/us.svg" width="18" alt=""> United States vs <img src="MondialXboost.Web/wwwroot/flags/4x3/au.svg" width="18" alt=""> Australia | Jun 19 19:00 UTC | 1-1 | 36 % | 26 % | 37 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/py.svg" width="18" alt=""> Paraguay vs <img src="MondialXboost.Web/wwwroot/flags/4x3/tr.svg" width="18" alt=""> Turkey | Jun 20 03:00 UTC | 1-1 | 34 % | 27 % | 38 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/py.svg" width="18" alt=""> Paraguay vs <img src="MondialXboost.Web/wwwroot/flags/4x3/au.svg" width="18" alt=""> Australia | Jun 26 02:00 UTC | 0-1 | 22 % | 29 % | 49 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/us.svg" width="18" alt=""> United States vs <img src="MondialXboost.Web/wwwroot/flags/4x3/tr.svg" width="18" alt=""> Turkey | Jun 26 02:00 UTC | 1-1 | 53 % | 22 % | 25 % |

</details>

<details open>
<summary><strong>Group E</strong></summary>

| Match | Status | Result / Pick | H | D | A |
| --- | --- | --- | ---: | ---: | ---: |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/de.svg" width="18" alt=""> Germany vs <img src="MondialXboost.Web/wwwroot/flags/4x3/cw.svg" width="18" alt=""> Curacao | FT | **7-1** <br><sub>Prediction: 2-0</sub> | 75 % | 15 % | 10 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/ci.svg" width="18" alt=""> Ivory Coast vs <img src="MondialXboost.Web/wwwroot/flags/4x3/ec.svg" width="18" alt=""> Ecuador | Jun 14 23:00 UTC | 0-0 | 34 % | 32 % | 34 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/de.svg" width="18" alt=""> Germany vs <img src="MondialXboost.Web/wwwroot/flags/4x3/ci.svg" width="18" alt=""> Ivory Coast | Jun 20 20:00 UTC | 1-1 | 41 % | 27 % | 32 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/cw.svg" width="18" alt=""> Curacao vs <img src="MondialXboost.Web/wwwroot/flags/4x3/ec.svg" width="18" alt=""> Ecuador | Jun 21 00:00 UTC | 0-1 | 15 % | 23 % | 62 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/cw.svg" width="18" alt=""> Curacao vs <img src="MondialXboost.Web/wwwroot/flags/4x3/ci.svg" width="18" alt=""> Ivory Coast | Jun 25 20:00 UTC | 0-2 | 13 % | 21 % | 66 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/de.svg" width="18" alt=""> Germany vs <img src="MondialXboost.Web/wwwroot/flags/4x3/ec.svg" width="18" alt=""> Ecuador | Jun 25 20:00 UTC | 1-1 | 43 % | 28 % | 29 % |

</details>

<details open>
<summary><strong>Group F</strong></summary>

| Match | Status | Result / Pick | H | D | A |
| --- | --- | --- | ---: | ---: | ---: |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/nl.svg" width="18" alt=""> Netherlands vs <img src="MondialXboost.Web/wwwroot/flags/4x3/jp.svg" width="18" alt=""> Japan | Jun 14 20:00 UTC | 1-1 | 35 % | 27 % | 38 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/se.svg" width="18" alt=""> Sweden vs <img src="MondialXboost.Web/wwwroot/flags/4x3/tn.svg" width="18" alt=""> Tunisia | Jun 15 02:00 UTC | 1-1 | 33 % | 28 % | 38 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/nl.svg" width="18" alt=""> Netherlands vs <img src="MondialXboost.Web/wwwroot/flags/4x3/se.svg" width="18" alt=""> Sweden | Jun 20 17:00 UTC | 2-1 | 59 % | 21 % | 20 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/jp.svg" width="18" alt=""> Japan vs <img src="MondialXboost.Web/wwwroot/flags/4x3/tn.svg" width="18" alt=""> Tunisia | Jun 21 04:00 UTC | 1-0 | 51 % | 27 % | 21 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/jp.svg" width="18" alt=""> Japan vs <img src="MondialXboost.Web/wwwroot/flags/4x3/se.svg" width="18" alt=""> Sweden | Jun 25 23:00 UTC | 2-1 | 62 % | 20 % | 18 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/nl.svg" width="18" alt=""> Netherlands vs <img src="MondialXboost.Web/wwwroot/flags/4x3/tn.svg" width="18" alt=""> Tunisia | Jun 25 23:00 UTC | 1-0 | 49 % | 27 % | 24 % |

</details>

<details open>
<summary><strong>Group G</strong></summary>

| Match | Status | Result / Pick | H | D | A |
| --- | --- | --- | ---: | ---: | ---: |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/be.svg" width="18" alt=""> Belgium vs <img src="MondialXboost.Web/wwwroot/flags/4x3/eg.svg" width="18" alt=""> Egypt | Jun 15 19:00 UTC | 1-0 | 47 % | 28 % | 25 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/ir.svg" width="18" alt=""> Iran vs <img src="MondialXboost.Web/wwwroot/flags/4x3/nz.svg" width="18" alt=""> New Zealand | Jun 16 01:00 UTC | 1-1 | 52 % | 25 % | 23 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/be.svg" width="18" alt=""> Belgium vs <img src="MondialXboost.Web/wwwroot/flags/4x3/ir.svg" width="18" alt=""> Iran | Jun 21 19:00 UTC | 1-1 | 38 % | 27 % | 36 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/eg.svg" width="18" alt=""> Egypt vs <img src="MondialXboost.Web/wwwroot/flags/4x3/nz.svg" width="18" alt=""> New Zealand | Jun 22 01:00 UTC | 1-1 | 39 % | 30 % | 32 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/be.svg" width="18" alt=""> Belgium vs <img src="MondialXboost.Web/wwwroot/flags/4x3/nz.svg" width="18" alt=""> New Zealand | Jun 27 03:00 UTC | 1-1 | 54 % | 24 % | 22 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/eg.svg" width="18" alt=""> Egypt vs <img src="MondialXboost.Web/wwwroot/flags/4x3/ir.svg" width="18" alt=""> Iran | Jun 27 03:00 UTC | 0-1 | 26 % | 29 % | 45 % |

</details>

<details open>
<summary><strong>Group H</strong></summary>

| Match | Status | Result / Pick | H | D | A |
| --- | --- | --- | ---: | ---: | ---: |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/sa.svg" width="18" alt=""> Saudi Arabia vs <img src="MondialXboost.Web/wwwroot/flags/4x3/uy.svg" width="18" alt=""> Uruguay | Jun 15 22:00 UTC | 0-1 | 22 % | 31 % | 47 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/es.svg" width="18" alt=""> Spain vs <img src="MondialXboost.Web/wwwroot/flags/4x3/sa.svg" width="18" alt=""> Saudi Arabia | Jun 21 16:00 UTC | 1-0 | 59 % | 25 % | 17 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/es.svg" width="18" alt=""> Spain vs <img src="MondialXboost.Web/wwwroot/flags/4x3/uy.svg" width="18" alt=""> Uruguay | Jun 27 00:00 UTC | 1-1 | 45 % | 28 % | 28 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/cv.svg" width="18" alt=""> Cape Verde vs <img src="MondialXboost.Web/wwwroot/flags/4x3/sa.svg" width="18" alt=""> Saudi Arabia | Scheduled | 0-1 | 27 % | 31 % | 41 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/cv.svg" width="18" alt=""> Cape Verde vs <img src="MondialXboost.Web/wwwroot/flags/4x3/uy.svg" width="18" alt=""> Uruguay | Scheduled | 0-1 | 17 % | 27 % | 56 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/es.svg" width="18" alt=""> Spain vs <img src="MondialXboost.Web/wwwroot/flags/4x3/cv.svg" width="18" alt=""> Cape Verde | Scheduled | 2-0 | 68 % | 20 % | 12 % |

</details>

<details open>
<summary><strong>Group I</strong></summary>

| Match | Status | Result / Pick | H | D | A |
| --- | --- | --- | ---: | ---: | ---: |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/fr.svg" width="18" alt=""> France vs <img src="MondialXboost.Web/wwwroot/flags/4x3/sn.svg" width="18" alt=""> Senegal | Jun 16 19:00 UTC | 1-1 | 41 % | 27 % | 32 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/iq.svg" width="18" alt=""> Iraq vs <img src="MondialXboost.Web/wwwroot/flags/4x3/no.svg" width="18" alt=""> Norway | Jun 16 22:00 UTC | 0-1 | 22 % | 27 % | 51 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/fr.svg" width="18" alt=""> France vs <img src="MondialXboost.Web/wwwroot/flags/4x3/iq.svg" width="18" alt=""> Iraq | Jun 22 21:00 UTC | 1-0 | 56 % | 26 % | 18 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/sn.svg" width="18" alt=""> Senegal vs <img src="MondialXboost.Web/wwwroot/flags/4x3/no.svg" width="18" alt=""> Norway | Jun 23 00:00 UTC | 1-1 | 38 % | 27 % | 36 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/fr.svg" width="18" alt=""> France vs <img src="MondialXboost.Web/wwwroot/flags/4x3/no.svg" width="18" alt=""> Norway | Jun 26 19:00 UTC | 1-1 | 43 % | 25 % | 32 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/sn.svg" width="18" alt=""> Senegal vs <img src="MondialXboost.Web/wwwroot/flags/4x3/iq.svg" width="18" alt=""> Iraq | Jun 26 19:00 UTC | 1-0 | 50 % | 29 % | 21 % |

</details>

<details open>
<summary><strong>Group J</strong></summary>

| Match | Status | Result / Pick | H | D | A |
| --- | --- | --- | ---: | ---: | ---: |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/ar.svg" width="18" alt=""> Argentina vs <img src="MondialXboost.Web/wwwroot/flags/4x3/dz.svg" width="18" alt=""> Algeria | Jun 17 01:00 UTC | 1-1 | 39 % | 26 % | 34 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/at.svg" width="18" alt=""> Austria vs <img src="MondialXboost.Web/wwwroot/flags/4x3/jo.svg" width="18" alt=""> Jordan | Jun 17 04:00 UTC | 1-1 | 51 % | 25 % | 24 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/ar.svg" width="18" alt=""> Argentina vs <img src="MondialXboost.Web/wwwroot/flags/4x3/at.svg" width="18" alt=""> Austria | Jun 22 17:00 UTC | 1-1 | 46 % | 26 % | 27 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/dz.svg" width="18" alt=""> Algeria vs <img src="MondialXboost.Web/wwwroot/flags/4x3/jo.svg" width="18" alt=""> Jordan | Jun 23 03:00 UTC | 1-1 | 60 % | 22 % | 18 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/dz.svg" width="18" alt=""> Algeria vs <img src="MondialXboost.Web/wwwroot/flags/4x3/at.svg" width="18" alt=""> Austria | Jun 28 02:00 UTC | 1-1 | 42 % | 28 % | 30 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/ar.svg" width="18" alt=""> Argentina vs <img src="MondialXboost.Web/wwwroot/flags/4x3/jo.svg" width="18" alt=""> Jordan | Jun 28 02:00 UTC | 2-0 | 63 % | 21 % | 16 % |

</details>

<details open>
<summary><strong>Group K</strong></summary>

| Match | Status | Result / Pick | H | D | A |
| --- | --- | --- | ---: | ---: | ---: |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/pt.svg" width="18" alt=""> Portugal vs <img src="MondialXboost.Web/wwwroot/flags/4x3/cd.svg" width="18" alt=""> Congo DR | Jun 17 17:00 UTC | 1-0 | 52 % | 29 % | 19 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/uz.svg" width="18" alt=""> Uzbekistan vs <img src="MondialXboost.Web/wwwroot/flags/4x3/co.svg" width="18" alt=""> Colombia | Jun 18 02:00 UTC | 1-1 | 26 % | 28 % | 46 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/pt.svg" width="18" alt=""> Portugal vs <img src="MondialXboost.Web/wwwroot/flags/4x3/uz.svg" width="18" alt=""> Uzbekistan | Jun 23 17:00 UTC | 1-1 | 46 % | 28 % | 26 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/cd.svg" width="18" alt=""> Congo DR vs <img src="MondialXboost.Web/wwwroot/flags/4x3/co.svg" width="18" alt=""> Colombia | Jun 24 02:00 UTC | 0-1 | 19 % | 29 % | 52 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/cd.svg" width="18" alt=""> Congo DR vs <img src="MondialXboost.Web/wwwroot/flags/4x3/uz.svg" width="18" alt=""> Uzbekistan | Jun 27 23:30 UTC | 0-0 | 26 % | 35 % | 39 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/pt.svg" width="18" alt=""> Portugal vs <img src="MondialXboost.Web/wwwroot/flags/4x3/co.svg" width="18" alt=""> Colombia | Jun 27 23:30 UTC | 1-1 | 37 % | 26 % | 37 % |

</details>

<details open>
<summary><strong>Group L</strong></summary>

| Match | Status | Result / Pick | H | D | A |
| --- | --- | --- | ---: | ---: | ---: |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/gb-eng.svg" width="18" alt=""> England vs <img src="MondialXboost.Web/wwwroot/flags/4x3/hr.svg" width="18" alt=""> Croatia | Jun 17 20:00 UTC | 1-1 | 46 % | 26 % | 28 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/gh.svg" width="18" alt=""> Ghana vs <img src="MondialXboost.Web/wwwroot/flags/4x3/pa.svg" width="18" alt=""> Panama | Jun 17 23:00 UTC | 1-1 | 30 % | 26 % | 44 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/gb-eng.svg" width="18" alt=""> England vs <img src="MondialXboost.Web/wwwroot/flags/4x3/gh.svg" width="18" alt=""> Ghana | Jun 23 20:00 UTC | 2-0 | 68 % | 20 % | 12 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/hr.svg" width="18" alt=""> Croatia vs <img src="MondialXboost.Web/wwwroot/flags/4x3/pa.svg" width="18" alt=""> Panama | Jun 23 23:00 UTC | 1-1 | 53 % | 24 % | 23 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/hr.svg" width="18" alt=""> Croatia vs <img src="MondialXboost.Web/wwwroot/flags/4x3/gh.svg" width="18" alt=""> Ghana | Jun 27 21:00 UTC | 1-0 | 59 % | 24 % | 18 % |
| <img src="MondialXboost.Web/wwwroot/flags/4x3/gb-eng.svg" width="18" alt=""> England vs <img src="MondialXboost.Web/wwwroot/flags/4x3/pa.svg" width="18" alt=""> Panama | Jun 27 21:00 UTC | 2-0 | 64 % | 20 % | 16 % |

</details>
<!-- mondial-xboost:snapshots:end -->

---

## ML Pipeline

### Datos

| Fuente | Partidos | Rango | Notas |
|--------|----------|-------|-------|
| `MondialXboost.Web/Data/historical_results.csv` | ~49,000 | 1974-2026 | Dataset canónico desduplicado y validado |

Historialmente el proyecto consumía datos de football-data.co.uk (~125k partidos de 38 ligas). El pipeline canónico actual usa `historical_results.csv`, que se depura activamente (eliminación de duplicados, normalización de nombres, validación de scores contradictorios).

### Feature Engineering

22 features por partido, calculadas usando solo información anterior al partido (sin data leakage):

| Categoría | Features | Descripción |
|-----------|----------|-------------|
| **Elo** | `elo_diff` | Rating dinámico tipo eloratings.net. K variable por importancia del partido. Home advantage +100 pts **solo si `neutral=False`**. |
| **Forma local (5/10)** | `home_points_avg_5`, `home_points_avg_10`, `home_goals_scored_avg_10`, `home_goals_conceded_avg_10`, `home_win_rate_10`, `home_draw_rate_10`, `home_loss_rate_10` | Rolling windows con `shift(1)`. Puntos promedio, goles anotados/recibidos, tasas W/D/L **del equipo local**. |
| **Forma visitante (5/10)** | `away_points_avg_5`, `away_points_avg_10`, `away_goals_scored_avg_10`, `away_goals_conceded_avg_10`, `away_win_rate_10`, `away_draw_rate_10`, `away_loss_rate_10` | Mismo set que local pero **del equipo visitante**. |
| **H2H** | `h2h_wins_diff`, `h2h_goals_avg`, `h2h_last_result`, `h2h_years_since` | Historial directo entre los dos equipos. Solo partidos anteriores. |
| **Contexto** | `home_matches_played`, `away_matches_played`, `neutral` | Partidos jugados por cada equipo y bandera de partido neutral. |

### Algoritmo

**XGBoost** es el único motor de ML del proyecto. Descartamos LightGBM, CatBoost, RandomForest y GradientBoosting de sklearn tras evaluar que:

- XGBoost ofrece la mejor relación velocidad/precisión para nuestro dataset.
- RandomForest sufre de overfitting severo (gap train-test > 15%).
- GradientBoosting de sklearn es demasiado lento para el volumen de datos.
- LightGBM/CatBoost aportan mejoras marginales pero añaden dependencias extra.

Optuna ajusta los hiperparámetros de XGBoost en cada batch:

| Hiperparámetro | Rango |
|----------------|-------|
| `n_estimators` | 100 - 1000 |
| `max_depth` | 3 - 12 |
| `learning_rate` | 0.01 - 0.3 |
| `subsample` | 0.5 - 1.0 |
| `colsample_bytree` | 0.5 - 1.0 |
| `reg_alpha` / `reg_lambda` | 1e-4 - 10 |
| `min_child_weight` | 1 - 10 |
| `gamma` | 0 - 1 |

### Validación

- **3-way temporal split:** train (< 2023) / val (2023) / test (>= 2024)
- **Walk-forward validation** con 3 folds temporales
- **Data leakage audit:** K-factor sin goal_diff, `shift(1)` en rolling, H2H solo partidos anteriores, Elo actualizado secuencialmente en orden cronológico.

### Modelo canónico y manifest

Cada vez que se entrena un modelo se genera `data/models/model_manifest.json` con:

- Fecha de entrenamiento y hash del dataset (`sha256` del CSV canónico).
- Lista exacta de `feature_cols` usada.
- Hiperparámetros de cada submodelo (outcome, home goals, away goals).
- Métricas de entrenamiento (`log_loss`, `accuracy`) etiquetadas explícitamente como *training*.
- Hashes de los artefactos `.pkl` para reproducibilidad.

Esto permite auditar qué modelo está en producción y reproducirlo a partir del código y los datos.

### Loop Engineering

Cada batch de `N` trials de Optuna genera un reporte HTML con:
- Configuración de entrada (dataset, features, split)
- Resultados de cada trial de XGBoost (accuracy, log loss, brier, overfit gap)
- Hiperparámetros del mejor modelo
- Feature importance top 15
- CSV parcial guardado en cada trial (resistente a cortes de Colab)

**Target:** 85% accuracy (baseline random: 33%, actual: ~58%)

### CLI local

El repo incluye wrappers para ejecutar el CLI desde el directorio del proyecto:

- **Git Bash / WSL / Linux / macOS:** `./mondial`
- **Windows cmd:** `mondial.cmd`
- **Windows PowerShell:** `\.mondial.cmd`

**Importante:** tenés que estar parado en la carpeta raíz del proyecto (`d:\martin\Proyectos\Mondial-xBoost` o donde lo hayas clonado).

**Si no sabés por dónde empezar, ejecutá el menú interactivo:**

```bash
# Git Bash
./mondial

# PowerShell
\.mondial.cmd

# cmd
mondial.cmd
```

También podés ver la guía de uso en cualquier momento:

```bash
./mondial guia
```

Comandos principales:

```bash
./mondial instalar                       # instala dependencias y el paquete
./mondial entrenar                       # entrena el modelo canónico
./mondial entrenar --elo-decay 4 --elo-recent 8   # Elo con decay temporal
./mondial entrenar-gpu                   # entrena con GPU (XGBOOST_DEVICE=cuda)
./mondial entrenar-cold-start            # entrena modelo de cold-start
./mondial predecir --home Brazil --away Morocco
./mondial predecir --home Brazil --away Morocco --blend
./mondial test                           # pytest tests/
./mondial lint                           # ruff check
./mondial gates                          # verify_gates
./mondial backtest                       # backtest de World Cup
./mondial bridge                         # smoke test del bridge C# <-> Python
./mondial elo                            # compara Elo contra World Football Elo
./mondial auditar                        # audita leakage temporal
./mondial loop --trials 50               # tuning con Optuna
./mondial data-council                   # revisión del data council
./mondial dashboard                      # dashboard de entrenamiento
./mondial servidor                       # levanta el bridge FastAPI
./mondial health                         # consulta /health del servidor
./mondial manifest                       # muestra model_manifest.json
./mondial limpiar                        # borra caché y artefactos de test
./mondial info                           # información del entorno
```

Cada comando tiene su propia ayuda con ejemplos:

```bash
./mondial entrenar --help
./mondial predecir --help
./mondial loop --help
```

#### Usar `mondial` como comando global

Si querés escribir solo `mondial` desde cualquier carpeta, instalá el paquete en modo editable (con el entorno virtual activado):

```bash
# Windows
venv\Scripts\python -m pip install -e .
venv\Scripts\mondial --help

# Git Bash / WSL / Linux / macOS
./venv/bin/python -m pip install -e .
./venv/bin/mondial --help
```

Si el entorno virtual está activado (`venv\Scripts\activate` o `source venv/bin/activate`), simplemente:

```bash
pip install -e .
mondial --help
```

### Google Colab

El entrenamiento corre en Google Colab con T4 GPU:

```
colab/mondial-xboost_xgboost.ipynb    # Notebook XGBoost único (7 celdas)
colab/data/all_matches.parquet   # Dataset (1.2 MB, 125k partidos)
```

**Setup:**
1. Subir `all_matches.parquet` a Google Drive (`Mondial-Xboost/data/`)
2. Subir notebook a Colab
3. Runtime → T4 GPU → Run All
4. Reporte HTML se guarda en Drive (`batch_N.html`)

---

## Arquitectura Original (.NET)

El sistema original es una app .NET 9 Blazor Server que genera predicciones con simulación Monte Carlo:

### Qué hace

- Importa datos seed desde CSV: grupos, resultados históricos, rankings FIFA, ratings Elo.
- Construye predicciones a través de modelos escalonados:
  - baseline uniforme
  - ranking FIFA
  - Elo
  - forma reciente
  - modelo Poisson con ajuste Dixon-Coles para bajos marcadores
  - modelo de goles ajustado por contexto reciente y disponibilidad de jugadores
- Selecciona el modelo más alto usable como oráculo final.
- Ejecuta simulación Monte Carlo repetible del torneo completo.
- Guarda predicciones y las evalúa después con Brier score, RPS, log loss, y accuracy de top-pick.

### Tech Stack

- .NET 9
- Blazor Server con MudBlazor
- Entity Framework Core 9
- SQLite
- CsvHelper
- xUnit

### Pantallas principales

- `/` - overview y model ladder
- `/lab` - comparar dos equipos a través del prediction ladder
- `/matches` - fixtures de grupos, snapshots de predicciones, refresh de contexto, ingreso de resultados
- `/fixture` - vista completa de fixture
- `/tournament` - ejecutar simulación Monte Carlo del torneo
- `/tournament/snapshots` - inspeccionar proyecciones guardadas
- `/performance` - métricas de evaluación de predicciones
- `/data` - import CSV, refresh rankings, refresh API-Football, refresh disponibilidad

### Estructura del proyecto

```
Mondial-Xboost.sln
MondialXboost.Web/
  Components/          Blazor pages, layout, y UI compartida
  DAL/                 EF Core DbContext
  Data/                CSV seed data y notas de video
  Helpers/             Parsing CSV, normalización de nombres, crypto
  Models/              Domain, CSV, API-Football, snapshot, y evaluación
  Predictors/          Model ladder y selector final
  Probability/         Outcome, scoreline, y probabilidad de torneo
  Services/            Import, predicción, rankings, API, disponibilidad, snapshots, evaluación
    Simulation/        Bracket de World Cup y motor Monte Carlo
MondialXboost.Web.Tests/   Tests xUnit
```

### Getting Started

```bash
dotnet restore
dotnet run --project MondialXboost.Web
```

La base de datos SQLite se crea automáticamente al inicio, y los datos seed se importan cuando se necesitan.

### Configuración

Settings en `MondialXboost.Web/appsettings.json` bajo la sección `Mondial-Xboost`:

- `SimulationCount` y `SimulationSeed`
- `RecentResultCount`
- `GoalModelYearsWindow`
- `RankingRefreshOnStartup`
- `FifaRankingsRawUrl`
- `EloRankingsBaseUrl`
- `ApiFootballApiKey`
- `OpenRouterApiKey`
- `AvailabilitySourceUrls`

### Testing

```bash
dotnet test
```

### Data Sources

CSV seed data en `MondialXboost.Web/Data`:

- `wc2026_groups.csv`
- `historical_results.csv`
- `fifa_rankings.csv`
- `elo_snapshot.csv`

### Loop Engineering

El proyecto usa un sistema de gates ejecutables para garantizar calidad antes de
mergear o desplegar:

```bash
# Ejecutar todos los gates (Python; .NET se salta si dotnet no está disponible)
python scripts/verify_gates.py

# Data Council
python scripts/run_data_council.py

# Backtest gate
python scripts/run_backtest_gate.py

# Bridge smoke test
python scripts/run_bridge_smoke_test.py
```

Configuración de gates: `.agents/skills/loop-engineering/gates.json`  
Estado del pipeline: `.agents/logs/pipeline-state.json`

### Documentación

- Pipeline de datos: `docs/vault/03-Architecture/data-pipeline-flow.md`
- Algoritmo Elo: `docs/vault/03-Architecture/elo-algorithm.md`
- Ponderación de jugadores: `docs/vault/03-Architecture/player-weighting.md`
- Flow HTML interactivo: `docs/flows/mondial-xboost-data-flow.html`
- Consejo de agentes: `.agents/agents/data-council.md`
- Comparación Elo vs World Football Elo: `backtest/results/elo_comparison.json`
- Estrategia de investigación y experimentos: `docs/vault/05-Research/model-coverage-strategy.md`
