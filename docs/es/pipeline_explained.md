# Pipeline Explicado: Guía de Implementación Paso a Paso

## Resumen General

Este documento recorre el pipeline de BSAD desde la perspectiva de un data scientist/ML engineer. Cubriremos cada paso en detalle, incluyendo el código, el razonamiento y consejos prácticos.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Generar   │───▶│  Construir  │───▶│  Entrenar   │───▶│  Puntuar    │───▶│   Evaluar   │
│    Datos    │    │  Features   │    │   Modelo    │    │  Anomalías  │    │  Resultados │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## Paso 1: Generación de Datos

### Propósito
Crear logs de eventos de seguridad sintéticos con patrones de ataque realistas para desarrollo y pruebas del modelo.

### Ubicación del Código
`src/bsad/steps.py` → `generate_data()`

### Entrada/Salida
```python
Entrada: Settings(n_entities=200, n_days=30, attack_rate=0.02)
Salida:  (events_df, attacks_df)  # pd.DataFrames
```

### Cómo Funciona

```python
def generate_data(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera eventos de seguridad sintéticos con ataques inyectados.

    El generador crea:
    1. Tasas base de entidad desde distribución Gamma
    2. Ventanas temporales con patrones día/noche
    3. Conteos de eventos por entidad-ventana
    4. Ataques inyectados a la tasa especificada
    """
    rng = np.random.default_rng(settings.random_seed)

    # 1. Crear entidades con tasas base variables
    entity_rates = rng.gamma(shape=2, scale=5, size=settings.n_entities)

    # 2. Generar ventanas temporales
    start_date = pd.Timestamp('2024-01-01')
    windows = pd.date_range(start_date, periods=settings.n_days * 24, freq='H')

    # 3. Generar eventos por entidad-ventana
    events = []
    for entity_id in range(settings.n_entities):
        for window in windows:
            # Aplicar patrón día/noche
            hour = window.hour
            time_factor = 0.5 + 0.5 * np.sin(np.pi * (hour - 6) / 12)

            # Generar conteo
            rate = entity_rates[entity_id] * time_factor
            count = rng.poisson(rate)

            events.append({
                'entity_id': f'user_{entity_id:04d}',
                'time_window': window,
                'event_count': count,
            })

    events_df = pd.DataFrame(events)

    # 4. Inyectar ataques
    attacks_df = _inject_attacks(events_df, settings, rng)

    return events_df, attacks_df
```

### Lógica de Inyección de Ataques

```python
def _inject_attacks(df: pd.DataFrame, settings: Settings, rng) -> pd.DataFrame:
    """Inyecta patrones de ataque en un subconjunto de entidad-ventanas."""

    n_attack_windows = int(len(df) * settings.attack_rate)
    attack_indices = rng.choice(len(df), size=n_attack_windows, replace=False)

    attack_types = ['brute_force', 'credential_stuffing', 'geo_anomaly', 'device_anomaly']
    attack_multipliers = {
        'brute_force': (10, 50),       # Alta intensidad
        'credential_stuffing': (3, 8), # Intensidad moderada
        'geo_anomaly': (1, 3),         # Baja intensidad (marcado por ubicación)
        'device_anomaly': (1, 2),      # Baja intensidad (marcado por dispositivo)
    }

    attacks = []
    for idx in attack_indices:
        attack_type = rng.choice(attack_types)
        mult_range = attack_multipliers[attack_type]
        multiplier = rng.uniform(*mult_range)

        # Amplificar conteo de eventos
        df.loc[idx, 'event_count'] = int(df.loc[idx, 'event_count'] * multiplier)
        df.loc[idx, 'is_attack'] = True
        df.loc[idx, 'attack_type'] = attack_type

        attacks.append({
            'index': idx,
            'attack_type': attack_type,
            'multiplier': multiplier,
        })

    return pd.DataFrame(attacks)
```

### Consejos Prácticos

1. **Semillar Todo**: Siempre pasar `random_seed` para reproducibilidad
2. **Patrones Realistas**: El sinusoide día/noche imita comportamiento real de usuarios
3. **Intensidades Variadas**: Diferentes tipos de ataque tienen diferente detectabilidad
4. **Ground Truth**: Guardar `attacks_df` para evaluación posterior

---

## Paso 2: Ingeniería de Features

### Propósito
Transformar eventos crudos en una tabla lista para modelado con una fila por entidad-ventana.

### Ubicación del Código
`src/bsad/steps.py` → `build_features()`

### Entrada/Salida
```python
Entrada: events_df (eventos crudos), Settings
Salida:  (modeling_df, metadata)  # pd.DataFrame, dict
```

### Cómo Funciona

```python
def build_features(events_df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict]:
    """
    Construye tabla de features para modelado.

    Transformaciones clave:
    1. Agregar eventos por (entidad, ventana_tiempo)
    2. Crear índices numéricos de entidad para PyMC
    3. Calcular estadísticas resumidas
    """
    # 1. Agregar (si no está ya agregado)
    if 'event_count' not in events_df.columns:
        modeling_df = events_df.groupby(['entity_id', 'time_window']).agg(
            event_count=('event_id', 'count'),
            has_attack=('is_attack', 'any'),
        ).reset_index()
    else:
        modeling_df = events_df.copy()

    # 2. Crear mapeo de índices de entidad
    entity_map = {eid: idx for idx, eid in enumerate(modeling_df['entity_id'].unique())}
    modeling_df['entity_idx'] = modeling_df['entity_id'].map(entity_map)

    # 3. Calcular metadata
    metadata = {
        'n_entities': len(entity_map),
        'n_windows': modeling_df['time_window'].nunique(),
        'n_observations': len(modeling_df),
        'entity_map': entity_map,
    }

    return modeling_df, metadata
```

### Extracción de Arrays del Modelo

```python
def get_model_arrays(modeling_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Extrae arrays numpy para modelo PyMC.

    Retorna dict con:
    - 'y': conteos de eventos observados
    - 'entity_idx': índice de entidad por observación
    - 'n_entities': total de entidades únicas
    """
    return {
        'y': modeling_df['event_count'].values.astype(np.int64),
        'entity_idx': modeling_df['entity_idx'].values.astype(np.int64),
        'n_entities': modeling_df['entity_idx'].nunique(),
    }
```

### Esquema de Tabla de Features

| Columna | Tipo | Propósito |
|---------|------|-----------|
| `entity_id` | str | Identificador legible |
| `entity_idx` | int | Indexado desde cero para PyMC |
| `time_window` | datetime | Tiempo de inicio de ventana |
| `event_count` | int | **Feature principal** |
| `has_attack` | bool | Etiqueta ground truth |
| `attack_type` | str | Categoría de ataque (opcional) |

### Consejos Prácticos

1. **Conteos Enteros**: NegativeBinomial de PyMC espera conteos enteros
2. **Índices Contiguos**: `entity_idx` debe ser 0 a n-1 sin espacios
3. **Eficiencia de Memoria**: Convertir a arrays numpy antes de entrenar

---

## Paso 3: Entrenamiento del Modelo

### Propósito
Ajustar un modelo Bayesiano jerárquico para aprender tasas específicas de entidad con pooling parcial.

### Ubicación del Código
`src/bsad/steps.py` → `train_model()`

### Entrada/Salida
```python
Entrada: arrays (dict de get_model_arrays), Settings
Salida:  az.InferenceData  # objeto trace de ArviZ
```

### Cómo Funciona

```python
def train_model(arrays: dict, settings: Settings) -> az.InferenceData:
    """
    Entrena modelo jerárquico Binomial Negativo.

    Estructura del modelo:
    - Población: mu (tasa media), alpha (concentración)
    - Entidad: theta[i] (tasa específica de entidad)
    - Observación: y ~ NegBinomial(theta[entidad], phi)
    """
    with pm.Model() as model:
        # ===== Priors a Nivel de Población =====
        # mu: tasa de eventos esperada en todas las entidades
        mu = pm.Exponential('mu', lam=0.1)

        # alpha: controla fuerza del pooling
        # Alpha alto → tasas de entidad se agrupan alrededor de mu
        # Alpha bajo → tasas de entidad varían ampliamente
        alpha = pm.HalfNormal('alpha', sigma=2)

        # ===== Parámetros a Nivel de Entidad =====
        # theta: tasa específica de entidad (pooling parcial desde Gamma)
        # E[theta] = mu, Var[theta] = mu/alpha
        theta = pm.Gamma(
            'theta',
            alpha=mu * alpha,  # forma
            beta=alpha,         # tasa
            shape=arrays['n_entities']
        )

        # ===== Sobredispersión =====
        # phi: controla varianza más allá de Poisson
        # Varianza NegBinom = mu + mu^2/phi
        phi = pm.HalfNormal('phi', sigma=1)

        # ===== Verosimilitud =====
        y_obs = pm.NegativeBinomial(
            'y_obs',
            mu=theta[arrays['entity_idx']],
            alpha=phi,
            observed=arrays['y']
        )

        # ===== Muestreo =====
        trace = pm.sample(
            draws=settings.n_samples,
            tune=settings.n_tune,
            chains=settings.n_chains,
            random_seed=settings.random_seed,
            cores=settings.n_cores,
            target_accept=settings.target_accept,
            return_inferencedata=True,
        )

    return trace
```

### Entendiendo los Priors

```
Nivel de Población
==================
mu ~ Exp(0.1)
  → Media = 10
  → Cubre tasas de eventos típicas (1-100+)

alpha ~ HalfNormal(2)
  → Concentración moderada
  → Permite al modelo aprender fuerza del pooling

Nivel de Entidad
================
theta[i] ~ Gamma(mu*alpha, alpha)
  → E[theta] = mu (todas las entidades comparten media poblacional)
  → Var[theta] = mu/alpha (varianza controlada por alpha)

Nivel de Observación
====================
y ~ NegBinomial(theta[entidad], phi)
  → E[y] = theta
  → Var[y] = theta + theta²/phi (sobredispersión)
```

### Diagnósticos de Convergencia

```python
def get_diagnostics(trace: az.InferenceData) -> dict:
    """Verificar convergencia MCMC."""
    summary = az.summary(trace, var_names=['mu', 'alpha', 'phi'])

    return {
        'r_hat_max': float(summary['r_hat'].max()),
        'ess_bulk_min': int(summary['ess_bulk'].min()),
        'ess_tail_min': int(summary['ess_tail'].min()),
        'divergences': int(trace.sample_stats.diverging.sum()),
    }
```

**Umbrales de Calidad:**
| Métrica | Bueno | Malo |
|---------|-------|------|
| R-hat | < 1.01 | > 1.1 |
| ESS (bulk) | > 400 | < 100 |
| ESS (tail) | > 400 | < 100 |
| Divergencias | 0 | > 0 |

### Consejos Prácticos

1. **Empezar Pequeño**: Probar con 100 muestras primero para detectar errores
2. **Vigilar Divergencias**: Divergencias no cero → reparametrizar
3. **Target Accept**: Aumentar a 0.95 si ocurren divergencias
4. **Backend JAX**: Usar `pm.sample(nuts_sampler='numpyro')` para 10x más rápido

---

## Paso 4: Scoring de Anomalías

### Propósito
Calcular scores de anomalía e intervalos de incertidumbre usando la distribución predictiva posterior.

### Ubicación del Código
`src/bsad/steps.py` → `compute_scores()`, `compute_intervals()`

### Entrada/Salida
```python
Entrada: y (conteos), trace, entity_idx
Salida:  dict con 'anomaly_score', 'mean', 'std'
```

### Cómo Funciona

```python
def compute_scores(y: np.ndarray, trace: az.InferenceData,
                   entity_idx: np.ndarray) -> dict[str, np.ndarray]:
    """
    Calcula scores de anomalía desde la predictiva posterior.

    Score = -log P(y | posterior)

    Interpretación:
    - Score bajo (0-3): Comportamiento normal
    - Score medio (3-6): Inusual
    - Score alto (6+): Altamente anómalo
    """
    from scipy.stats import nbinom
    from scipy.special import logsumexp

    # Extraer muestras posteriores
    theta = trace.posterior['theta'].values  # (cadenas, muestras, entidades)
    phi = trace.posterior['phi'].values      # (cadenas, muestras)

    # Aplanar a través de cadenas
    theta_flat = theta.reshape(-1, theta.shape[-1])  # (muestras, entidades)
    phi_flat = phi.flatten()                          # (muestras,)

    n_samples = len(phi_flat)
    scores = np.zeros(len(y))

    for i, (y_i, idx_i) in enumerate(zip(y, entity_idx)):
        # Obtener muestras de tasa de la entidad
        theta_i = theta_flat[:, idx_i]

        # Parametrización NegBinom: n=phi, p=phi/(phi+mu)
        n_param = phi_flat
        p_param = phi_flat / (phi_flat + theta_i)

        # Log probabilidad bajo cada muestra posterior
        log_probs = nbinom.logpmf(y_i, n=n_param, p=p_param)

        # Promedio sobre posterior (truco log-sum-exp)
        avg_log_prob = logsumexp(log_probs) - np.log(n_samples)

        # Score de anomalía = log probabilidad negativa
        scores[i] = -avg_log_prob

    return {
        'anomaly_score': scores,
        'mean_score': scores.mean(),
        'std_score': scores.std(),
    }
```

### Intervalos de Credibilidad

```python
def compute_intervals(trace: az.InferenceData, entity_idx: np.ndarray,
                      credible_mass: float = 0.9) -> dict[str, np.ndarray]:
    """
    Calcula intervalos predictivos posteriores para cada observación.

    Retorna límites inferior/superior del intervalo de credibilidad del 90%.
    """
    from scipy.stats import nbinom

    theta = trace.posterior['theta'].values.reshape(-1, -1)
    phi = trace.posterior['phi'].values.flatten()

    n_obs = len(entity_idx)
    lower = np.zeros(n_obs)
    upper = np.zeros(n_obs)
    median = np.zeros(n_obs)

    alpha = (1 - credible_mass) / 2  # ej., 0.05 para IC del 90%

    for i, idx in enumerate(entity_idx):
        theta_i = theta[:, idx]

        # Generar muestras predictivas
        n_param = phi
        p_param = phi / (phi + theta_i)
        y_pred = nbinom.rvs(n=n_param, p=p_param)

        lower[i] = np.quantile(y_pred, alpha)
        upper[i] = np.quantile(y_pred, 1 - alpha)
        median[i] = np.median(y_pred)

    return {
        'lower': lower,
        'upper': upper,
        'median': median,
    }
```

### Creando el DataFrame con Scores

```python
def create_scored_df(modeling_df: pd.DataFrame, scores: dict,
                     intervals: dict) -> pd.DataFrame:
    """Combina datos de modelado con scores e intervalos."""
    scored_df = modeling_df.copy()

    scored_df['anomaly_score'] = scores['anomaly_score']
    scored_df['pred_lower'] = intervals['lower']
    scored_df['pred_upper'] = intervals['upper']
    scored_df['pred_median'] = intervals['median']

    # Ordenar por score (más alto primero)
    scored_df = scored_df.sort_values('anomaly_score', ascending=False)

    return scored_df
```

### Consejos Prácticos

1. **Vectorizar Cuando Sea Posible**: El bucle es lento; operaciones por lotes ayudan
2. **Usar Log-Sum-Exp**: Evitar underflow numérico con probabilidades pequeñas
3. **Ancho de IC**: Intervalos anchos → entidad incierta, estrechos → confiado
4. **Calibración de Scores**: Los scores no son probabilidades; calibrar si es necesario

---

## Paso 5: Evaluación

### Propósito
Medir el rendimiento de detección contra las etiquetas ground truth.

### Ubicación del Código
`src/bsad/steps.py` → `evaluate()`

### Entrada/Salida
```python
Entrada: scored_df (con 'anomaly_score' y 'has_attack')
Salida:  dict con métricas
```

### Cómo Funciona

```python
def evaluate(scored_df: pd.DataFrame, k_values: list[int] | None = None) -> dict:
    """
    Evalúa el rendimiento de detección de anomalías.

    Métricas:
    - PR-AUC: Área Bajo la Curva Precisión-Recall
    - ROC-AUC: AUC de Característica Operativa del Receptor
    - Recall@K: Fracción de ataques en los top K scores
    """
    from sklearn.metrics import (
        precision_recall_curve, roc_curve, auc
    )

    y_true = scored_df['has_attack'].astype(int).values
    scores = scored_df['anomaly_score'].values

    # PR-AUC (métrica principal para eventos raros)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)

    # ROC-AUC (para referencia)
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    # Recall@K
    k_values = k_values or [50, 100, 200]
    recall_at_k = {}
    total_attacks = y_true.sum()

    for k in k_values:
        if k <= len(scored_df):
            top_k = scored_df.head(k)['has_attack'].sum()
            recall_at_k[f'recall_at_{k}'] = top_k / total_attacks if total_attacks > 0 else 0

    return {
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'attack_rate': float(y_true.mean()),
        'n_attacks': int(total_attacks),
        'n_total': len(scored_df),
        **recall_at_k,
    }
```

### ¿Por Qué PR-AUC Sobre ROC-AUC?

```
Ejemplo con 2% de tasa de ataque:

Modelo que siempre predice "no ataque":
- Exactitud: 98%
- ROC-AUC: ~0.50 (aleatorio)
- PR-AUC: ~0.02 (igual a tasa de ataque)

Nuestro modelo:
- ROC-AUC: 0.93 (¡se ve genial!)
- PR-AUC: 0.85 (¡realmente genial!)

PR-AUC es más difícil de manipular con desbalance de clases.
```

### Consejos Prácticos

1. **Ordenar Primero**: `scored_df` debe estar ordenado por score (descendente)
2. **Elegir K Sabiamente**: K debe coincidir con capacidad del analista (50-100 típico)
3. **Comparación con Baseline**: PR-AUC aleatorio = tasa_ataque
4. **Por Tipo de Ataque**: Desglosar métricas por tipo de ataque para insights

---

## Ejecución Completa del Pipeline

### Usando la Clase Pipeline

```python
from bsad import Settings, Pipeline

# Configurar
settings = Settings(
    n_entities=200,
    n_days=30,
    n_samples=2000,
    random_seed=42,
)

# Ejecutar
pipeline = Pipeline(settings)
state = pipeline.run_demo()

# Acceder a resultados
print(f"PR-AUC: {state.metrics['pr_auc']:.3f}")
print(f"Top anomalía: {state.scored_df.iloc[0]['entity_id']}")
```

### Usando el CLI

```bash
# Demo completo
bsad demo --n-entities 200 --n-days 30 --samples 2000

# Paso a paso
bsad generate-data --output data/events.parquet
bsad train --input data/events.parquet --output outputs/model.nc
bsad score --model outputs/model.nc --output outputs/scores.parquet
bsad evaluate --scores outputs/scores.parquet
```

### Usando Funciones de Pasos Directamente

```python
from bsad import steps, io
from bsad.config import Settings

settings = Settings(n_entities=100, n_days=14)

# Paso 1: Generar
events_df, attacks_df = steps.generate_data(settings)

# Paso 2: Features
modeling_df, metadata = steps.build_features(events_df, settings)
arrays = steps.get_model_arrays(modeling_df)

# Paso 3: Entrenar
trace = steps.train_model(arrays, settings)

# Paso 4: Puntuar
scores = steps.compute_scores(arrays['y'], trace, arrays['entity_idx'])
intervals = steps.compute_intervals(trace, arrays['entity_idx'])
scored_df = steps.create_scored_df(modeling_df, scores, intervals)

# Paso 5: Evaluar
metrics = steps.evaluate(scored_df)
```

---

## Solución de Problemas

### Problemas Comunes

| Problema | Causa | Solución |
|----------|-------|----------|
| Divergencias | Problemas de geometría posterior | Aumentar `target_accept` a 0.95 |
| ESS bajo | Muestras correlacionadas | Aumentar `n_samples` |
| Muestreo lento | Dataset grande | Usar backend JAX |
| Error de memoria | Demasiadas entidades | Reducir tamaño de lote o usar VI |
| Score NaN | Observación con probabilidad cero | Agregar epsilon pequeño al log |

### Modo Debug

```python
# Ejecución mínima para debugging
settings = Settings(
    n_entities=20,
    n_days=7,
    n_samples=100,
    n_tune=100,
    n_chains=1,
)
```

---

## Resumen

| Paso | Entrada | Salida | Función Clave |
|------|---------|--------|---------------|
| 1. Generar | Settings | events_df, attacks_df | `generate_data()` |
| 2. Features | events_df | modeling_df, arrays | `build_features()` |
| 3. Entrenar | arrays | trace (InferenceData) | `train_model()` |
| 4. Puntuar | y, trace | scored_df | `compute_scores()` |
| 5. Evaluar | scored_df | dict métricas | `evaluate()` |

El pipeline está diseñado para ser:
- **Modular**: Cada paso es una función pura
- **Transparente**: Inspeccionar cualquier estado intermedio
- **Reproducible**: Semillar todas las operaciones aleatorias
- **Extensible**: Agregar nuevos pasos sin modificar los existentes
