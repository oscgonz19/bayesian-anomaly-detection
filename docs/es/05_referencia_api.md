# Referencia API

## Tabla de Contenidos
1. [Módulo de Generación de Datos](#módulo-de-generación-de-datos)
2. [Módulo de Características](#módulo-de-características)
3. [Módulo del Modelo](#módulo-del-modelo)
4. [Módulo de Puntuación](#módulo-de-puntuación)
5. [Módulo de Evaluación](#módulo-de-evaluación)
6. [Comandos CLI](#comandos-cli)

---

## Módulo de Generación de Datos

`bsad.data_generator`

### Clases

#### `GeneratorConfig`

Dataclass de configuración para generación de datos sintéticos.

```python
@dataclass
class GeneratorConfig:
    n_users: int = 200                    # Número de entidades usuario
    n_ips: int = 100                      # Número de direcciones IP
    n_endpoints: int = 50                 # Número de endpoints API
    n_days: int = 30                      # Días a simular
    events_per_user_day_mean: float = 5.0 # Media eventos por usuario por día
    events_per_user_day_std: float = 3.0  # Desviación estándar
    attack_rate: float = 0.02             # Fracción de ventanas-entidad con ataques
    random_seed: int = 42                 # Semilla para reproducibilidad

    # Parámetros de ataque
    brute_force_multiplier: tuple[int, int] = (50, 200)
    credential_stuffing_users: tuple[int, int] = (10, 30)
    credential_stuffing_events_per_user: tuple[int, int] = (3, 10)
    geo_anomaly_locations: int = 5
    device_anomaly_new_devices: tuple[int, int] = (3, 8)
```

### Funciones

#### `generate_synthetic_data`

```python
def generate_synthetic_data(
    config: GeneratorConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generar dataset completo de eventos de seguridad sintéticos.

    Parámetros
    ----------
    config : GeneratorConfig, opcional
        Configuración de generación. Usa valores por defecto si es None.

    Retorna
    -------
    events_df : pd.DataFrame
        Todos los eventos con columnas:
        - timestamp: datetime
        - user_id: str
        - ip_address: str
        - endpoint: str
        - status_code: int
        - location: str
        - device_fingerprint: str
        - bytes_transferred: int
        - is_attack: bool
        - attack_type: str

    attacks_df : pd.DataFrame
        Metadatos de ataques con columnas:
        - attack_type: str
        - target_entity: str o lista
        - source_ip: str
        - window_start: datetime
        - n_events: int

    Ejemplo
    -------
    >>> from bsad.data_generator import GeneratorConfig, generate_synthetic_data
    >>> config = GeneratorConfig(n_users=100, n_days=14, attack_rate=0.05)
    >>> events_df, attacks_df = generate_synthetic_data(config)
    >>> print(f"Generados {len(events_df)} eventos con {attacks_df['n_events'].sum()} eventos de ataque")
    """
```

---

## Módulo de Características

`bsad.features`

### Clases

#### `FeatureConfig`

```python
@dataclass
class FeatureConfig:
    entity_column: str = "user_id"        # Columna para agrupar
    window_size: Literal["1H", "6H", "1D"] = "1D"  # Ventana de agregación
    include_temporal: bool = True          # Añadir hora, día_semana, etc.
    include_categorical: bool = True       # Añadir codificaciones categóricas
```

### Funciones

#### `build_modeling_table`

```python
def build_modeling_table(
    events_df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Construir tabla de modelado completa desde eventos crudos.

    Realiza:
    1. Agregación por ventana de tiempo
    2. Extracción de características temporales
    3. Estadísticas a nivel de entidad
    4. Codificación de IDs de entidad

    Parámetros
    ----------
    events_df : pd.DataFrame
        Logs de eventos crudos
    config : FeatureConfig, opcional
        Configuración de ingeniería de características

    Retorna
    -------
    modeling_df : pd.DataFrame
        Tabla de características con columnas:
        - entity_idx: int (ID de entidad codificado)
        - window: datetime
        - event_count: int (variable objetivo)
        - unique_ips, unique_endpoints, etc.
        - hour, day_of_week, is_weekend, is_business_hours
        - entity_mean_count, entity_std_count, count_zscore
        - has_attack: bool
        - attack_type: str

    metadata : dict
        - entity_column: str
        - entity_mapping: dict[str, int]
        - n_entities: int
        - n_windows: int
        - attack_rate: float
        - feature_columns: list[str]
    """
```

#### `get_model_arrays`

```python
def get_model_arrays(
    modeling_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Extraer arrays numpy para modelo PyMC.

    Retorna
    -------
    dict con claves:
        - y: Conteos de eventos (int64), forma (n_obs,)
        - entity_idx: Índices de entidad (int64), forma (n_obs,)
        - is_attack: Etiquetas verdad conocida (bool), forma (n_obs,)
        - window_idx: Índices de ventana (int64), forma (n_obs,)
    """
```

---

## Módulo del Modelo

`bsad.model`

### Clases

#### `ModelConfig`

```python
@dataclass
class ModelConfig:
    # Parámetros de muestreo
    n_samples: int = 2000         # Muestras posteriores por cadena
    n_tune: int = 1000            # Muestras de calentamiento/ajuste
    n_chains: int = 4             # Cadenas MCMC independientes
    target_accept: float = 0.9    # Tasa de aceptación objetivo NUTS
    random_seed: int = 42         # Semilla de reproducibilidad

    # Parámetros de prior
    mu_prior_rate: float = 0.1           # Tasa Exponencial para μ
    alpha_prior_sd: float = 2.0          # σ HalfNormal para α
    overdispersion_prior_sd: float = 2.0 # σ HalfNormal para φ
```

### Funciones

#### `build_hierarchical_negbinom_model`

```python
def build_hierarchical_negbinom_model(
    y: np.ndarray,
    entity_idx: np.ndarray,
    n_entities: int,
    config: ModelConfig | None = None,
) -> pm.Model:
    """
    Construir modelo Binomial Negativo jerárquico.

    Estructura del modelo:
        μ ~ Exponential(0.1)           # Media poblacional
        α ~ HalfNormal(2)              # Concentración
        θ_i ~ Gamma(μα, α)             # Tasas por entidad
        φ ~ HalfNormal(2)              # Sobredispersión
        y ~ NegativeBinomial(θ, φ)     # Observaciones

    Parámetros
    ----------
    y : np.ndarray, forma (n_obs,)
        Conteos de eventos por observación
    entity_idx : np.ndarray, forma (n_obs,)
        Índice de entidad para cada observación (0 a n_entities-1)
    n_entities : int
        Número total de entidades únicas
    config : ModelConfig, opcional
        Configuración del modelo

    Retorna
    -------
    pm.Model
        Objeto modelo PyMC (aún no ajustado)
    """
```

#### `fit_model`

```python
def fit_model(
    model: pm.Model,
    config: ModelConfig | None = None,
) -> az.InferenceData:
    """
    Ajustar modelo usando sampler NUTS.

    Realiza:
    1. Muestreo MCMC con ajuste automático
    2. Muestreo predictivo posterior
    3. Diagnósticos de convergencia

    Retorna
    -------
    az.InferenceData
        InferenceData ArviZ con grupos:
        - posterior: Muestras de parámetros
        - posterior_predictive: Observaciones simuladas
        - sample_stats: Diagnósticos MCMC
        - observed_data: Datos de entrada
    """
```

#### `get_model_diagnostics`

```python
def get_model_diagnostics(trace: az.InferenceData) -> dict:
    """
    Calcular diagnósticos del modelo.

    Retorna
    -------
    dict con claves:
        - r_hat_max: R-hat máximo entre parámetros (objetivo: < 1.05)
        - ess_bulk_min: ESS bulk mínimo (objetivo: > 400)
        - ess_tail_min: ESS tail mínimo (objetivo: > 400)
        - divergences: Número de transiciones divergentes (objetivo: 0)
        - converged: bool, True si r_hat_max < 1.05
    """
```

---

## Módulo de Puntuación

`bsad.scoring`

### Funciones

#### `compute_anomaly_scores`

```python
def compute_anomaly_scores(
    y_observed: np.ndarray,
    trace: az.InferenceData,
    entity_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Calcular puntuaciones de anomalía desde predictivo posterior.

    puntuación = -log p(y_observado | predictivo posterior)

    Puntuaciones más altas indican observaciones más anómalas.

    Parámetros
    ----------
    y_observed : np.ndarray, forma (n_obs,)
        Conteos de eventos observados
    trace : az.InferenceData
        Trace del modelo ajustado con muestras posteriores
    entity_idx : np.ndarray, forma (n_obs,)
        Índice de entidad para cada observación

    Retorna
    -------
    dict con claves:
        - anomaly_score: np.ndarray, forma (n_obs,)
            Puntuaciones estimadas (media posterior)
        - score_std: np.ndarray, forma (n_obs,)
            Desviación estándar de puntuación entre posterior
        - score_lower: np.ndarray, forma (n_obs,)
            Percentil 5 de puntuaciones
        - score_upper: np.ndarray, forma (n_obs,)
            Percentil 95 de puntuaciones
    """
```

#### `create_scored_dataframe`

```python
def create_scored_dataframe(
    modeling_df: pd.DataFrame,
    scores: dict[str, np.ndarray],
    intervals: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """
    Crear DataFrame con puntuaciones de anomalía y metadatos.

    Añade columnas:
    - anomaly_score, score_std, score_lower, score_upper
    - predicted_mean, predicted_lower, predicted_upper
    - anomaly_rank: Ranking por puntuación (1 = más anómalo)
    - exceeds_interval: bool, conteo > predicted_upper

    Retorna DataFrame ordenado por anomaly_score descendente.
    """
```

---

## Módulo de Evaluación

`bsad.evaluation`

### Funciones

#### `compute_all_metrics`

```python
def compute_all_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    k_values: list[int] | None = None,
) -> dict:
    """
    Calcular métricas de evaluación comprehensivas.

    Parámetros
    ----------
    y_true : np.ndarray
        Etiquetas verdad conocida binarias (0/1)
    scores : np.ndarray
        Puntuaciones de anomalía (mayor = más anómalo)
    k_values : list[int], opcional
        Valores K para Recall@K (default: [10, 25, 50, 100])

    Retorna
    -------
    dict con claves:
        - pr_auc: AUC Precisión-Recall
        - roc_auc: AUC ROC
        - n_observations: Total observaciones
        - n_positives: Número de ataques
        - attack_rate: Fracción de ataques
        - recall_at_{k}: Recall en top K
        - precision_at_{k}: Precisión en top K
        - pr_curve: dict con arrays precisión, recall, umbrales
    """
```

---

## Comandos CLI

### `bsad generate-data`

```bash
bsad generate-data [OPCIONES]

Opciones:
  -n, --n-entities INTEGER   Número de entidades usuario [default: 200]
  -d, --n-days INTEGER       Número de días a simular [default: 30]
  -a, --attack-rate FLOAT    Fracción de ventanas-entidad con ataques [default: 0.02]
  -s, --seed INTEGER         Semilla aleatoria [default: 42]
  -o, --output PATH          Ruta de salida para eventos [default: data/events.parquet]
```

### `bsad train`

```bash
bsad train [OPCIONES]

Opciones:
  -i, --input PATH           Archivo de eventos entrada [default: data/events.parquet]
  -o, --output PATH          Ruta de salida del modelo [default: outputs/model.nc]
  -s, --samples INTEGER      Muestras posteriores [default: 2000]
  -t, --tune INTEGER         Muestras de ajuste [default: 1000]
  -c, --chains INTEGER       Cadenas MCMC [default: 4]
  --seed INTEGER             Semilla aleatoria [default: 42]
```

### `bsad score`

```bash
bsad score [OPCIONES]

Opciones:
  -m, --model PATH           Ruta del modelo entrenado [default: outputs/model.nc]
  -i, --input PATH           Ruta de tabla de modelado [default: outputs/modeling_table.parquet]
  -o, --output PATH          Ruta de salida de puntuaciones [default: outputs/scores.parquet]
```

### `bsad evaluate`

```bash
bsad evaluate [OPCIONES]

Opciones:
  -s, --scores PATH          Archivo de datos puntuados [default: outputs/scores.parquet]
  -o, --output PATH          Ruta de salida de métricas [default: outputs/metrics.json]
  -p, --plots PATH           Directorio de salida de gráficos [default: outputs/plots]
```

### `bsad demo`

```bash
bsad demo [OPCIONES]

Ejecutar pipeline completo: generar → entrenar → puntuar → evaluar

Opciones:
  -o, --output-dir PATH      Directorio de salida [default: outputs]
  -n, --n-entities INTEGER   Número de entidades [default: 200]
  -d, --n-days INTEGER       Número de días [default: 30]
  -s, --samples INTEGER      Muestras posteriores [default: 1000]
  --seed INTEGER             Semilla aleatoria [default: 42]
```

---

## Siguiente: [Tutorial](06_tutorial.md)
