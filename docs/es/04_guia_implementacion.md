# Guía de Implementación

## Tabla de Contenidos
1. [Generación de Datos Sintéticos](#generación-de-datos-sintéticos)
2. [Pipeline de Ingeniería de Características](#pipeline-de-ingeniería-de-características)
3. [Entrenamiento del Modelo](#entrenamiento-del-modelo)
4. [Puntuación de Anomalías](#puntuación-de-anomalías)
5. [Métricas de Evaluación](#métricas-de-evaluación)

---

## Generación de Datos Sintéticos

### Descripción General

El generador de datos crea logs de eventos de seguridad realistas con patrones de ataque conocidos. Esto permite:
- Desarrollo de modelos sin datos reales sensibles
- Evaluación controlada con etiquetas de verdad conocida
- Experimentos reproducibles

### Patrones de Ataque

#### 1. Ataque de Fuerza Bruta

```python
def inject_brute_force_attack(df, config, rng):
    """
    Características:
    - Una IP → Un usuario
    - 50-200 eventos en ventana de 1 hora
    - 90% intentos fallidos (401), 10% éxito al final
    - Ubicación fuente inusual (TOR, VPN, Desconocido)
    """
    n_events = rng.integers(50, 200)
    for i in range(n_events):
        status = 401 if i < n_events - 1 else 200  # Éxito final
        # Todos los eventos concentrados en una sola hora
```

#### 2. Credential Stuffing (Relleno de Credenciales)

```python
def inject_credential_stuffing_attack(df, config, rng):
    """
    Características:
    - Una IP → Múltiples usuarios (10-30)
    - 3-10 intentos por usuario
    - Menor tasa de éxito (15%)
    - Distribuido durante el día
    """
    n_target_users = rng.integers(10, 30)
    for user in target_users:
        n_attempts = rng.integers(3, 10)
        # Intentos distribuidos a lo largo del día
```

#### 3. Anomalía Geográfica

```python
def inject_geo_anomaly_attack(df, config, rng):
    """
    Características:
    - Credenciales legítimas desde ubicación sospechosa
    - Acceso exitoso (cuenta comprometida)
    - Mayor transferencia de datos (exfiltración)
    - Ubicaciones inusuales: Corea del Norte, Irán, TOR, VPN
    """
    locations = ["North-Korea", "Iran", "TOR-Exit", "Suspicious-Proxy"]
    bytes_transferred = rng.lognormal(8, 1)  # Mayor que normal
```

#### 4. Anomalía de Dispositivo

```python
def inject_device_anomaly_attack(df, config, rng):
    """
    Características:
    - Un usuario con muchas nuevas huellas de dispositivo
    - Indica compartición de cuenta o compromiso
    - Múltiples nuevos dispositivos en período corto (3-8)
    """
    n_new_devices = rng.integers(3, 8)
    for device in new_devices:
        # Generar huella única
        # Múltiples eventos por dispositivo
```

### Esquema de Eventos

| Campo | Tipo | Descripción |
|-------|------|-------------|
| timestamp | datetime | Marca de tiempo del evento |
| user_id | string | Identificador de usuario (user_0001) |
| ip_address | string | IP fuente (ip_0042 o attack_ip_XXXX) |
| endpoint | string | Endpoint API (/api/v1/login) |
| status_code | int | Estado HTTP (200, 401, 403, 500) |
| location | string | Región geográfica |
| device_fingerprint | string | Identificador de dispositivo (hash MD5) |
| bytes_transferred | int | Tamaño solicitud/respuesta |
| is_attack | bool | Etiqueta verdad conocida |
| attack_type | string | Categoría de ataque o "none" |

---

## Pipeline de Ingeniería de Características

### Agregación por Ventana de Tiempo

```python
def create_time_windows(df: pd.DataFrame, config: FeatureConfig):
    """
    Agregar eventos en características por entidad-ventana.

    Entrada: Eventos crudos (una fila por evento)
    Salida: Características agregadas (una fila por entidad-ventana)
    """
    # Crear identificador de ventana
    df["window"] = df["timestamp"].dt.floor(config.window_size)  # "1D", "1H"

    # Agregar
    grouped = df.groupby([entity_column, "window"]).agg({
        "timestamp": "count",           # event_count
        "ip_address": "nunique",        # unique_ips
        "endpoint": "nunique",          # unique_endpoints
        "device_fingerprint": "nunique",# unique_devices
        "location": "nunique",          # unique_locations
        "bytes_transferred": "sum",     # bytes_total
        "is_attack": "any",             # has_attack (verdad conocida)
    })
```

### Definiciones de Características

| Característica | Cálculo | Señal de Anomalía |
|----------------|---------|-------------------|
| event_count | COUNT(*) | Alto conteo → fuerza bruta |
| unique_ips | COUNT(DISTINCT ip) | Muchas IPs → ataque distribuido |
| unique_endpoints | COUNT(DISTINCT endpoint) | Muchos endpoints → reconocimiento |
| unique_devices | COUNT(DISTINCT device) | Muchos dispositivos → compartición cuenta |
| unique_locations | COUNT(DISTINCT location) | Muchas ubicaciones → anomalía geo |
| failed_count | COUNT(status IN (4xx, 5xx)) | Muchos fallos → fuerza bruta |
| bytes_total | SUM(bytes) | Muchos bytes → exfiltración |

### Características Temporales

```python
def add_temporal_features(df: pd.DataFrame):
    """Añadir características basadas en tiempo para detección de patrones."""
    df["hour"] = df["window"].dt.hour           # 0-23
    df["day_of_week"] = df["window"].dt.dayofweek  # 0=Lun, 6=Dom
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["is_business_hours"] = (
        (df["hour"] >= 9) &
        (df["hour"] <= 17) &
        (~df["is_weekend"])
    )
```

---

## Entrenamiento del Modelo

### Pipeline de Entrenamiento

```python
def train_pipeline(events_df: pd.DataFrame, config: ModelConfig):
    # 1. Ingeniería de características
    modeling_df, metadata = build_modeling_table(events_df)
    arrays = get_model_arrays(modeling_df)

    # 2. Construir modelo
    model = build_hierarchical_negbinom_model(
        y=arrays["y"],
        entity_idx=arrays["entity_idx"],
        n_entities=metadata["n_entities"],
        config=config,
    )

    # 3. Muestreo MCMC
    trace = fit_model(model, config)

    # 4. Diagnósticos
    diagnostics = get_model_diagnostics(trace)

    return trace, modeling_df, diagnostics
```

### Estructura del Trace

El `InferenceData` retornado contiene:

```python
trace.posterior          # Muestras posteriores
  - mu:    (chain, draw)           # Forma: (4, 2000)
  - alpha: (chain, draw)           # Forma: (4, 2000)
  - phi:   (chain, draw)           # Forma: (4, 2000)
  - theta: (chain, draw, entity)   # Forma: (4, 2000, n_entidades)

trace.posterior_predictive
  - y:     (chain, draw, obs)      # Forma: (4, 2000, n_obs)

trace.sample_stats
  - diverging: (chain, draw)       # Banderas de divergencia
  - energy:    (chain, draw)       # Energía Hamiltoniana
```

---

## Puntuación de Anomalías

### Algoritmo de Puntuación

```python
def compute_anomaly_scores(y_observed, trace, entity_idx):
    """
    Calcular puntuaciones de anomalía desde predictivo posterior.

    Puntuación = -log p(y_observado | posterior)

    Mayor puntuación = más anómalo (menos probable bajo el modelo)
    """
    # Extraer muestras posteriores
    theta = trace.posterior["theta"].values  # (cadenas, muestras, entidades)
    phi = trace.posterior["phi"].values      # (cadenas, muestras)

    # Aplanar cadenas: (cadenas * muestras, ...)
    theta_flat = theta.reshape(-1, n_entities)
    phi_flat = phi.reshape(-1)

    n_samples = theta_flat.shape[0]  # 8000 muestras (4 cadenas × 2000 muestras)

    # Calcular log-verosimilitud para cada muestra posterior
    log_likelihoods = np.zeros((n_samples, n_obs))

    for s in range(n_samples):
        mu_s = theta_flat[s, entity_idx]  # Tasas específicas por entidad
        phi_s = phi_flat[s]

        # Log PMF Binomial Negativa
        log_likelihoods[s, :] = stats.nbinom.logpmf(
            y_observed,
            n=phi_s,                    # parámetro n de scipy
            p=phi_s / (phi_s + mu_s)    # parámetro p de scipy
        )

    # Log-verosimilitud promedio (log-sum-exp para estabilidad numérica)
    avg_log_lik = logsumexp(log_likelihoods, axis=0) - np.log(n_samples)

    # Puntuación anomalía = log verosimilitud negativo
    anomaly_scores = -avg_log_lik

    return anomaly_scores
```

### Interpretación de Puntuación

```
puntuación = -log p(y | modelo)

puntuación = 0:   p = 1.0    (perfectamente esperado)
puntuación = 2:   p = 0.14   (algo improbable)
puntuación = 5:   p = 0.007  (muy improbable)
puntuación = 10:  p = 4.5e-5 (extremadamente improbable)
```

---

## Métricas de Evaluación

### ¿Por Qué Estas Métricas?

| Métrica | Propósito | Por Qué Importante |
|---------|-----------|---------------------|
| PR-AUC | Calidad general de ranking | Maneja desbalance de clases mejor que ROC-AUC |
| Recall@K | Efectividad operacional | "¿Cuántos ataques en las top K alertas?" |
| Precision@K | Calidad de alertas | "¿Qué fracción del top K son ataques reales?" |

### Implementación PR-AUC

```python
def compute_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Área Bajo la Curva Precisión-Recall.

    Preferida sobre ROC-AUC para datos desbalanceados porque:
    1. Se enfoca en clase minoritaria (ataques)
    2. No inflada por verdaderos negativos
    3. Mide directamente tradeoff precisión/recall
    """
    return average_precision_score(y_true, scores)
```

### Implementación Recall@K

```python
def compute_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """
    Fracción de todos los ataques capturados en las top K puntuaciones.

    Recall@K = (ataques en top K) / (total ataques)

    Operacionalmente: "Si investigamos las top K alertas diarias,
                      ¿qué fracción de ataques atrapamos?"
    """
    n_positives = y_true.sum()
    if n_positives == 0:
        return 0.0

    top_k_idx = np.argsort(scores)[-k:]  # Índices de top K puntuaciones
    tp_at_k = y_true[top_k_idx].sum()    # Verdaderos positivos en top K

    return tp_at_k / n_positives
```

---

## Siguiente: [Referencia API](05_referencia_api.md)
