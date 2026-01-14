# Tutorial: Guía Paso a Paso

## Tabla de Contenidos
1. [Instalación](#instalación)
2. [Inicio Rápido](#inicio-rápido)
3. [Pipeline Paso a Paso](#pipeline-paso-a-paso)
4. [Interpretación de Resultados](#interpretación-de-resultados)
5. [Personalización](#personalización)
6. [Solución de Problemas](#solución-de-problemas)

---

## Instalación

### Usando Conda (Recomendado)

```bash
# Clonar repositorio
git clone https://github.com/yourusername/bayesian-security-anomaly-detection.git
cd bayesian-security-anomaly-detection

# Crear ambiente conda
conda env create -f environment.yml

# Activar ambiente
conda activate bsad

# Verificar instalación
bsad --help
```

### Usando pip

```bash
# Clonar repositorio
git clone https://github.com/yourusername/bayesian-security-anomaly-detection.git
cd bayesian-security-anomaly-detection

# Crear ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o: .venv\Scripts\activate  # Windows

# Instalar paquete
pip install -e ".[dev]"

# Verificar instalación
bsad --help
```

### Verificar Instalación

```bash
# Verificar CLI disponible
bsad --help

# Verificar imports de Python
python -c "from bsad import __version__; print(f'Versión BSAD: {__version__}')"

# Verificar instalación PyMC
python -c "import pymc as pm; print(f'Versión PyMC: {pm.__version__}')"
```

---

## Inicio Rápido

### Demo con Un Comando

```bash
# Ejecutar pipeline completo con valores por defecto
make demo

# O con parámetros personalizados
bsad demo --n-entities 100 --n-days 14 --samples 500
```

Esto:
1. Generará eventos de seguridad sintéticos
2. Entrenará el modelo Bayesiano
3. Puntuará todas las observaciones
4. Evaluará rendimiento
5. Generará gráficos

Salida en directorio `outputs/`.

### Salida Esperada

```
================================================================
DETECCIÓN BAYESIANA DE ANOMALÍAS DE SEGURIDAD - DEMO
================================================================

Paso 1/4: Generando eventos de seguridad sintéticos
  Generados 45,231 eventos
  Eventos de ataque: 1,847

Paso 2/4: Entrenando modelo Bayesiano jerárquico
  Entidades: 200
  Ventanas: 5,892
  Muestreando 1000 draws posteriores (puede tomar varios minutos)...
  R-hat: 1.002, Divergencias: 0

Paso 3/4: Puntuando observaciones
  Puntuadas 5,892 ventanas-entidad

Paso 4/4: Evaluando rendimiento
  PR-AUC: 0.847
  Recall@50: 0.412
  Recall@100: 0.623

================================================================
DEMO COMPLETO
================================================================

Artefactos Generados:
  Eventos:   outputs/data/events.parquet
  Modelo:    outputs/model.nc
  Puntuaciones: outputs/scores.parquet
  Métricas:  outputs/metrics.json
  Gráficos:  outputs/plots/
```

---

## Pipeline Paso a Paso

### Paso 1: Generar Datos Sintéticos

```bash
bsad generate-data \
    --n-entities 200 \
    --n-days 30 \
    --attack-rate 0.02 \
    --seed 42 \
    --output data/events.parquet
```

**Qué hace:**
- Crea 200 usuarios sintéticos con patrones de actividad heterogéneos
- Simula 30 días de eventos de seguridad
- Inyecta ~2% ventanas de ataque (fuerza bruta, credential stuffing, etc.)
- Guarda eventos en archivo Parquet

**Verificar:**
```python
import pandas as pd

events = pd.read_parquet("data/events.parquet")
print(f"Total eventos: {len(events):,}")
print(f"Eventos de ataque: {events['is_attack'].sum():,}")
print(f"Tipos de ataque: {events[events['is_attack']]['attack_type'].value_counts().to_dict()}")
```

### Paso 2: Entrenar Modelo

```bash
bsad train \
    --input data/events.parquet \
    --output outputs/model.nc \
    --samples 2000 \
    --tune 1000 \
    --chains 4
```

**Qué hace:**
- Carga eventos y construye tabla de modelado (ingeniería de características)
- Construye modelo Binomial Negativo jerárquico
- Ejecuta muestreo MCMC (NUTS) con 4 cadenas × 2000 muestras
- Guarda muestras posteriores en archivo NetCDF

**Duración esperada:** 5-15 minutos dependiendo del tamaño de datos y hardware.

**Verificar:**
```python
import arviz as az

trace = az.from_netcdf("outputs/model.nc")
print(az.summary(trace, var_names=["mu", "alpha", "phi"]))
```

### Paso 3: Puntuar Observaciones

```bash
bsad score \
    --model outputs/model.nc \
    --input outputs/modeling_table.parquet \
    --output outputs/scores.parquet
```

**Qué hace:**
- Carga modelo entrenado (muestras posteriores)
- Calcula puntuaciones de anomalía para cada ventana-entidad
- Calcula incertidumbre de puntuación (std, IC 90%)
- Rankea observaciones por puntuación de anomalía

**Verificar:**
```python
import pandas as pd

scores = pd.read_parquet("outputs/scores.parquet")
print(f"Top 5 anomalías:")
print(scores[["user_id", "window", "event_count", "anomaly_score", "has_attack"]].head())
```

### Paso 4: Evaluar Rendimiento

```bash
bsad evaluate \
    --scores outputs/scores.parquet \
    --output outputs/metrics.json \
    --plots outputs/plots
```

**Qué hace:**
- Calcula métricas PR-AUC, ROC-AUC, Recall@K
- Genera gráficos de diagnóstico
- Guarda métricas en archivo JSON

---

## Interpretación de Resultados

### Entendiendo Puntuaciones de Anomalía

```python
scores_df = pd.read_parquet("outputs/scores.parquet")

# Interpretación de puntuación
# Mayor puntuación = más anómalo = menos probable bajo el modelo

# Puntuaciones benignas típicas: 2-5
# Puntuaciones de ataque típicas: 6-15+

print(scores_df.groupby("has_attack")["anomaly_score"].describe())
```

### Componentes de Puntuación

Cada observación tiene:

| Campo | Significado |
|-------|-------------|
| `anomaly_score` | Estimación puntual (-log probabilidad) |
| `score_std` | Incertidumbre en puntuación |
| `score_lower` | Percentil 5 |
| `score_upper` | Percentil 95 |
| `predicted_mean` | Conteo esperado de eventos |
| `predicted_lower` | Límite inferior (IC 90%) |
| `predicted_upper` | Límite superior (IC 90%) |
| `exceeds_interval` | Observado > predicted_upper |

### Evaluando Calidad de Detección

```python
# Buena detección separa distribuciones ataque/benigno
import matplotlib.pyplot as plt

benign = scores_df[~scores_df["has_attack"]]["anomaly_score"]
attack = scores_df[scores_df["has_attack"]]["anomaly_score"]

plt.hist(benign, bins=50, alpha=0.7, label="Benigno", density=True)
plt.hist(attack, bins=50, alpha=0.7, label="Ataque", density=True)
plt.legend()
plt.xlabel("Puntuación de Anomalía")
plt.ylabel("Densidad")
plt.show()
```

---

## Personalización

### Tasa de Ataque Personalizada

```python
from bsad.data_generator import GeneratorConfig, generate_synthetic_data

# Mayor tasa de ataque para pruebas
config = GeneratorConfig(
    n_users=100,
    n_days=14,
    attack_rate=0.10,  # 10% tasa de ataque
)
events_df, attacks_df = generate_synthetic_data(config)
```

### Diferentes Tamaños de Ventana

```python
from bsad.features import FeatureConfig, build_modeling_table

# Ventanas por hora (más granular)
config = FeatureConfig(window_size="1H")
modeling_df, metadata = build_modeling_table(events_df, config)

# Ventanas de 6 horas
config = FeatureConfig(window_size="6H")
modeling_df, metadata = build_modeling_table(events_df, config)
```

### Priors Personalizados

```python
from bsad.model import ModelConfig, build_hierarchical_negbinom_model

# Prior de pooling más fuerte (más similitud entre entidades)
config = ModelConfig(
    alpha_prior_sd=1.0,  # Prior de concentración más estrecho
)

# Muestreo más rápido (menos muestras)
config = ModelConfig(
    n_samples=1000,
    n_tune=500,
    n_chains=2,
)
```

### Análisis a Nivel de IP

```python
# Agrupar por IP en lugar de usuario
config = FeatureConfig(entity_column="ip_address")
modeling_df, metadata = build_modeling_table(events_df, config)
```

---

## Solución de Problemas

### Muestreo Lento

**Síntoma:** MCMC toma >30 minutos

**Soluciones:**
1. Reducir tamaño de muestra:
   ```bash
   bsad train --samples 500 --tune 250 --chains 2
   ```

2. Reducir tamaño de datos:
   ```bash
   bsad generate-data --n-entities 50 --n-days 7
   ```

3. Usar hardware más rápido (GPU no soportado para NUTS)

### Advertencias de Convergencia

**Síntoma:** R-hat > 1.05 o muchas divergencias

**Soluciones:**
1. Aumentar target_accept:
   ```python
   config = ModelConfig(target_accept=0.95)
   ```

2. Aumentar tuning:
   ```python
   config = ModelConfig(n_tune=2000)
   ```

3. Verificar valores extremos en datos (outliers)

### Errores de Memoria

**Síntoma:** Sin memoria durante muestreo

**Soluciones:**
1. Reducir cadenas:
   ```bash
   bsad train --chains 2
   ```

2. Reducir tamaño de datos
3. Usar máquina con más RAM (16GB+ recomendado)

### Rendimiento de Detección Pobre

**Síntoma:** PR-AUC bajo (<0.5)

**Posibles causas:**
1. Datos insuficientes: Aumentar n_days o n_entities
2. Tasa de ataque baja: Difícil distinguir con pocos positivos
3. Patrones de ataque similares a baseline: Verificar inyección de ataques

**Diagnóstico:**
```python
# Verificar si ataques tienen firmas distintas
print(scores_df.groupby("attack_type")["event_count"].describe())
print(scores_df.groupby("attack_type")["anomaly_score"].describe())
```

---

## Siguientes Pasos

- Leer [Fundamentos Teóricos](02_fundamentos_teoricos.md) para entendimiento más profundo
- Explorar [Arquitectura del Modelo](03_arquitectura_modelo.md) para opciones de personalización
- Consultar [Referencia API](05_referencia_api.md) para uso programático
