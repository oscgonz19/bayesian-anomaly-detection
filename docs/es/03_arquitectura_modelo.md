# Arquitectura del Modelo

## Tabla de Contenidos
1. [Especificación del Modelo](#especificación-del-modelo)
2. [Justificación de Selección de Priors](#justificación-de-selección-de-priors)
3. [Detalles de Parametrización](#detalles-de-parametrización)
4. [Implementación del Modelo](#implementación-del-modelo)
5. [Configuración de Inferencia](#configuración-de-inferencia)
6. [Diagnósticos del Modelo](#diagnósticos-del-modelo)

---

## Especificación del Modelo

### Modelo Gráfico

![Diagrama del Modelo Jerárquico](../images/hierarchical_model_diagram.png)
*Estructura jerárquica de tres niveles: Población → Entidad → Observación*

```
                    ┌─────────┐
                    │   μ     │  Tasa media poblacional
                    └────┬────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
         ┌────────┐ ┌────────┐ ┌────────┐
         │  θ_1   │ │  θ_2   │ │  θ_N   │  Tasas específicas por entidad
         └────┬───┘ └────┬───┘ └────┬───┘
              │          │          │
              │          │          │          ┌─────────┐
              │          │          │          │   φ     │  Sobredispersión
              ▼          ▼          ▼          └────┬────┘
         ┌────────┐ ┌────────┐ ┌────────┐         │
         │  y_1   │ │  y_2   │ │  y_N   │◀────────┘
         └────────┘ └────────┘ └────────┘  Conteos observados

    Hiperpriors: μ ~ Exp(0.1), α ~ HalfNormal(2)
    Priors entidad: θ_i ~ Gamma(μα, α)
    Verosimilitud: y_i ~ NegBinom(θ_i, φ)
```

### Especificación Matemática

**Nivel 1: Hiperpriors**
```
μ ~ Exponential(λ = 0.1)
α ~ HalfNormal(σ = 2)
φ ~ HalfNormal(σ = 2)
```

**Nivel 2: Parámetros de Entidad**
```
θ_i ~ Gamma(forma = μ·α, tasa = α)    para i = 1, ..., N_entidades
```

**Nivel 3: Observaciones**
```
y_ij ~ NegativeBinomial(mu = θ_i, alpha = φ)    para j = 1, ..., n_i
```

Donde:
- μ = tasa media de eventos poblacional
- α = parámetro de concentración (controla varianza entre entidades)
- φ = parámetro de sobredispersión
- θ_i = tasa media de eventos de entidad i
- y_ij = eventos observados para entidad i en ventana j

### Distribuciones Inducidas

**Prior marginal sobre θ_i:**
```
E[θ_i] = μ
Var(θ_i) = μ/α
```

Mayor α significa que las entidades son más similares entre sí (menos heterogeneidad).

**Distribución marginal de y_ij:**
```
E[y_ij] = θ_i
Var(y_ij) = θ_i + θ_i²/φ
```

La varianza excede la media, capturando sobredispersión.

---

## Justificación de Selección de Priors

### Media Poblacional μ ~ Exponential(0.1)

**Justificación:**
- Media de Exponential(0.1) = 10 eventos/ventana
- 95% de masa prior en [0.25, 30]
- Débilmente informativo: permite que los datos dominen mientras previene valores extremos

**Alternativa considerada:**
```
μ ~ Gamma(2, 0.2)  # Más concentrada alrededor de 10
```

Elegimos Exponential por simplicidad y mínimas suposiciones.

### Concentración α ~ HalfNormal(2)

**Justificación:**
- HalfNormal(2) da 95% de masa en [0, 4]
- α < 1: Alta heterogeneidad entre entidades
- α > 2: Pooling moderado
- Permite que los datos determinen nivel apropiado de pooling

**Efecto de α:**
```
α = 0.1  → CV(θ) ≈ 3.16 (muy heterogéneo)
α = 1.0  → CV(θ) ≈ 1.00 (heterogeneidad moderada)
α = 10   → CV(θ) ≈ 0.32 (bastante homogéneo)
```

Donde CV = coeficiente de variación = σ/μ.

### Sobredispersión φ ~ HalfNormal(2)

**Justificación:**
- Controla ratio varianza-a-media
- φ → ∞: Binomial Negativa → Poisson
- φ = 1: Varianza = 2×media
- Permite ajuste flexible de sobredispersión

### Verificaciones Predictivas del Prior

Antes de ajustar, podemos muestrear de los priors para verificar sensatez:

```python
with model:
    prior_samples = pm.sample_prior_predictive(samples=1000)

# Verificar: ¿Son razonables los conteos simulados?
# Esperamos: Mayormente rango 0-50, valores más altos ocasionales
```

---

## Detalles de Parametrización

### Parametrización Binomial Negativa

PyMC usa parametrización (μ, α):

```
NegativeBinomial(mu=μ, alpha=α)

P(y=k) = Γ(k+α)/(Γ(α)k!) × (α/(α+μ))^α × (μ/(α+μ))^k
```

Esto difiere de la parametrización (n, p) en scipy:
```
scipy: n = α, p = α/(α+μ)
pymc:  mu = μ, alpha = α
```

### Parametrización Gamma para θ

Parametrizamos la Gamma para que:
- E[θ] = μ (hereda media poblacional)
- Var(θ) = μ/α (controlada por concentración)

```python
# Parametrización forma-tasa
θ ~ Gamma(alpha=μ*α, beta=α)

# Esto da:
# E[θ] = (μ*α)/α = μ
# Var(θ) = (μ*α)/α² = μ/α
```

### Variables de Índice

Para cómputo eficiente, usamos variables de índice:

```python
# entity_idx: mapea cada observación a su entidad
# Forma: (n_observaciones,)
# Valores: enteros en [0, n_entidades-1]

# Uso en modelo:
θ[entity_idx]  # Transmite tasas de entidad a observaciones
```

---

## Implementación del Modelo

### Código PyMC del Modelo

```python
def build_hierarchical_negbinom_model(
    y: np.ndarray,
    entity_idx: np.ndarray,
    n_entities: int,
    config: ModelConfig,
) -> pm.Model:
    """
    Construir modelo Binomial Negativo jerárquico.

    Parámetros
    ----------
    y : array de forma (n_obs,)
        Conteos de eventos por observación
    entity_idx : array de forma (n_obs,)
        Índice de entidad para cada observación
    n_entities : int
        Número total de entidades únicas
    config : ModelConfig
        Configuración del modelo
    """
    coords = {
        "entity": np.arange(n_entities),
        "obs": np.arange(len(y)),
    }

    with pm.Model(coords=coords) as model:
        # === Datos ===
        entity_idx_data = pm.Data("entity_idx", entity_idx, dims="obs")
        y_data = pm.Data("y_obs", y, dims="obs")

        # === Hiperpriors ===
        # Tasa media poblacional
        mu = pm.Exponential("mu", lam=config.mu_prior_rate)

        # Concentración (controla heterogeneidad entre entidades)
        alpha = pm.HalfNormal("alpha", sigma=config.alpha_prior_sd)

        # === Tasas a nivel de entidad ===
        # Pooling parcial: θ_i ~ Gamma con E[θ] = μ
        theta = pm.Gamma(
            "theta",
            alpha=mu * alpha,  # parámetro de forma
            beta=alpha,         # parámetro de tasa
            dims="entity",
        )

        # === Sobredispersión ===
        phi = pm.HalfNormal("phi", sigma=config.overdispersion_prior_sd)

        # === Verosimilitud ===
        pm.NegativeBinomial(
            "y",
            mu=theta[entity_idx_data],  # tasa específica por entidad
            alpha=phi,                   # sobredispersión compartida
            observed=y_data,
            dims="obs",
        )

    return model
```

---

## Configuración de Inferencia

### Configuración por Defecto

```python
@dataclass
class ModelConfig:
    # Muestreo
    n_samples: int = 2000    # Muestras posteriores por cadena
    n_tune: int = 1000       # Muestras de calentamiento/adaptación
    n_chains: int = 4        # Cadenas MCMC independientes
    target_accept: float = 0.9  # Tasa de aceptación objetivo NUTS
    random_seed: int = 42

    # Priors
    mu_prior_rate: float = 0.1
    alpha_prior_sd: float = 2.0
    overdispersion_prior_sd: float = 2.0
```

### Proceso de Muestreo

```python
with model:
    trace = pm.sample(
        draws=config.n_samples,
        tune=config.n_tune,
        chains=config.n_chains,
        target_accept=config.target_accept,
        random_seed=seed,
        cores=min(config.n_chains, os.cpu_count()),
        return_inferencedata=True,
    )
```

**Fases:**
1. **Ajuste (1000 iteraciones)**: Adaptar tamaño de paso y matriz de masa
2. **Muestreo (2000 iteraciones)**: Generar muestras posteriores
3. **Post-procesamiento**: Calcular diagnósticos, añadir predictivo posterior

![Explicación del Scoring](../images/scoring_explanation.png)
*Cómo se calculan las puntuaciones de anomalía: de distribución a salida rankeada con incertidumbre*

---

## Diagnósticos del Modelo

### Verificaciones de Convergencia

```python
def get_model_diagnostics(trace: az.InferenceData) -> dict:
    summary = az.summary(trace, var_names=["mu", "alpha", "phi"])

    return {
        "r_hat_max": float(summary["r_hat"].max()),
        "ess_bulk_min": float(summary["ess_bulk"].min()),
        "ess_tail_min": float(summary["ess_tail"].min()),
        "divergences": int(trace.sample_stats["diverging"].sum()),
        "converged": bool(summary["r_hat"].max() < 1.05),
    }
```

### Umbrales de Diagnóstico

| Métrica | Bueno | Advertencia | Problema |
|---------|-------|-------------|----------|
| R-hat | < 1.01 | 1.01-1.05 | > 1.05 |
| ESS bulk | > 400 | 100-400 | < 100 |
| ESS tail | > 400 | 100-400 | < 100 |
| Divergencias | 0 | 1-10 | > 10 |

### Solución de Problemas

**R-hat alto:**
- Aumentar `n_samples` y `n_tune`
- Verificar posterior multimodal
- Reparametrizar modelo

**ESS bajo:**
- Aumentar `n_samples`
- Verificar alta autocorrelación
- Considerar parametrización no centrada

**Divergencias:**
- Aumentar `target_accept` (ej., 0.95, 0.99)
- Reparametrizar (no centrada)
- Usar priors más fuertes

---

## Siguiente: [Guía de Implementación](04_guia_implementacion.md)
