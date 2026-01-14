# Fundamentos Teóricos

## Tabla de Contenidos
1. [Fundamentos de Estadística Bayesiana](#fundamentos-de-estadística-bayesiana)
2. [Distribuciones para Datos de Conteo](#distribuciones-para-datos-de-conteo)
3. [Modelos Jerárquicos](#modelos-jerárquicos)
4. [Monte Carlo vía Cadenas de Markov (MCMC)](#monte-carlo-vía-cadenas-de-markov-mcmc)
5. [Teoría de Detección de Anomalías](#teoría-de-detección-de-anomalías)

---

## Fundamentos de Estadística Bayesiana

### El Teorema de Bayes

La base de la inferencia Bayesiana es el teorema de Bayes, que relaciona probabilidades condicionales:

```
P(θ|y) = P(y|θ) × P(θ) / P(y)
```

En el contexto de inferencia estadística:

| Término | Nombre | Interpretación |
|---------|--------|----------------|
| P(θ\|y) | **Posterior** | Probabilidad de los parámetros dados los datos observados |
| P(y\|θ) | **Verosimilitud** | Probabilidad de los datos dados los parámetros |
| P(θ) | **Prior** | Probabilidad de los parámetros antes de ver datos |
| P(y) | **Verosimilitud Marginal** | Constante normalizadora (evidencia) |

Dado que P(y) es constante con respecto a θ, frecuentemente escribimos:

```
posterior ∝ verosimilitud × prior
```

### Interpretación Frecuentista vs. Bayesiana

| Aspecto | Frecuentista | Bayesiana |
|---------|--------------|-----------|
| **Parámetros** | Constantes fijas desconocidas | Variables aleatorias con distribuciones |
| **Datos** | Aleatorios (varían entre muestras) | Fijos (observados una vez) |
| **Inferencia** | Estimaciones puntuales + intervalos de confianza | Distribución posterior completa |
| **Interpretación** | "IC 95% significa: si repitiéramos este experimento muchas veces, 95% de los ICs contendrían el valor verdadero" | "Intervalo creíble 95% significa: hay 95% de probabilidad de que el parámetro esté en este rango dados nuestros datos" |

### Distribuciones Prior

Los priors codifican nuestras creencias antes de ver datos. Elecciones comunes:

#### Priors Débilmente Informativos
Regularizan la inferencia sin dominar la verosimilitud:

```python
# Para parámetros de tasa positivos
μ ~ Exponential(0.1)  # Media = 10, permite amplio rango

# Para parámetros de escala
σ ~ HalfNormal(2)     # Positivo, concentrado cerca de 0-4
```

#### Priors Conjugados
Cuando prior y posterior están en la misma familia:

| Verosimilitud | Prior Conjugado | Posterior |
|---------------|-----------------|-----------|
| Binomial | Beta | Beta |
| Poisson | Gamma | Gamma |
| Normal (σ conocida) | Normal | Normal |
| Binomial Negativa | Beta-Binomial Negativa | Beta-Binomial Negativa |

#### Por Qué Importan los Priors

1. **Regularización**: Previenen sobreajuste restringiendo el espacio de parámetros
2. **Incorporación de conocimiento del dominio**: Los expertos en seguridad conocen tasas de eventos típicas
3. **Manejo de datos escasos**: El prior domina cuando los datos son limitados, previniendo estimaciones extremas

### Cálculo del Posterior

El posterior frecuentemente es intratable analíticamente:

```
P(θ|y) = P(y|θ) × P(θ) / ∫ P(y|θ) P(θ) dθ
                         ↑
                    Frecuentemente intratable
```

Soluciones:
1. **Priors conjugados**: Posterior en forma cerrada (aplicabilidad limitada)
2. **Inferencia variacional**: Aproximar posterior con distribución más simple
3. **MCMC**: Generar muestras del posterior (nuestro enfoque)

---

## Distribuciones para Datos de Conteo

### La Distribución de Poisson

Para datos de conteo, la distribución de Poisson es un punto de partida natural:

```
P(y=k|λ) = (λ^k × e^(-λ)) / k!
```

Propiedades:
- Soporte: y ∈ {0, 1, 2, ...}
- Media: E[y] = λ
- Varianza: Var(y) = λ

**Limitación**: La varianza iguala la media (equidispersión). Los datos de seguridad típicamente están **sobredispersos** (varianza > media).

### La Distribución Binomial Negativa

Maneja sobredispersión introduciendo un parámetro de dispersión:

```
P(y=k|μ,α) = Γ(k+α) / (Γ(α)k!) × (α/(α+μ))^α × (μ/(α+μ))^k
```

Donde:
- μ = media
- α = parámetro de dispersión (mayor = menos sobredispersión)

Propiedades:
- Media: E[y] = μ
- Varianza: Var(y) = μ + μ²/α

Cuando α → ∞, Binomial Negativa → Poisson.

### ¿Por Qué Binomial Negativa para Datos de Seguridad?

Los conteos de eventos de seguridad exhiben:

1. **Sobredispersión**: Varianza excede la media debido a:
   - Patrones de ataque explosivos
   - Heterogeneidad en comportamiento de usuarios
   - Efectos estacionales

2. **Inflación de ceros**: Muchas ventanas entidad tienen cero eventos

3. **Colas pesadas**: Valores extremos ocasionales (ataques)

Ejemplo de datos típicos de seguridad:
```
Media eventos/ventana:     8.3
Varianza:                  47.2
Ratio Varianza/Media:      5.7  (sobredisperso)
```

La Binomial Negativa captura esto naturalmente a través de su estructura de varianza flexible.

### Alternativa: Mezcla Poisson-Gamma

La Binomial Negativa puede derivarse como Poisson con tasa distribuida Gamma:

```
λ ~ Gamma(α, α/μ)
y | λ ~ Poisson(λ)

Marginalizando λ:
y ~ NegativeBinomial(μ, α)
```

Esta interpretación es útil para modelos jerárquicos donde las tasas específicas por entidad vienen de una distribución poblacional.

---

## Modelos Jerárquicos

### El Paradigma Jerárquico

En modelos jerárquicos (multinivel), los parámetros mismos tienen distribuciones:

```
Nivel 3 (Hiperpriors):    μ, α ~ hiperpriors poblacionales
                              ↓
Nivel 2 (Params entidad): θ_i ~ distribución poblacional(μ, α)
                              ↓
Nivel 1 (Datos):          y_ij ~ verosimilitud(θ_i)
```

### Pooling Parcial

Los modelos jerárquicos implementan **pooling parcial**—un compromiso entre:

| Enfoque | Descripción | Problema |
|---------|-------------|----------|
| **Sin pooling** | Estimar θ_i separado para cada entidad | Sobreajusta con datos escasos |
| **Pooling completo** | θ único para todas las entidades | Ignora heterogeneidad entre entidades |
| **Pooling parcial** | θ_i de entidad extraído de distribución poblacional | Lo mejor de ambos mundos |

#### Formulación Matemática

Las tasas por entidad se extraen de una distribución poblacional:

```
θ_i ~ Gamma(μα, α)
```

Donde:
- μ = tasa media poblacional
- α = concentración (mayor = menos variación entre entidades)

El posterior para cada θ_i se convierte en:

```
θ_i | y_i, μ, α ∝ Verosimilitud(y_i|θ_i) × Gamma(θ_i|μα, α)
```

Entidades con:
- **Muchas observaciones**: Verosimilitud domina, θ_i refleja datos específicos de entidad
- **Pocas observaciones**: Prior domina, θ_i se contrae hacia media poblacional μ

### Ilustración de Contracción (Shrinkage)

Considere tres entidades con diferentes cantidades de datos:

```
Entidad A: 1000 eventos observados, media muestral = 15
Entidad B: 10 eventos observados, media muestral = 15
Entidad C: 2 eventos observados, media muestral = 15
Media poblacional μ = 10

Medias posteriores (pooling parcial):
Entidad A: ~14.8 (principalmente sus propios datos)
Entidad B: ~12.1 (mezcla)
Entidad C: ~10.4 (contraída hacia población)
```

### ¿Por Qué Modelos Jerárquicos para Seguridad?

1. **Estructura natural**: Usuarios/IPs forman una población con características compartidas
2. **Préstamo de fuerza**: Nuevas entidades se benefician del aprendizaje a nivel poblacional
3. **Líneas base adaptativas**: Cada entidad obtiene línea base "normal" personalizada
4. **Cuantificación de incertidumbre**: Datos escasos = intervalos creíbles más amplios

### Nuestro Modelo Jerárquico

```
# Hiperpriors (nivel poblacional)
μ ~ Exponential(0.1)      # Tasa media de eventos poblacional
α ~ HalfNormal(2)         # Concentración (variabilidad entre entidades)

# Parámetros a nivel de entidad
θ_i ~ Gamma(μα, α)        # Tasa específica por entidad, para i = 1,...,N

# Sobredispersión
φ ~ HalfNormal(2)         # Compartido entre entidades

# Observaciones
y_ij ~ NegativeBinomial(θ_i, φ)  # Eventos para entidad i, ventana j
```

---

## Monte Carlo vía Cadenas de Markov (MCMC)

### El Problema de Muestreo

Necesitamos muestras del posterior:

```
θ^(1), θ^(2), ..., θ^(S) ~ P(θ|y)
```

El muestreo directo usualmente es imposible. MCMC construye una cadena de Markov cuya distribución estacionaria es el posterior objetivo.

### Fundamentos de Cadenas de Markov

Una cadena de Markov es una secuencia donde cada estado depende solo del anterior:

```
P(θ^(t) | θ^(t-1), θ^(t-2), ..., θ^(1)) = P(θ^(t) | θ^(t-1))
```

Con el kernel de transición apropiado, la cadena converge a una distribución estacionaria.

### Algoritmo Metropolis-Hastings

El algoritmo MCMC fundamental:

```
1. Inicializar θ^(0)
2. Para t = 1, 2, ..., S:
   a. Proponer θ* de distribución de propuesta q(θ*|θ^(t-1))
   b. Calcular probabilidad de aceptación:
      α = min(1, [P(θ*|y) × q(θ^(t-1)|θ*)] / [P(θ^(t-1)|y) × q(θ*|θ^(t-1))])
   c. Con probabilidad α: θ^(t) = θ* (aceptar)
      De lo contrario: θ^(t) = θ^(t-1) (rechazar)
```

### Monte Carlo Hamiltoniano (HMC)

Mejora sobre Metropolis de caminata aleatoria usando información del gradiente:

1. **Aumentar** espacio de parámetros con variables de momento
2. **Simular dinámica Hamiltoniana** para proponer estados distantes
3. **Aceptar/rechazar** basado en conservación del Hamiltoniano

Ventajas:
- Explora espacio de parámetros más eficientemente
- Menos correlación entre muestras
- Escala mejor a altas dimensiones

### Sampler No-U-Turn (NUTS)

NUTS (nuestro sampler) ajusta HMC automáticamente:

- **Longitud de trayectoria adaptativa**: Simula hasta que la trayectoria "da vuelta"
- **Tamaño de paso automático**: Promediado dual durante calentamiento
- **Sin ajuste manual**: Elimina selección difícil de hiperparámetros

```python
# PyMC usa NUTS por defecto
trace = pm.sample(
    draws=2000,      # Muestras posteriores
    tune=1000,       # Calentamiento para adaptación
    chains=4,        # Cadenas independientes
    target_accept=0.9  # Objetivo de tasa de aceptación
)
```

### Diagnósticos de Convergencia

#### R-hat (Estadístico Gelman-Rubin)

Compara varianza dentro de cadenas y entre cadenas:

```
R-hat = √(Var(combinada) / Var(dentro-cadena))
```

- R-hat ≈ 1.0: Las cadenas han convergido a la misma distribución
- R-hat > 1.05: Potenciales problemas de convergencia

#### Tamaño de Muestra Efectivo (ESS)

Contabiliza autocorrelación en muestras MCMC:

```
ESS = S / (1 + 2 × Σ_k ρ_k)
```

Donde ρ_k es la autocorrelación de lag-k.

Directrices:
- ESS > 400 para resúmenes posteriores confiables
- ESS > 1000 para probabilidades de colas

#### Divergencias

Diagnóstico específico de HMC. Las divergencias indican:
- Problemas de geometría del posterior
- Tamaño de paso muy grande
- Forma patológica del posterior

Soluciones:
- Aumentar `target_accept` (pasos más pequeños)
- Reparametrizar modelo
- Usar parametrización no centrada

---

## Teoría de Detección de Anomalías

### ¿Qué es una Anomalía?

Una anomalía (outlier) es una observación que se desvía significativamente del comportamiento esperado. En nuestro marco probabilístico:

```
Anomalía = observación con baja probabilidad bajo el modelo aprendido
```

### Distribución Predictiva Posterior

La predictiva posterior para nueva observación ỹ:

```
P(ỹ|y) = ∫ P(ỹ|θ) P(θ|y) dθ
```

Esto integra la incertidumbre de parámetros, dando predicciones que consideran la incertidumbre del modelo.

Aproximada vía Monte Carlo:

```
P(ỹ|y) ≈ (1/S) × Σ_s P(ỹ|θ^(s))
```

### Puntuación de Anomalías

Nuestra puntuación de anomalía es la probabilidad predictiva posterior logarítmica negativa:

```
puntuación(y_obs) = -log P(y_obs | y_train)
                  = -log [ (1/S) × Σ_s P(y_obs | θ^(s)) ]
```

Propiedades:
- **Mayor puntuación = más anómalo**: Menos probable bajo el modelo
- **Incorpora incertidumbre**: Promediado sobre muestras posteriores
- **Interpretable**: Basado en teoría de probabilidad

### Truco Log-Sum-Exp

Para estabilidad numérica, calculamos:

```
puntuación = -log Σ_s exp(log_lik_s) + log(S)

Usando log-sum-exp:
logsumexp(x) = max(x) + log(Σ exp(x - max(x)))
```

### Por Qué Funciona Esta Puntuación

1. **Fundamento probabilístico**: Mide directamente "sorpresa" bajo el modelo
2. **Maneja incertidumbre**: Una observación podría tener puntuación alta si:
   - Está lejos de la media, O
   - El modelo está confiado de que la media está en otro lugar
3. **Específica por entidad**: El comportamiento esperado de cada entidad se aprende de datos
4. **Calibración automática**: Las puntuaciones son comparables entre entidades

### Incertidumbre de Puntuación

También reportamos incertidumbre de puntuación por variación posterior:

```python
# Para cada muestra posterior s
puntuación_s = -log P(y_obs | θ^(s))

# Estadísticas resumen
puntuación_media = mean(puntuación_s)
puntuación_std = std(puntuación_s)
puntuación_IC95 = [percentile(puntuación_s, 2.5), percentile(puntuación_s, 97.5)]
```

Alta puntuación + baja incertidumbre → detección de anomalía confiable
Alta puntuación + alta incertidumbre → marcado pero investigar más

---

## Referencias

1. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.

2. McElreath, R. (2020). *Statistical Rethinking: A Bayesian Course with Examples in R and Stan* (2nd ed.). CRC Press.

3. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(1), 1593-1623.

4. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. *arXiv preprint arXiv:1701.02434*.

5. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey. *ACM Computing Surveys*, 41(3), 1-58.

6. Hilbe, J. M. (2011). *Negative Binomial Regression* (2nd ed.). Cambridge University Press.

---

## Siguiente: [Arquitectura del Modelo](03_arquitectura_modelo.md)
