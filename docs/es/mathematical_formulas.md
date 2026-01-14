# Fórmulas Matemáticas: Especificación Estadística Completa

## Resumen General

Este documento proporciona la especificación matemática completa del modelo de Detección Bayesiana de Anomalías de Seguridad. Está dirigido a estadísticos, investigadores cuantitativos y aquellos que deseen entender los fundamentos teóricos.

---

## 1. Especificación del Modelo

### 1.1 Notación

| Símbolo | Descripción |
|---------|-------------|
| $i$ | Índice de entidad, $i \in \{1, \ldots, N\}$ |
| $t$ | Índice de ventana temporal, $t \in \{1, \ldots, T\}$ |
| $y_{it}$ | Conteo de eventos observado para entidad $i$ en ventana $t$ |
| $\theta_i$ | Parámetro de tasa específico de entidad |
| $\mu$ | Tasa media poblacional |
| $\alpha$ | Parámetro de concentración (fuerza del pooling) |
| $\phi$ | Parámetro de sobredispersión |

### 1.2 Modelo Jerárquico

El modelo jerárquico completo es:

$$
\begin{aligned}
\mu &\sim \text{Exponential}(\lambda = 0.1) \\
\alpha &\sim \text{HalfNormal}(\sigma = 2) \\
\phi &\sim \text{HalfNormal}(\sigma = 1) \\
\theta_i &\sim \text{Gamma}(\text{shape} = \mu\alpha, \text{rate} = \alpha) \\
y_{it} &\sim \text{NegativeBinomial}(\mu = \theta_i, \alpha = \phi)
\end{aligned}
$$

---

## 2. Distribuciones Prior

### 2.1 Media Poblacional ($\mu$)

$$\mu \sim \text{Exponential}(\lambda = 0.1)$$

**Propiedades:**
- Soporte: $\mu > 0$
- Media: $\mathbb{E}[\mu] = 1/\lambda = 10$
- Varianza: $\text{Var}[\mu] = 1/\lambda^2 = 100$

**PDF:**
$$p(\mu) = \lambda e^{-\lambda \mu} = 0.1 e^{-0.1\mu}$$

**Justificación:** Prior débilmente informativo que permite tasas medias poblacionales desde casi cero hasta 50+ eventos por ventana.

### 2.2 Parámetro de Concentración ($\alpha$)

$$\alpha \sim \text{HalfNormal}(\sigma = 2)$$

**Propiedades:**
- Soporte: $\alpha > 0$
- Media: $\mathbb{E}[\alpha] = \sigma\sqrt{2/\pi} \approx 1.60$
- Moda: $\alpha = 0$

**PDF:**
$$p(\alpha) = \frac{2}{\sigma\sqrt{2\pi}} \exp\left(-\frac{\alpha^2}{2\sigma^2}\right) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{\alpha^2}{8}\right)$$

**Justificación:** Permite tanto pooling fuerte ($\alpha$ grande) como pooling débil ($\alpha$ pequeño), dejando que los datos determinen el encogimiento apropiado.

### 2.3 Parámetro de Sobredispersión ($\phi$)

$$\phi \sim \text{HalfNormal}(\sigma = 1)$$

**Propiedades:**
- Soporte: $\phi > 0$
- Controla la varianza relativa a Poisson

**PDF:**
$$p(\phi) = \frac{2}{\sqrt{2\pi}} \exp\left(-\frac{\phi^2}{2}\right)$$

**Justificación:** Los conteos de eventos de seguridad típicamente muestran sobredispersión moderada; este prior concentra masa en valores razonables.

---

## 3. Distribución a Nivel de Entidad

### 3.1 Prior Gamma para Tasas de Entidad

$$\theta_i \sim \text{Gamma}(\text{shape} = \mu\alpha, \text{rate} = \alpha)$$

**Propiedades:**
$$
\begin{aligned}
\mathbb{E}[\theta_i] &= \frac{\mu\alpha}{\alpha} = \mu \\
\text{Var}[\theta_i] &= \frac{\mu\alpha}{\alpha^2} = \frac{\mu}{\alpha}
\end{aligned}
$$

**PDF:**
$$p(\theta_i | \mu, \alpha) = \frac{\alpha^{\mu\alpha}}{\Gamma(\mu\alpha)} \theta_i^{\mu\alpha - 1} e^{-\alpha\theta_i}$$

### 3.2 Interpretación del Pooling Parcial

El parámetro de concentración $\alpha$ controla el grado de pooling:

| Valor de $\alpha$ | Efecto | Tasas de Entidad |
|-------------------|--------|------------------|
| $\alpha \to 0$ | Sin pooling | $\theta_i$ varían libremente |
| $\alpha \approx 1$ | Pooling moderado | Encogimiento balanceado |
| $\alpha \to \infty$ | Pooling completo | $\theta_i \to \mu$ |

**Factor de Encogimiento:**
$$\text{Encogimiento} = \frac{\alpha}{\alpha + n_i}$$

donde $n_i$ es el número de observaciones para la entidad $i$.

---

## 4. Función de Verosimilitud

### 4.1 Distribución Binomial Negativa

$$y_{it} \sim \text{NegativeBinomial}(\mu = \theta_i, \alpha = \phi)$$

**Parametrización (media-dispersión):**
$$p(y | \mu, \alpha) = \binom{y + \alpha - 1}{y} \left(\frac{\alpha}{\alpha + \mu}\right)^\alpha \left(\frac{\mu}{\alpha + \mu}\right)^y$$

**Propiedades:**
$$
\begin{aligned}
\mathbb{E}[y] &= \mu = \theta_i \\
\text{Var}[y] &= \mu + \frac{\mu^2}{\alpha} = \theta_i + \frac{\theta_i^2}{\phi}
\end{aligned}
$$

### 4.2 Razón de Sobredispersión

$$\text{Sobredispersión} = \frac{\text{Var}[y]}{\mathbb{E}[y]} = 1 + \frac{\mu}{\phi}$$

Para $\phi = 1, \mu = 10$: Sobredispersión = 11 (varianza es 11x la media)

### 4.3 ¿Por Qué No Poisson?

Poisson asume $\text{Var}[y] = \mathbb{E}[y]$. Los logs de seguridad exhiben:
- Comportamiento en ráfagas (varianza >> media)
- Colas pesadas (conteos extremos más comunes)

La Binomial Negativa generaliza Poisson con un parámetro de dispersión extra.

---

## 5. Distribución Posterior

### 5.1 Posterior Conjunta

Por el teorema de Bayes:

$$p(\mu, \alpha, \phi, \boldsymbol{\theta} | \mathbf{y}) \propto p(\mathbf{y} | \boldsymbol{\theta}, \phi) \cdot p(\boldsymbol{\theta} | \mu, \alpha) \cdot p(\mu) \cdot p(\alpha) \cdot p(\phi)$$

**Expandida:**
$$
p(\mu, \alpha, \phi, \boldsymbol{\theta} | \mathbf{y}) \propto
\left[\prod_{i,t} p(y_{it} | \theta_i, \phi)\right]
\left[\prod_i p(\theta_i | \mu, \alpha)\right]
p(\mu) \cdot p(\alpha) \cdot p(\phi)
$$

### 5.2 Log-Posterior (para MCMC)

$$
\log p(\mu, \alpha, \phi, \boldsymbol{\theta} | \mathbf{y}) = \sum_{i,t} \log p(y_{it} | \theta_i, \phi) + \sum_i \log p(\theta_i | \mu, \alpha) + \log p(\mu) + \log p(\alpha) + \log p(\phi) + C
$$

### 5.3 Por qué MCMC es Necesario

La integral posterior es **intratable**:

$$p(\mu, \alpha, \phi, \boldsymbol{\theta} | \mathbf{y}) = \frac{p(\mathbf{y} | \boldsymbol{\theta}, \phi) p(\boldsymbol{\theta} | \mu, \alpha) p(\mu) p(\alpha) p(\phi)}{\int \int \int \int p(\mathbf{y} | \boldsymbol{\theta}', \phi') p(\boldsymbol{\theta}' | \mu', \alpha') p(\mu') p(\alpha') p(\phi') \, d\boldsymbol{\theta}' d\mu' d\alpha' d\phi'}$$

El denominador requiere integrar sobre espacio de $(N + 3)$ dimensiones donde $N$ es el número de entidades.

**Solución MCMC**: Muestrear del posterior en lugar de computarlo analíticamente.

---

## 5A. Teoría de Markov Chain Monte Carlo

### 5A.1 Cadenas de Markov

Una **cadena de Markov** es una secuencia $\{X_0, X_1, X_2, \ldots\}$ que satisface la propiedad de Markov:

$$P(X_{t+1} | X_t, X_{t-1}, \ldots, X_0) = P(X_{t+1} | X_t) = T(X_{t+1} | X_t)$$

**Distribución Estacionaria**: Una distribución $\pi$ es estacionaria si:

$$\pi(x') = \int T(x' | x) \pi(x) \, dx$$

**Teorema Ergódico**: Para una cadena ergódica con distribución estacionaria $\pi$:

$$\lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^{T} f(X_t) = \mathbb{E}_\pi[f(X)] = \int f(x) \pi(x) \, dx$$

**Insight Clave**: Podemos estimar expectativas bajo $\pi$ promediando sobre una sola cadena larga.

### 5A.2 Diagnósticos MCMC

**R-hat (Estadística de Gelman-Rubin)**:

Para $C$ cadenas de longitud $N$ cada una:

$$\hat{R} = \sqrt{\frac{\widehat{\text{Var}}^+(\theta)}{W}}$$

donde $W$ es varianza intra-cadena y $\widehat{\text{Var}}^+$ es estimación de varianza agrupada.

**Interpretación**:
- $\hat{R} \approx 1$: Las cadenas han convergido
- $\hat{R} > 1.01$: Cadenas explorando diferentes regiones
- $\hat{R} > 1.1$: Problemas serios de convergencia

**Tamaño de Muestra Efectivo (ESS)**:

$$\text{ESS} = \frac{MN}{1 + 2\sum_{k=1}^{\infty} \rho_k}$$

donde $\rho_k$ es autocorrelación en lag $k$.

**Usamos NUTS** (No-U-Turn Sampler), una variante adaptativa de HMC que ajusta automáticamente el tamaño de paso.

---

## 6. Scoring de Anomalías

### 6.1 Distribución Predictiva Posterior

Para una nueva observación $y^*$ de la entidad $i$:

$$p(y^* | \mathbf{y}) = \int p(y^* | \theta_i, \phi) \cdot p(\theta_i, \phi | \mathbf{y}) \, d\theta_i \, d\phi$$

**Aproximación Monte Carlo:**
$$p(y^* | \mathbf{y}) \approx \frac{1}{S} \sum_{s=1}^{S} p(y^* | \theta_i^{(s)}, \phi^{(s)})$$

donde $(\theta_i^{(s)}, \phi^{(s)})$ son muestras posteriores.

### 6.2 Definición del Score de Anomalía

$$\text{score}(y_{it}) = -\log p(y_{it} | \mathbf{y}_{-it})$$

**Interpretación:**
- El score es la "sorpresa" de observar $y_{it}$
- Score más alto = menos probable bajo el modelo
- Escala: logaritmo natural (nats)

### 6.3 Cálculo Numérico

Usando log-sum-exp para estabilidad:

$$
\log p(y | \mathbf{y}) = \log \left(\frac{1}{S} \sum_{s=1}^{S} p(y | \theta^{(s)}, \phi^{(s)})\right)
= \text{logsumexp}\left(\log p(y | \theta^{(s)}, \phi^{(s)})\right) - \log S
$$

---

## 7. Intervalos de Credibilidad

### 7.1 Intervalos Predictivos Posteriores

Para la entidad $i$, el intervalo de credibilidad $(1-\alpha)$ $[L_i, U_i]$:

$$P(L_i \leq y^*_i \leq U_i | \mathbf{y}) = 1 - \alpha$$

**Cálculo:**
1. Muestrear $\theta_i^{(s)}, \phi^{(s)}$ de la posterior
2. Generar $y^{*(s)} \sim \text{NegBinomial}(\theta_i^{(s)}, \phi^{(s)})$
3. $L_i = \text{cuantil}(y^{*(1:S)}, \alpha/2)$
4. $U_i = \text{cuantil}(y^{*(1:S)}, 1 - \alpha/2)$

### 7.2 Ancho del Intervalo como Incertidumbre

$$\text{Incertidumbre}_i = U_i - L_i$$

Entidades con:
- **Intervalos estrechos**: Predicciones confiadas (muchos datos)
- **Intervalos anchos**: Predicciones inciertas (datos escasos → más pooling)

---

## 8. Métricas de Evaluación

### 8.1 Métricas Precisión-Recall

**Precisión:**
$$\text{Precisión} = \frac{TP}{TP + FP} = P(\text{Ataque} | \text{Marcado})$$

**Recall:**
$$\text{Recall} = \frac{TP}{TP + FN} = P(\text{Marcado} | \text{Ataque})$$

**PR-AUC:**
$$\text{PR-AUC} = \int_0^1 P(r) \, dr$$

donde $P(r)$ es la precisión al nivel de recall $r$.

### 8.2 Recall@K

$$\text{Recall@K} = \frac{|\{\text{ataques en top } K\}|}{|\{\text{todos los ataques}\}|}$$

**Interpretación operacional:** Si los analistas investigan $K$ alertas por día, ¿qué fracción de ataques detectan?

### 8.3 Comparación con Baseline

**PR-AUC de baseline aleatorio:**
$$\text{PR-AUC}_{\text{aleatorio}} = \frac{|\{\text{ataques}\}|}{|\{\text{todas las observaciones}\}|} = \text{tasa de ataque}$$

Nuestro modelo debería superar significativamente este baseline.

---

## 9. Propiedades del Modelo

### 9.1 Conjugacidad

El modelo Gamma-Binomial Negativo no es completamente conjugado, pero:
- Gamma es conjugada a Gamma (para tasas jerárquicas)
- Permite pasos Gibbs eficientes si se desea

### 9.2 Intercambiabilidad

Las entidades son intercambiables a priori:
$$p(\theta_1, \ldots, \theta_N | \mu, \alpha) = p(\theta_{\pi(1)}, \ldots, \theta_{\pi(N)} | \mu, \alpha)$$

para cualquier permutación $\pi$.

### 9.3 Consistencia Posterior

A medida que aumentan los datos, la posterior se concentra en los parámetros verdaderos:
$$p(\theta_i | \mathbf{y}) \xrightarrow{n_i \to \infty} \delta_{\theta_i^*}$$

---

## 10. Extensiones

### 10.1 Componente Temporal

Agregar estructura autorregresiva:
$$\theta_{i,t} = \rho \theta_{i,t-1} + (1-\rho)\mu_i + \epsilon_{it}$$

### 10.2 Modelo Multi-Feature

Extender a conteos multivariados:
$$\mathbf{y}_{it} \sim \text{MultivariateNegBinom}(\boldsymbol{\theta}_i, \boldsymbol{\Sigma})$$

### 10.3 Aproximación Variacional

Reemplazar MCMC con inferencia variacional:
$$q^*(\boldsymbol{\theta}) = \arg\min_{q \in \mathcal{Q}} \text{KL}(q || p(\boldsymbol{\theta} | \mathbf{y}))$$

---

## Tabla Resumen

| Componente | Distribución | Parámetros |
|------------|--------------|------------|
| Media poblacional | $\mu \sim \text{Exp}(0.1)$ | $\mathbb{E}[\mu] = 10$ |
| Concentración | $\alpha \sim \text{HalfNormal}(2)$ | Controla pooling |
| Sobredispersión | $\phi \sim \text{HalfNormal}(1)$ | Controla varianza |
| Tasa de entidad | $\theta_i \sim \text{Gamma}(\mu\alpha, \alpha)$ | $\mathbb{E}[\theta_i] = \mu$ |
| Observación | $y_{it} \sim \text{NegBinom}(\theta_i, \phi)$ | $\mathbb{E}[y] = \theta_i$ |
| Score de anomalía | $-\log p(y | \text{posterior})$ | Mayor = más anómalo |

---

## Referencias

1. Gelman, A., et al. (2013). *Bayesian Data Analysis, 3rd Edition*. Capítulo 5 (Modelos Jerárquicos).

2. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler. *JMLR*.

3. Hilbe, J. M. (2011). *Negative Binomial Regression, 2nd Edition*.

4. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. *arXiv:1701.02434*.
