# Inferencia Bayesiana vs Clásica

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENFOQUE CLÁSICO (FRECUENTISTA)                        │
└─────────────────────────────────────────────────────────────────────────┘

    Datos (y)  ──────────────────────────┐
                                        ▼
                            ┌───────────────────────┐
                            │  Estimación Máxima    │
                            │  Verosimilitud (MLE)  │
                            └───────────────────────┘
                                        │
                                        ▼
                            ┌───────────────────────┐
                            │   Estimación θ̂        │
                            │   (Valor Único)       │
                            └───────────────────────┘
                                        │
                                        ▼
                            ┌───────────────────────┐
                            │  Intervalo Confianza  │
                            │   (de distribución    │
                            │    muestral)          │
                            └───────────────────────┘

    ❌ LIMITACIONES:
    • No incorpora conocimiento previo
    • Estimaciones puntuales no cuantifican incertidumbre
    • Intervalos de confianza son sobre muestreo, no parámetros
    • Problemas con datos escasos (sobreajuste)
    • Sin mecanismo de regularización


┌─────────────────────────────────────────────────────────────────────────┐
│                         ENFOQUE BAYESIANO                                │
└─────────────────────────────────────────────────────────────────────────┘

    Prior P(θ)     Datos (y)
         │              │
         └──────┬───────┘
                ▼
    ┌───────────────────────┐
    │   Teorema de Bayes    │
    │                       │
    │  P(θ|y) ∝ P(y|θ)P(θ) │
    │  ─────────────────    │
    │       P(y)            │
    └───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Posterior P(θ|y)     │
    │  (Distribución        │
    │   Completa)           │
    │      ▁▂▃▅▇▅▃▂▁       │
    │     ╱         ╲      │
    │    ╱           ╲     │
    │   ╱             ╲    │
    └───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Intervalo Creíble    │
    │  (95% de la masa      │
    │   posterior aquí)     │
    └───────────────────────┘

    ✅ VENTAJAS:
    • Incorpora conocimiento previo/restricciones
    • Cuantificación completa de incertidumbre
    • Regularización natural (priors previenen sobreajuste)
    • Funciona bien con datos escasos
    • Interpretable: "95% probabilidad θ está en [a, b]"


┌─────────────────────────────────────────────────────────────────────────┐
│                      EJEMPLO: DETECCIÓN DE ANOMALÍAS                     │
└─────────────────────────────────────────────────────────────────────────┘

CLÁSICO:
    Entidad con 5 observaciones: [2, 3, 2, 50, 3]
    → Media = 12, Desv = 21
    → Umbral = Media + 2×Desv = 54
    → Resultado: Nada marcado (50 < 54)
    ❌ ¡Sobreestima por el propio outlier!

BAYESIANO:
    Prior: θ ~ Gamma(μα, α) donde μ = 5 (media poblacional)
    Verosimilitud: y ~ NegBin(θ, φ)
    → Posterior: Regularizado hacia μ = 5
    → Media predicha ≈ 7 (reducida desde 12)
    → IC 95%: [3, 15]
    → Observación de 50 está muy fuera del IC
    ✅ ¡Correctamente marca como anomalía!


┌─────────────────────────────────────────────────────────────────────────┐
│                         DIFERENCIAS CLAVE                                │
└─────────────────────────────────────────────────────────────────────────┘

| Aspecto             | Clásico                | Bayesiano                 |
|---------------------|------------------------|---------------------------|
| **Salida**          | Estimación puntual     | Distribución completa     |
| **Incertidumbre**   | Intervalo confianza    | Intervalo creíble         |
| **Interpretación**  | Basada en muestreo     | Declaración probabilidad  |
| **Conocimiento**    | No se usa              | Via priors                |
| **Datos Escasos**   | Estimaciones poco      | Regularizado via pooling  |
|                     | confiables             |                           |
| **Computación**     | Analítica (rápida)     | MCMC (más lenta)          |
| **Caso de Uso**     | Datasets grandes       | Cualquier tamaño          |
