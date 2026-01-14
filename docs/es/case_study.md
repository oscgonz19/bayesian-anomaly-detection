# Caso de Estudio: Detección Bayesiana de Anomalías en Logs de Seguridad

## Resumen del Proyecto

**Nombre del Proyecto:** Bayesian Security Anomaly Detection (BSAD)
**Dominio:** Ciberseguridad / Machine Learning
**Técnicas:** Inferencia Bayesiana, Modelado Jerárquico, MCMC
**Stack:** Python, PyMC, Pandas, ArviZ, Typer CLI

---

## El Problema

Los equipos de seguridad en organizaciones de todos los tamaños enfrentan un desafío abrumador: **identificar actividad maliciosa oculta entre millones de eventos legítimos**. Los sistemas tradicionales basados en reglas y umbrales simples fallan en la práctica:

| Desafío | Impacto |
|---------|---------|
| **Altas tasas de falsos positivos** | Fatiga de alertas; analistas ignoran advertencias |
| **Sin cuantificación de incertidumbre** | No se pueden priorizar investigaciones |
| **Umbrales genéricos** | No detectan ataques dirigidos a usuarios de baja actividad |
| **Pobre generalización** | Sobreajuste en entidades con historial escaso |

Considera un escenario real: Un usuario que típicamente genera 10 eventos de login por hora de repente genera 50. ¿Es un ataque? Con un umbral fijo de 100, esta anomalía pasa desapercibida. Pero para este usuario específico, 50 eventos representa un aumento de 5x—altamente sospechoso.

---

## La Solución

BSAD implementa un **enfoque Bayesiano jerárquico** que aborda estos desafíos mediante modelado probabilístico principiado:

### Innovación Clave: Pooling Parcial

En lugar de tratar cada usuario independientemente (separación completa) o aplicar umbrales globales (pooling completo), usamos **pooling parcial**:

```
Comportamiento específico de entidad ← Aprendido de historial individual
                                    ← Regularizado por estadísticas poblacionales
```

Esto significa:
- **Usuarios de alta actividad** obtienen líneas base personalizadas de sus propios datos
- **Usuarios de baja actividad** toman fuerza de la población, evitando sobreajuste
- **El modelo decide automáticamente** cuánto combinar basándose en disponibilidad de datos

### Puntuación Consciente de Incertidumbre

Nuestros scores de anomalía no son números arbitrarios—se derivan de **probabilidades predictivas posteriores**:

```
anomaly_score = -log P(conteo_observado | posterior)
```

Un score de 8.5 significa "esta observación es extremadamente improbable dado todo lo que sabemos sobre esta entidad y la población." La base probabilística nos permite:

- Proporcionar intervalos de credibilidad junto con estimaciones puntuales
- Cuantificar la confianza del modelo para cada predicción
- Tomar decisiones principiadas bajo incertidumbre

---

## Enfoque Técnico

### Arquitectura del Modelo

```
Modelo Jerárquico Binomial Negativo
===================================

Nivel Poblacional:
  μ ~ Exponential(0.1)      # Tasa media poblacional
  α ~ HalfNormal(2)         # Parámetro de concentración

Nivel de Entidad (pooling parcial):
  θ_entidad ~ Gamma(μ, α)   # Tasa específica de entidad

Nivel de Observación:
  y_conteo ~ NegBinomial(θ_entidad, sobredispersión)
```

**¿Por qué Binomial Negativo?** Los conteos de eventos de seguridad exhiben sobredispersión (varianza > media) debido a comportamiento en ráfagas. El Binomial Negativo maneja esto naturalmente, a diferencia de modelos Poisson.

### Patrones de Ataque Detectados

El sistema identifica cuatro firmas de ataque comunes:

| Patrón | Señal de Detección |
|--------|-------------------|
| **Fuerza Bruta** | Ráfaga extrema de eventos desde una fuente |
| **Credential Stuffing** | Actividad elevada en múltiples objetivos |
| **Anomalía Geográfica** | Actividad desde ubicaciones inusuales |
| **Anomalía de Dispositivo** | Aparición de nuevas huellas de dispositivo |

---

## Resultados

### Métricas de Rendimiento

En datos sintéticos con 2% de tasa de ataque:

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **PR-AUC** | 0.847 | Fuerte balance precisión-recall |
| **Recall@50** | 0.412 | 41% de ataques en top 50 alertas |
| **Recall@100** | 0.623 | 62% de ataques en top 100 alertas |

### ¿Por Qué Estas Métricas?

Para eventos raros como ataques de seguridad, **PR-AUC es más significativo que ROC-AUC**:

- ROC-AUC puede ser engañosamente alto (0.95+) incluso con pobre precisión
- PR-AUC refleja directamente el balance precisión/recall que enfrentan los analistas
- Recall@K responde la pregunta operacional: "¿Cuántos ataques detectaré si investigo K alertas por día?"

---

## Aspectos Destacados de Implementación

### Arquitectura de Pipeline Limpia

El código prioriza **simplicidad y legibilidad**:

```
src/bsad/
├── config.py      # Toda la configuración en un dataclass
├── io.py          # Helpers de I/O de archivos
├── steps.py       # Funciones puras para cada paso
├── pipeline.py    # Orquestación (el ÚNICO coordinador)
└── cli.py         # Capa CLI delgada
```

**Principio de Diseño:** Los pasos son funciones puras que no se llaman entre sí. La clase Pipeline controla toda la orquestación.

### Reproducibilidad

Cada ejecución es reproducible:

```bash
# El mismo comando siempre produce los mismos resultados
bsad demo --seed 42 --n-entities 200 --n-days 30
```

Todo el estado aleatorio está controlado mediante semillas explícitas.

### Herramientas de Diagnóstico

Diagnósticos MCMC integrados aseguran calidad del modelo:

- **R-hat < 1.01**: Las cadenas han convergido
- **ESS > 400**: Suficientes muestras efectivas
- **Divergencias = 0**: Sin problemas numéricos

---

## Valor de Negocio

### Para Equipos de Seguridad

| Beneficio | Impacto |
|-----------|---------|
| **Reducción de fatiga de alertas** | Líneas base específicas = menos falsos positivos |
| **Investigaciones priorizadas** | Cuantificación de incertidumbre guía el enfoque |
| **Alertas explicables** | "La actividad de este usuario está 3σ sobre su normal" |

### Para Organizaciones

| Beneficio | Impacto |
|-----------|---------|
| **Mejor asignación de recursos** | Enfoque del tiempo de analistas en alertas de alta confianza |
| **Rastro de auditoría** | Scores probabilísticos proveen decisiones defendibles |
| **Adaptabilidad** | El modelo se ajusta automáticamente a cambios de comportamiento |

---

## Habilidades Demostradas

Este proyecto demuestra:

### Machine Learning y Estadística
- Inferencia Bayesiana y muestreo MCMC
- Modelado jerárquico/multinivel
- Programación probabilística con PyMC
- Diagnósticos de modelo y análisis de convergencia

### Ingeniería de Software
- Arquitectura limpia y mantenible
- Diseño de pipeline funcional puro
- CLI completo con Typer
- Experimentos reproducibles

### Conocimiento de Dominio
- Análisis de logs de eventos de seguridad
- Reconocimiento de patrones de ataque
- Métricas de evaluación para clasificación desbalanceada

### Comunicación
- Documentación técnica (¡la estás leyendo!)
- Visualización de conceptos probabilísticos complejos
- Clara separación de contenido para diferentes audiencias

---

## Pruébalo Tú Mismo

```bash
# Clonar e instalar
git clone https://github.com/yourusername/bayesian-security-anomaly-detection.git
cd bayesian-security-anomaly-detection
pip install -e ".[dev]"

# Ejecutar demo completo
bsad demo --output-dir outputs/

# Ver resultados
cat outputs/metrics.json
ls outputs/plots/
```

---

## Direcciones Futuras

| Mejora | Beneficio |
|--------|-----------|
| **Modelado temporal** | Capturar deriva de comportamiento en el tiempo |
| **Extensión multi-feature** | Incluir bytes, endpoints, duración de sesión |
| **Aprendizaje online** | Actualizaciones incrementales con nuevos datos |
| **Inferencia variacional** | Escalar a millones de entidades |

---

## Contacto

Para preguntas sobre este proyecto u oportunidades de colaboración, por favor contacta vía GitHub Issues o LinkedIn.

---

*Este caso de estudio es parte de un portafolio que demuestra machine learning aplicado a ciberseguridad.*
