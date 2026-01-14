<div align="center">

# 🛡️ BSAD: Detección Bayesiana de Anomalías de Seguridad

**Detección de eventos raros en datos de conteo de seguridad usando modelado Bayesiano jerárquico**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyMC](https://img.shields.io/badge/PyMC-5.10+-orange.svg)](https://www.pymc.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![UNSW-NB15](https://img.shields.io/badge/dataset-UNSW--NB15-purple.svg)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

[Comenzar Aquí 🗺️](#-resumen-del-proyecto--navegación) •
[El Problema](#-el-problema) •
[Cuándo Usar](#-cuándo-usar-bsad) •
[Resultados](#-resultados) •
[Inicio Rápido](#-inicio-rápido)

[**🇬🇧 English Version**](README.md)

</div>

---

## 🎯 Resumen en Una Línea

**BSAD detecta ANOMALÍAS de CONTEO raras por ENTIDAD con cuantificación de incertidumbre—logrando +30 puntos PR-AUC sobre métodos clásicos en su dominio.**

---

## 🗺️ Resumen del Proyecto & Navegación

Este proyecto demuestra **cuándo y por qué** usar modelado Bayesiano jerárquico para detección de anomalías, usando datos de tráfico de red UNSW-NB15 como caso de estudio comprensivo.

### 📚 Tres Caminos de Aprendizaje

Elige tu ruta según tus necesidades:

| Camino | Comienza Aquí | Qué Aprenderás |
|--------|---------------|----------------|
| **🎓 Teoría & Práctica** | [`01_end_to_end_walkthrough.ipynb`](notebooks/01_end_to_end_walkthrough.ipynb) | Tutorial completo de BSAD: inferencia Bayesiana, MCMC, modelos jerárquicos, con datos sintéticos |
| **📊 Aplicación a Datos Reales** | [`02_unsw_nb15_real_data.ipynb`](notebooks/02_unsw_nb15_real_data.ipynb) | Transformación de UNSW-NB15 desde clasificación (64% ataques) a detección de eventos raros (1-5% ataques) |
| **⚖️ Selección de Método** | [`03_model_comparison.ipynb`](notebooks/03_model_comparison.ipynb) | Cuándo BSAD gana (+30 PR-AUC) vs cuándo métodos clásicos ganan |

### 📖 Profundizaciones

| Documento | Propósito |
|-----------|-----------|
| [`docs/assets/unsw_nb15_dataset_description.md`](docs/assets/unsw_nb15_dataset_description.md) | **¿Qué son los flujos de red?** Documentación comprensiva del dataset explicando por qué el contexto importa |
| [`docs/assets/model_comparison.md`](docs/assets/model_comparison.md) | Marco de decisión: BSAD vs Isolation Forest vs One-Class SVM vs LOF |
| [`docs/assets/posterior_predictive_scoring.md`](docs/assets/posterior_predictive_scoring.md) | Cómo funcionan las puntuaciones de BSAD: `-log P(y \| posterior)` |

### 🎯 Decisión Rápida: ¿Debo Usar BSAD?

**✅ SÍ** si tus datos tienen **TODOS** estos elementos:
- Datos de CONTEO (enteros: logins, requests, paquetes)
- Estructura de entidades (usuarios, IPs, servicios, dispositivos)
- Anomalías raras (<5% tasa de ataque)
- Sobredispersión (Varianza >> Media)

**❌ NO** si tienes:
- Features multivariadas continuas → Usa **Isolation Forest** o **One-Class SVM**
- Tasas altas de ataque (>10%) → Esto es clasificación, usa **Random Forest** o **XGBoost**
- Sin estructura de entidades → Usa detección de anomalías clásica

---

## ❌ El Problema

### No Toda la Detección de Anomalías es Igual

Hay **dos problemas fundamentalmente diferentes** que se confunden como "detección de anomalías":

| Aspecto | Clasificación (Incorrecto para BSAD) | Detección de Eventos Raros (Dominio de BSAD) |
|---------|--------------------------------------|----------------------------------------------|
| **Tasa de Ataque** | 50-70% | <5% |
| **Tipo de Datos** | Vectores de features | Datos de CONTEO |
| **Estructura** | Muestras independientes | Jerarquías de entidades |
| **Ejemplo** | Clasificación de flujos de red | Intentos de login por usuario |
| **Mejor Herramienta** | Random Forest, SVM | **BSAD** |

### La Intuición Crítica

**BSAD es un ESPECIALISTA, no un generalista.**

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ❌ CASO DE USO INCORRECTO (Clasificación)                     │
│   ────────────────────────────────────────                      │
│   Dataset: 64% ataques, 36% normal                             │
│   Problema: "¿Es este flujo malicioso?"                        │
│   Mejor Herramienta: Random Forest, XGBoost, Redes Neuronales  │
│                                                                 │
│   ✅ CASO DE USO CORRECTO (Detección de Eventos Raros)         │
│   ───────────────────────────────────────────────               │
│   Dataset: 2% ataques, 98% normal                              │
│   Problema: "¿Es inusual el conteo de actividad del usuario?"  │
│   Mejor Herramienta: BSAD (Bayesiano Jerárquico)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ Cuándo Usar BSAD

### Marco de Decisión

```
                    ┌─────────────────────────────────────┐
                    │   ¿Qué tipo de datos tienes?        │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        ┌─────────────────────┐           ┌─────────────────────┐
        │  DATOS DE CONTEO    │           │  VECTORES FEATURES  │
        │  (enteros)          │           │  (continuos)        │
        └─────────────────────┘           └─────────────────────┘
                    │                                   │
                    ▼                                   ▼
        ┌─────────────────────┐           ┌─────────────────────┐
        │  ¿Estructura de     │           │  Usa Clásicos:      │
        │   entidades?        │           │  • Isolation Forest │
        │  (usuarios, IPs)    │           │  • One-Class SVM    │
        └─────────────────────┘           │  • LOF              │
                    │                     └─────────────────────┘
          ┌────────┴────────┐
          ▼                 ▼
     ┌─────────┐      ┌─────────────┐
     │   SÍ    │      │     NO      │
     │ → BSAD  │      │ → Clásicos  │
     └─────────┘      └─────────────┘
```

### Lista de Verificación BSAD

Usa BSAD cuando se cumplan **TODOS** estos criterios:

- [x] **Datos de CONTEO**: Eventos, requests, paquetes, logins (enteros)
- [x] **Estructura de entidades**: Usuarios, IPs, servicios, dispositivos
- [x] **Anomalías RARAS**: Tasa de ataque < 5%
- [x] **Sobredispersión**: Varianza >> Media
- [x] **Necesitas incertidumbre**: Se requieren intervalos de confianza

### Casos de Uso Perfectos

| Dominio | Entidad | Variable de Conteo | ¿Perfecto para BSAD? |
|---------|---------|-------------------|----------------------|
| SOC | ID de Usuario | Intentos de login/hora | ✓ |
| Seguridad API | Endpoint | Requests/minuto | ✓ |
| Red | IP origen | Conexiones/ventana | ✓ |
| IoT | ID de Dispositivo | Mensajes/intervalo | ✓ |
| Costos Cloud | Servicio | Gasto por hora | ✓ |

---

## 📊 Caso de Estudio: UNSW-NB15

### El Dataset

**UNSW-NB15** es un dataset ampliamente usado de detección de intrusiones de red del Centro Australiano de Ciberseguridad.

> **📖 Descripción Completa del Dataset**: Ver [`docs/assets/unsw_nb15_dataset_description.md`](docs/assets/unsw_nb15_dataset_description.md) para documentación comprensiva sobre qué son los flujos de red, estructura del dataset, y por qué el contexto importa.

| Propiedad | Original | Problema |
|-----------|----------|----------|
| Registros | 257,673 flujos | |
| Tasa de Ataque | **64%** | ❌ Esto es CLASIFICACIÓN |
| Features | 49 features | ❌ No son datos de conteo nativamente |
| Entidades | Ninguna explícita | ❌ Sin jerarquía (pero implícita en `proto_service`) |

**Entendimiento Crítico**: UNSW-NB15 contiene *flujos de red*, no paquetes. Cada fila es una historia completa de comunicación entre dos máquinas. El dataset tiene estructura de entidad implícita a través de tipos de tráfico (`proto_service`), que puede ser explotada para modelado Bayesiano.

### Nuestra Transformación: Régimen de Ataques Raros

Creamos datasets apropiados de detección de anomalías mediante remuestreo:

```
Original (64% ataques)  →  Régimen de Ataques Raros
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        ├─ 1% ataques (939 muestras)
Mantener TODOS         ├─ 2% ataques (1,897 muestras)
los normales     →     └─ 5% ataques (4,894 muestras)
Submuestrear ataques
```

**Archivos Creados:**
- `data/unsw_nb15_rare_attack_1pct.parquet`
- `data/unsw_nb15_rare_attack_2pct.parquet`
- `data/unsw_nb15_rare_attack_5pct.parquet`

### Por Qué Esto Importa

| Régimen | Tasa de Ataque | Naturaleza | Rendimiento BSAD |
|---------|----------------|------------|------------------|
| Clasificación | 64% | Los ataques son NORMALES | ❌ Pobre ajuste |
| Evento Raro | 1-5% | Los ataques son ANOMALÍAS | ✅ Excelente |

---

## 🏆 Resultados

### Escenario A: Datos de Conteo con Estructura de Entidad (Dominio de BSAD)

**Configuración**: 50 entidades, 200 ventanas de tiempo, anomalías raras (1-5%)

```
📊 Resultados PR-AUC:
                      1%      2%      5%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BSAD (Bayesiano)    0.985   0.989   0.985  👑 GANADOR
Isolation Forest   0.631   0.672   0.683
One-Class SVM      0.570   0.697   0.651
LOF                0.031   0.034   0.100

📈 Ventaja de BSAD: +30 puntos PR-AUC sobre el mejor clásico
```

### Escenario B: Features Multivariadas (Dominio Clásico)

**Configuración**: UNSW-NB15 con 8 features continuas

```
📊 Resultados PR-AUC (5% tasa de ataque):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
One-Class SVM      0.052  👑 GANADOR
Isolation Forest   0.025
LOF                0.015
BSAD (Bayesiano)   0.005  (fuera de su dominio)
```

### Intuición Clave

| Escenario | Ganador | Ventaja |
|-----------|---------|---------|
| Datos de conteo + Entidades | **BSAD** | +30 pts PR-AUC |
| Features multivariadas | **Clásicos** | Mejor ajuste |

**BSAD es un especialista que domina en su dominio.**

---

## 🔬 Cómo Funciona BSAD

### El Modelo: Binomial Negativo Jerárquico

```
Nivel Poblacional:
    μ ~ Exponential(0.1)         # Media de toda la población
    α ~ HalfNormal(2.0)          # Fuerza de agrupamiento

Nivel de Entidad:
    θ_e ~ Gamma(μ·α, α)          # Tasa específica por entidad
                                 # (pooling parcial automático)

Nivel de Observación:
    φ ~ HalfNormal(5.0)          # Parámetro de sobredispersión
    y_{e,t} ~ NegBinomial(θ_e, φ)  # Conteos observados
```

### Tres Capacidades Únicas

#### 1. Líneas Base Específicas por Entidad

Los métodos clásicos usan un único límite de decisión. BSAD aprende que:
- `udp_dns` normalmente tiene 2-3 paquetes
- `tcp_http` normalmente tiene 100+ paquetes
- `tcp_smtp` normalmente tiene 40-50 paquetes

**El mismo conteo puede ser normal para una entidad pero anómalo para otra.**

#### 2. Cuantificación de Incertidumbre

```python
anomaly_score = -log P(y | posterior)
credible_interval = [percentil_5, percentil_95]
```

Los métodos clásicos dan una puntuación. BSAD da una **distribución completa**.

#### 3. Pooling Parcial (Compartir Información Inteligente)

```
Entidad con pocos datos  →  Toma prestada fuerza del promedio poblacional
Entidad con muchos datos →  Sigue su propio patrón
```

Esto previene sobreajuste en entidades con datos escasos.

---

## 🚀 Inicio Rápido

### Instalación

```bash
# Clonar
git clone git@github.com:oscgonz19/bayesian-anomaly-detection.git
cd bayesian-anomaly-detection

# Instalar
pip install -e ".[dev]"

# Verificar
python -c "from bsad import Pipeline; print('OK')"
```

### Ejecutar Demo

```bash
# Generar datos sintéticos y entrenar modelo
bsad demo --output-dir outputs/

# O con Python
from bsad import Pipeline, Settings

settings = Settings(n_entities=200, n_days=30, attack_rate=0.02)
pipeline = Pipeline(settings)
pipeline.run_all()
```

### Explorar Notebooks

**Ver la sección [📚 Tres Caminos de Aprendizaje](#-resumen-del-proyecto--navegación) arriba para guía detallada sobre qué notebook comenzar.**

| Notebook | Conceptos Clave | Output |
|----------|-----------------|--------|
| **01. Recorrido End-to-End** | Inferencia Bayesiana, MCMC, modelos jerárquicos, pooling parcial, verificaciones predictivas posteriores | Demo con datos sintéticos con teoría completa |
| **02. Datos Reales UNSW-NB15** | Regímenes estadísticos (64% → 1-5%), flujos de red, sobredispersión, estructura de entidad, transformación ataques raros | Demuestra por qué BSAD necesita configuración apropiada de detección de anomalías |
| **03. Comparación de Modelos** | Escenario A (BSAD gana), Escenario B (Clásicos ganan), cuantificación de incertidumbre, líneas base por entidad | Cara a cara: +30 PR-AUC de ventaja en dominio de BSAD |

**Outputs Visuales Creados:**
- 📊 `outputs/eda_case_study/` - 5 visualizaciones EDA comprensivas
- 📈 `outputs/rare_attack_comparison/` - Gráficos de comparación de modelos
- 🎯 Todos los resultados demuestran: **BSAD es un especialista, no un generalista**

---

## 📁 Estructura del Proyecto

```
bayesian-security-anomaly-detection/
├── src/bsad/
│   ├── config.py          # Configuración de settings
│   ├── steps.py           # Funciones puras (datos, modelo, scoring)
│   ├── pipeline.py        # Orquestación
│   ├── cli.py             # Interfaz de línea de comandos
│   └── unsw_adapter.py    # Adaptador de datos UNSW-NB15
├── notebooks/
│   ├── 01_end_to_end_walkthrough.ipynb
│   ├── 02_unsw_nb15_real_data.ipynb
│   └── 03_model_comparison.ipynb
├── data/
│   ├── unsw_nb15_rare_attack_1pct.parquet
│   ├── unsw_nb15_rare_attack_2pct.parquet
│   └── unsw_nb15_rare_attack_5pct.parquet
├── outputs/
│   ├── eda_case_study/         # Visualizaciones EDA
│   └── rare_attack_comparison/ # Resultados comparación
├── docs/
│   ├── assets/
│   │   ├── unsw_nb15_dataset_description.md
│   │   ├── model_comparison.md
│   │   └── posterior_predictive_scoring.md
│   ├── en/  # Documentación técnica en inglés
│   └── es/  # Documentación técnica en español
└── tests/
```

---

## 📚 Documentación Completa

### En Español
- **[Índice Principal](docs/es/README.md)** - Punto de entrada a toda la documentación en español
- **[Visión General](docs/es/01_vision_general.md)** - Introducción al sistema
- **[Fundamentos Teóricos](docs/es/02_fundamentos_teoricos.md)** - Estadística Bayesiana, MCMC
- **[Arquitectura del Modelo](docs/es/03_arquitectura_modelo.md)** - Especificación del modelo

### En Inglés
- **[Main Index](docs/en/README.md)** - Entry point to all English documentation
- **[Overview](docs/en/01_overview.md)** - System introduction
- **[Theoretical Foundations](docs/en/02_theoretical_foundations.md)** - Bayesian statistics, MCMC
- **[Model Architecture](docs/en/03_model_architecture.md)** - Model specification

---

## 🎓 Conceptos Clave para Recordar

### 1. BSAD es un Especialista

No uses BSAD para todo. Úsalo cuando tus datos coincidan con su dominio:
- ✅ Datos de CONTEO con estructura de ENTIDAD
- ❌ No para features multivariadas continuas

### 2. El Régimen Estadístico Importa Más que el Dataset

- UNSW-NB15 al 64% de ataques = Clasificación
- UNSW-NB15 al 1-5% de ataques = Detección de Anomalías
- **El mismo dataset, problema diferente**

### 3. El Contexto Define la Normalidad

En datos de red:
- 50 paquetes es normal para ARP
- 50 paquetes es anómalo para DNS
- 50 paquetes es irrelevante para HTTP

**Los números no tienen significado sin contexto.**

### 4. La Incertidumbre es una Feature, No un Bug

BSAD te dice:
- "Esta es anómala (puntuación alta) y estoy seguro (intervalo estrecho)"
- "Esta puede ser anómala (puntuación media) pero soy incierto (intervalo ancho)"

Los métodos clásicos solo dan la puntuación.

---

## 📖 Citación

Si usas BSAD en tu investigación, por favor cita:

```bibtex
@software{bsad2024,
  title={BSAD: Bayesian Security Anomaly Detection},
  author={González, Oscar},
  year={2024},
  url={https://github.com/oscgonz19/bayesian-anomaly-detection}
}
```

Para el dataset UNSW-NB15:

```bibtex
@inproceedings{moustafa2015unsw,
  title={UNSW-NB15: a comprehensive data set for network intrusion detection systems},
  author={Moustafa, Nour and Slay, Jill},
  booktitle={2015 Military Communications and Information Systems Conference (MilCIS)},
  pages={1--6},
  year={2015},
  organization={IEEE}
}
```

---

## 🤝 Contribuir

¿Encontraste un error o quieres mejorar el proyecto?
1. Fork el repositorio
2. Crea tu feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

---

## 🙏 Agradecimientos

- **PyMC Team** - Por el increíble framework de programación probabilística
- **ACCS UNSW** - Por el dataset UNSW-NB15
- **Comunidad de Seguridad** - Por retroalimentación y casos de uso

---

<div align="center">

**BSAD: La herramienta correcta para la detección de eventos raros**

[⬆️ Volver arriba](#️-bsad-detección-bayesiana-de-anomalías-de-seguridad)

</div>
