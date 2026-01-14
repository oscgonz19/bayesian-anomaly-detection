# Documentación - Español

## Detección Bayesiana de Anomalías de Seguridad (BSAD)

Bienvenido a la documentación de BSAD. Esta guía cubre todo desde los fundamentos teóricos hasta la implementación práctica.

---

## Navegación Rápida

### Primeros Pasos
- **[Tutorial](06_tutorial.md)** - Guía paso a paso para usar BSAD

### Entendiendo el Sistema
- **[Visión General](01_vision_general.md)** - Introducción al sistema y arquitectura
- **[Fundamentos Teóricos](02_fundamentos_teoricos.md)** - Estadística Bayesiana, MCMC, modelos jerárquicos
- **[Arquitectura del Modelo](03_arquitectura_modelo.md)** - Especificación detallada del modelo

### Implementación
- **[Guía de Implementación](04_guia_implementacion.md)** - Recorrido por el código
- **[Referencia API](05_referencia_api.md)** - Documentación completa de funciones

---

## Resumen de Documentos

| Documento | Contenido | Audiencia |
|-----------|-----------|-----------|
| Visión General | Planteamiento del problema, arquitectura, componentes | Todos los usuarios |
| Fundamentos Teóricos | Inferencia Bayesiana, MCMC, distribución Binomial Negativa | Científicos de datos, investigadores |
| Arquitectura del Modelo | Especificación matemática, implementación PyMC | Ingenieros ML |
| Guía de Implementación | Generación de datos, ingeniería de características, puntuación | Desarrolladores |
| Referencia API | Firmas de funciones, parámetros, ejemplos | Desarrolladores |
| Tutorial | Instalación, inicio rápido, solución de problemas | Todos los usuarios |

---

## Orden de Lectura

### Para Nuevos Usuarios
1. [Tutorial](06_tutorial.md) - Comenzar rápidamente
2. [Visión General](01_vision_general.md) - Entender qué hace el sistema
3. [Referencia API](05_referencia_api.md) - Aprender el CLI y funciones

### Para Científicos de Datos
1. [Fundamentos Teóricos](02_fundamentos_teoricos.md) - Entender las matemáticas
2. [Arquitectura del Modelo](03_arquitectura_modelo.md) - Entender el modelo
3. [Guía de Implementación](04_guia_implementacion.md) - Entender el código

### Para Profesionales de Seguridad
1. [Visión General](01_vision_general.md) - Planteamiento del problema y enfoque
2. [Tutorial](06_tutorial.md) - Ejecutar la demo
3. [Guía de Implementación](04_guia_implementacion.md) - Sección de patrones de ataque

---

## Conceptos Clave

### Enfoque Bayesiano
- Las distribuciones prior codifican suposiciones
- El posterior combina prior con datos observados
- La incertidumbre se cuantifica, no se oculta

### Modelo Jerárquico
- El pooling parcial comparte información entre entidades
- Entidades con datos escasos toman prestada fuerza de la población
- Entidades ricas en datos siguen sus propios patrones

### Puntuación de Anomalías
- Puntuación = -log p(observación | modelo)
- Mayor puntuación = menos probable = más anómalo
- La incertidumbre en puntuaciones refleja confianza del modelo

---

## Contribuir

¿Encontraste un error o quieres mejorar la documentación?
Por favor envía un issue o pull request en GitHub.
