# CERTAMEN 3 - OPTIMIZACIÓN COMPUTACIONAL

## 🎯 Descripción del Proyecto

Este proyecto implementa y combina dos paradigmas fundamentales de la inteligencia artificial aplicados a problemas de optimización:

1. **Satisfacción de Restricciones (CSP)** usando el algoritmo AC-3
2. **Metaheurísticas de Optimización por Enjambre** con GGO y PSO

### 📋 Problema Principal: Mezcla Publicitaria

Optimización de la distribución de anuncios en 5 medios diferentes:

- **x1**: TV tarde (0-15)
- **x2**: TV noche (0-10)
- **x3**: Diario (0-25)
- **x4**: Revista (0-4)
- **x5**: Radio (0-30)

**Restricciones presupuestarias:**

- TV: `200·x1 + 350·x2 ≤ 3,800`
- Diario+Revista: `80·x3 + 120·x4 ≤ 2,800`
- Diario+Radio: `80·x3 + 20·x5 ≤ 3,500`

**Función objetivo:** `Z = Q - C` (calidad - costo)

## 🔧 Archivos del Proyecto

### 📊 Algoritmos Implementados

| Archivo          | Descripción                          | Algoritmo                                                   |
| ---------------- | ------------------------------------ | ----------------------------------------------------------- |
| `SI.py`          | **GGO (Greylag Goose Optimization)** | Metaheurística principal con estructura de grupos de gansos |
| `PSO-KP.py`      | **PSO para Problema de Mochila**     | Particle Swarm Optimization aplicado al Knapsack Problem    |
| `AC-3.py`        | **AC-3 para Mezcla Publicitaria**    | Arc Consistency 3 para reducción de dominios                |
| `Ejemplo-AC3.py` | **AC-3 Didáctico**                   | Ejemplo básico de funcionamiento del AC-3                   |

### 📚 Documentación

| Archivo                               | Contenido                                      |
| ------------------------------------- | ---------------------------------------------- |
| `Enunciado.pdf`                       | Especificaciones del problema y requerimientos |
| `ICI514 - Rúbrica Video.pdf`          | Criterios de evaluación del proyecto           |
| `optimizacion_certamen_final-2.pdf`   | Material de examen y evaluación                |
| `1-s2.0-S0957417423026490-main-2.pdf` | Paper académico de referencia                  |

## 🚀 Uso

### Ejecutar GGO (Algoritmo Principal)

```bash
python SI.py
```

### Ejecutar PSO para Problema de Mochila

```bash
python PSO-KP.py
```

### Ejecutar AC-3 para Mezcla Publicitaria

```bash
python AC-3.py
```

### Ejecutar Ejemplo AC-3

```bash
python Ejemplo-AC3.py
```

## 📈 Resultados Esperados

### GGO (Greylag Goose Optimization)

- **Población:** 40 individuos
- **Iteraciones:** 1000
- **Grupos:** 4 grupos de 10 gansos cada uno
- **Rotación de líder:** Cada 20 iteraciones

### Ejemplo de Resultado Óptimo

```
Mejor solución x: [0, 0, 7, 3, 30]
Calidad total Q: 1560
Costo total C: 1520
Función objetivo Z: 40
```

## 🔬 Características Técnicas

### GGO Implementation

- ✅ Control de factibilidad automático
- ✅ Manejo de dominios con `keep_domain()`
- ✅ Estructura de grupos con líderes rotativos
- ✅ Operador de movimiento: `x_i ← x_i + r1·(leader_i - x_i) + r2·N(0,σ)`

### PSO Implementation

- ✅ Codificación binaria para problema de mochila
- ✅ Función sigmoide para discretización
- ✅ Control de factibilidad con reparación

### AC-3 Implementation

- ✅ Propagación de restricciones
- ✅ Reducción de dominios
- ✅ Verificación de consistencia de arcos

## 🎓 Contexto Académico

**Curso:** Optimización Computacional  
**Evaluación:** Certamen 3  
**Universidad:** ICI514

## 📊 Integración de Paradigmas

El proyecto demuestra la **sinergia entre CSP y metaheurísticas**:

1. **AC-3** reduce el espacio de búsqueda eliminando valores inconsistentes
2. **GGO/PSO** exploran eficientemente el espacio reducido
3. **Resultado:** Optimización más eficiente y robusta

---
