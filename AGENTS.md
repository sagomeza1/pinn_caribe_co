# AGENTS.md — PINN Predicción Meteorológica Caribe Colombia

## ⚠️ Instrucciones Copilot — Uso Obligatorio

> **LECTURA OBLIGATORIA ANTES DE CUALQUIER ACCIÓN.**
> El agente **DEBE** leer y aplicar siempre `.github/copilot-instructions.md`.
> Ese archivo define la identidad canónica del proyecto, anti-patterns y convenciones de código.
> **Ambos archivos aplican simultáneamente. Este `AGENTS.md` prevalece ante conflicto.**

---

## Identidad del proyecto

- **Nombre:** Predicción de condiciones meteorológicas usando PINNs
- **Propósito:** Predecir velocidad de viento en X, velocidad de viento en Y y presión sobre una malla
  espacial en el Caribe de Colombia, usando registros de estaciones meteorológicas y restricciones
  físicas de Navier-Stokes.
- **Contexto académico:** Trabajo de grado — Maestría en Analítica de Datos, Universidad Central.
- **Stack:** Python 3.x · PyTorch · pip (`requirements.txt`)
- **Fuente de datos:** Datos Abiertos Colombia (https://www.datos.gov.co/)

### Entradas y salidas oficiales (invariables)

| Rol | Variables |
|-----|-----------|
| Entradas | Posición X, Posición Y, Registro temporal (t) |
| Salidas | Velocidad viento en X (u_x), Velocidad viento en Y (u_y), Presión (p) |

---

## Protocolo obligatorio para agentes

### Fase 0 — Planificación estratégica (ANTES de escribir código)

Toda tarea mediana o grande debe pasar por esta fase antes de ejecutar cambios.
**Es obligatoria y no se puede omitir.**

**0.1 Preguntas de optimización del plan:**
Antes de proponer cualquier implementación, el agente DEBE formular preguntas para cubrir:
- ¿El cambio afecta la arquitectura del modelo, el preprocesamiento o ambos?
- ¿Se mantendrá la transformación dirección/magnitud → componentes X/Y existente?
- ¿Hay restricciones en cómo se ponderan observaciones vs. pérdida física?
- ¿Cuál es el criterio de validación para considerar el cambio correcto?
- ¿Hay datos faltantes involucrados que requieran decisión explícita?

Si el usuario no aporta contexto suficiente, insistir en al menos 2 preguntas clave antes de continuar.
**No asumir; preguntar.**

**0.2 Revisión holística antes de proponer un plan:**
1. Mapear módulos afectados: `src/data/`, `src/model/`, `src/training/`, `src/evaluation/`.
2. Verificar que las entradas del modelo siguen siendo exclusivamente X, Y, t.
3. Verificar que las salidas siguen siendo exclusivamente u_x, u_y, presión.
4. Confirmar que la physics loss (Navier-Stokes) no se elimina ni se hace opcional.
5. Si el impacto abarca más de 2 módulos, proponer un plan por fases con validación intermedia.
6. Documentar hallazgos en la sección `## Análisis de impacto` del PLAN correspondiente.

### Fases operativas

| Fase | Cuándo | Acción |
|------|--------|--------|
| Inicio de sesión | Al comenzar | Leer documentos dinámicos en `solutions/` si existen |
| Planificación | Antes de escribir código | Crear o actualizar PLAN con pasos y análisis de impacto |
| Ejecución | Durante la sesión | Actualizar PLAN: `[x]` completado, `[ ]` pendiente |
| Éxito | Al completar bloque | Actualizar INFORME correspondiente |
| Pausa / fix crítico | Al interrumpir | Generar CHECKPOINT con estado actual |
| Cierre de PLAN | Al finalizar | Evaluar si hay hallazgos para actualizar `.github/copilot-instructions.md` |

> Los documentos dinámicos (PLAN, INFORME, CHECKPOINT) viven en `solutions/` (en `.gitignore`).
> El agente no debe preocuparse por commitearlos.

---

## Invariantes del proyecto — no negociables

El agente NUNCA debe proponer cambios que violen alguna de estas condiciones:

1. Siempre existe una predicción de u_x, u_y y presión a partir de X, Y y t.
2. La transformación de dirección y magnitud del viento a componentes X/Y se mantiene.
3. Las observaciones y las restricciones físicas (Navier-Stokes) se usan **conjuntamente**.
4. La malla de predicción se deriva de la posición de las estaciones meteorológicas.
5. Los datos tienen trazabilidad explícita hacia Datos Abiertos Colombia.
6. Los datos de aprendizaje están normalizados y centralizados antes del entrenamiento.
7. Los datos faltantes no se eliminan automáticamente.
8. Los datos ambiguos no se completan por inferencia implícita.
9. No hay prioridad fija entre observaciones y restricciones físicas en la función de pérdida.

---

## Lo que el agente NUNCA debe hacer

- Añadir variables de entrada distintas a X, Y, t (ni temperatura, ni humedad, ni ninguna otra).
- Añadir variables de salida distintas a u_x, u_y, presión.
- Usar dirección o magnitud del viento como salida directa sin descomposición.
- Eliminar o hacer opcional la pérdida física (physics loss) basada en Navier-Stokes.
- Redefinir, reemplazar o ignorar la fuente Datos Abiertos Colombia.
- Asumir que los datos faltantes deben eliminarse.
- Completar datos ambiguos por inferencia cuando no hay certeza.
- Definir una malla que no se derive de posiciones de estaciones meteorológicas.
- Romper la trazabilidad entre datos originales (`data/raw/`) y transformados (`data/processed/`).

---

## Convenciones de código

- **Idioma:** Comentarios, docstrings y mensajes de log siempre en español.
- **Docstrings:** Toda función pública documentada con `:param`, `:return:`, `:raises:`.
- **Constantes:** Centralizadas en módulo de configuración; nunca hardcodear valores de dominio.
- **Entorno:** Gestión de dependencias exclusivamente mediante `requirements.txt`.
- **Notebooks:** Solo para exploración/prototipado; el código de producción va en `src/`.

## Convención obligatoria de reportes visuales (`reports/`)

- Cada vez que se guarde una imagen o GIF en `reports/`, se debe crear (si no existe) una carpeta cuyo nombre relacione explícitamente la iteración (por ejemplo: `iter_03`, `epoca_2000`, `iter1_vs_iter3`).
- Todo archivo visual generado debe guardarse dentro de la carpeta de su iteración correspondiente.
- No se deben guardar imágenes o GIFs directamente en la raíz de `reports/`.
- Cada carpeta de iteración debe contener únicamente resultados asociados a esa iteración del modelo.

---

## Estructura del repositorio

```
pinn/
  data/
    raw/           ← datos originales sin modificar (Datos Abiertos Colombia)
    processed/     ← datos transformados, componentes X/Y calculadas
  src/
    data/          ← carga, preprocesamiento, descomposición viento → X/Y
    model/         ← arquitectura PINN, physics loss (Navier-Stokes)
    training/      ← bucle de entrenamiento, optimizadores, callbacks
    evaluation/    ← métricas, visualizaciones sobre la malla
  notebooks/       ← exploración y prototipado (no producción)
  docs/
    01_definicion_proyecto.md
    02_instrucciones_canonicas.md
  solutions/       ← documentación dinámica (en .gitignore)
  requirements.txt
  AGENTS.md
```

---

## Deuda técnica conocida

_(Actualizar al descubrir o resolver items durante sesiones de desarrollo)_

| Item | Descripción | Estado |
|------|-------------|--------|
| — | Sin items registrados aún | — |
