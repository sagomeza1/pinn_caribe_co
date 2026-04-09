# Guía para crear y optimizar archivos de instrucciones de GitHub Copilot

> **Propósito de este documento:** Servir como referencia metodológica para crear,
> evaluar y mejorar archivos de instrucciones personalizadas (`copilot-instructions.md`,
> `AGENTS.md` y `*.instructions.md`) en cualquier repositorio.
>
> ⚠️ **Este archivo NO es un archivo de instrucciones.** No debe ubicarse en `.github/`
> ni en la raíz del repositorio como `AGENTS.md`. Debe almacenarse en una ubicación
> de documentación separada (ej: `docs/`, un repositorio de referencia personal, o fuera
> del workspace) para evitar que contamine el contexto de Copilot.

---

## 1. Tipos de archivos de instrucciones disponibles

GitHub Copilot soporta múltiples tipos de instrucciones personalizadas que se combinan
entre sí (no se excluyen mutuamente):

### 1.1 Instrucciones de repositorio (`.github/copilot-instructions.md`)

- **Alcance:** Todo el repositorio. Se inyectan automáticamente en cada solicitud.
- **Ubicación:** `.github/copilot-instructions.md` (ruta exacta, no modificable).
- **Formato:** Markdown con lenguaje natural.
- **Usado por:** Copilot Chat, Copilot Code Completions (parcial), Copilot Code Review,
  Copilot Coding Agent.
- **Verificación:** Aparece en "References" debajo de cada respuesta en Chat.

### 1.2 Instrucciones por ruta (`.github/instructions/*.instructions.md`)

- **Alcance:** Archivos que coincidan con el patrón glob definido en el frontmatter.
- **Ubicación:** `.github/instructions/` (puede tener subdirectorios).
- **Formato:** Markdown con frontmatter YAML obligatorio (`applyTo`).
- **Cuándo usar:** Cuando distintas áreas del proyecto necesitan reglas diferentes
  (frontend vs backend, tests, configs, etc.).

Ejemplo de frontmatter:
```yaml
---
applyTo: "src/**/*.py"
---
```

Patrones glob útiles:
| Patrón | Efecto |
|--------|--------|
| `**/*.py` | Todos los `.py` recursivamente |
| `src/*.py` | Solo `.py` en `src/` (no subdirectorios) |
| `src/**/*.py` | Todos los `.py` en `src/` recursivamente |
| `**/*.py,**/*.pyx` | Múltiples patrones separados por coma |
| `**/tests/**/*.py` | Archivos de test en cualquier directorio `tests/` |

### 1.3 Instrucciones de agente (`AGENTS.md`)

- **Alcance:** Agentes de IA operando en el repositorio (Agent Mode en VS Code,
  Coding Agent en GitHub.com, Copilot CLI).
- **Ubicación principal:** Raíz del repositorio (`/AGENTS.md`). GitHub Copilot trata el
  `AGENTS.md` en la raíz como instrucciones **primarias**. Si se coloca en `.github/`,
  no tiene esa jerarquía preferente.
- **Múltiples ubicaciones:** Se puede tener `AGENTS.md` en subdirectorios. Copilot
  busca desde el directorio del archivo en edición hacia la raíz; el más cercano
  tiene precedencia. Útil para monorepos o proyectos con módulos independientes.
- **Formato:** Markdown, siguiendo la especificación de [openai/agents.md](https://github.com/openai/agents.md).
- **Relación con copilot-instructions.md:** Ambos se usan juntos.
- **Compatibilidad:** GitHub Copilot también lee `CLAUDE.md` y `GEMINI.md` en la raíz.

### 1.4 Instrucciones personales

- **VS Code:** Settings → buscar "copilot" → sección de instrucciones personales.
  También configurable en `settings.json` con la propiedad
  `github.copilot.chat.codeGeneration.instructions`.
- **Local (CLI/global):** `$HOME/.copilot/copilot-instructions.md`.
- **Alcance:** Todas tus interacciones con Copilot, sin importar el repositorio.

### 1.5 Instrucciones de organización

- **Alcance:** Todos los miembros de una organización de GitHub.
- **Configuración:** Administrada por el admin de la organización desde GitHub.com.
- **Requisito:** Copilot Business o Enterprise.

### Jerarquía de combinación

Todas las instrucciones se **combinan** (no se reemplazan). Si hay conflictos,
la resolución es **no determinista**. Evitar contradicciones entre archivos.

```
Org Instructions + Personal Instructions + copilot-instructions.md
  + *.instructions.md (si coincide ruta) + AGENTS.md + Prompt del usuario
```

---

## 2. Requisitos para que las instrucciones funcionen

### Checklist de activación (VS Code)

- [ ] **Setting activado:** `Settings` → buscar `instruction file` → marcar
  **"Code Generation: Use Instruction Files"** (activado por defecto).
- [ ] **Workspace abierto como carpeta:** Abrir la carpeta raíz del proyecto,
  no un archivo suelto ni un workspace multi-root sin la raíz correcta.
- [ ] **Archivo en ruta correcta:** Exactamente `.github/copilot-instructions.md`
  (respetar mayúsculas/minúsculas).
- [ ] **Verificar en References:** Después de cada respuesta en Chat, expandir
  "References" y confirmar que el archivo aparece listado.
- [ ] **VS Code actualizado:** Versión reciente de VS Code y de la extensión
  GitHub Copilot / GitHub Copilot Chat.
- [ ] **Nueva sesión de chat:** Si las instrucciones no aparecen, iniciar una
  nueva sesión de chat (`+` en el panel).

### Configuración en `.vscode/settings.json` (recomendada)

Configuración mínima:

```jsonc
{
  "github.copilot.chat.codeGeneration.useInstructionFiles": true,
  "chat.instructionFilesLocations": {
    ".github/instructions": true
  }
}
```

Configuración avanzada (patrón recomendado para proyectos con agentes IA):

```jsonc
{
  // ── Entorno Python ─────────────────────────────────────────────
  "python-envs.defaultEnvManager": "ms-python.python:conda",
  "python-envs.defaultPackageManager": "ms-python.python:conda",

  // ── Editor ─────────────────────────────────────────────────────
  "files.autoSave": "afterDelay",
  "files.autoSaveDelay": 10,
  "editor.formatOnSave": true,
  "editor.rulers": [100],
  "[python]": {
    "editor.defaultFormatter": "ms-python.python"
  },

  // ── GitHub Copilot ─────────────────────────────────────────────
  // Activa la lectura automática de archivos de instrucciones
  "github.copilot.chat.codeGeneration.useInstructionFiles": true,
  // Registra las ubicaciones de instrucciones por ruta
  "chat.instructionFilesLocations": {
    ".github/instructions": true
  },
  // Instrucciones inline + referencia a archivos principales
  "github.copilot.chat.codeGeneration.instructions": [
    // Referenciar explícitamente los dos archivos de instrucciones principales
    { "file": ".github/copilot-instructions.md" },
    { "file": "AGENTS.md" },
    // Recordatorio inline sobre restricciones críticas del entorno (redundancia ante olvidos)
    { "text": "Este proyecto se ejecuta en un devcontainer con Conda ([nombre-env]). NUNCA proponer crear entornos virtuales adicionales (venv, virtualenv, pipenv, poetry). Gestionar dependencias exclusivamente mediante environment.yml." },
    // Recordatorio de idioma y estilo de documentación
    { "text": "Siempre responder en español. Documentar toda función pública con docstring reST (:param:, :return:, :raises:). Usar prefijos emoji en logging: ✅ éxito, ❌ error, ⚠️ warning, 🔄 progreso, 📝 info." },
    // Recordatorio sobre documentación dinámica (crítico para sesiones multi-sesión)
    { "text": "La carpeta solutions/ contiene documentación dinámica (INFORME, PLAN, CHECKPOINT, REPORTE) para preservar contexto entre sesiones de agente. No se sube al remoto (.gitignore) pero es esencial para el desarrollo. Al iniciar sesiones medianas/grandes, consultar los documentos existentes en solutions/." }
  ]
}
```

**Por qué usar instrucciones inline además de los archivos:**
Las entradas `"text"` en `instructions` actúan como **recordatorios redundantes** que
se mantienen visibles incluso cuando la ventana de contexto se llena. Son especialmente
útilas para las restricciones más críticas (entorno Conda, idioma, `solutions/`).

**Differentiar proyectos visualmente (Peacock):**
En workspaces multi-proyecto con devcontainers distintos, usar la extensión
[Peacock](https://marketplace.visualstudio.com/items?itemName=johnpapa.vscode-peacock)
para asignar un color único a cada proyecto. Esto evita confusión al trabajar
en múltiples terminales o ventanas de VS Code simultáneamente.

```jsonc
// En .vscode/settings.json de cada proyecto:
"peacock.remoteColor": "#f9e64f",  // ← color único por proyecto
"workbench.colorCustomizations": { /* colores derivados */ }
```

Este archivo aplica a **todos los que abran el workspace**, garantizando
consistencia en el equipo.

---

## 3. Metodología para crear instrucciones desde cero

### Paso 1: Analizar el repositorio

Antes de escribir instrucciones, responder estas preguntas:

1. **¿Cuál es el propósito del proyecto?** (una línea).
2. **¿Qué stack tecnológico usa?** (lenguaje, frameworks, bases de datos, herramientas).
3. **¿Cuál es la estructura de directorios principal?**
4. **¿Hay un entorno de desarrollo específico?** (devcontainer, Docker, Conda, etc.)
5. **¿Qué convenciones de código existen?** (naming, formato, patrones de diseño).
6. **¿Qué errores comunes comete Copilot en este proyecto?** (anti-patterns observados).
7. **¿Cómo se ejecutan los tests?** ¿Cómo se hace build? ¿Cómo se despliega?
8. **¿Hay reglas de negocio que Copilot debe respetar siempre?**
9. **¿Hay documentación que debe mantenerse actualizada junto con el código?**

### Paso 2: Estructurar el `copilot-instructions.md`

Estructura recomendada (adaptar según proyecto):

```markdown
# Copilot Instructions — [Nombre del Proyecto]

## Proyecto
[Descripción en 2-3 líneas. Stack tecnológico.]

## Entorno de desarrollo
[Restricciones absolutas: qué NUNCA hacer respecto al entorno.]

## Arquitectura / Estructura
[Diagrama simplificado. Estructura de directorios principal.]

## Constantes y configuración
[Dónde se centralizan. Cómo se importan. Variables de entorno.]

## Anti-patterns (rechazar siempre)
[Tabla de ❌ No hacer / ✅ Hacer en su lugar.]

## Convenciones de código
[Tipos de datos, naming, logging, paths, docstrings.]

## Ejecución
[Comandos de build, test, deploy. Scripts principales.]

## Documentación
[Qué documentación actualizar al hacer cambios.]
```

### Paso 3: Crear el `AGENTS.md`

El `AGENTS.md` complementa a `copilot-instructions.md` y está orientado a
agentes autónomos. Debe incluir información más operativa:

```markdown
# AGENTS.md — [Nombre del Proyecto]

## Identidad del proyecto
[Nombre, propósito, stack, arquitectura en resumen.]

## Entorno de ejecución
[Cómo se construye, qué NO crear, restricciones del contenedor.]

## Cómo construir, testear y ejecutar
[Comandos exactos, paso a paso.]

## Estructura del repositorio
[Árbol de directorios con descripción de cada carpeta clave.]

## Reglas de codificación para agentes
[Paradigma, constantes, convenciones, anti-patterns.]

## Planificación estratégica (Fase 0 — antes de escribir código)
[Ver sección 5b.1 para detalle completo. Resumen:]
- Preguntas obligatorias al usuario para optimizar el plan.
- Revisión holística del proyecto para detectar riesgos.
- Documentar análisis de impacto en el PLAN.

## Antes de hacer commit o proponer cambios
[Checklist de validación pre-commit.]

## Lo que NUNCA debe hacer un agente
[Lista explícita de prohibiciones.]
```

### Paso 4: Crear instrucciones por ruta (opcional)

Solo si el proyecto tiene áreas con reglas claramente distintas.
Ejemplo de estructura:

```
.github/instructions/
  python-etl.instructions.md       → applyTo: "preprocessing/**/*.py"
  scripts.instructions.md          → applyTo: "scripts/**/*.py"
  documentation.instructions.md    → applyTo: "docs/**/*.md,docs/**/*.rst"
  schemas.instructions.md          → applyTo: "**/*schema*.py,schemas/**"
```

---

## 4. Principios para escribir instrucciones efectivas

### Hacer ✅

- **Ser conciso y específico.** Cada instrucción debe ser una declaración
  autocontenida. Evitar párrafos largos explicativos.
- **Usar lenguaje imperativo.** "Usar X", "Nunca hacer Y", "Siempre incluir Z".
- **Priorizar lo que Copilot suele equivocar.** Observar patrones de error
  recurrentes y crear instrucciones específicas contra ellos.
- **Incluir ejemplos concretos** cuando la regla pueda ser ambigua
  (ej: formato de import, nombre de constante, comando exacto).
- **Mantener instrucciones actualizadas.** Revisarlas al menos al cambiar stack,
  agregar dependencias o modificar la arquitectura.
- **Usar tablas de anti-patterns.** El formato `❌ No hacer / ✅ Hacer` es
  altamente efectivo para que el modelo lo interprete.
- **Incluir la estructura de directorios.** Copilot necesita saber dónde
  vive cada cosa para generar imports y paths correctos.
- **Incluir una fase de planificación estratégica.** Instruir al agente para
  que en tareas medianas/grandes: (1) formule preguntas al usuario para
  optimizar el plan antes de ejecutar, y (2) realice una revisión holística
  del proyecto completo para identificar dependencias, riesgos de ruptura y
  conflictos con la arquitectura existente. Esto previene cambios críticos
  no anticipados y mejora significativamente la tasa de éxito.

### Evitar ❌

- **No pedir que consulte recursos externos** ("revisa styleguide.md en otro repo").
  Las instrucciones deben ser autocontenidas.
- **No incluir instrucciones de estilo de respuesta** ("responde de forma amigable",
  "usa lenguaje informal"). Esto contamina y no agrega valor.
- **No imponer límites de longitud** ("responde en menos de 1000 caracteres").
  Esto degrada la calidad de las respuestas.
- **No duplicar información entre archivos.** Si algo está en `copilot-instructions.md`,
  en `AGENTS.md` puede hacer referencia o resumir, pero no copiar textualmente.
- **No hacer el archivo excesivamente largo.** Las instrucciones se envían con
  cada mensaje; un archivo muy extenso consume ventana de contexto y puede
  causar que se ignoren instrucciones al final.
- **No incluir información volátil** (versiones de paquetes que cambian seguido,
  conteos de filas que se actualizan). Esto genera desincronización.

---

## 5. Cómo mitigar que Copilot "olvide" instrucciones en sesiones largas

El problema principal reportado es que en sesiones de chat prolongadas, Copilot
tiende a ignorar las instrucciones personalizadas. Esto ocurre porque:

1. **La ventana de contexto se llena** con mensajes anteriores del chat, y las
   instrucciones (que se inyectan al inicio) pierden peso relativo.
2. **El contexto de archivos abiertos** compite con las instrucciones por espacio.

### Estrategias de mitigación

| Estrategia | Cómo aplicarla |
|------------|----------------|
| **Sesiones cortas y enfocadas** | Iniciar una nueva sesión de chat (`+`) para cada tarea distinta. No reutilizar sesiones de horas. |
| **Instrucciones concisas** | Reducir el archivo a las reglas más críticas. Mover detalles a `*.instructions.md` por ruta. |
| **Refuerzo manual selectivo** | En prompts largos, agregar "Recuerda seguir las copilot-instructions" como recordatorio puntual. |
| **Separar por ruta** | Usar `*.instructions.md` para que solo se inyecten las instrucciones relevantes al archivo actual. |
| **Verificar References** | Si las instrucciones no aparecen en References, la sesión las perdió → iniciar nueva sesión. |
| **Instrucciones personales** | Las más críticas (ej: "siempre en español", "nunca crear venv") van también en settings personales como redundancia. |

---

## 5b. Documentación dinámica para sesiones de agentes

Cuando el proyecto opera con agentes de IA (Agent Mode, Coding Agent) en sesiones de
desarrollo largas o multi-sesión, es fundamental mantener **documentación dinámica**
que preserve contexto entre sesiones y guíe el progreso.

### Problema que resuelve

Los agentes no retienen contexto entre sesiones de chat. Cada sesión nueva arranca
sin conocimiento del estado anterior del desarrollo. La documentación dinámica actúa
como **memoria persistente** que los agentes consultan al inicio de cada sesión y
actualizan al avanzar.

### Tipos de documentos dinámicos

| Tipo | Convención de nombre | Propósito | Ciclo de vida |
|------|----------------------|-----------|---------------|
| **INFORME** | `informe_*_{año}.md` | Documenta grandes bloques de cambios. Evidencia y contexto organizacional. | Se actualiza progresivamente al completar PLANes exitosos. |
| **PLAN** | `plan_*_{año}.md` | Define pasos de un bloque de desarrollo. Guía sesiones largas. | Se crea al inicio de sesiones medianas/grandes y se actualiza con traza. |
| **CHECKPOINT** | `{tipo}_{fecha}[_{hora}].md` | Preserva contexto ante pausa, fix o cierre de sesión. | Único por evento, snapshot puntual. |
| **REPORTE** | `reporte_*_{año}.md` | Presenta resultados y valor generado para directivos. Preciso, sin redundancias, centrado en aciertos. | Se crea al completar un entregable o hito mayor. Puede editarse manualmente. |

### Flujo entre tipos

```
PLAN (inicio sesión)
  → ejecución (actualizar PLAN con traza - [x] / - [ ])
    → CHECKPOINT (si pausa, fix crítico o cierre de sesión)
      → INFORME (al completar bloque exitosamente)
        → REPORTE (al completar entregable para directivos)
          → Evaluar mejoras a copilot-instructions.md / AGENTS.md
```

El flujo es cíclico: cada PLAN cerrado genera aprendizajes (anti-patterns descubiertos,
deuda técnica resuelta, nuevas reglas de codificación) que deben evaluarse como
candidatos a actualizar los archivos de instrucciones del repositorio.

### Retroalimentación hacia archivos de instrucciones

Al cerrar un PLAN exitosamente, el agente debe evaluar si la sesión produjo
hallazgos que mejoren las instrucciones permanentes del repositorio:

| Tipo de hallazgo | Destino | Ejemplo |
|-----------------|---------|---------|
| Nuevo anti-pattern descubierto | `copilot-instructions.md` §Anti-patterns | Regex `\\d` incompatible con Java regex engine |
| Regla de rendimiento validada | `copilot-instructions.md` §Rendimiento | UDFs Python causan `Py4JNetworkError` bajo carga |
| Deuda técnica resuelta | `AGENTS.md` §Deuda técnica | Función duplicada centralizada |
| Deuda técnica descubierta | `AGENTS.md` §Deuda técnica | Nuevo duplicado identificado |
| Módulo deprecado o legacy | `AGENTS.md` §Módulos legacy | Módulo reemplazado por alternativa |
| Cambio en protocolo operativo | `AGENTS.md` §Protocolo agentes | Nueva regla de flujo de trabajo |

**Regla clave:** Los cambios a instrucciones se **proponen al usuario** antes de aplicarse,
ya que afectan permanentemente el comportamiento del agente en todas las sesiones futuras.

### Protocolo operativo para agentes (estructura base)

Todo `AGENTS.md` que opere con documentación dinámica debe incluir un protocolo
con estas fases mínimas:

```markdown
### Protocolo obligatorio para agentes

#### Fase 0 — Planificación estratégica (ANTES de escribir código)

Toda tarea mediana o grande debe pasar por esta fase antes de crear un PLAN
o ejecutar cambios. Es obligatoria y no se puede omitir.

**0.1 Preguntas de optimización del plan:**
Antes de proponer un plan, el agente DEBE formular preguntas al usuario
para optimizarlo. Mínimo cubrir:
- Alcance: ¿La tarea se limita a los archivos mencionados o puede impactar
  otros módulos/dominios?
- Prioridad: ¿Qué resultado es más importante: velocidad de entrega,
  robustez, mantenibilidad o rendimiento?
- Restricciones: ¿Hay archivos, tablas o flujos que NO deben tocarse?
- Criterios de éxito: ¿Cómo se verificará que el cambio es correcto?
- Dependencias: ¿Hay tareas previas pendientes que podrían interactuar?

Si el usuario no aporta suficiente contexto, el agente debe insistir en al
menos 2 preguntas clave antes de continuar. No asumir; preguntar.

**0.2 Revisión holística del proyecto:**
Antes de confirmar el plan, el agente DEBE revisar panorámicamente el proyecto:
1. Mapear dependencias: módulos, scripts y tablas afectados directa e
   indirectamente.
2. Evaluar riesgos de ruptura: imports, contratos de datos (schemas, tipos),
   flujos entre capas, scripts CLI y sus flags.
3. Revisar deuda técnica documentada para no agravarla.
4. Validar coherencia con la arquitectura (paradigma, flujos, constantes,
   convenciones de código).
5. Si detecta riesgos, recomendar ajustes o alternativas antes de proceder.
   Nunca ejecutar un plan con riesgos identificados sin aprobación explícita.
6. Si el impacto abarca >3 archivos o >2 dominios, dividir en fases
   incrementales con validación intermedia.
7. Documentar hallazgos en una sección "## Análisis de impacto" del PLAN.

#### Fases operativas

1. **Al iniciar sesión:** Leer INFORME y PLAN existentes para contexto.
2. **Al planificar cambios:** Crear o actualizar un PLAN con pasos.
3. **Durante ejecución:** Actualizar PLAN con traza de progreso.
4. **Al completar con éxito:** Actualizar INFORME correspondiente.
5. **Al completar entregable o hito mayor:** Crear REPORTE orientado a directivos.
6. **Al pausar o hacer fix crítico:** Generar CHECKPOINT.
7. **Al modificar lógica de negocio:** Actualizar documentación permanente.
8. **Al cerrar PLAN exitoso:** Evaluar mejoras a archivos de instrucciones
   (copilot-instructions.md, AGENTS.md). Proponer cambios antes de aplicar.
```

Este protocolo asegura que:
- El agente **nunca arranca sin contexto** (lee documentos existentes).
- El agente **no ejecuta cambios a ciegas** (pregunta y revisa primero).
- Los **riesgos se detectan antes** de escribir código (revisión holística).
- El progreso **siempre queda registrado** (trazabilidad).
- Los hallazgos **retroalimentan** las instrucciones permanentes (mejora continua).

### 5b.1 Detalle de la fase de planificación estratégica

La Fase 0 es la adición más crítica al protocolo operativo. Su objetivo es
prevenir que el agente ejecute cambios que rompan el proyecto o que no se
alineen con las necesidades reales del usuario.

#### ¿Por qué es necesaria?

Los agentes de IA tienden a:
1. **Asumir alcance** — interpretan la tarea de forma literal sin considerar
   efectos secundarios en otros módulos.
2. **Ignorar dependencias cruzadas** — modifican un archivo sin verificar
   quién lo importa o lo consume.
3. **No cuestionar el plan** — ejecutan lo pedido sin evaluar si hay una
   forma mejor, más segura o más mantenible.
4. **Omitir riesgos arquitectónicos** — no validan contra convenciones,
   anti-patterns o flujos establecidos del proyecto.

#### ¿Cómo documentarlo en los archivos de instrucciones?

En `AGENTS.md`:
- Incluir la Fase 0 completa como primer bloque del protocolo obligatorio.
- Listar las preguntas mínimas que el agente debe hacer.
- Describir los pasos de la revisión holística adaptados al proyecto.
- Definir la regla de umbral (ej: >3 archivos → plan por fases).

En `copilot-instructions.md`:
- Incluir una versión compacta en las reglas de uso para agentes.
- Referenciar `AGENTS.md` para el detalle completo.
- Enfatizar que nunca se ejecuta un plan con riesgos sin aprobación.

#### Ejemplo de preguntas adaptadas a un proyecto

| Pregunta | Por qué es importante |
|----------|----------------------|
| ¿El cambio afecta solo este dominio o cruza a otros catálogos? | Evita ruptura de flujos Medallion entre catálogos |
| ¿Se debe mantener compatibilidad con scripts CLI existentes? | Previene que flags o argumentos dejen de funcionar |
| ¿Hay tablas con paridad obligatoria que puedan verse afectadas? | Detecta si el cambio rompe invariantes de datos |
| ¿Se prefiere velocidad de entrega o robustez máxima? | Calibra el nivel de validación y testing del plan |
| ¿Hay deuda técnica conocida en los archivos involucrados? | Evita agravar problemas o duplicar código legacy |

### Ubicación recomendada

Estos documentos deben vivir en una carpeta que:
- **NO se suba al repositorio remoto** (`.gitignore`) — son temporales y del entorno local.
- **SÉ accesible al workspace** — para que los agentes puedan leerlos y actualizarlos.
- Ejemplo: `solutions/`, `docs/dynamic/`, o cualquier carpeta que cumpla ambos criterios.

Los documentos se eliminan (o archivan externamente) al cumplir su propósito.

### Baselines temporales

Para refactorizaciones grandes, es útil mantener una copia del código original
en la misma carpeta dinámica (e.g. `solutions/old_scripts_*/`, `solutions/old_documents/`).
Estas rotan según la fuente o etapa en refactorización activa.

### Cómo documentar esto en los archivos de instrucciones

En `copilot-instructions.md`:
- Describir **qué** tipos de documentación dinámica existen, sus convenciones y cuándo actualizarlos.
- Incluir una tabla de eventos → acciones documentales.
- Incluir un evento de cierre de PLAN → evaluar mejoras a archivos de instrucciones.
- Incluir la fase de planificación estratégica (versión compacta) en las reglas
  de uso para agentes: preguntas obligatorias + revisión holística + documentar
  análisis de impacto.

En `AGENTS.md`:
- Incluir un **protocolo operativo** con Fase 0 (planificación estratégica) como
  primer bloque: preguntas de optimización + revisión holística + análisis de impacto.
- Luego las fases operativas: consultar al inicio, crear/actualizar durante,
  cerrar al finalizar, **retroalimentar instrucciones** al cerrar con éxito.
- Indicar que la carpeta está en `.gitignore` para que los agentes sepan que no deben
  preocuparse por commits de estos archivos.
- Mantener una sección de **deuda técnica conocida** que se actualice al resolver
  o descubrir items durante sesiones de desarrollo.

En `.vscode/settings.json`:
- Agregar una instrucción inline que refuerce la importancia de `solutions/` para
  los agentes, de modo que el contexto esté disponible incluso sin el `copilot-instructions.md`.

---

## 5c. Implementación en el workspace OIC — Nivel general

El workspace OIC es un **monorepo multi-proyecto**. Las instrucciones operan en dos niveles:

### Arquitectura de instrucciones del workspace

```
andres_campos/                          ← workspace raíz
  AGENTS.md                            ← instrucciones primarias globales para agentes
  .github/copilot-instructions.md      ← convenciones transversales (stack, entorno, anti-patterns)
  solutions/                           ← documentación dinámica del workspace raíz
    Enero/                             ← informes por período (docs contractuales)
    plan_*.md                          ← planes activos de desarrollo
    informe_*.md                       ← informes completados
  03_procesamiento_fuentes_primarias_oic/
    AGENTS.md                          ← prevalece sobre el raíz para tareas dentro de 03
    .github/copilot-instructions.md
    solutions/                         ← doc. dinámica específica del proyecto 03
  04_geo_procesamiento_fuentes_oic/
    AGENTS.md
    .github/copilot-instructions.md
    solutions/
  lake_house_oic/
    AGENTS.md
    .github/copilot-instructions.md
    .github/copilot/                   ← documentos de uso recurrente (informes, commits)
    solutions/
```

### Carpeta `solutions/` en la raíz del workspace

La carpeta `solutions/` en la raíz del workspace fue originalmente `Evidencias_2026/`.
Se renombró para alinearse con la convención de documentación dinámica usada en los proyectos.

| Característica | Detalle |
|----------------|---------|
| **En `.gitignore`** | No se sube al repositorio remoto |
| **Propósito** | Documentación dinámica del workspace (PLANes, INFORMEs inter-proyectos) |
| **Subcarpetas** | Por período (`Enero/`, `Febrero/`) para informes contractuales |
| **Scripts de apoyo** | Herramientas como `md_to_pdf.py` para generar documentos de entrega |
| **Diferencia con proyectos** | No contiene `old_*`; usa fechas en el nombre del archivo cuando es necesario |

**Regla:** La carpeta `solutions/` raíz contiene documentos que abarcan múltiples proyectos
o que son de nivel workspace (ej: informe de infraestructura, planes de migración globales).
Los documentos específicos de un proyecto van en `solutions/` del proyecto correspondiente.

### Jerarquía de prevalencia para agentes

```
Proyecto AGENTS.md
  > Workspace AGENTS.md
  > Proyecto copilot-instructions.md
  > Workspace copilot-instructions.md
  > settings.json inline instructions
  > Instrucciones personales
```

**Regla clave:** El `AGENTS.md` más cercano al archivo en edición siempre prevalece.
Ante conflicto, el agente debe aplicar el más específico y referirse al más general
solamente para lo que no esté cubierto por el específico.

### Referencia cruzada obligatoria en AGENTS.md

Todo `AGENTS.md` (raíz o de proyecto) debe incluir en sus primeras líneas
una sección de referencia explícita a `copilot-instructions.md`:

```markdown
## ⚠️ Instrucciones Copilot — Uso Obligatorio

> **LECTURA OBLIGATORIA ANTES DE CUALQUIER ACCIÓN.**
> El agente **DEBE** leer y aplicar siempre `.github/copilot-instructions.md`.
> Ese archivo define las convenciones transversales de código, anti-patterns,
> restricciones de entorno y arquitectura del workspace.
> **Ambos archivos aplican simultáneamente. Este `AGENTS.md` prevalece ante conflicto.**
```

Esta sección se incluye al inicio de todos los `AGENTS.md` del workspace OIC, inmediatamente
después del bloque introductorio y antes de la sección de Identidad.

---

## 5d. Implementación por proyecto — Workspace OIC

Cada proyecto del workspace configura sus instrucciones de forma autónoma.
A continuación, las características específicas de cada uno.

### Proyecto `03_procesamiento_fuentes_primarias_oic`

| Aspecto | Detalle |
|---------|----------|
| **Stack** | Python 3.12 · PySpark · Delta Lake 3.2.1 · Unity Catalog · Conda `oic_lakehouse_spark` |
| **Color Peacock** | Amarillo `#f9e64f` — devcontainer alfanumérico |
| **Arquitectura** | Medallion: RAW → Pre-Bronze → Bronze → Silver → Gold → Entregables |
| **Anti-pattern clave** | Usar `\d` en regex PySpark (Java no lo reconoce); siempre `[0-9]` |
| **Anti-pattern clave** | `df.count()` después de `saveAsTable`; usar `df.cache().count()` antes |
| **Anti-pattern clave** | `BooleanType()` para flags; usar `LongType()` con `1`/`0` |
| **Constantes** | Todas en `preprocessing/snr/config/constants.py` — nunca hardcodear |
| **Logger** | `PreproOICLogger` (singleton) de `preprocessing.common.logger` |
| **Paradigma** | Funcional puro; scripts como entry points CLI, funciones en módulos |
| **solutions/** | Contiene INFORME pipeline SNR, REPORTE entregables, baselines (`old_*`) |

**Instrucciones inline de settings.json:**
```jsonc
{ "text": "Este proyecto se ejecuta en un devcontainer con Conda (oic_lakehouse_spark). NUNCA proponer crear entornos virtuales adicionales. Gestionar dependencias exclusivamente mediante environment.yml." },
{ "text": "Siempre responder en español. Documentar toda función pública con docstring (:param, :return:, :raises:)." },
{ "text": "La carpeta solutions/ contiene documentación dinámica crítica (INFORME, PLAN, CHECKPOINT) para mantener contexto entre sesiones de Copilot. Está en .gitignore pero es esencial para el desarrollo. Al iniciar sesiones medianas/grandes, consultar los documentos existentes en solutions/." }
```

**Anti-patterns exclusivos de este proyecto:**

| ❌ Nunca hacer | ✅ Hacer en su lugar |
|----------------|---------------------|
| Hardcodear años (`2025`, `2024`) | `YEAR_MIN`/`YEAR_MAX` de `constants.py` |
| `pq.read_table()` para contar filas en >10 GB | `pq.ParquetFile(path).metadata.num_rows` |
| `multiprocessing.Pool` sin `spark.stop()` previo | `spark.stop()` libera ~17 GB JVM heap primero |
| Spark + `multiprocessing` concurrentes | Detener Spark antes de lanzar workers CPU-bound |
| `DELETE FROM` para full refresh | `DROP TABLE` + `shutil.rmtree` |

---

### Proyecto `04_geo_procesamiento_fuentes_oic`

| Aspecto | Detalle |
|---------|----------|
| **Stack** | Python 3.10 · Spark 3.5 · Sedona 1.7.2 · Delta Lake 3.3.1 · Unity Catalog · GeoPandas · Conda `sedona-spark` |
| **Color Peacock** | Verde agua `#00897b` — devcontainer geoespacial |
| **Arquitectura** | Medallion: Bronze → Silver → Gold en catálogo `oic_geo` |
| **CRS canónico** | `EPSG:4686` (MAGNA-SIRGAS) — toda geometría Silver/Gold debe estar en este CRS |
| **Regla clave** | Nunca `SparkSession.builder` directamente; siempre `utils.spark.create_spark_session()` |
| **Modo test** | Tablas con sufijo `_test_YYYYMMDD_HHMMSS`; llamar `set_test_timestamp()` UNA sola vez |
| **solutions/** | `README.md` de estado; sin baselines (proyecto más joven) |

**Instrucciones inline adicionales (específicas del stack):**
```jsonc
{ "text": "Stack: Python 3.10 + Apache Spark 3.5 + Apache Sedona 1.7.2 + Delta Lake 3.3.1 + Unity Catalog. Catálogo principal: oic_geo. CRS canónico de salida: EPSG:4686 (MAGNA-SIRGAS). Usar siempre create_spark_session() de utils.spark, nunca SparkSession.builder directamente." }
```

**Anti-patterns exclusivos de este proyecto:**

| ❌ Nunca hacer | ✅ Hacer en su lugar |
|----------------|---------------------|
| UDF Python para operaciones geométricas | Funciones nativas Sedona SQL |
| Mezclar GeoPandas y Sedona sin necesidad | Elegir el motor según escala (GeoPandas local, Sedona distribuido) |
| Hardcodear `"oic_geo"`, `"bronze"` como string | Importar de `utils.config` (`CATALOG`, `SCHEMAS`) |
| Crear tabla de test sin sufijo | `config.get_table_name(layer, test_mode=True)` |
| Levantar 03 y 04 devcontainers simultáneamente | Comparten recursos del host; solo uno activo a la vez |

---

### Proyecto `lake_house_oic`

| Aspecto | Detalle |
|---------|----------|
| **Stack** | Docker Compose · PySpark 3.5.5 · Delta Lake 3.2.1/3.3.1 · Unity Catalog 0.2.1 · Sedona 1.7.2 · MLflow 3.1.4 · Jupyter Lab |
| **Arquitectura** | Multi-contenedor: UC Server, UC UI, MLflow, FileBrowser, Jupyter PySpark, Jupyter Sedona |
| **CIFS/SMB** | `uid=100:gid=101` para UC; `uid=1000:gid=1000` para datos/Jupyter |
| **Regla crítica** | Al agregar un catálogo UC, **actualizar los 4 archivos Spark** simultáneamente |
| `.github/copilot/` | Carpeta de documentos recurrentes: commits, informes de contratista, arquitectura |
| **solutions/** | Diagramas de arquitectura, README de estado |

**Los 4 archivos Spark que deben sincronizarse al agregar un catálogo:**
1. `pyspark/startup_spark.py`
2. `apache-sedona/startup_spark_sedona.py`
3. `oic_project/src/utils/spark.py`
4. `oic_geo_project/src/utils/spark.py`

**Catálogos Unity Catalog del workspace:**

| Catálogo | Dominio |
|----------|---------|
| `oic_snr` | Transacciones inmobiliarias SNR |
| `oic_catastrales` | Registros catastrales |
| `oic_dane` | Datos demográficos DANE/DIVIPOLA |
| `oic_superservicios` | Servicios públicos |
| `oic_ofertas_web` | Ofertas web scraping |
| `oic_geo` | Datos geoespaciales |
| `actualizacion_catastral` | Actualización catastral |
| `user_client` / `user_geo_client` | Sandboxes |

**Anti-patterns exclusivos de este proyecto:**

| ❌ Nunca hacer | ✅ Hacer en su lugar |
|----------------|---------------------|
| Hardcodear IP del servidor | Usar `UNITY_CATALOG_URI` de `.env` |
| Montar UC con `uid=1000` | UC requiere `uid=100:gid=101` |
| Omitir flags CIFS de resiliencia | `hard,retrans=5,actimeo=60,closetimeo=10` |
| Editar solo 1 de 4 archivos Spark al agregar catálogo | Actualizar los 4 simultáneamente |
| Usar `localhost:8080` en contenedores bridge | `http://server:8080` con alias DNS |
| Exponer puerto 4040 en múltiples contenedores | Solo uno puede usar Spark UI a la vez |
| Mezclar versiones Delta (Maven ≠ pip) | Asegurar que JAR y pip coincidan |

**Comandos de usuario especiales (implementados en `copilot-instructions.md`):**

| Comando | Acción generada |
|---------|-----------------|
| `PREPARA MI COMMIT` | Mensaje de commit en `.github/copilot/commit_to_send.md` con emojis y timestamp |
| `PREPARA MI INFORME DE CONTRATISTA` | Informe técnico en `.github/copilot/informe_contratista.md` (analiza git log) |
| `PREPARA MI DOCUMENTO DE ACTIVIDAD CONTRACTUAL` | Documento detallado en `.github/copilot/documento_actividad_contractual.md` |

Esta funcionalidad se implementa documentando el comando en texto plano dentro
del `copilot-instructions.md` del proyecto, con las acciones exactas a ejecutar.

---

## 6. Checklist de mejora continua

Revisar periódicamente (se recomienda cada sprint, cada cambio significativo,
o al cerrar un PLAN exitoso):

**Archivos de instrucciones:**
- [ ] ¿Las instrucciones reflejan el stack actual del proyecto?
- [ ] ¿Hay anti-patterns nuevos observados que no están documentados?
- [ ] ¿La estructura de directorios del archivo coincide con la real?
- [ ] ¿Hay instrucciones que Copilot ignora consistentemente? → Reformularlas
      con lenguaje más directo o moverlas más arriba en el archivo.
- [ ] ¿El archivo es demasiado largo? → Considerar mover secciones a
      instrucciones por ruta.
- [ ] ¿Hay contradicciones entre `copilot-instructions.md` y `AGENTS.md`?
- [ ] ¿Todos los `AGENTS.md` tienen la sección `⚠️ Instrucciones Copilot — Uso Obligatorio`?
- [ ] ¿Se agregaron nuevas dependencias, tablas o módulos que deben reflejarse?

**Documentación dinámica:**
- [ ] ¿Se resolvió deuda técnica que sigue listada como pendiente en `AGENTS.md`?
- [ ] ¿Se descubrieron nuevas reglas de rendimiento o compatibilidad durante la sesión?
- [ ] ¿El protocolo operativo de agentes refleja el flujo real de trabajo?
- [ ] ¿Los documentos de `solutions/` cubren el estado actual del proyecto?
- [ ] ¿Hay PLANes abiertos que deberían cerrarse como INFORME?

**VS Code settings:**
- [ ] ¿Las instrucciones inline del `settings.json` reflejan el entorno Conda correcto?
- [ ] ¿La referencia a `solutions/` en las instrucciones inline sigue siendo válida?
- [ ] ¿El color Peacock diferencia correctamente este proyecto de los demás?

**Workspace multi-proyecto:**
- [ ] ¿Los `AGENTS.md` de los proyectos referencian correctamente `copilot-instructions.md`?
- [ ] ¿La carpeta `solutions/` raíz está en `.gitignore` del workspace?
- [ ] ¿Al agregar un catálogo UC se actualizaron los 4 archivos Spark en `lake_house_oic`?
- [ ] ¿La jerarquía de prevalencia se mantiene (proyecto > workspace)?

---

## 7. Prompt sugerido para que Copilot genere/mejore instrucciones

Cuando quieras que Copilot te ayude a crear o mejorar archivos de instrucciones
para un repositorio, puedes usar este prompt como punto de partida (adjuntando
este archivo como contexto):

```
Analiza la estructura de este repositorio, el stack tecnológico, las convenciones
de código existentes y los patrones de diseño utilizados. Luego genera:

1. Un archivo `.github/copilot-instructions.md` optimizado con:
   - Descripción del proyecto y stack
   - Restricciones del entorno de desarrollo
   - Estructura de directorios
   - Constantes y configuración centralizada
   - Anti-patterns específicos del proyecto
   - Convenciones de código
   - Comandos de ejecución y validación
   - Reglas de documentación

2. Un archivo `AGENTS.md` en la raíz con:
   - Identidad del proyecto
   - Entorno de ejecución y restricciones
   - Comandos de build, test y ejecución
   - Estructura del repositorio
   - Reglas de codificación para agentes
   - Checklist pre-commit
   - Lista de prohibiciones

Las instrucciones deben ser concisas, en lenguaje imperativo, y priorizadas
por importancia (lo más crítico primero). Usar tablas de anti-patterns
con formato ❌/✅.
```

---

## 8. Referencias oficiales

- [Adding repository custom instructions](https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions)
- [Adding personal custom instructions](https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-personal-instructions)
- [About customizing GitHub Copilot responses](https://docs.github.com/en/copilot/concepts/prompting/response-customization)
- [Support for different types of custom instructions](https://docs.github.com/en/copilot/reference/custom-instructions-support)
- [Adding organization custom instructions](https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-organization-instructions)
- [OpenAI AGENTS.md specification](https://github.com/openai/agents.md)
- [Copilot customization cheat sheet](https://docs.github.com/en/copilot/reference/customization-cheat-sheet)
- [About custom agents](https://docs.github.com/en/copilot/concepts/agents/coding-agent/about-custom-agents)